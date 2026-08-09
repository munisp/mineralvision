"""
Real minimal plant Digital Twin: state model, physics-lite simulation,
anomaly hooks.

Replaces placeholder internals with an actual deterministic model:

- **State model** — zones (stockpiles) and equipment entities with
  telemetry time-series (flow rate + grade) recorded at every dt step.
- **Physics-lite** — deterministic ODE-style update rules integrated with
  explicit Euler dt stepping:
    * mass balance:      dM/dt  = inflow - outflow              [t/h]
    * metal balance:     dQ/dt  = inflow*g_in - outflow*g_out   [t-metal/h]
    * grade:             g      = Q / M
    * equipment wear:    dw/dt  = wear_rate (efficiency decays linearly)
    * throughput:        f      = min(f_nominal, capacity * (1 - wear))
- **Anomaly hooks** — per-stream rolling z-score plus static thresholds;
  anomalies are emitted as structured events, never silently.

Determinism: no randomness anywhere. Same inputs => same outputs.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TelemetryPoint:
    """One telemetry sample."""
    time: float                # simulation time [h]
    timestamp: str             # wall-clock ISO timestamp
    flow: float                # [t/h]
    grade: float               # mass fraction of target metal [0-1]


@dataclass
class AnomalyEvent:
    """A detected anomaly on a telemetry stream."""
    stream: str
    kind: str                  # 'zscore' | 'threshold_high' | 'threshold_low'
    value: float
    score: float
    time: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream": self.stream,
            "kind": self.kind,
            "value": self.value,
            "score": self.score,
            "time": self.time,
            "message": self.message,
        }


class TelemetryStream:
    """
    Telemetry time-series for one flow stream with rolling statistics
    and anomaly detection (z-score + thresholds).
    """

    def __init__(self, name: str, history: int = 500,
                 z_threshold: float = 4.0,
                 flow_limits: Tuple[float, float] = (0.0, float("inf")),
                 grade_limits: Tuple[float, float] = (0.0, 1.0)):
        self.name = name
        self.points: Deque[TelemetryPoint] = deque(maxlen=history)
        self.z_threshold = z_threshold
        self.flow_limits = flow_limits
        self.grade_limits = grade_limits

    def record(self, point: TelemetryPoint) -> List[AnomalyEvent]:
        """Record a sample; returns any anomaly events it triggered."""
        anomalies: List[AnomalyEvent] = []

        # Threshold checks
        if not (self.flow_limits[0] <= point.flow <= self.flow_limits[1]):
            kind = "threshold_high" if point.flow > self.flow_limits[1] else "threshold_low"
            anomalies.append(AnomalyEvent(
                stream=self.name, kind=kind, value=point.flow,
                score=abs(point.flow), time=point.time,
                message=f"flow {point.flow:.3f} outside {self.flow_limits}",
            ))
        if not (self.grade_limits[0] <= point.grade <= self.grade_limits[1]):
            kind = "threshold_high" if point.grade > self.grade_limits[1] else "threshold_low"
            anomalies.append(AnomalyEvent(
                stream=self.name, kind=kind, value=point.grade,
                score=abs(point.grade), time=point.time,
                message=f"grade {point.grade:.4f} outside {self.grade_limits}",
            ))

        # Z-score check against rolling history (needs a baseline).
        # A near-zero-variance baseline still fires on real spikes: floor
        # the std at 1% of the mean magnitude.
        if len(self.points) >= 10:
            flows = np.array([p.flow for p in self.points])
            mean, std = float(flows.mean()), float(flows.std())
            std = max(std, abs(mean) * 0.01, 1e-9)
            z = (point.flow - mean) / std
            if abs(z) > self.z_threshold:
                    anomalies.append(AnomalyEvent(
                        stream=self.name, kind="zscore", value=point.flow,
                        score=float(z), time=point.time,
                        message=f"flow z-score {z:.2f} exceeds {self.z_threshold}",
                    ))

        self.points.append(point)
        return anomalies

    def series(self) -> Dict[str, List[float]]:
        return {
            "time": [p.time for p in self.points],
            "flow": [p.flow for p in self.points],
            "grade": [p.grade for p in self.points],
        }


class Zone:
    """A stockpile zone: mass + contained metal (grade = metal/mass)."""

    def __init__(self, zone_id: str, mass: float = 0.0, grade: float = 0.0):
        self.zone_id = zone_id
        self.mass = float(mass)                 # [t]
        self.metal = float(mass) * float(grade) # [t of metal]

    @property
    def grade(self) -> float:
        return self.metal / self.mass if self.mass > 1e-12 else 0.0

    def withdraw(self, amount: float) -> Tuple[float, float]:
        """Withdraw up to `amount` t; returns (mass, metal) actually removed."""
        amount = min(amount, self.mass)
        grade = self.grade
        self.mass -= amount
        self.metal -= amount * grade
        if self.mass < 1e-9:
            self.mass = 0.0
            self.metal = 0.0
        return amount, amount * grade

    def deposit(self, mass: float, metal: float) -> None:
        self.mass += mass
        self.metal += metal

    def to_dict(self) -> Dict[str, Any]:
        return {"zone_id": self.zone_id, "mass": self.mass, "grade": self.grade}


class Equipment:
    """
    Processing equipment moving material between zones.

    Throughput is capacity-limited and degrades with wear; wear grows
    linearly with operating time (deterministic).
    """

    def __init__(self, equipment_id: str, source: Zone, destination: Zone,
                 nominal_flow: float, capacity: float,
                 efficiency: float = 1.0, wear_rate: float = 0.0):
        self.equipment_id = equipment_id
        self.source = source
        self.destination = destination
        self.nominal_flow = float(nominal_flow)   # [t/h]
        self.capacity = float(capacity)           # [t/h]
        self.efficiency = float(efficiency)
        self.wear_rate = float(wear_rate)         # [1/h]
        self.wear = 0.0                           # [0-1]
        self.operating_hours = 0.0

    def current_capacity(self) -> float:
        return self.capacity * (1.0 - self.wear)

    def step(self, dt: float) -> Tuple[float, float, float]:
        """
        Advance dt hours; returns (flow [t/h], mass [t], metal [t]) moved.
        """
        flow = min(self.nominal_flow, self.current_capacity()) * self.efficiency
        flow = max(flow, 0.0)

        mass = flow * dt
        moved_mass, moved_metal = self.source.withdraw(mass)
        if mass > 1e-12 and moved_mass < mass:
            # Source depleted: scale flow to what actually moved
            flow = moved_mass / dt

        self.destination.deposit(moved_mass, moved_metal)

        self.operating_hours += dt
        self.wear = min(1.0, self.wear + self.wear_rate * dt)

        out_grade = moved_metal / moved_mass if moved_mass > 1e-12 else 0.0
        return flow, moved_mass, out_grade

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equipment_id": self.equipment_id,
            "nominal_flow": self.nominal_flow,
            "capacity": self.capacity,
            "current_capacity": self.current_capacity(),
            "efficiency": self.efficiency,
            "wear": self.wear,
            "operating_hours": self.operating_hours,
        }


class PlantTwin:
    """
    Digital twin of a processing plant: zones + equipment + telemetry,
    advanced with deterministic dt stepping. Total mass and metal in the
    system are conserved (closed loop) unless external feeds/outputs exist.
    """

    def __init__(self, start_time: datetime = None):
        self.zones: Dict[str, Zone] = {}
        self.equipment: Dict[str, Equipment] = {}
        self.streams: Dict[str, TelemetryStream] = {}
        self.anomalies: List[AnomalyEvent] = []
        self.time = 0.0  # simulation time [h]
        self._start_time = start_time or datetime(2024, 1, 1)
        # external (non-conserved) fluxes tracked explicitly
        self.external_mass_in = 0.0
        self.external_mass_out = 0.0
        self.external_metal_in = 0.0
        self.external_metal_out = 0.0

    # -- construction ---------------------------------------------------

    def add_zone(self, zone_id: str, mass: float = 0.0, grade: float = 0.0) -> Zone:
        zone = Zone(zone_id, mass, grade)
        self.zones[zone_id] = zone
        return zone

    def add_equipment(self, equipment_id: str, source_id: str, destination_id: str,
                      nominal_flow: float, capacity: float,
                      efficiency: float = 1.0, wear_rate: float = 0.0) -> Equipment:
        eq = Equipment(
            equipment_id, self.zones[source_id], self.zones[destination_id],
            nominal_flow, capacity, efficiency, wear_rate,
        )
        self.equipment[equipment_id] = eq
        self.streams.setdefault(equipment_id, TelemetryStream(equipment_id))
        return eq

    def add_external_feed(self, stream_id: str, destination_id: str,
                          flow: float, grade: float) -> None:
        """Register a constant external feed into a zone."""
        self._external_feeds = getattr(self, "_external_feeds", [])
        self._external_feeds.append((stream_id, destination_id, float(flow), float(grade)))
        self.streams.setdefault(stream_id, TelemetryStream(stream_id))

    # -- simulation -----------------------------------------------------

    def step(self, dt: float) -> Dict[str, Any]:
        """Advance the plant by dt hours (explicit Euler)."""
        timestamp = (self._start_time + timedelta(hours=self.time)).isoformat()
        new_anomalies: List[AnomalyEvent] = []

        # External feeds first
        for stream_id, dest_id, flow, grade in getattr(self, "_external_feeds", []):
            mass = flow * dt
            metal = mass * grade
            self.zones[dest_id].deposit(mass, metal)
            self.external_mass_in += mass
            self.external_metal_in += metal
            new_anomalies += self.streams[stream_id].record(
                TelemetryPoint(self.time, timestamp, flow, grade)
            )

        # Equipment transfers
        for eq_id, eq in self.equipment.items():
            flow, moved_mass, out_grade = eq.step(dt)
            new_anomalies += self.streams[eq_id].record(
                TelemetryPoint(self.time, timestamp, flow, out_grade)
            )

        self.anomalies.extend(new_anomalies)
        self.time += dt

        return {
            "time": self.time,
            "timestamp": timestamp,
            "zones": {zid: z.to_dict() for zid, z in self.zones.items()},
            "equipment": {eid: e.to_dict() for eid, e in self.equipment.items()},
            "new_anomalies": [a.to_dict() for a in new_anomalies],
        }

    def run(self, duration: float, dt: float) -> List[Dict[str, Any]]:
        """Run for `duration` hours in steps of `dt`; returns per-step states."""
        states = []
        steps = int(round(duration / dt))
        for _ in range(steps):
            states.append(self.step(dt))
        return states

    # -- accounting -----------------------------------------------------

    def total_mass(self) -> float:
        return sum(z.mass for z in self.zones.values())

    def total_metal(self) -> float:
        return sum(z.metal for z in self.zones.values())

    def mass_balance_error(self, initial_mass: float) -> float:
        """
        Conservation error: current mass vs expected
        (initial + external_in - external_out). Should be ~0.
        """
        expected = initial_mass + self.external_mass_in - self.external_mass_out
        return abs(self.total_mass() - expected)

    def metal_balance_error(self, initial_metal: float) -> float:
        expected = initial_metal + self.external_metal_in - self.external_metal_out
        return abs(self.total_metal() - expected)

    def telemetry(self, stream_id: str) -> Dict[str, List[float]]:
        return self.streams[stream_id].series()

    def get_anomalies(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.anomalies]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "zones": {zid: z.to_dict() for zid, z in self.zones.items()},
            "equipment": {eid: e.to_dict() for eid, e in self.equipment.items()},
            "anomaly_count": len(self.anomalies),
            "total_mass": self.total_mass(),
            "total_metal": self.total_metal(),
        }


def create_plant_twin(start_time: datetime = None) -> PlantTwin:
    """Factory function to create a PlantTwin."""
    return PlantTwin(start_time)
