"""Pure logic for commodity discovery — wraps the real gold/lithium engines.

No HTTP types here. All heavy computation is delegated to:
  - src/api/ml/gold_exploration.py    (GoldPathfinderElements, AlterationIndices, RegolithModel)
  - src/api/ml/lithium_exploration.py (LithiumPathfinderElements, BrineChemistry)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # app context (repo root on sys.path)
    from src.api.ml.gold_exploration import (
        AlterationIndices,
        AlterationType,
        GeochemSample,
        GoldDepositType,
        GoldPathfinderElements,
        RegolithModel,
    )
    from src.api.ml.lithium_exploration import (
        BrineChemistry,
        BrineSample,
        BrineType,
        LithiumDepositType,
        LithiumPathfinderElements,
        PegmatiteSample,
    )
except ImportError:  # test context (MineralVision_Final_Package/src on sys.path)
    from api.ml.gold_exploration import (
        AlterationIndices,
        AlterationType,
        GeochemSample,
        GoldDepositType,
        GoldPathfinderElements,
        RegolithModel,
    )
    from api.ml.lithium_exploration import (
        BrineChemistry,
        BrineSample,
        BrineType,
        LithiumDepositType,
        LithiumPathfinderElements,
        PegmatiteSample,
    )


# ---------------------------------------------------------------------------
# Sample builders (dict -> engine dataclass)
# ---------------------------------------------------------------------------

def build_geochem_sample(d: Dict[str, Any]) -> GeochemSample:
    return GeochemSample(
        sample_id=d["sample_id"],
        x=float(d["x"]),
        y=float(d["y"]),
        z=d.get("z"),
        sample_type=d.get("sample_type", "soil"),
        elements={k: float(v) for k, v in d.get("elements", {}).items()},
        units=d.get("units", {}),
    )


def build_pegmatite_sample(d: Dict[str, Any]) -> PegmatiteSample:
    fields = (
        "li", "cs", "rb", "ta", "nb", "sn", "be", "b", "f", "p",
        "k", "na", "al", "si", "fe",
    )
    kwargs = {f: float(d[f]) for f in fields if f in d and d[f] is not None}
    return PegmatiteSample(
        sample_id=d["sample_id"],
        x=float(d["x"]),
        y=float(d["y"]),
        z=float(d.get("z", 0.0)),
        sample_type=d.get("sample_type", "rock"),
        minerals_identified=d.get("minerals_identified", []),
        li2o_percent=float(d.get("li2o_percent", 0.0)),
        **kwargs,
    )


def build_brine_sample(d: Dict[str, Any]) -> BrineSample:
    brine_type_raw = d.get("brine_type", BrineType.CONTINENTAL_SALAR.value)
    brine_type = BrineType(brine_type_raw)
    ions = (
        "lithium", "sodium", "potassium", "magnesium", "calcium",
        "chloride", "sulfate", "bicarbonate", "boron", "tds",
    )
    kwargs = {f: float(d[f]) for f in ions if f in d and d[f] is not None}
    from datetime import datetime
    date_raw = d.get("sample_date")
    sample_date = (
        datetime.fromisoformat(date_raw) if isinstance(date_raw, str) else datetime.utcnow()
    )
    return BrineSample(
        sample_id=d["sample_id"],
        x=float(d["x"]),
        y=float(d["y"]),
        z=float(d.get("z", 0.0)),
        sample_date=sample_date,
        brine_type=brine_type,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Gold: pathfinder scoring
# ---------------------------------------------------------------------------

def score_gold_samples(
    samples: List[Dict[str, Any]],
    deposit_type: GoldDepositType,
) -> List[Dict[str, Any]]:
    """Score geochem samples with the real GoldPathfinderElements engine."""
    engine = GoldPathfinderElements(deposit_type)
    results = []
    for d in samples:
        sample = build_geochem_sample(d)
        score = engine.compute_pathfinder_score(sample, deposit_type)
        ratios = engine.compute_element_ratios(sample, deposit_type)
        results.append({
            "sample_id": sample.sample_id,
            "x": sample.x,
            "y": sample.y,
            "pathfinder_score": round(float(score), 6),
            "element_ratios": {k: round(float(v), 6) for k, v in ratios.items()},
        })
    # Rank: 1 = best
    order = sorted(range(len(results)),
                   key=lambda i: results[i]["pathfinder_score"], reverse=True)
    for rank, idx in enumerate(order, start=1):
        results[idx]["rank"] = rank
    return results


# ---------------------------------------------------------------------------
# Gold: alteration
# ---------------------------------------------------------------------------

def compute_gold_alteration(
    hyperspectral_data: Optional[List[List[List[float]]]],
    wavelengths: Optional[List[float]],
    spectral_indices: Optional[List[str]],
    geochem_samples: Optional[List[Dict[str, Any]]],
    geochem_indices: Optional[List[str]],
) -> Dict[str, Any]:
    """Spectral and/or geochemical alteration analysis via AlterationIndices."""
    engine = AlterationIndices()
    out: Dict[str, Any] = {"spectral": {}, "geochemical": {}, "classification": []}

    index_values: Dict[str, float] = {}

    if hyperspectral_data is not None:
        if wavelengths is None:
            raise ValueError("wavelengths required with hyperspectral_data")
        data = np.asarray(hyperspectral_data, dtype=float)
        wl = np.asarray(wavelengths, dtype=float)
        if data.ndim != 3:
            raise ValueError("hyperspectral_data must be a 3D array (rows, cols, bands)")
        if data.shape[2] != wl.shape[0]:
            raise ValueError("wavelengths length must match number of bands")
        names = spectral_indices or list(engine.SPECTRAL_INDICES.keys())
        spectral_out = {}
        for name in names:
            grid = engine.compute_spectral_index(data, wl, name)
            mean_val = float(np.mean(grid))
            spectral_out[name] = {
                "mean": round(mean_val, 6),
                "max": round(float(np.max(grid)), 6),
                "min": round(float(np.min(grid)), 6),
            }
        out["spectral"] = spectral_out

    if geochem_samples:
        samples = [build_geochem_sample(d) for d in geochem_samples]
        names = geochem_indices or list(engine.GEOCHEM_INDICES.keys())
        geochem_out = {}
        for name in names:
            values = engine.compute_geochem_index(samples, name)
            clean = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
            mean_val = float(np.mean(clean)) if clean else 0.0
            geochem_out[name] = {
                "per_sample": [None if (isinstance(v, float) and np.isnan(v))
                               else round(float(v), 6) for v in values],
                "mean": round(mean_val, 6),
            }
            index_values[name] = mean_val
        out["geochemical"] = geochem_out

        classified = engine.classify_alteration(index_values)
        out["classification"] = [a.value for a in classified]

    return out


# ---------------------------------------------------------------------------
# Gold: regolith
# ---------------------------------------------------------------------------

def compute_regolith(
    dem: List[List[float]],
    slope: Optional[List[List[float]]],
    curvature: Optional[List[List[float]]],
    rainfall: float,
    cell_size: float,
    drainage_distance: Optional[List[List[float]]],
) -> Dict[str, Any]:
    """Regolith thickness + classification from a DEM via RegolithModel."""
    engine = RegolithModel()
    dem_arr = np.asarray(dem, dtype=float)
    if dem_arr.ndim != 2:
        raise ValueError("dem must be a 2D array")

    if slope is not None:
        slope_arr = np.asarray(slope, dtype=float)
    else:
        # Slope in degrees from DEM gradient
        gy, gx = np.gradient(dem_arr, cell_size)
        slope_arr = np.degrees(np.arctan(np.hypot(gx, gy)))

    if curvature is not None:
        curv_arr = np.asarray(curvature, dtype=float)
    else:
        gy, gx = np.gradient(slope_arr, cell_size)
        curv_arr = np.gradient(gx, cell_size, axis=1) + np.gradient(gy, cell_size, axis=0)

    drain_arr = (np.asarray(drainage_distance, dtype=float)
                 if drainage_distance is not None else None)

    thickness = engine.estimate_regolith_thickness(dem_arr, slope_arr, rainfall=rainfall)
    classes = engine.classify_regolith(dem_arr, slope_arr, curv_arr, drain_arr)

    unique, counts = np.unique(classes, return_counts=True)
    class_counts = {str(u): int(c) for u, c in zip(unique, counts)}

    return {
        "thickness_mean_m": round(float(np.mean(thickness)), 4),
        "thickness_max_m": round(float(np.max(thickness)), 4),
        "thickness_min_m": round(float(np.min(thickness)), 4),
        "thickness_grid": np.round(thickness, 4).tolist(),
        "regolith_class_grid": classes.tolist(),
        "regolith_class_counts": class_counts,
    }


# ---------------------------------------------------------------------------
# Lithium: pegmatite scoring
# ---------------------------------------------------------------------------

def classify_zonation(fractionation_index: float) -> str:
    """Zonation class from fractionation index (LCT pegmatite evolution).

    Higher fractionation index = more evolved (lower K/Rb) = closer to the
    Li-rich core zone of a zoned LCT pegmatite.
    """
    if fractionation_index >= 80:
        return "core_zone"
    if fractionation_index >= 50:
        return "intermediate_zone"
    if fractionation_index >= 20:
        return "outer_zone"
    return "border_zone"


def score_pegmatite_samples(
    samples: List[Dict[str, Any]],
    deposit_type: LithiumDepositType,
    sample_medium: str = "rock",
) -> List[Dict[str, Any]]:
    engine = LithiumPathfinderElements(deposit_type)
    results = []
    for d in samples:
        sample = build_pegmatite_sample(d)
        score = engine.compute_pathfinder_score(sample, sample_medium)
        fi = engine.compute_fractionation_index(sample)
        k_rb = (sample.k * 10000 / sample.rb) if (sample.k > 0 and sample.rb > 0) else None
        results.append({
            "sample_id": sample.sample_id,
            "x": sample.x,
            "y": sample.y,
            "pathfinder_score": round(float(score), 6),
            "fractionation_index": round(float(fi), 6),
            "k_rb_ratio": round(float(k_rb), 4) if k_rb is not None else None,
            "zonation": classify_zonation(fi),
        })
    order = sorted(range(len(results)),
                   key=lambda i: results[i]["pathfinder_score"], reverse=True)
    for rank, idx in enumerate(order, start=1):
        results[idx]["rank"] = rank
    return results


# ---------------------------------------------------------------------------
# Lithium: brine chemistry
# ---------------------------------------------------------------------------

def analyze_brine_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chem = BrineChemistry()
    # Mg/Li interpretation needs the pathfinder engine; deposit-type independent.
    mg_engine = LithiumPathfinderElements(LithiumDepositType.BRINE_SALAR)
    results = []
    for d in samples:
        sample = build_brine_sample(d)
        brine_type = chem.classify_brine_type(sample)
        evap = chem.compute_evaporation_index(sample)
        enrichment = chem.compute_lithium_enrichment(sample)
        mg_li, quality = mg_engine.compute_mg_li_ratio(sample)
        results.append({
            "sample_id": sample.sample_id,
            "brine_classification": brine_type,
            "lithium_mg_l": sample.lithium,
            "mg_li_ratio": (round(float(mg_li), 4)
                            if np.isfinite(mg_li) else None),
            "mg_li_interpretation": quality,
            "evaporation_index": round(float(evap), 6),
            "lithium_enrichment_vs_seawater": round(float(enrichment), 4),
        })
    order = sorted(range(len(results)),
                   key=lambda i: results[i]["lithium_mg_l"], reverse=True)
    for rank, idx in enumerate(order, start=1):
        results[idx]["rank"] = rank
    return results


# ---------------------------------------------------------------------------
# Discovery workflow
# ---------------------------------------------------------------------------

def run_discovery_workflow(
    commodity: str,
    samples: List[Dict[str, Any]],
    deposit_type: Optional[str],
    cell_size: float,
) -> Dict[str, Any]:
    """Score -> normalize -> cluster into zones -> rank, with explanations."""
    commodity = commodity.lower()

    if commodity == "gold":
        dt = GoldDepositType(deposit_type) if deposit_type else GoldDepositType.OROGENIC
        engine = GoldPathfinderElements(dt)
        config = engine.get_pathfinders(dt)
        scored = []
        for d in samples:
            s = build_geochem_sample(d)
            scored.append({
                "sample_id": s.sample_id, "x": s.x, "y": s.y,
                "score": float(engine.compute_pathfinder_score(s, dt)),
                "elements": s.elements,
            })
        thresholds = config["thresholds"]
        primary = config["primary"]
    elif commodity == "lithium":
        dt = (LithiumDepositType(deposit_type) if deposit_type
              else LithiumDepositType.PEGMATITE_LCT)
        engine = LithiumPathfinderElements(dt)
        scored = []
        for d in samples:
            s = build_pegmatite_sample(d)
            scored.append({
                "sample_id": s.sample_id, "x": s.x, "y": s.y,
                "score": float(engine.compute_pathfinder_score(s, "rock")),
                "fractionation_index": float(engine.compute_fractionation_index(s)),
            })
        thresholds = {}
        primary = list(engine.pathfinders.get("primary", {}).keys())
    else:
        raise ValueError(f"unsupported commodity: {commodity}")

    if not scored:
        raise ValueError("no samples provided")

    # Normalize scores to 0..1
    raw = np.array([s["score"] for s in scored], dtype=float)
    span = float(raw.max() - raw.min())
    for i, s in enumerate(scored):
        s["normalized_score"] = float((raw[i] - raw.min()) / span) if span > 0 else 1.0

    # Cluster into zones on a regular grid of cell_size
    xs = np.array([s["x"] for s in scored])
    ys = np.array([s["y"] for s in scored])
    gx = np.floor((xs - xs.min()) / cell_size).astype(int)
    gy = np.floor((ys - ys.min()) / cell_size).astype(int)
    zones: Dict[Tuple[int, int], List[int]] = {}
    for i, key in enumerate(zip(gx.tolist(), gy.tolist())):
        zones.setdefault(key, []).append(i)

    zone_list = []
    for key, idxs in zones.items():
        zone_samples = [scored[i] for i in idxs]
        zone_score = float(np.mean([s["normalized_score"] for s in zone_samples]))
        best = max(zone_samples, key=lambda s: s["normalized_score"])
        zone_list.append({
            "zone_id": f"Z{key[0]}_{key[1]}",
            "n_samples": len(zone_samples),
            "centroid_x": round(float(np.mean([s["x"] for s in zone_samples])), 4),
            "centroid_y": round(float(np.mean([s["y"] for s in zone_samples])), 4),
            "zone_score": round(zone_score, 6),
            "best_sample_id": best["sample_id"],
        })

    zone_list.sort(key=lambda z: z["zone_score"], reverse=True)
    for rank, z in enumerate(zone_list, start=1):
        z["rank"] = rank
        z["explanation"] = _zone_explanation(
            z, commodity, scored, zones, thresholds, primary)

    return {
        "commodity": commodity,
        "deposit_type": dt.value,
        "n_samples": len(scored),
        "scored_samples": [
            {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in s.items() if k != "elements"}
            for s in scored
        ],
        "ranked_zones": zone_list,
    }


def _zone_explanation(zone, commodity, scored, zones, thresholds, primary) -> str:
    key = tuple(int(p) for p in zone["zone_id"][1:].split("_"))
    idxs = zones.get(key, [])
    if commodity == "gold" and idxs:
        enriched = []
        for i in idxs:
            elems = scored[i].get("elements", {})
            for el, thr in thresholds.items():
                if elems.get(el, 0) > thr:
                    enriched.append(f"{el}>{thr}ppm")
        drivers = ", ".join(sorted(set(enriched))[:4]) or "no pathfinders above threshold"
        return (
            f"Zone {zone['zone_id']} ranks #{zone['rank']} (score "
            f"{zone['zone_score']:.3f}) with {zone['n_samples']} sample(s); "
            f"diagnostic enrichment: {drivers}."
        )
    return (
        f"Zone {zone['zone_id']} ranks #{zone['rank']} (score "
        f"{zone['zone_score']:.3f}) on {zone['n_samples']} sample(s); "
        f"primary pathfinders: {', '.join(primary[:5])}."
    )


# ---------------------------------------------------------------------------
# Deposit type catalogue
# ---------------------------------------------------------------------------

def list_deposit_types() -> Dict[str, Any]:
    gold = []
    for dt in GoldDepositType:
        config = GoldPathfinderElements.PATHFINDERS.get(dt)
        gold.append({
            "deposit_type": dt.value,
            "diagnostic_elements": {
                "primary": config["primary"] if config else [],
                "secondary": config["secondary"] if config else [],
                "thresholds_ppm": config["thresholds"] if config else {},
            },
            "has_pathfinder_model": config is not None,
        })
    lithium = []
    for dt in LithiumDepositType:
        config = LithiumPathfinderElements.PATHFINDERS.get(dt, {})
        lithium.append({
            "deposit_type": dt.value,
            "diagnostic_elements": {
                "primary": list(config.get("primary", {}).keys()),
                "secondary": list(config.get("secondary", {}).keys()),
                "ratios": config.get("ratios", {}),
            },
        })
    return {"gold": gold, "lithium": lithium}
