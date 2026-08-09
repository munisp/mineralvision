"""
Inversion job worker — real compute wrapping ``api.geophysics.inversion``.

The job payload fully describes a linear inverse problem: mesh geometry,
survey (observation locations, observed values, uncertainties, method type)
and regularization/solver parameters.  The worker builds the core objects
(``InversionMesh``, ``SurveyData``, ``ForwardModeler``,
``RegularizationOperator``, ``DepthWeighting``, ``GaussNewtonSolver``) and
runs the Gauss-Newton Tikhonov inversion, recording staged progress and the
final artifact (model vector, predicted data, misfit, convergence history).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict

import numpy as np
from sqlalchemy.orm import Session

from api.geophysics.inversion import (
    DepthWeighting,
    ForwardModeler,
    GaussNewtonSolver,
    InversionMesh,
    InversionParameters,
    InversionType,
    ObservationPoint,
    Point3D,
    RegularizationOperator,
    SurveyData,
)

from .models import InversionJob

SUPPORTED_TYPES = {t.value for t in InversionType}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update(session: Session, job: InversionJob, **fields) -> None:
    for k, v in fields.items():
        setattr(job, k, v)
    job.updated_at = _now()
    session.commit()


def run_inversion_compute(config: Dict[str, Any],
                          progress: Callable[[float], None] = lambda p: None,
                          ) -> Dict[str, Any]:
    """Execute the inversion described by ``config``; returns the artifact.

    Raises ValueError for invalid configs (recorded as job error by caller).
    """
    mesh_cfg = config.get("mesh") or {}
    survey_cfg = config.get("survey") or {}
    param_cfg = config.get("parameters") or {}

    n_cells = tuple(mesh_cfg.get("n_cells", ()))
    cell_sizes = tuple(mesh_cfg.get("cell_sizes", ()))
    origin = tuple(mesh_cfg.get("origin", (0.0, 0.0, 0.0)))
    if len(n_cells) != 3 or len(cell_sizes) != 3 or len(origin) != 3:
        raise ValueError("mesh requires n_cells, cell_sizes, origin (3-vectors)")
    n_total = int(np.prod(n_cells))
    if n_total > 20000:
        raise ValueError(f"mesh has {n_total} cells; job-service cap is 20000")

    survey_type = survey_cfg.get("survey_type", "gravity")
    if survey_type not in SUPPORTED_TYPES:
        raise ValueError(f"unsupported survey_type: {survey_type}")
    observations = survey_cfg.get("observations") or []
    if not observations:
        raise ValueError("survey requires at least one observation")

    progress(0.1)
    mesh = InversionMesh(
        origin=Point3D(*[float(v) for v in origin]),
        cell_sizes=tuple(float(v) for v in cell_sizes),
        n_cells=tuple(int(v) for v in n_cells),
    )
    obs = [
        ObservationPoint(
            location=Point3D(float(o["x"]), float(o["y"]), float(o["z"])),
            observed_value=float(o["value"]),
            uncertainty=max(1e-12, float(o.get("uncertainty", 1.0))),
        )
        for o in observations
    ]
    survey = SurveyData(
        name=str(survey_cfg.get("name", "job-survey")),
        survey_type=InversionType(survey_type),
        observations=obs,
        inclination=float(survey_cfg.get("inclination", 0.0)),
        declination=float(survey_cfg.get("declination", 0.0)),
        field_strength=float(survey_cfg.get("field_strength", 50000.0)),
    )

    params = InversionParameters(
        max_iterations=int(param_cfg.get("max_iterations", 30)),
        target_misfit=float(param_cfg.get("target_misfit", 1.0)),
        beta_initial=float(param_cfg.get("beta_initial", 1.0)),
        beta_cooling=float(param_cfg.get("beta_cooling", 0.5)),
        alpha_s=float(param_cfg.get("alpha_s", 1.0)),
        alpha_x=float(param_cfg.get("alpha_x", 1.0)),
        alpha_y=float(param_cfg.get("alpha_y", 1.0)),
        alpha_z=float(param_cfg.get("alpha_z", 1.0)),
        tolerance=float(param_cfg.get("tolerance", 1e-6)),
    )

    progress(0.2)
    forward = ForwardModeler(mesh, survey)
    forward.compute_sensitivity_matrix()
    progress(0.5)

    regularization = RegularizationOperator(mesh, params)
    depth_weighting = DepthWeighting(mesh, params, forward.sensitivity_matrix)
    solver = GaussNewtonSolver(forward, regularization, depth_weighting, params)

    initial = np.zeros(mesh.n_active_cells)
    reference = None
    if param_cfg.get("reference_model") is not None:
        reference = np.asarray(param_cfg["reference_model"], dtype=float)
        if len(reference) != mesh.n_active_cells:
            raise ValueError("reference_model length mismatch")

    result = solver.solve(initial, reference)
    progress(0.9)

    return {
        "success": bool(result.success),
        "message": result.message,
        "n_iterations": int(result.n_iterations),
        "data_misfit": float(result.data_misfit),
        "model_norm": float(result.model_norm),
        "objective_function": float(result.objective_function),
        "computation_time": float(result.computation_time),
        "final_model": [float(v) for v in result.final_model],
        "predicted_data": [float(v) for v in result.predicted_data],
        "convergence_history": result.convergence_history,
        "mesh": {"n_cells": list(n_cells), "cell_sizes": list(cell_sizes),
                 "origin": list(origin)},
        "survey_type": survey_type,
    }


def run_job(job_id: str, session_factory) -> None:
    """Background worker entry point: run job, persist artifact or error."""
    session = session_factory()
    try:
        job = session.get(InversionJob, job_id)
        if job is None or job.status != "pending":
            return
        _update(session, job, status="running", progress=0.05)
        try:
            config = json.loads(job.params_json)
            artifact = run_inversion_compute(
                config, progress=lambda p: _update(session, job, progress=p))
            _update(session, job, status="completed", progress=1.0,
                    result_json=json.dumps(artifact))
        except Exception as exc:  # noqa: BLE001 — recorded as job error
            _update(session, job, status="failed", error=str(exc))
    finally:
        session.close()
