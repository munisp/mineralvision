"""HTTP layer for the inversion job service (thin; see logic.py / models.py)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from .logic import run_job
from .models import InversionJob, get_session

router = APIRouter(prefix="/innovations/inversion_jobs", tags=["inversion_jobs"])


class MeshIn(BaseModel):
    origin: List[float] = Field(min_length=3, max_length=3)
    cell_sizes: List[float] = Field(min_length=3, max_length=3)
    n_cells: List[int] = Field(min_length=3, max_length=3)


class ObservationIn(BaseModel):
    x: float
    y: float
    z: float
    value: float
    uncertainty: float = Field(default=1.0, gt=0)


class SurveyIn(BaseModel):
    name: str = "job-survey"
    survey_type: str = "gravity"
    observations: List[ObservationIn]
    inclination: float = 0.0
    declination: float = 0.0
    field_strength: float = 50000.0


class ParametersIn(BaseModel):
    max_iterations: int = Field(default=30, ge=1, le=200)
    target_misfit: float = Field(default=1.0, ge=0)
    beta_initial: float = Field(default=1.0, gt=0)
    beta_cooling: float = Field(default=0.5, gt=0, le=1.0)
    alpha_s: float = Field(default=1.0, ge=0)
    alpha_x: float = Field(default=1.0, ge=0)
    alpha_y: float = Field(default=1.0, ge=0)
    alpha_z: float = Field(default=1.0, ge=0)
    tolerance: float = Field(default=1e-6, gt=0)
    reference_model: Optional[List[float]] = None


class SubmitRequest(BaseModel):
    mesh: MeshIn
    survey: SurveyIn
    parameters: ParametersIn = ParametersIn()


@router.post("/jobs", status_code=201)
def submit_job(req: SubmitRequest,
               background_tasks: BackgroundTasks,
               session: Session = Depends(get_session)) -> Dict[str, Any]:
    """Submit an inversion job; executed asynchronously in the background."""
    config = {
        "mesh": req.mesh.model_dump(),
        "survey": req.survey.model_dump(),
        "parameters": req.parameters.model_dump(),
    }
    job = InversionJob(status="pending", progress=0.0,
                       params_json=json.dumps(config))
    session.add(job)
    session.commit()
    session.refresh(job)

    # Run the worker against the same database bind as this request's
    # session so test overrides (e.g. in-memory SQLite) propagate.
    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    background_tasks.add_task(run_job, job.id, factory)
    return {"job_id": job.id, "status": job.status}


@router.get("/jobs")
def list_jobs(session: Session = Depends(get_session)) -> Dict[str, Any]:
    jobs = session.query(InversionJob).order_by(InversionJob.created_at).all()
    return {"n_jobs": len(jobs), "jobs": [j.to_dict() for j in jobs]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str,
            session: Session = Depends(get_session)) -> Dict[str, Any]:
    job = session.get(InversionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()


@router.get("/jobs/{job_id}/result")
def get_result(job_id: str,
               session: Session = Depends(get_session)) -> Dict[str, Any]:
    job = session.get(InversionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == "failed":
        raise HTTPException(status_code=409,
                            detail=f"job failed: {job.error}")
    if job.status != "completed":
        raise HTTPException(status_code=409,
                            detail=f"job not completed (status={job.status})")
    return json.loads(job.result_json)


@router.get("/jobs/{job_id}/artifact")
def download_artifact(job_id: str,
                      session: Session = Depends(get_session)) -> Response:
    """Download the inversion artifact (model + diagnostics) as JSON file."""
    job = session.get(InversionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.result_json is None:
        raise HTTPException(status_code=409, detail="no artifact available")
    return Response(
        content=job.result_json,
        media_type="application/json",
        headers={"Content-Disposition":
                 f"attachment; filename=inversion_{job_id}.json"},
    )
