"""
Journeys API Endpoints for MineralVision

FastAPI endpoints for managing and executing user journeys through the
Temporal orchestration layer.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field

from ..orchestration.journeys import (
    JourneyManifest,
    JourneyStep,
    JourneyRegistry,
    get_journey_registry,
)
from ..orchestration.temporal import (
    WorkflowManager,
    WorkflowRun,
    WorkflowStatus,
    get_workflow_manager,
)
from ..orchestration.middleware import (
    MiddlewareIntegration,
    MiddlewareStatus,
    get_middleware_integration,
    initialize_middleware,
)
from ..auth_middleware import TokenPayload, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/journeys", tags=["journeys"])


# Request/Response Models

class JourneyListResponse(BaseModel):
    """Response for listing journeys."""
    journeys: List[Dict[str, Any]]
    total: int
    categories: List[str]


class JourneyDetailResponse(BaseModel):
    """Response for journey details."""
    id: str
    name: str
    description: str
    category: str
    steps: List[Dict[str, Any]]
    ui_entry_point: str
    required_permissions: List[str]
    estimated_duration_minutes: int
    tags: List[str]


class StartJourneyRequest(BaseModel):
    """Request to start a journey."""
    journey_id: str = Field(..., description="ID of the journey to start")
    project_id: str = Field(..., description="Project context for the journey")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters")


class StartJourneyResponse(BaseModel):
    """Response after starting a journey."""
    workflow_id: str
    run_id: str
    journey_id: str
    status: str
    started_at: str


class WorkflowStatusResponse(BaseModel):
    """Response for workflow status."""
    workflow_id: str
    run_id: str
    journey_id: str
    status: str
    current_step: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    outputs: Dict[str, Any]


class ApprovalRequest(BaseModel):
    """Request to approve/reject a step."""
    step_id: str = Field(..., description="ID of the step to approve")
    approved: bool = Field(..., description="Whether to approve or reject")
    comment: Optional[str] = Field(None, description="Optional comment")


class MiddlewareStatusResponse(BaseModel):
    """Response for middleware status."""
    status: Dict[str, str]
    connected_count: int
    total_count: int


# Endpoints

@router.get("/", response_model=JourneyListResponse)
async def list_journeys(
    category: Optional[str] = Query(None, description="Filter by category"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
):
    """List all available user journeys."""
    registry = get_journey_registry()
    
    if category:
        journeys = registry.list_by_category(category)
    else:
        journeys = registry.list_all()
    
    if tag:
        journeys = [j for j in journeys if tag in j.tags]
    
    # Get unique categories
    all_journeys = registry.list_all()
    categories = list(set(j.category for j in all_journeys))
    
    return JourneyListResponse(
        journeys=[j.to_dict() for j in journeys],
        total=len(journeys),
        categories=sorted(categories),
    )


@router.get("/categories")
async def list_categories():
    """List all journey categories."""
    registry = get_journey_registry()
    journeys = registry.list_all()
    categories = {}
    
    for journey in journeys:
        if journey.category not in categories:
            categories[journey.category] = {
                "name": journey.category.replace("_", " ").title(),
                "count": 0,
                "journeys": [],
            }
        categories[journey.category]["count"] += 1
        categories[journey.category]["journeys"].append({
            "id": journey.id,
            "name": journey.name,
        })
    
    return {"categories": list(categories.values())}


@router.get("/{journey_id}", response_model=JourneyDetailResponse)
async def get_journey(journey_id: str):
    """Get details of a specific journey."""
    registry = get_journey_registry()
    journey = registry.get(journey_id)
    
    if not journey:
        raise HTTPException(status_code=404, detail=f"Journey {journey_id} not found")
    
    return JourneyDetailResponse(
        id=journey.id,
        name=journey.name,
        description=journey.description,
        category=journey.category,
        steps=[s.to_dict() for s in journey.steps],
        ui_entry_point=journey.ui_entry_point,
        required_permissions=journey.required_permissions,
        estimated_duration_minutes=journey.estimated_duration_minutes,
        tags=journey.tags,
    )


@router.post("/start", response_model=StartJourneyResponse)
async def start_journey(
    request: StartJourneyRequest,
    background_tasks: BackgroundTasks,
    user: TokenPayload = Depends(require_auth),
):
    """Start a new journey workflow."""
    registry = get_journey_registry()
    journey = registry.get(request.journey_id)
    
    if not journey:
        raise HTTPException(status_code=404, detail=f"Journey {request.journey_id} not found")
    
    # Get workflow manager
    manager = get_workflow_manager()
    
    # Initialize if needed
    if not manager.client.is_connected:
        await manager.initialize()
    
    # Bind the orchestration run to the authenticated caller for auditability.
    if not user.user_id:
        raise HTTPException(status_code=401, detail="Authenticated token does not contain a user id")

    run = await manager.start_journey(
        journey_id=request.journey_id,
        user_id=user.user_id,
        project_id=request.project_id,
        inputs=request.inputs,
    )
    
    # Publish event to Kafka
    middleware = get_middleware_integration()
    background_tasks.add_task(
        middleware.publish_kafka,
        "mineralvision.journeys.started",
        {
            "workflow_id": run.workflow_id,
            "journey_id": request.journey_id,
            "project_id": request.project_id,
            "user_id": user_id,
        },
    )
    
    return StartJourneyResponse(
        workflow_id=run.workflow_id,
        run_id=run.run_id,
        journey_id=run.journey_id,
        status=run.status.value,
        started_at=run.started_at or datetime.utcnow().isoformat(),
    )


@router.get("/runs/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get the status of a running workflow."""
    manager = get_workflow_manager()
    run = await manager.get_run(workflow_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    current_step = await manager.get_current_step(workflow_id)
    
    return WorkflowStatusResponse(
        workflow_id=run.workflow_id,
        run_id=run.run_id,
        journey_id=run.journey_id,
        status=run.status.value,
        current_step=current_step,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error=run.error,
        outputs=run.outputs,
    )


@router.get("/runs")
async def list_workflow_runs(
    journey_id: Optional[str] = Query(None, description="Filter by journey ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Maximum number of results"),
):
    """List workflow runs with optional filters."""
    manager = get_workflow_manager()
    
    status_filter = None
    if status:
        try:
            status_filter = WorkflowStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    runs = await manager.list_runs(journey_id=journey_id, status=status_filter)
    
    return {
        "runs": [
            {
                "workflow_id": r.workflow_id,
                "run_id": r.run_id,
                "journey_id": r.journey_id,
                "status": r.status.value,
                "started_at": r.started_at,
                "completed_at": r.completed_at,
            }
            for r in runs[:limit]
        ],
        "total": len(runs),
    }


@router.post("/runs/{workflow_id}/approve")
async def approve_step(
    workflow_id: str,
    request: ApprovalRequest,
    background_tasks: BackgroundTasks,
):
    """Approve or reject a step that requires human approval."""
    manager = get_workflow_manager()
    run = await manager.get_run(workflow_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    if run.status != WorkflowStatus.WAITING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Workflow is not waiting for approval (status: {run.status.value})"
        )
    
    await manager.approve_step(
        workflow_id=workflow_id,
        step_id=request.step_id,
        approved=request.approved,
        comment=request.comment,
    )
    
    # Publish event
    middleware = get_middleware_integration()
    background_tasks.add_task(
        middleware.publish_kafka,
        "mineralvision.journeys.step.approved" if request.approved else "mineralvision.journeys.step.rejected",
        {
            "workflow_id": workflow_id,
            "step_id": request.step_id,
            "approved": request.approved,
            "comment": request.comment,
        },
    )
    
    return {"status": "ok", "approved": request.approved}


@router.post("/runs/{workflow_id}/cancel")
async def cancel_workflow(
    workflow_id: str,
    background_tasks: BackgroundTasks,
):
    """Cancel a running workflow."""
    manager = get_workflow_manager()
    run = await manager.get_run(workflow_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    if run.status not in [WorkflowStatus.RUNNING, WorkflowStatus.WAITING_APPROVAL, WorkflowStatus.PENDING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel workflow with status: {run.status.value}"
        )
    
    await manager.cancel_run(workflow_id)
    
    # Publish event
    middleware = get_middleware_integration()
    background_tasks.add_task(
        middleware.publish_kafka,
        "mineralvision.journeys.cancelled",
        {"workflow_id": workflow_id},
    )
    
    return {"status": "cancelled", "workflow_id": workflow_id}


@router.get("/middleware/status", response_model=MiddlewareStatusResponse)
async def get_middleware_status():
    """Get the status of all middleware connections."""
    middleware = get_middleware_integration()
    status = middleware.get_status()
    
    connected = sum(1 for s in status.values() if s == MiddlewareStatus.CONNECTED)
    
    return MiddlewareStatusResponse(
        status={k: v.value for k, v in status.items()},
        connected_count=connected,
        total_count=len(status),
    )


@router.post("/middleware/connect")
async def connect_middleware():
    """Connect to all middleware components."""
    results = await initialize_middleware()
    
    connected = sum(1 for v in results.values() if v)
    
    return {
        "results": results,
        "connected_count": connected,
        "total_count": len(results),
    }


@router.get("/validate")
async def validate_journeys():
    """Validate that all journey steps map to existing endpoints/modules."""
    registry = get_journey_registry()
    journeys = registry.list_all()
    
    validation_results = []
    
    for journey in journeys:
        journey_result = {
            "id": journey.id,
            "name": journey.name,
            "valid": True,
            "steps": [],
        }
        
        for step in journey.steps:
            step_result = {
                "id": step.id,
                "name": step.name,
                "valid": True,
                "issues": [],
            }
            
            # Check if endpoint exists (would need OpenAPI introspection)
            if step.endpoint:
                # For now, assume endpoints are valid
                step_result["endpoint"] = step.endpoint
            
            # Check if module exists
            if step.module:
                try:
                    import importlib
                    parts = step.module.rsplit(".", 1)
                    if len(parts) == 2:
                        importlib.import_module(parts[0])
                        step_result["module"] = step.module
                except ImportError as e:
                    step_result["valid"] = False
                    step_result["issues"].append(f"Module not found: {step.module}")
                    journey_result["valid"] = False
            
            journey_result["steps"].append(step_result)
        
        validation_results.append(journey_result)
    
    valid_count = sum(1 for r in validation_results if r["valid"])
    
    return {
        "valid_count": valid_count,
        "total_count": len(validation_results),
        "all_valid": valid_count == len(validation_results),
        "journeys": validation_results,
    }
