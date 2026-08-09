"""
Temporal Workflow Integration for MineralVision

Provides Temporal client, workflow definitions, and activity implementations
for orchestrating user journeys across the platform.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# Temporal SDK imports (with fallback for when SDK not installed)
try:
    from temporalio import workflow, activity
    from temporalio.client import Client as TemporalClientSDK
    from temporalio.worker import Worker
    from temporalio.common import RetryPolicy
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    logger.warning("Temporal SDK not installed. Using mock implementation.")


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"


@dataclass
class WorkflowRun:
    """Represents a running or completed workflow."""
    run_id: str
    workflow_id: str
    journey_id: str
    status: WorkflowStatus
    current_step: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    outputs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.outputs is None:
            self.outputs = {}


class TemporalConfig:
    """Configuration for Temporal connection."""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        namespace: str = None,
        task_queue: str = None,
    ):
        self.host = host or os.environ.get("TEMPORAL_HOST", "localhost")
        self.port = port or int(os.environ.get("TEMPORAL_PORT", "7233"))
        self.namespace = namespace or os.environ.get("TEMPORAL_NAMESPACE", "mineralvision")
        self.task_queue = task_queue or os.environ.get("TEMPORAL_TASK_QUEUE", "mineralvision-journeys")
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class TemporalClient:
    """Client for interacting with Temporal server."""
    
    def __init__(self, config: TemporalConfig = None):
        self.config = config or TemporalConfig()
        self._client: Optional[TemporalClientSDK] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Temporal server."""
        if not TEMPORAL_AVAILABLE:
            logger.warning("Temporal SDK not available, using mock mode")
            self._connected = False
            return False
        
        try:
            self._client = await TemporalClientSDK.connect(
                self.config.address,
                namespace=self.config.namespace,
            )
            self._connected = True
            logger.info(f"Connected to Temporal at {self.config.address}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Temporal: {e}")
            self._connected = False
            return False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def start_workflow(
        self,
        workflow_name: str,
        workflow_id: str,
        args: Dict[str, Any],
        task_queue: str = None,
    ) -> str:
        """Start a new workflow execution."""
        if not self._connected or not self._client:
            # Mock mode - return a fake run ID
            import uuid
            run_id = f"mock-{uuid.uuid4().hex[:8]}"
            logger.info(f"Mock workflow started: {workflow_id} -> {run_id}")
            return run_id
        
        handle = await self._client.start_workflow(
            workflow_name,
            args,
            id=workflow_id,
            task_queue=task_queue or self.config.task_queue,
        )
        return handle.result_run_id
    
    async def get_workflow_status(self, workflow_id: str) -> WorkflowStatus:
        """Get the status of a workflow."""
        if not self._connected or not self._client:
            return WorkflowStatus.COMPLETED
        
        try:
            handle = self._client.get_workflow_handle(workflow_id)
            desc = await handle.describe()
            status_map = {
                "RUNNING": WorkflowStatus.RUNNING,
                "COMPLETED": WorkflowStatus.COMPLETED,
                "FAILED": WorkflowStatus.FAILED,
                "CANCELLED": WorkflowStatus.CANCELLED,
                "TERMINATED": WorkflowStatus.CANCELLED,
            }
            return status_map.get(desc.status.name, WorkflowStatus.PENDING)
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return WorkflowStatus.FAILED
    
    async def signal_workflow(
        self,
        workflow_id: str,
        signal_name: str,
        args: Any = None,
    ):
        """Send a signal to a running workflow."""
        if not self._connected or not self._client:
            logger.info(f"Mock signal sent to {workflow_id}: {signal_name}")
            return
        
        handle = self._client.get_workflow_handle(workflow_id)
        await handle.signal(signal_name, args)
    
    async def query_workflow(
        self,
        workflow_id: str,
        query_name: str,
    ) -> Any:
        """Query a running workflow."""
        if not self._connected or not self._client:
            return {"status": "mock", "current_step": "unknown"}
        
        handle = self._client.get_workflow_handle(workflow_id)
        return await handle.query(query_name)
    
    async def cancel_workflow(self, workflow_id: str):
        """Cancel a running workflow."""
        if not self._connected or not self._client:
            logger.info(f"Mock workflow cancelled: {workflow_id}")
            return
        
        handle = self._client.get_workflow_handle(workflow_id)
        await handle.cancel()


class WorkflowManager:
    """Manages workflow executions and provides high-level operations."""
    
    def __init__(self, client: TemporalClient = None):
        self.client = client or TemporalClient()
        self._runs: Dict[str, WorkflowRun] = {}
    
    async def initialize(self):
        """Initialize the workflow manager."""
        await self.client.connect()
    
    async def start_journey(
        self,
        journey_id: str,
        user_id: str,
        project_id: str,
        inputs: Dict[str, Any] = None,
    ) -> WorkflowRun:
        """Start a user journey workflow."""
        import uuid
        from datetime import datetime
        
        workflow_id = f"journey-{journey_id}-{uuid.uuid4().hex[:8]}"
        
        args = {
            "journey_id": journey_id,
            "user_id": user_id,
            "project_id": project_id,
            "inputs": inputs or {},
        }
        
        run_id = await self.client.start_workflow(
            "JourneyWorkflow",
            workflow_id,
            args,
        )
        
        run = WorkflowRun(
            run_id=run_id,
            workflow_id=workflow_id,
            journey_id=journey_id,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.utcnow().isoformat(),
        )
        
        self._runs[workflow_id] = run
        return run
    
    async def get_run(self, workflow_id: str) -> Optional[WorkflowRun]:
        """Get a workflow run by ID."""
        if workflow_id in self._runs:
            run = self._runs[workflow_id]
            run.status = await self.client.get_workflow_status(workflow_id)
            return run
        return None
    
    async def list_runs(
        self,
        journey_id: str = None,
        status: WorkflowStatus = None,
    ) -> List[WorkflowRun]:
        """List workflow runs with optional filters."""
        runs = list(self._runs.values())
        
        if journey_id:
            runs = [r for r in runs if r.journey_id == journey_id]
        
        if status:
            runs = [r for r in runs if r.status == status]
        
        return runs
    
    async def approve_step(
        self,
        workflow_id: str,
        step_id: str,
        approved: bool,
        comment: str = None,
    ):
        """Approve or reject a step that requires human approval."""
        await self.client.signal_workflow(
            workflow_id,
            "step_approval",
            {
                "step_id": step_id,
                "approved": approved,
                "comment": comment,
            },
        )
    
    async def cancel_run(self, workflow_id: str):
        """Cancel a running workflow."""
        await self.client.cancel_workflow(workflow_id)
        if workflow_id in self._runs:
            self._runs[workflow_id].status = WorkflowStatus.CANCELLED
    
    async def get_current_step(self, workflow_id: str) -> Optional[str]:
        """Get the current step of a running workflow."""
        try:
            result = await self.client.query_workflow(workflow_id, "current_step")
            return result.get("step_id")
        except Exception:
            return None


# Activity definitions for Python workers
if TEMPORAL_AVAILABLE:
    
    @activity.defn
    async def call_api_endpoint(
        endpoint: str,
        method: str,
        payload: Dict[str, Any],
        headers: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Activity to call a FastAPI endpoint."""
        import httpx
        
        base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
        url = f"{base_url}{endpoint}"
        
        async with httpx.AsyncClient() as client:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers, params=payload)
            elif method.upper() == "POST":
                response = await client.post(url, headers=headers, json=payload)
            elif method.upper() == "PUT":
                response = await client.put(url, headers=headers, json=payload)
            elif method.upper() == "DELETE":
                response = await client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
    
    @activity.defn
    async def run_ml_inference(
        module_path: str,
        function_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Activity to run ML inference from a module."""
        import importlib
        
        module = importlib.import_module(module_path)
        func = getattr(module, function_name)
        
        if asyncio.iscoroutinefunction(func):
            result = await func(**inputs)
        else:
            result = func(**inputs)
        
        return result
    
    @activity.defn
    async def publish_kafka_event(
        topic: str,
        event: Dict[str, Any],
    ) -> bool:
        """Activity to publish an event to Kafka."""
        # Import middleware integration
        from ..middleware import get_middleware_integration
        
        middleware = get_middleware_integration()
        return await middleware.publish_kafka(topic, event)
    
    @activity.defn
    async def publish_fluvio_event(
        topic: str,
        event: Dict[str, Any],
    ) -> bool:
        """Activity to publish an event to Fluvio."""
        from ..middleware import get_middleware_integration
        
        middleware = get_middleware_integration()
        return await middleware.publish_fluvio(topic, event)
    
    @activity.defn
    async def check_permission(
        user_id: str,
        permission: str,
        resource_id: str = None,
    ) -> bool:
        """Activity to check user permission via Permify."""
        from ..middleware import get_middleware_integration
        
        middleware = get_middleware_integration()
        return await middleware.check_permission(user_id, permission, resource_id)
    
    @activity.defn
    async def write_ledger_entry(
        entry_type: str,
        data: Dict[str, Any],
    ) -> str:
        """Activity to write an entry to TigerBeetle ledger."""
        from ..middleware import get_middleware_integration
        
        middleware = get_middleware_integration()
        return await middleware.write_ledger(entry_type, data)
    
    @activity.defn
    async def store_to_lakehouse(
        table: str,
        data: Dict[str, Any],
    ) -> str:
        """Activity to store data to the lakehouse."""
        from ..middleware import get_middleware_integration
        
        middleware = get_middleware_integration()
        return await middleware.store_lakehouse(table, data)
    
    @activity.defn
    async def cache_to_redis(
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
    ) -> bool:
        """Activity to cache data in Redis."""
        from ..middleware import get_middleware_integration
        
        middleware = get_middleware_integration()
        return await middleware.cache_redis(key, value, ttl_seconds)


# Global instances
_temporal_client: Optional[TemporalClient] = None
_workflow_manager: Optional[WorkflowManager] = None


def get_temporal_client() -> TemporalClient:
    """Get the global Temporal client instance."""
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = TemporalClient()
    return _temporal_client


def get_workflow_manager() -> WorkflowManager:
    """Get the global workflow manager instance."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager(get_temporal_client())
    return _workflow_manager
