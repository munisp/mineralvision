"""
Temporal Workflow Integration — Compatibility Shim
===================================================

DEPRECATED LOCATION. The canonical Temporal implementation lives at:

    MineralVision_Final_Package/src/api/orchestration/temporal/

This module is a thin compatibility shim: it re-exports the canonical
client/manager and keeps the historical public names working. The
duplicated Temporal client, mock client and workflow engine logic that
previously lived here has been DELETED (temporal dedupe, wave 2).

Import path handling
--------------------
The canonical module lives in a sibling package tree. This shim computes
the repository root from its own file location and appends
``<repo>/MineralVision_Final_Package`` to ``sys.path`` so the canonical
module is importable as ``src.api.orchestration.temporal``. Alternatively,
set ``PYTHONPATH=<repo>/MineralVision_Final_Package`` yourself.

Real-client-first contract
--------------------------
The canonical ``TemporalClient.connect()`` raises ``RuntimeError`` when
the temporalio SDK is missing or the server is unreachable, UNLESS
``MV_ALLOW_MOCK_FALLBACK=true`` is explicitly set (mock mode, degraded).
This shim inherits that behavior.

Public name mapping
--------------------
- ``WorkflowStatus``, ``WorkflowRun``, ``TemporalConfig``,
  ``TemporalClient``, ``WorkflowManager``, ``get_temporal_client``,
  ``get_workflow_manager`` — re-exported from the canonical module.
- ``TemporalWorkflowEngine`` — thin adapter over the canonical
  ``TemporalClient``/``WorkflowManager`` preserving the legacy API.
- ``MockTemporalClient`` / ``MockWorkflowHandle`` — REMOVED (no silent
  mocks); accessing them raises a descriptive AttributeError.
- MineralVision-specific workflow/activity definitions and registries
  (``DataProcessingWorkflow``, ``MLTrainingWorkflow``,
  ``ReportGenerationWorkflow``, ``SensorFusionWorkflow``,
  ``ActivityRegistry``, ``WorkflowRegistry``,
  ``ScheduledWorkflowManager``, decorators) are kept here unchanged —
  they are domain definitions, not duplicated Temporal client logic.
"""

import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate and import the canonical Temporal module
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FINAL_PACKAGE = _REPO_ROOT / "MineralVision_Final_Package"

if str(_FINAL_PACKAGE) not in sys.path:
    sys.path.append(str(_FINAL_PACKAGE))

try:
    from src.api.orchestration.temporal import (
        WorkflowStatus,
        WorkflowRun,
        TemporalConfig,
        TemporalClient,
        WorkflowManager,
        get_temporal_client,
        get_workflow_manager,
        TEMPORAL_AVAILABLE,
    )
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Cannot import the canonical Temporal module "
        "(src.api.orchestration.temporal). Expected repository layout: "
        f"{_FINAL_PACKAGE}. Set PYTHONPATH=<repo>/MineralVision_Final_Package "
        f"or restore the canonical module. Original error: {exc}"
    ) from exc

# Legacy alias kept for name compatibility
WorkflowExecution = WorkflowRun


class TemporalWorkflowEngine:
    """
    Thin compatibility adapter over the canonical TemporalClient.

    Preserves the legacy MineralVision_Enhanced engine API while delegating
    ALL client logic to MineralVision_Final_Package/src/api/orchestration/
    temporal. Mock mode is only available via MV_ALLOW_MOCK_FALLBACK=true
    (see canonical module); check ``engine.degraded``.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        namespace = self.config.get('namespace')
        target_host = self.config.get('target_host')  # e.g. "localhost:7233"
        host = port = None
        if target_host and ":" in target_host:
            host, port_str = target_host.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                host, port = None, None

        temporal_config = TemporalConfig(
            host=host,
            port=port,
            namespace=namespace,
            task_queue=self.config.get('default_task_queue'),
        )
        self.client = TemporalClient(temporal_config)
        self.manager = WorkflowManager(self.client)
        self._workers: List[Any] = []

    @property
    def degraded(self) -> bool:
        """True when running in explicit mock mode (no real Temporal)."""
        return self.client.degraded

    @property
    def _connected(self) -> bool:
        return self.client.is_connected or self.client.degraded

    async def connect(self) -> 'TemporalWorkflowEngine':
        """Connect to the Temporal server via the canonical client."""
        await self.client.connect()
        return self

    async def start_workflow(self, workflow_type: str,
                            args: List[Any] = None,
                            workflow_id: str = None,
                            task_queue: str = None,
                            **kwargs) -> str:
        """Start a workflow execution; returns the run ID."""
        if not self._connected:
            await self.connect()

        workflow_id = workflow_id or f"{workflow_type}-{uuid.uuid4()}"
        run_id = await self.client.start_workflow(
            workflow_type,
            workflow_id,
            {"args": args or [], **kwargs},
            task_queue=task_queue,
        )
        logger.info(f"Started workflow {workflow_type} with ID {workflow_id}")
        return run_id

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowRun]:
        """Get a workflow run by ID."""
        return await self.manager.get_run(workflow_id)

    async def list_workflows(self, query: str = None,
                            status: WorkflowStatus = None) -> List[WorkflowRun]:
        """List workflow executions (journey_id filter via `query`)."""
        return await self.manager.list_runs(journey_id=query, status=status)

    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel a running workflow."""
        await self.client.cancel_workflow(workflow_id)
        logger.info(f"Cancelled workflow {workflow_id}")

    async def terminate_workflow(self, workflow_id: str, reason: str = None) -> None:
        """Terminate a workflow (mapped to cancel)."""
        await self.client.cancel_workflow(workflow_id)
        logger.info(f"Terminated workflow {workflow_id}: {reason}")

    async def signal_workflow(self, workflow_id: str, signal_name: str, *args) -> None:
        """Send a signal to a workflow."""
        payload = args[0] if args else None
        await self.client.signal_workflow(workflow_id, signal_name, payload)

    async def query_workflow(self, workflow_id: str, query_name: str, *args) -> Any:
        """Query a workflow."""
        return await self.client.query_workflow(workflow_id, query_name)

    async def start_worker(self, task_queue: str = None,
                          activities: List[Callable] = None,
                          workflows: List[Type] = None) -> None:
        """Start a workflow worker (requires the real temporalio SDK)."""
        if not TEMPORAL_AVAILABLE:
            raise RuntimeError(
                "temporalio SDK is not installed — cannot start a worker. "
                "install temporalio"
            )
        from temporalio.worker import Worker

        if not self.client.is_connected:
            raise RuntimeError(
                "Worker requires a connected real Temporal client "
                f"(degraded={self.degraded})"
            )

        task_queue = task_queue or self.config.get('default_task_queue', 'mineralvision-workflows')
        worker = Worker(
            self.client._client,
            task_queue=task_queue,
            activities=activities or [],
            workflows=workflows or []
        )
        self._workers.append(worker)
        import asyncio
        asyncio.create_task(worker.run())
        logger.info(f"Started worker on task queue {task_queue}")

    async def shutdown(self) -> None:
        """Shutdown the workflow engine."""
        for worker in self._workers:
            if hasattr(worker, 'shutdown'):
                await worker.shutdown()
        self._workers.clear()
        logger.info("Temporal workflow engine shutdown complete")


# Removed mocks — fail loudly if legacy code asks for them
_REMOVED_NAMES = {"MockTemporalClient", "MockWorkflowHandle"}


def __getattr__(name: str):
    if name in _REMOVED_NAMES:
        raise AttributeError(
            f"{name} was removed (no silent mocks). Use TemporalClient with "
            "a real Temporal server, or set MV_ALLOW_MOCK_FALLBACK=true to "
            "explicitly enable the canonical mock mode (degraded)."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# MineralVision-specific workflow/activity definitions (domain logic, kept)
# ---------------------------------------------------------------------------


class ActivityType(Enum):
    """Types of activities in MineralVision workflows."""
    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    GEOSPATIAL_PROCESSING = "geospatial_processing"
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"
    REPORT_GENERATION = "report_generation"
    NOTIFICATION = "notification"
    DATA_EXPORT = "data_export"
    QUALITY_CHECK = "quality_check"
    SENSOR_FUSION = "sensor_fusion"


@dataclass
class RetryConfig:
    """Retry configuration for activities."""
    initial_interval: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    backoff_coefficient: float = 2.0
    maximum_interval: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    maximum_attempts: int = 5
    non_retryable_errors: List[str] = field(default_factory=list)


@dataclass
class ActivityConfig:
    """Configuration for workflow activities."""
    name: str
    activity_type: ActivityType
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    heartbeat_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    task_queue: str = "mineralvision-activities"


@dataclass
class WorkflowConfig:
    """Configuration for workflows."""
    name: str
    task_queue: str = "mineralvision-workflows"
    execution_timeout: timedelta = field(default_factory=lambda: timedelta(hours=24))
    run_timeout: timedelta = field(default_factory=lambda: timedelta(hours=1))
    task_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    retry_policy: Optional[RetryConfig] = None
    cron_schedule: Optional[str] = None
    memo: Dict[str, Any] = field(default_factory=dict)
    search_attributes: Dict[str, Any] = field(default_factory=dict)


class ActivityRegistry:
    """Registry for workflow activities."""

    def __init__(self):
        self._activities: Dict[str, Callable] = {}
        self._configs: Dict[str, ActivityConfig] = {}

    def register(self, config: ActivityConfig) -> Callable:
        """Decorator to register an activity."""
        def decorator(func: Callable) -> Callable:
            self._activities[config.name] = func
            self._configs[config.name] = config

            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper
        return decorator

    def get_activity(self, name: str) -> Optional[Callable]:
        return self._activities.get(name)

    def get_config(self, name: str) -> Optional[ActivityConfig]:
        return self._configs.get(name)

    def list_activities(self) -> List[str]:
        return list(self._activities.keys())


class WorkflowRegistry:
    """Registry for workflows."""

    def __init__(self):
        self._workflows: Dict[str, Type] = {}
        self._configs: Dict[str, WorkflowConfig] = {}

    def register(self, config: WorkflowConfig) -> Callable:
        """Decorator to register a workflow."""
        def decorator(cls: Type) -> Type:
            self._workflows[config.name] = cls
            self._configs[config.name] = config
            return cls
        return decorator

    def get_workflow(self, name: str) -> Optional[Type]:
        return self._workflows.get(name)

    def get_config(self, name: str) -> Optional[WorkflowConfig]:
        return self._configs.get(name)

    def list_workflows(self) -> List[str]:
        return list(self._workflows.keys())


# Global registries
activity_registry = ActivityRegistry()
workflow_registry = WorkflowRegistry()


class DataProcessingWorkflow:
    """
    Workflow for processing geological data.

    Steps:
    1. Validate input data
    2. Transform and clean data
    3. Run quality checks
    4. Store processed data
    5. Generate processing report
    """

    @staticmethod
    async def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the data processing workflow."""
        results = {
            'workflow_id': str(uuid.uuid4()),
            'start_time': datetime.now().isoformat(),
            'steps': []
        }

        results['steps'].append({'name': 'validate', 'status': 'completed', 'duration_ms': 150})
        results['steps'].append({'name': 'transform', 'status': 'completed', 'duration_ms': 500})
        results['steps'].append({'name': 'quality_check', 'status': 'completed', 'duration_ms': 200})
        results['steps'].append({'name': 'store', 'status': 'completed', 'duration_ms': 300})
        results['steps'].append({'name': 'report', 'status': 'completed', 'duration_ms': 100})

        results['end_time'] = datetime.now().isoformat()
        results['status'] = 'completed'

        return results


class MLTrainingWorkflow:
    """
    Workflow for ML model training.

    Steps:
    1. Prepare training data
    2. Configure model
    3. Train model
    4. Evaluate model
    5. Register model
    6. Deploy model (optional)
    """

    @staticmethod
    async def run(config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the ML training workflow."""
        model_name = config.get('model_name', 'mineral_classifier')
        epochs = config.get('epochs', 10)

        results = {
            'workflow_id': str(uuid.uuid4()),
            'model_name': model_name,
            'start_time': datetime.now().isoformat(),
            'training_config': config,
            'metrics': {}
        }

        for epoch in range(min(epochs, 5)):
            results['metrics'][f'epoch_{epoch+1}'] = {
                'loss': 1.0 - (epoch * 0.15),
                'accuracy': 0.5 + (epoch * 0.08)
            }

        results['final_metrics'] = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85
        }

        results['model_artifact'] = f"models/{model_name}/v1"
        results['end_time'] = datetime.now().isoformat()
        results['status'] = 'completed'

        return results


class ReportGenerationWorkflow:
    """
    Workflow for generating geological reports.

    Steps:
    1. Gather data
    2. Run analyses
    3. Generate visualizations
    4. Compile report
    5. Export to formats (PDF, DOCX)
    6. Distribute report
    """

    @staticmethod
    async def run(report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the report generation workflow."""
        report_type = report_config.get('report_type', 'exploration_summary')

        results = {
            'workflow_id': str(uuid.uuid4()),
            'report_type': report_type,
            'start_time': datetime.now().isoformat(),
            'sections': []
        }

        sections = ['executive_summary', 'methodology', 'results', 'conclusions', 'appendices']
        for section in sections:
            results['sections'].append({
                'name': section,
                'status': 'completed',
                'page_count': 5
            })

        results['output_formats'] = ['pdf', 'docx', 'html']
        results['total_pages'] = 25
        results['end_time'] = datetime.now().isoformat()
        results['status'] = 'completed'

        return results


class SensorFusionWorkflow:
    """
    Workflow for fusing multi-sensor data.

    Steps:
    1. Ingest sensor data streams
    2. Synchronize timestamps
    3. Apply calibration
    4. Fuse data
    5. Validate fusion results
    6. Store fused data
    """

    @staticmethod
    async def run(sensor_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the sensor fusion workflow."""
        sensors = sensor_config.get('sensors', ['magnetometer', 'spectrometer', 'lidar'])

        results = {
            'workflow_id': str(uuid.uuid4()),
            'sensors': sensors,
            'start_time': datetime.now().isoformat(),
            'fusion_results': {}
        }

        for sensor in sensors:
            results['fusion_results'][sensor] = {
                'records_processed': 10000,
                'quality_score': 0.95,
                'calibration_applied': True
            }

        results['fused_dataset_id'] = f"fused-{uuid.uuid4()}"
        results['total_records'] = len(sensors) * 10000
        results['end_time'] = datetime.now().isoformat()
        results['status'] = 'completed'

        return results


# Register workflows
workflow_registry.register(WorkflowConfig(name="data_processing"))(DataProcessingWorkflow)
workflow_registry.register(WorkflowConfig(name="ml_training"))(MLTrainingWorkflow)
workflow_registry.register(WorkflowConfig(name="report_generation"))(ReportGenerationWorkflow)
workflow_registry.register(WorkflowConfig(name="sensor_fusion"))(SensorFusionWorkflow)


# Activity definitions

@activity_registry.register(ActivityConfig(
    name="validate_data",
    activity_type=ActivityType.DATA_VALIDATION,
    timeout=timedelta(minutes=5)
))
async def validate_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input data."""
    return {
        'valid': True,
        'records_checked': len(data.get('records', [])),
        'errors': []
    }


@activity_registry.register(ActivityConfig(
    name="process_geospatial",
    activity_type=ActivityType.GEOSPATIAL_PROCESSING,
    timeout=timedelta(minutes=30),
    heartbeat_timeout=timedelta(minutes=5)
))
async def process_geospatial(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process geospatial data."""
    return {
        'processed': True,
        'features_count': 1000,
        'crs': 'EPSG:4326'
    }


@activity_registry.register(ActivityConfig(
    name="run_ml_inference",
    activity_type=ActivityType.ML_INFERENCE,
    timeout=timedelta(minutes=10)
))
async def run_ml_inference(model_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Run ML model inference."""
    return {
        'model_id': model_id,
        'predictions': [],
        'confidence_scores': []
    }


@activity_registry.register(ActivityConfig(
    name="send_notification",
    activity_type=ActivityType.NOTIFICATION,
    timeout=timedelta(seconds=30)
))
async def send_notification(notification: Dict[str, Any]) -> Dict[str, Any]:
    """Send notification."""
    return {
        'sent': True,
        'channel': notification.get('channel', 'email'),
        'recipients': notification.get('recipients', [])
    }


# Factory functions

def create_temporal_engine(config: Dict[str, Any] = None) -> TemporalWorkflowEngine:
    """Create a Temporal workflow engine instance."""
    return TemporalWorkflowEngine(config)


async def create_and_connect_engine(config: Dict[str, Any] = None) -> TemporalWorkflowEngine:
    """Create and connect a Temporal workflow engine."""
    engine = TemporalWorkflowEngine(config)
    await engine.connect()
    return engine


# Workflow decorators for easy definition

def workflow_definition(name: str, **kwargs):
    """Decorator to define a workflow."""
    config = WorkflowConfig(name=name, **kwargs)
    return workflow_registry.register(config)


def activity_definition(name: str, activity_type: ActivityType, **kwargs):
    """Decorator to define an activity."""
    config = ActivityConfig(name=name, activity_type=activity_type, **kwargs)
    return activity_registry.register(config)


# Scheduled workflow support

class ScheduledWorkflowManager:
    """Manager for scheduled/cron workflows."""

    def __init__(self, engine: TemporalWorkflowEngine):
        self.engine = engine
        self._schedules: Dict[str, Dict[str, Any]] = {}

    async def schedule_workflow(self, workflow_type: str,
                               cron_expression: str,
                               args: List[Any] = None,
                               workflow_id_prefix: str = None) -> str:
        """
        Schedule a workflow to run on a cron schedule.

        Returns:
            Schedule ID
        """
        schedule_id = f"schedule-{uuid.uuid4()}"

        self._schedules[schedule_id] = {
            'workflow_type': workflow_type,
            'cron_expression': cron_expression,
            'args': args or [],
            'workflow_id_prefix': workflow_id_prefix or workflow_type,
            'created_at': datetime.now().isoformat(),
            'last_run': None,
            'next_run': None,
            'enabled': True
        }

        logger.info(f"Scheduled workflow {workflow_type} with cron {cron_expression}")
        return schedule_id

    async def pause_schedule(self, schedule_id: str) -> None:
        """Pause a scheduled workflow."""
        if schedule_id in self._schedules:
            self._schedules[schedule_id]['enabled'] = False

    async def resume_schedule(self, schedule_id: str) -> None:
        """Resume a scheduled workflow."""
        if schedule_id in self._schedules:
            self._schedules[schedule_id]['enabled'] = True

    async def delete_schedule(self, schedule_id: str) -> None:
        """Delete a scheduled workflow."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]

    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all scheduled workflows."""
        return [
            {'schedule_id': sid, **schedule}
            for sid, schedule in self._schedules.items()
        ]


def create_scheduled_manager(engine: TemporalWorkflowEngine) -> ScheduledWorkflowManager:
    """Create a scheduled workflow manager."""
    return ScheduledWorkflowManager(engine)


__all__ = [
    # Canonical re-exports
    "WorkflowStatus",
    "WorkflowRun",
    "WorkflowExecution",
    "TemporalConfig",
    "TemporalClient",
    "WorkflowManager",
    "get_temporal_client",
    "get_workflow_manager",
    "TEMPORAL_AVAILABLE",
    # Adapter engine
    "TemporalWorkflowEngine",
    "create_temporal_engine",
    "create_and_connect_engine",
    # Domain definitions
    "ActivityType",
    "RetryConfig",
    "ActivityConfig",
    "WorkflowConfig",
    "ActivityRegistry",
    "WorkflowRegistry",
    "activity_registry",
    "workflow_registry",
    "DataProcessingWorkflow",
    "MLTrainingWorkflow",
    "ReportGenerationWorkflow",
    "SensorFusionWorkflow",
    "validate_data",
    "process_geospatial",
    "run_ml_inference",
    "send_notification",
    "workflow_definition",
    "activity_definition",
    "ScheduledWorkflowManager",
    "create_scheduled_manager",
]
