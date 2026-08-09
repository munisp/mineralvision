"""
Data Lineage Tracking Module
============================

Production-grade data lineage with:
- OpenLineage-compatible event model
- Job and dataset tracking
- Column-level lineage
- Lineage graph visualization
- Marquez integration support
"""

import os
import json
import logging
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import threading

logger = logging.getLogger(__name__)

from .._mock_fallback import real_client_unavailable


class EventType(Enum):
    """Lineage event types."""
    START = "START"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ABORT = "ABORT"
    FAIL = "FAIL"


class DatasetType(Enum):
    """Dataset types."""
    DB_TABLE = "DB_TABLE"
    FILE = "FILE"
    STREAM = "STREAM"
    API = "API"


@dataclass
class DatasetFacet:
    """Dataset facet for additional metadata."""
    schema: Optional[Dict] = None
    data_source: Optional[Dict] = None
    documentation: Optional[str] = None
    ownership: Optional[Dict] = None
    quality_metrics: Optional[Dict] = None
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        if self.schema:
            result['schema'] = self.schema
        if self.data_source:
            result['dataSource'] = self.data_source
        if self.documentation:
            result['documentation'] = {'description': self.documentation}
        if self.ownership:
            result['ownership'] = self.ownership
        if self.quality_metrics:
            result['dataQualityMetrics'] = self.quality_metrics
        result.update(self.custom)
        return result


@dataclass
class ColumnLineage:
    """Column-level lineage information."""
    output_column: str
    input_columns: List[Dict[str, str]]  # [{'dataset': 'name', 'column': 'col'}]
    transformation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'outputColumn': self.output_column,
            'inputColumns': self.input_columns,
            'transformation': self.transformation
        }


@dataclass
class Dataset:
    """Represents a dataset in lineage."""
    namespace: str
    name: str
    dataset_type: DatasetType = DatasetType.DB_TABLE
    facets: DatasetFacet = field(default_factory=DatasetFacet)
    
    @property
    def qualified_name(self) -> str:
        """Get fully qualified name."""
        return f"{self.namespace}.{self.name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenLineage format."""
        return {
            'namespace': self.namespace,
            'name': self.name,
            'facets': self.facets.to_dict()
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Dataset':
        """Create from dictionary."""
        return cls(
            namespace=d['namespace'],
            name=d['name'],
            facets=DatasetFacet(**d.get('facets', {}))
        )


@dataclass
class JobFacet:
    """Job facet for additional metadata."""
    sql: Optional[str] = None
    source_code: Optional[str] = None
    source_code_location: Optional[str] = None
    documentation: Optional[str] = None
    ownership: Optional[Dict] = None
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        if self.sql:
            result['sql'] = {'query': self.sql}
        if self.source_code:
            result['sourceCode'] = {'code': self.source_code}
        if self.source_code_location:
            result['sourceCodeLocation'] = {'url': self.source_code_location}
        if self.documentation:
            result['documentation'] = {'description': self.documentation}
        if self.ownership:
            result['ownership'] = self.ownership
        result.update(self.custom)
        return result


@dataclass
class Job:
    """Represents a job/transformation in lineage."""
    namespace: str
    name: str
    facets: JobFacet = field(default_factory=JobFacet)
    
    @property
    def qualified_name(self) -> str:
        """Get fully qualified name."""
        return f"{self.namespace}.{self.name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenLineage format."""
        return {
            'namespace': self.namespace,
            'name': self.name,
            'facets': self.facets.to_dict()
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Job':
        """Create from dictionary."""
        return cls(
            namespace=d['namespace'],
            name=d['name'],
            facets=JobFacet(**d.get('facets', {}))
        )


@dataclass
class RunFacet:
    """Run facet for additional metadata."""
    nominal_time: Optional[datetime] = None
    parent: Optional[Dict] = None
    error_message: Optional[str] = None
    processing_engine: Optional[Dict] = None
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        if self.nominal_time:
            result['nominalTime'] = {
                'nominalStartTime': self.nominal_time.isoformat()
            }
        if self.parent:
            result['parent'] = self.parent
        if self.error_message:
            result['errorMessage'] = {'message': self.error_message}
        if self.processing_engine:
            result['processing_engine'] = self.processing_engine
        result.update(self.custom)
        return result


@dataclass
class Run:
    """Represents a job run."""
    run_id: str
    facets: RunFacet = field(default_factory=RunFacet)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenLineage format."""
        return {
            'runId': self.run_id,
            'facets': self.facets.to_dict()
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Run':
        """Create from dictionary."""
        return cls(
            run_id=d['runId'],
            facets=RunFacet(**d.get('facets', {}))
        )


@dataclass
class LineageEvent:
    """OpenLineage-compatible event."""
    event_type: EventType
    event_time: datetime
    job: Job
    run: Run
    inputs: List[Dataset] = field(default_factory=list)
    outputs: List[Dataset] = field(default_factory=list)
    column_lineage: List[ColumnLineage] = field(default_factory=list)
    producer: str = "mineralvision"
    schema_url: str = "https://openlineage.io/spec/1-0-5/OpenLineage.json"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenLineage format."""
        result = {
            'eventType': self.event_type.value,
            'eventTime': self.event_time.isoformat() + 'Z',
            'producer': self.producer,
            'schemaURL': self.schema_url,
            'job': self.job.to_dict(),
            'run': self.run.to_dict(),
            'inputs': [i.to_dict() for i in self.inputs],
            'outputs': [o.to_dict() for o in self.outputs]
        }
        
        if self.column_lineage:
            result['columnLineage'] = {
                'fields': [cl.to_dict() for cl in self.column_lineage]
            }
        
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LineageEvent':
        """Create from dictionary."""
        return cls(
            event_type=EventType(d['eventType']),
            event_time=datetime.fromisoformat(d['eventTime'].rstrip('Z')),
            job=Job.from_dict(d['job']),
            run=Run.from_dict(d['run']),
            inputs=[Dataset.from_dict(i) for i in d.get('inputs', [])],
            outputs=[Dataset.from_dict(o) for o in d.get('outputs', [])],
            producer=d.get('producer', 'mineralvision'),
            schema_url=d.get('schemaURL', '')
        )


class LineageStore(ABC):
    """Abstract base class for lineage storage."""
    
    @abstractmethod
    def store_event(self, event: LineageEvent):
        """Store a lineage event."""
        pass
    
    @abstractmethod
    def get_events(self, job_name: str = None, run_id: str = None,
                  start_time: datetime = None, end_time: datetime = None) -> List[LineageEvent]:
        """Get lineage events."""
        pass
    
    @abstractmethod
    def get_upstream(self, dataset: Dataset, depth: int = 1) -> List[Dataset]:
        """Get upstream datasets."""
        pass
    
    @abstractmethod
    def get_downstream(self, dataset: Dataset, depth: int = 1) -> List[Dataset]:
        """Get downstream datasets."""
        pass


class LocalLineageStore(LineageStore):
    """
    Local file-based lineage store.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self._events: List[LineageEvent] = []
        self._dataset_graph: Dict[str, Dict[str, Set[str]]] = {}  # {dataset: {upstream: set, downstream: set}}
        self._lock = threading.Lock()
        
        self._load_events()
    
    def _load_events(self):
        """Load events from storage."""
        events_path = os.path.join(self.storage_path, 'events.json')
        
        if os.path.exists(events_path):
            try:
                with open(events_path, 'r') as f:
                    data = json.load(f)
                
                for event_dict in data.get('events', []):
                    event = LineageEvent.from_dict(event_dict)
                    self._events.append(event)
                    self._update_graph(event)
                
                logger.info(f"Loaded {len(self._events)} lineage events")
                
            except Exception as e:
                logger.error(f"Error loading lineage events: {e}")
    
    def _save_events(self):
        """Save events to storage."""
        events_path = os.path.join(self.storage_path, 'events.json')
        
        data = {
            'events': [e.to_dict() for e in self._events]
        }
        
        with open(events_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_graph(self, event: LineageEvent):
        """Update lineage graph with event."""
        for output in event.outputs:
            output_name = output.qualified_name
            
            if output_name not in self._dataset_graph:
                self._dataset_graph[output_name] = {'upstream': set(), 'downstream': set()}
            
            for input_ds in event.inputs:
                input_name = input_ds.qualified_name
                
                if input_name not in self._dataset_graph:
                    self._dataset_graph[input_name] = {'upstream': set(), 'downstream': set()}
                
                # Output depends on input (input is upstream of output)
                self._dataset_graph[output_name]['upstream'].add(input_name)
                self._dataset_graph[input_name]['downstream'].add(output_name)
    
    def store_event(self, event: LineageEvent):
        """Store a lineage event."""
        with self._lock:
            self._events.append(event)
            self._update_graph(event)
            self._save_events()
            
            logger.debug(f"Stored lineage event: {event.job.name} ({event.event_type.value})")
    
    def get_events(self, job_name: str = None, run_id: str = None,
                  start_time: datetime = None, end_time: datetime = None) -> List[LineageEvent]:
        """Get lineage events with optional filters."""
        results = self._events
        
        if job_name:
            results = [e for e in results if e.job.name == job_name]
        
        if run_id:
            results = [e for e in results if e.run.run_id == run_id]
        
        if start_time:
            results = [e for e in results if e.event_time >= start_time]
        
        if end_time:
            results = [e for e in results if e.event_time <= end_time]
        
        return results
    
    def get_upstream(self, dataset: Dataset, depth: int = 1) -> List[Dataset]:
        """Get upstream datasets."""
        result = []
        visited = set()
        
        def traverse(ds_name: str, current_depth: int):
            if current_depth > depth or ds_name in visited:
                return
            
            visited.add(ds_name)
            
            if ds_name in self._dataset_graph:
                for upstream_name in self._dataset_graph[ds_name]['upstream']:
                    if upstream_name not in visited:
                        # Parse namespace and name
                        parts = upstream_name.split('.', 1)
                        namespace = parts[0] if len(parts) > 1 else 'default'
                        name = parts[1] if len(parts) > 1 else parts[0]
                        
                        result.append(Dataset(namespace=namespace, name=name))
                        traverse(upstream_name, current_depth + 1)
        
        traverse(dataset.qualified_name, 0)
        return result
    
    def get_downstream(self, dataset: Dataset, depth: int = 1) -> List[Dataset]:
        """Get downstream datasets."""
        result = []
        visited = set()
        
        def traverse(ds_name: str, current_depth: int):
            if current_depth > depth or ds_name in visited:
                return
            
            visited.add(ds_name)
            
            if ds_name in self._dataset_graph:
                for downstream_name in self._dataset_graph[ds_name]['downstream']:
                    if downstream_name not in visited:
                        parts = downstream_name.split('.', 1)
                        namespace = parts[0] if len(parts) > 1 else 'default'
                        name = parts[1] if len(parts) > 1 else parts[0]
                        
                        result.append(Dataset(namespace=namespace, name=name))
                        traverse(downstream_name, current_depth + 1)
        
        traverse(dataset.qualified_name, 0)
        return result


class MarquezClient(LineageStore):
    """
    Marquez lineage server client.
    """
    
    def __init__(self, url: str, namespace: str = "default"):
        self.url = url.rstrip('/')
        self.namespace = namespace
        self._session = None
        self._degraded = False
        self._initialize()
    
    def _initialize(self):
        """Initialize HTTP session."""
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                'Content-Type': 'application/json'
            })
            
            logger.info(f"Initialized Marquez client: {self.url}")
            
        except ImportError as exc:
            # Real-client-first: mock (events dropped) only when explicitly allowed
            if real_client_unavailable("Marquez lineage client", "requests package not installed", exc):
                self._degraded = True
                self._session = None
    
    def store_event(self, event: LineageEvent):
        """Store a lineage event via Marquez API."""
        if self._session is None:
            logger.warning("Marquez client not initialized")
            return
        
        try:
            response = self._session.post(
                f"{self.url}/api/v1/lineage",
                json=event.to_dict()
            )
            response.raise_for_status()
            
            logger.debug(f"Sent lineage event to Marquez: {event.job.name}")
            
        except Exception as e:
            logger.error(f"Error sending lineage event: {e}")
    
    def get_events(self, job_name: str = None, run_id: str = None,
                  start_time: datetime = None, end_time: datetime = None) -> List[LineageEvent]:
        """Get lineage events from Marquez."""
        if self._session is None:
            return []
        
        try:
            if run_id:
                response = self._session.get(
                    f"{self.url}/api/v1/namespaces/{self.namespace}/jobs/{job_name}/runs/{run_id}"
                )
            elif job_name:
                response = self._session.get(
                    f"{self.url}/api/v1/namespaces/{self.namespace}/jobs/{job_name}/runs"
                )
            else:
                response = self._session.get(
                    f"{self.url}/api/v1/namespaces/{self.namespace}/jobs"
                )
            
            response.raise_for_status()
            data = response.json()
            
            # Convert to LineageEvent objects
            events = []
            for run_data in data.get('runs', [data]):
                # Simplified conversion
                events.append(LineageEvent(
                    event_type=EventType.COMPLETE,
                    event_time=datetime.fromisoformat(run_data.get('createdAt', datetime.utcnow().isoformat()).rstrip('Z')),
                    job=Job(namespace=self.namespace, name=job_name or 'unknown'),
                    run=Run(run_id=run_data.get('id', str(uuid.uuid4())))
                ))
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting lineage events: {e}")
            return []
    
    def get_upstream(self, dataset: Dataset, depth: int = 1) -> List[Dataset]:
        """Get upstream datasets from Marquez."""
        if self._session is None:
            return []
        
        try:
            response = self._session.get(
                f"{self.url}/api/v1/namespaces/{dataset.namespace}/datasets/{dataset.name}/lineage",
                params={'depth': depth}
            )
            response.raise_for_status()
            data = response.json()
            
            upstream = []
            for node in data.get('graph', []):
                if node.get('type') == 'DATASET' and node.get('id') != dataset.qualified_name:
                    parts = node['id'].split('.', 1)
                    upstream.append(Dataset(
                        namespace=parts[0] if len(parts) > 1 else 'default',
                        name=parts[1] if len(parts) > 1 else parts[0]
                    ))
            
            return upstream
            
        except Exception as e:
            logger.error(f"Error getting upstream lineage: {e}")
            return []
    
    def get_downstream(self, dataset: Dataset, depth: int = 1) -> List[Dataset]:
        """Get downstream datasets from Marquez."""
        if self._session is None:
            return []
        
        try:
            response = self._session.get(
                f"{self.url}/api/v1/namespaces/{dataset.namespace}/datasets/{dataset.name}/lineage",
                params={'depth': depth, 'direction': 'downstream'}
            )
            response.raise_for_status()
            data = response.json()
            
            downstream = []
            for node in data.get('graph', []):
                if node.get('type') == 'DATASET' and node.get('id') != dataset.qualified_name:
                    parts = node['id'].split('.', 1)
                    downstream.append(Dataset(
                        namespace=parts[0] if len(parts) > 1 else 'default',
                        name=parts[1] if len(parts) > 1 else parts[0]
                    ))
            
            return downstream
            
        except Exception as e:
            logger.error(f"Error getting downstream lineage: {e}")
            return []


class LineageTracker:
    """
    High-level lineage tracking interface.
    """
    
    def __init__(self, store: LineageStore = None,
                 marquez_url: str = None,
                 local_path: str = None,
                 namespace: str = "mineralvision"):
        self.namespace = namespace
        
        if store:
            self.store = store
        elif marquez_url:
            self.store = MarquezClient(marquez_url, namespace)
        else:
            self.store = LocalLineageStore(local_path or '/tmp/lineage')
        
        self._active_runs: Dict[str, Tuple[Job, Run, List[Dataset], datetime]] = {}
    
    def start_job(self, job_name: str, inputs: List[Dataset] = None,
                 facets: JobFacet = None) -> str:
        """
        Start tracking a job run.
        
        Args:
            job_name: Name of the job
            inputs: Input datasets
            facets: Job facets
            
        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())
        
        job = Job(
            namespace=self.namespace,
            name=job_name,
            facets=facets or JobFacet()
        )
        
        run = Run(run_id=run_id)
        
        event = LineageEvent(
            event_type=EventType.START,
            event_time=datetime.utcnow(),
            job=job,
            run=run,
            inputs=inputs or []
        )
        
        self.store.store_event(event)
        self._active_runs[run_id] = (job, run, inputs or [], datetime.utcnow())
        
        logger.info(f"Started job {job_name} with run_id {run_id}")
        
        return run_id
    
    def complete_job(self, run_id: str, outputs: List[Dataset] = None,
                    column_lineage: List[ColumnLineage] = None):
        """
        Mark a job run as complete.
        
        Args:
            run_id: Run ID from start_job
            outputs: Output datasets
            column_lineage: Column-level lineage
        """
        if run_id not in self._active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return
        
        job, run, inputs, start_time = self._active_runs.pop(run_id)
        
        event = LineageEvent(
            event_type=EventType.COMPLETE,
            event_time=datetime.utcnow(),
            job=job,
            run=run,
            inputs=inputs,
            outputs=outputs or [],
            column_lineage=column_lineage or []
        )
        
        self.store.store_event(event)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Completed job {job.name} (run_id={run_id}) in {duration:.2f}s")
    
    def fail_job(self, run_id: str, error_message: str = None):
        """
        Mark a job run as failed.
        
        Args:
            run_id: Run ID from start_job
            error_message: Error message
        """
        if run_id not in self._active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return
        
        job, run, inputs, start_time = self._active_runs.pop(run_id)
        
        run.facets.error_message = error_message
        
        event = LineageEvent(
            event_type=EventType.FAIL,
            event_time=datetime.utcnow(),
            job=job,
            run=run,
            inputs=inputs
        )
        
        self.store.store_event(event)
        
        logger.error(f"Failed job {job.name} (run_id={run_id}): {error_message}")
    
    def track_transformation(self, job_name: str, inputs: List[Dataset],
                            outputs: List[Dataset], sql: str = None,
                            column_lineage: List[ColumnLineage] = None):
        """
        Track a complete transformation in one call.
        
        Args:
            job_name: Name of the transformation job
            inputs: Input datasets
            outputs: Output datasets
            sql: SQL query if applicable
            column_lineage: Column-level lineage
        """
        run_id = self.start_job(
            job_name,
            inputs=inputs,
            facets=JobFacet(sql=sql) if sql else None
        )
        
        self.complete_job(run_id, outputs=outputs, column_lineage=column_lineage)
    
    def get_upstream(self, dataset: Dataset, depth: int = 1) -> List[Dataset]:
        """Get upstream datasets."""
        return self.store.get_upstream(dataset, depth)
    
    def get_downstream(self, dataset: Dataset, depth: int = 1) -> List[Dataset]:
        """Get downstream datasets."""
        return self.store.get_downstream(dataset, depth)
    
    def get_lineage_graph(self, dataset: Dataset, depth: int = 2) -> Dict[str, Any]:
        """
        Get complete lineage graph for a dataset.
        
        Args:
            dataset: Dataset to get lineage for
            depth: Depth to traverse
            
        Returns:
            Graph representation
        """
        upstream = self.get_upstream(dataset, depth)
        downstream = self.get_downstream(dataset, depth)
        
        nodes = [{'id': dataset.qualified_name, 'type': 'dataset', 'central': True}]
        edges = []
        
        for ds in upstream:
            nodes.append({'id': ds.qualified_name, 'type': 'dataset', 'direction': 'upstream'})
            edges.append({'source': ds.qualified_name, 'target': dataset.qualified_name})
        
        for ds in downstream:
            nodes.append({'id': ds.qualified_name, 'type': 'dataset', 'direction': 'downstream'})
            edges.append({'source': dataset.qualified_name, 'target': ds.qualified_name})
        
        return {
            'nodes': nodes,
            'edges': edges,
            'central_dataset': dataset.qualified_name
        }
    
    def get_impact_analysis(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Analyze impact of changes to a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Impact analysis results
        """
        downstream = self.get_downstream(dataset, depth=10)
        
        return {
            'dataset': dataset.qualified_name,
            'impacted_datasets': [ds.qualified_name for ds in downstream],
            'impact_count': len(downstream),
            'analysis_time': datetime.utcnow().isoformat()
        }


class LineageContext:
    """
    Context manager for automatic lineage tracking.
    """
    
    def __init__(self, tracker: LineageTracker, job_name: str,
                inputs: List[Dataset] = None, sql: str = None):
        self.tracker = tracker
        self.job_name = job_name
        self.inputs = inputs or []
        self.sql = sql
        self.run_id = None
        self.outputs: List[Dataset] = []
        self.column_lineage: List[ColumnLineage] = []
    
    def __enter__(self):
        self.run_id = self.tracker.start_job(
            self.job_name,
            inputs=self.inputs,
            facets=JobFacet(sql=self.sql) if self.sql else None
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.tracker.fail_job(self.run_id, str(exc_val))
        else:
            self.tracker.complete_job(
                self.run_id,
                outputs=self.outputs,
                column_lineage=self.column_lineage
            )
        return False
    
    def add_output(self, dataset: Dataset):
        """Add an output dataset."""
        self.outputs.append(dataset)
    
    def add_column_lineage(self, output_column: str,
                          input_columns: List[Dict[str, str]],
                          transformation: str = None):
        """Add column-level lineage."""
        self.column_lineage.append(ColumnLineage(
            output_column=output_column,
            input_columns=input_columns,
            transformation=transformation
        ))


def create_lineage_tracker(marquez_url: str = None,
                          local_path: str = None,
                          namespace: str = "mineralvision") -> LineageTracker:
    """Factory function to create lineage tracker."""
    return LineageTracker(
        marquez_url=marquez_url,
        local_path=local_path,
        namespace=namespace
    )


def create_dataset(namespace: str, name: str,
                  schema: Dict = None) -> Dataset:
    """Factory function to create dataset."""
    facets = DatasetFacet(schema=schema) if schema else DatasetFacet()
    return Dataset(namespace=namespace, name=name, facets=facets)
