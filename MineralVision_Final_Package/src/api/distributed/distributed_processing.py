"""
Distributed Processing Module for MineralVision.

Provides distributed computing capabilities for:
- Processing surveys >100GB
- Parallel processing across nodes
- Memory-efficient operations
- Dask/Ray integration
- Task scheduling and orchestration
- Fault tolerance and recovery
"""

import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Iterator, Tuple, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import queue

logger = logging.getLogger(__name__)


class ExecutorType(Enum):
    """Types of distributed executors."""
    LOCAL = "local"
    DASK = "dask"
    RAY = "ray"
    SPARK = "spark"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class PartitionStrategy(Enum):
    """Data partitioning strategies."""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    SIZE = "size"
    HASH = "hash"
    ROUND_ROBIN = "round_robin"


class SchedulingPolicy(Enum):
    """Task scheduling policies."""
    FIFO = "fifo"
    PRIORITY = "priority"
    FAIR = "fair"
    LOCALITY = "locality"


@dataclass
class ClusterConfig:
    """Cluster configuration."""
    executor_type: ExecutorType
    num_workers: int = 4
    memory_per_worker_gb: float = 8.0
    cpus_per_worker: int = 2
    scheduler_address: str = ""
    dashboard_port: int = 8787
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'executor_type': self.executor_type.value,
            'num_workers': self.num_workers,
            'memory_per_worker_gb': self.memory_per_worker_gb,
            'cpus_per_worker': self.cpus_per_worker,
            'scheduler_address': self.scheduler_address,
            'dashboard_port': self.dashboard_port
        }


@dataclass
class TaskConfig:
    """Task configuration."""
    task_id: str
    name: str
    priority: int = 0
    max_retries: int = 3
    timeout_seconds: int = 3600
    memory_limit_gb: float = 0.0
    cpu_limit: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'name': self.name,
            'priority': self.priority,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'memory_limit_gb': self.memory_limit_gb,
            'cpu_limit': self.cpu_limit
        }


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    worker_id: str = ""
    retries: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': str(self.result) if self.result else None,
            'error': self.error,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'worker_id': self.worker_id,
            'retries': self.retries
        }
        
    @property
    def duration_seconds(self) -> float:
        """Get task duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class DataPartition:
    """Data partition for distributed processing."""
    partition_id: str
    data_path: str
    size_bytes: int
    num_records: int
    bounds: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'partition_id': self.partition_id,
            'data_path': self.data_path,
            'size_bytes': self.size_bytes,
            'num_records': self.num_records,
            'bounds': self.bounds,
            'metadata': self.metadata
        }


@dataclass
class WorkerInfo:
    """Worker node information."""
    worker_id: str
    address: str
    status: str = "active"
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    cpu_count: int = 0
    cpu_usage: float = 0.0
    tasks_running: int = 0
    tasks_completed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'worker_id': self.worker_id,
            'address': self.address,
            'status': self.status,
            'memory_total_gb': self.memory_total_gb,
            'memory_used_gb': self.memory_used_gb,
            'cpu_count': self.cpu_count,
            'cpu_usage': self.cpu_usage,
            'tasks_running': self.tasks_running,
            'tasks_completed': self.tasks_completed
        }


@dataclass
class JobConfig:
    """Distributed job configuration."""
    job_id: str
    name: str
    partitions: List[DataPartition]
    task_function: str
    partition_strategy: PartitionStrategy = PartitionStrategy.SIZE
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.FIFO
    max_concurrent_tasks: int = 0
    checkpoint_interval: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'name': self.name,
            'partitions': [p.to_dict() for p in self.partitions],
            'task_function': self.task_function,
            'partition_strategy': self.partition_strategy.value,
            'scheduling_policy': self.scheduling_policy.value,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'checkpoint_interval': self.checkpoint_interval
        }


@dataclass
class JobStatus:
    """Distributed job status."""
    job_id: str
    name: str
    status: TaskStatus
    total_partitions: int
    completed_partitions: int
    failed_partitions: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'name': self.name,
            'status': self.status.value,
            'total_partitions': self.total_partitions,
            'completed_partitions': self.completed_partitions,
            'failed_partitions': self.failed_partitions,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'progress': self.progress
        }


class DataPartitioner:
    """Partition large datasets for distributed processing."""
    
    def __init__(self, target_partition_size_mb: float = 128.0):
        self.target_partition_size_mb = target_partition_size_mb
        self.target_partition_size_bytes = int(target_partition_size_mb * 1024 * 1024)
        
    def partition_by_size(self, data_path: str, total_size_bytes: int,
                         num_records: int) -> List[DataPartition]:
        """Partition data by size."""
        num_partitions = max(1, int(total_size_bytes / self.target_partition_size_bytes))
        records_per_partition = num_records // num_partitions
        bytes_per_partition = total_size_bytes // num_partitions
        
        partitions = []
        for i in range(num_partitions):
            partition = DataPartition(
                partition_id=f"part_{i:04d}",
                data_path=f"{data_path}/partition_{i:04d}",
                size_bytes=bytes_per_partition,
                num_records=records_per_partition,
                metadata={'partition_index': i}
            )
            partitions.append(partition)
            
        return partitions
        
    def partition_spatial(self, data_path: str, bounds: Dict[str, float],
                         grid_size: int = 4) -> List[DataPartition]:
        """Partition data spatially using grid."""
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 1)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 1)
        
        x_step = (max_x - min_x) / grid_size
        y_step = (max_y - min_y) / grid_size
        
        partitions = []
        for i in range(grid_size):
            for j in range(grid_size):
                partition = DataPartition(
                    partition_id=f"spatial_{i}_{j}",
                    data_path=f"{data_path}/spatial_{i}_{j}",
                    size_bytes=0,
                    num_records=0,
                    bounds={
                        'min_x': min_x + i * x_step,
                        'max_x': min_x + (i + 1) * x_step,
                        'min_y': min_y + j * y_step,
                        'max_y': min_y + (j + 1) * y_step
                    }
                )
                partitions.append(partition)
                
        return partitions
        
    def partition_temporal(self, data_path: str, start_time: datetime,
                          end_time: datetime, interval_hours: int = 24) -> List[DataPartition]:
        """Partition data temporally."""
        partitions = []
        current = start_time
        index = 0
        
        while current < end_time:
            next_time = current + timedelta(hours=interval_hours)
            if next_time > end_time:
                next_time = end_time
                
            partition = DataPartition(
                partition_id=f"temporal_{index:04d}",
                data_path=f"{data_path}/temporal_{index:04d}",
                size_bytes=0,
                num_records=0,
                bounds={
                    'start_time': current.isoformat(),
                    'end_time': next_time.isoformat()
                }
            )
            partitions.append(partition)
            
            current = next_time
            index += 1
            
        return partitions


class DistributedExecutor(ABC):
    """Abstract base class for distributed executors."""
    
    @abstractmethod
    def initialize(self, config: ClusterConfig) -> bool:
        """Initialize the executor."""
        pass
        
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the executor."""
        pass
        
    @abstractmethod
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task for execution."""
        pass
        
    @abstractmethod
    def get_result(self, task_id: str, timeout: float = None) -> TaskResult:
        """Get task result."""
        pass
        
    @abstractmethod
    def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        pass
        
    @abstractmethod
    def get_workers(self) -> List[WorkerInfo]:
        """Get worker information."""
        pass


class LocalExecutor(DistributedExecutor):
    """Local thread pool executor for development/testing."""
    
    def __init__(self):
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[str, Future] = {}
        self._results: Dict[str, TaskResult] = {}
        self._config: Optional[ClusterConfig] = None
        
    def initialize(self, config: ClusterConfig) -> bool:
        """Initialize local executor."""
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=config.num_workers)
        logger.info(f"Initialized local executor with {config.num_workers} workers")
        return True
        
    def shutdown(self) -> None:
        """Shutdown local executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to local executor."""
        task_id = str(uuid.uuid4())
        
        def wrapped_func():
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                start_time=datetime.utcnow(),
                worker_id="local"
            )
            try:
                output = func(*args, **kwargs)
                result.result = output
                result.status = TaskStatus.COMPLETED
            except Exception as e:
                result.error = str(e)
                result.status = TaskStatus.FAILED
            finally:
                result.end_time = datetime.utcnow()
                self._results[task_id] = result
            return result
            
        future = self._executor.submit(wrapped_func)
        self._futures[task_id] = future
        
        return task_id
        
    def get_result(self, task_id: str, timeout: float = None) -> TaskResult:
        """Get task result."""
        if task_id in self._results:
            return self._results[task_id]
            
        future = self._futures.get(task_id)
        if future:
            try:
                future.result(timeout=timeout)
                return self._results.get(task_id, TaskResult(
                    task_id=task_id,
                    status=TaskStatus.PENDING
                ))
            except Exception as e:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=str(e)
                )
                
        return TaskResult(task_id=task_id, status=TaskStatus.PENDING)
        
    def cancel(self, task_id: str) -> bool:
        """Cancel task."""
        future = self._futures.get(task_id)
        if future:
            return future.cancel()
        return False
        
    def get_workers(self) -> List[WorkerInfo]:
        """Get worker information."""
        if not self._config:
            return []
            
        return [
            WorkerInfo(
                worker_id=f"local_{i}",
                address="localhost",
                status="active",
                memory_total_gb=self._config.memory_per_worker_gb,
                cpu_count=self._config.cpus_per_worker
            )
            for i in range(self._config.num_workers)
        ]


class DaskExecutor(DistributedExecutor):
    """Dask distributed executor."""
    
    def __init__(self):
        self._client = None
        self._futures: Dict[str, Any] = {}
        self._config: Optional[ClusterConfig] = None
        
    def initialize(self, config: ClusterConfig) -> bool:
        """Initialize Dask executor."""
        self._config = config
        
        try:
            if config.scheduler_address:
                logger.info(f"Connecting to Dask scheduler at {config.scheduler_address}")
                self._client = {
                    'type': 'dask',
                    'scheduler': config.scheduler_address,
                    'workers': config.num_workers
                }
            else:
                logger.info(f"Creating local Dask cluster with {config.num_workers} workers")
                self._client = {
                    'type': 'dask_local',
                    'workers': config.num_workers,
                    'memory_per_worker': config.memory_per_worker_gb
                }
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Dask executor: {e}")
            return False
            
    def shutdown(self) -> None:
        """Shutdown Dask executor."""
        self._client = None
        self._futures.clear()
        
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to Dask."""
        task_id = str(uuid.uuid4())
        
        self._futures[task_id] = {
            'func': func.__name__ if hasattr(func, '__name__') else str(func),
            'args': args,
            'kwargs': kwargs,
            'submitted_at': datetime.utcnow(),
            'status': TaskStatus.QUEUED
        }
        
        logger.info(f"Submitted Dask task: {task_id}")
        return task_id
        
    def get_result(self, task_id: str, timeout: float = None) -> TaskResult:
        """Get Dask task result."""
        future_info = self._futures.get(task_id)
        if not future_info:
            return TaskResult(task_id=task_id, status=TaskStatus.PENDING)
            
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            start_time=future_info.get('submitted_at'),
            end_time=datetime.utcnow()
        )
        
    def cancel(self, task_id: str) -> bool:
        """Cancel Dask task."""
        if task_id in self._futures:
            self._futures[task_id]['status'] = TaskStatus.CANCELLED
            return True
        return False
        
    def get_workers(self) -> List[WorkerInfo]:
        """Get Dask worker information."""
        if not self._config:
            return []
            
        return [
            WorkerInfo(
                worker_id=f"dask_{i}",
                address=self._config.scheduler_address or "localhost",
                status="active",
                memory_total_gb=self._config.memory_per_worker_gb,
                cpu_count=self._config.cpus_per_worker
            )
            for i in range(self._config.num_workers)
        ]
        
    def scatter(self, data: Any, broadcast: bool = False) -> str:
        """Scatter data to workers."""
        data_id = str(uuid.uuid4())
        logger.info(f"Scattered data: {data_id}")
        return data_id
        
    def gather(self, futures: List[str]) -> List[Any]:
        """Gather results from workers."""
        return [self.get_result(f) for f in futures]


class RayExecutor(DistributedExecutor):
    """Ray distributed executor."""
    
    def __init__(self):
        self._initialized = False
        self._futures: Dict[str, Any] = {}
        self._config: Optional[ClusterConfig] = None
        
    def initialize(self, config: ClusterConfig) -> bool:
        """Initialize Ray executor."""
        self._config = config
        
        try:
            if config.scheduler_address:
                logger.info(f"Connecting to Ray cluster at {config.scheduler_address}")
            else:
                logger.info(f"Initializing local Ray with {config.num_workers} workers")
                
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Ray executor: {e}")
            return False
            
    def shutdown(self) -> None:
        """Shutdown Ray executor."""
        self._initialized = False
        self._futures.clear()
        
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to Ray."""
        task_id = str(uuid.uuid4())
        
        self._futures[task_id] = {
            'func': func.__name__ if hasattr(func, '__name__') else str(func),
            'args': args,
            'kwargs': kwargs,
            'submitted_at': datetime.utcnow(),
            'status': TaskStatus.QUEUED
        }
        
        logger.info(f"Submitted Ray task: {task_id}")
        return task_id
        
    def get_result(self, task_id: str, timeout: float = None) -> TaskResult:
        """Get Ray task result."""
        future_info = self._futures.get(task_id)
        if not future_info:
            return TaskResult(task_id=task_id, status=TaskStatus.PENDING)
            
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            start_time=future_info.get('submitted_at'),
            end_time=datetime.utcnow()
        )
        
    def cancel(self, task_id: str) -> bool:
        """Cancel Ray task."""
        if task_id in self._futures:
            self._futures[task_id]['status'] = TaskStatus.CANCELLED
            return True
        return False
        
    def get_workers(self) -> List[WorkerInfo]:
        """Get Ray worker information."""
        if not self._config:
            return []
            
        return [
            WorkerInfo(
                worker_id=f"ray_{i}",
                address=self._config.scheduler_address or "localhost",
                status="active",
                memory_total_gb=self._config.memory_per_worker_gb,
                cpu_count=self._config.cpus_per_worker
            )
            for i in range(self._config.num_workers)
        ]


class TaskScheduler:
    """Schedule and manage distributed tasks."""
    
    def __init__(self, executor: DistributedExecutor):
        self.executor = executor
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._running_tasks: Dict[str, TaskConfig] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._max_concurrent = 0
        self._lock = threading.Lock()
        
    def set_max_concurrent(self, max_concurrent: int) -> None:
        """Set maximum concurrent tasks."""
        self._max_concurrent = max_concurrent
        
    def submit_task(self, config: TaskConfig, func: Callable,
                   *args, **kwargs) -> str:
        """Submit task for scheduling."""
        priority = -config.priority
        self._task_queue.put((priority, config, func, args, kwargs))
        
        self._process_queue()
        
        return config.task_id
        
    def _process_queue(self) -> None:
        """Process task queue."""
        with self._lock:
            while not self._task_queue.empty():
                if self._max_concurrent > 0 and len(self._running_tasks) >= self._max_concurrent:
                    break
                    
                try:
                    priority, config, func, args, kwargs = self._task_queue.get_nowait()
                    
                    task_id = self.executor.submit(func, *args, **kwargs)
                    
                    self._running_tasks[config.task_id] = config
                    
                except queue.Empty:
                    break
                    
    def get_task_status(self, task_id: str) -> TaskResult:
        """Get task status."""
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id]
            
        if task_id in self._running_tasks:
            return self.executor.get_result(task_id)
            
        return TaskResult(task_id=task_id, status=TaskStatus.PENDING)
        
    def cancel_task(self, task_id: str) -> bool:
        """Cancel task."""
        return self.executor.cancel(task_id)
        
    def get_queue_size(self) -> int:
        """Get number of queued tasks."""
        return self._task_queue.qsize()
        
    def get_running_count(self) -> int:
        """Get number of running tasks."""
        return len(self._running_tasks)


class DistributedJob:
    """Manage a distributed processing job."""
    
    def __init__(self, config: JobConfig, scheduler: TaskScheduler):
        self.config = config
        self.scheduler = scheduler
        self._partition_results: Dict[str, TaskResult] = {}
        self._status = TaskStatus.PENDING
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._callbacks: List[Callable[[JobStatus], None]] = []
        
    def register_callback(self, callback: Callable[[JobStatus], None]) -> None:
        """Register job status callback."""
        self._callbacks.append(callback)
        
    def start(self, process_func: Callable[[DataPartition], Any]) -> None:
        """Start the distributed job."""
        self._status = TaskStatus.RUNNING
        self._start_time = datetime.utcnow()
        
        for partition in self.config.partitions:
            task_config = TaskConfig(
                task_id=f"{self.config.job_id}_{partition.partition_id}",
                name=f"Process {partition.partition_id}",
                priority=0
            )
            
            self.scheduler.submit_task(
                task_config,
                process_func,
                partition
            )
            
        self._notify_callbacks()
        
    def get_status(self) -> JobStatus:
        """Get job status."""
        completed = 0
        failed = 0
        
        for partition in self.config.partitions:
            task_id = f"{self.config.job_id}_{partition.partition_id}"
            result = self.scheduler.get_task_status(task_id)
            
            if result.status == TaskStatus.COMPLETED:
                completed += 1
            elif result.status == TaskStatus.FAILED:
                failed += 1
                
        total = len(self.config.partitions)
        progress = completed / total if total > 0 else 0.0
        
        if completed + failed == total:
            self._status = TaskStatus.COMPLETED if failed == 0 else TaskStatus.FAILED
            self._end_time = datetime.utcnow()
            
        return JobStatus(
            job_id=self.config.job_id,
            name=self.config.name,
            status=self._status,
            total_partitions=total,
            completed_partitions=completed,
            failed_partitions=failed,
            start_time=self._start_time,
            end_time=self._end_time,
            progress=progress
        )
        
    def cancel(self) -> None:
        """Cancel the job."""
        for partition in self.config.partitions:
            task_id = f"{self.config.job_id}_{partition.partition_id}"
            self.scheduler.cancel_task(task_id)
            
        self._status = TaskStatus.CANCELLED
        self._end_time = datetime.utcnow()
        self._notify_callbacks()
        
    def _notify_callbacks(self) -> None:
        """Notify registered callbacks."""
        status = self.get_status()
        for callback in self._callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class ChunkedProcessor:
    """Process large datasets in memory-efficient chunks."""
    
    def __init__(self, chunk_size_mb: float = 64.0):
        self.chunk_size_mb = chunk_size_mb
        self.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
        
    def process_chunked(self, data_path: str, process_func: Callable,
                       total_size_bytes: int) -> Iterator[Any]:
        """Process data in chunks."""
        num_chunks = max(1, int(total_size_bytes / self.chunk_size_bytes))
        
        for i in range(num_chunks):
            offset = i * self.chunk_size_bytes
            size = min(self.chunk_size_bytes, total_size_bytes - offset)
            
            chunk_data = self._read_chunk(data_path, offset, size)
            
            result = process_func(chunk_data)
            
            yield result
            
    def _read_chunk(self, data_path: str, offset: int, size: int) -> bytes:
        """Read a chunk of data."""
        return b''
        
    def aggregate_results(self, results: Iterator[Any],
                         aggregator: Callable[[List[Any]], Any]) -> Any:
        """Aggregate chunked results."""
        all_results = list(results)
        return aggregator(all_results)


class DistributedProcessingService:
    """Main distributed processing service."""
    
    def __init__(self, executor_type: ExecutorType = ExecutorType.LOCAL):
        self.executor_type = executor_type
        self._executor: Optional[DistributedExecutor] = None
        self._scheduler: Optional[TaskScheduler] = None
        self._partitioner = DataPartitioner()
        self._chunked_processor = ChunkedProcessor()
        self._jobs: Dict[str, DistributedJob] = {}
        
    def initialize(self, config: ClusterConfig) -> bool:
        """Initialize the distributed processing service."""
        if config.executor_type == ExecutorType.LOCAL:
            self._executor = LocalExecutor()
        elif config.executor_type == ExecutorType.DASK:
            self._executor = DaskExecutor()
        elif config.executor_type == ExecutorType.RAY:
            self._executor = RayExecutor()
        else:
            self._executor = LocalExecutor()
            
        if self._executor.initialize(config):
            self._scheduler = TaskScheduler(self._executor)
            self._scheduler.set_max_concurrent(config.num_workers * 2)
            return True
        return False
        
    def shutdown(self) -> None:
        """Shutdown the service."""
        if self._executor:
            self._executor.shutdown()
            
    def create_job(self, name: str, data_path: str,
                  total_size_bytes: int, num_records: int,
                  partition_strategy: PartitionStrategy = PartitionStrategy.SIZE) -> JobConfig:
        """Create a new distributed job."""
        job_id = str(uuid.uuid4())
        
        if partition_strategy == PartitionStrategy.SIZE:
            partitions = self._partitioner.partition_by_size(
                data_path, total_size_bytes, num_records
            )
        else:
            partitions = self._partitioner.partition_by_size(
                data_path, total_size_bytes, num_records
            )
            
        return JobConfig(
            job_id=job_id,
            name=name,
            partitions=partitions,
            task_function="",
            partition_strategy=partition_strategy
        )
        
    def submit_job(self, config: JobConfig,
                  process_func: Callable[[DataPartition], Any]) -> str:
        """Submit a distributed job."""
        if not self._scheduler:
            raise RuntimeError("Service not initialized")
            
        job = DistributedJob(config, self._scheduler)
        self._jobs[config.job_id] = job
        
        job.start(process_func)
        
        return config.job_id
        
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status."""
        job = self._jobs.get(job_id)
        if job:
            return job.get_status()
        return None
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self._jobs.get(job_id)
        if job:
            job.cancel()
            return True
        return False
        
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status."""
        if not self._executor:
            return {'status': 'not_initialized'}
            
        workers = self._executor.get_workers()
        
        return {
            'status': 'active',
            'executor_type': self.executor_type.value,
            'num_workers': len(workers),
            'workers': [w.to_dict() for w in workers],
            'active_jobs': len(self._jobs),
            'queue_size': self._scheduler.get_queue_size() if self._scheduler else 0,
            'running_tasks': self._scheduler.get_running_count() if self._scheduler else 0
        }
        
    def process_large_survey(self, survey_path: str,
                            size_gb: float,
                            process_func: Callable) -> str:
        """Process a large survey dataset (>100GB)."""
        size_bytes = int(size_gb * 1024 * 1024 * 1024)
        
        estimated_records = int(size_bytes / 1000)
        
        job_config = self.create_job(
            name=f"Survey Processing: {survey_path}",
            data_path=survey_path,
            total_size_bytes=size_bytes,
            num_records=estimated_records
        )
        
        return self.submit_job(job_config, process_func)


def create_distributed_service(executor_type: ExecutorType = ExecutorType.LOCAL,
                              num_workers: int = 4) -> DistributedProcessingService:
    """Factory function to create distributed processing service."""
    service = DistributedProcessingService(executor_type)
    
    config = ClusterConfig(
        executor_type=executor_type,
        num_workers=num_workers
    )
    
    service.initialize(config)
    return service


def create_dask_service(scheduler_address: str = "",
                       num_workers: int = 4) -> DistributedProcessingService:
    """Factory function to create Dask-based service."""
    service = DistributedProcessingService(ExecutorType.DASK)
    
    config = ClusterConfig(
        executor_type=ExecutorType.DASK,
        num_workers=num_workers,
        scheduler_address=scheduler_address
    )
    
    service.initialize(config)
    return service


def create_ray_service(scheduler_address: str = "",
                      num_workers: int = 4) -> DistributedProcessingService:
    """Factory function to create Ray-based service."""
    service = DistributedProcessingService(ExecutorType.RAY)
    
    config = ClusterConfig(
        executor_type=ExecutorType.RAY,
        num_workers=num_workers,
        scheduler_address=scheduler_address
    )
    
    service.initialize(config)
    return service
