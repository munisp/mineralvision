"""
Real-time Streaming Sensor Fusion for MineralVision.

This module provides streaming capabilities for continuous sensor data fusion,
supporting real-time processing, windowing, and incremental updates.
"""

import numpy as np
import asyncio
import threading
import queue
import time
from typing import List, Dict, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import logging
import json
import weakref

from .core import SensorData, SensorFusionAlgorithm, SensorType, DataDimension
from .kalman_fusion import StandardKalmanFilter, KalmanConfig, KalmanState

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming processing modes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"


class WindowType(Enum):
    """Types of windowing for streaming data."""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"
    GLOBAL = "global"


@dataclass
class StreamConfig:
    """Configuration for streaming fusion."""
    window_size: int = 100
    window_type: WindowType = WindowType.SLIDING
    slide_interval: int = 10
    session_gap: timedelta = timedelta(seconds=30)
    max_latency_ms: int = 100
    buffer_size: int = 1000
    num_workers: int = 4
    mode: StreamingMode = StreamingMode.ASYNCHRONOUS
    checkpoint_interval: int = 1000
    enable_backpressure: bool = True
    max_queue_size: int = 10000


@dataclass
class StreamingMetrics:
    """Metrics for monitoring streaming performance."""
    messages_processed: int = 0
    messages_dropped: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    fusion_count: int = 0
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def avg_latency_ms(self) -> float:
        if self.messages_processed == 0:
            return 0.0
        return self.total_latency_ms / self.messages_processed
        
    @property
    def throughput(self) -> float:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return 0.0
        return self.messages_processed / elapsed
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'messages_processed': self.messages_processed,
            'messages_dropped': self.messages_dropped,
            'avg_latency_ms': self.avg_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'min_latency_ms': self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0,
            'fusion_count': self.fusion_count,
            'errors': self.errors,
            'throughput': self.throughput,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }


class SensorMessage:
    """Message container for streaming sensor data."""
    
    def __init__(self, sensor_id: str, sensor_type: SensorType,
                 data: np.ndarray, timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict] = None):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.ingestion_time = datetime.now()
        
    @property
    def latency_ms(self) -> float:
        return (self.ingestion_time - self.timestamp).total_seconds() * 1000
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'data': self.data.tolist(),
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorMessage':
        return cls(
            sensor_id=data['sensor_id'],
            sensor_type=SensorType(data['sensor_type']),
            data=np.array(data['data']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class StreamingWindow:
    """Window for collecting and processing streaming data."""
    
    def __init__(self, window_type: WindowType, window_size: int,
                 slide_interval: int = 1, session_gap: timedelta = timedelta(seconds=30)):
        self.window_type = window_type
        self.window_size = window_size
        self.slide_interval = slide_interval
        self.session_gap = session_gap
        
        self.buffer: deque = deque(maxlen=window_size)
        self.last_message_time: Optional[datetime] = None
        self.window_start_time: Optional[datetime] = None
        self.message_count = 0
        
    def add(self, message: SensorMessage) -> bool:
        """
        Add a message to the window.
        
        Returns:
            True if window is ready for processing
        """
        self.buffer.append(message)
        self.message_count += 1
        
        if self.window_start_time is None:
            self.window_start_time = message.timestamp
            
        self.last_message_time = message.timestamp
        
        return self._is_window_ready()
        
    def _is_window_ready(self) -> bool:
        """Check if window is ready for processing."""
        if self.window_type == WindowType.TUMBLING:
            return len(self.buffer) >= self.window_size
            
        elif self.window_type == WindowType.SLIDING:
            return len(self.buffer) >= self.window_size and \
                   self.message_count % self.slide_interval == 0
                   
        elif self.window_type == WindowType.SESSION:
            if self.last_message_time and self.window_start_time:
                gap = self.last_message_time - self.window_start_time
                return gap >= self.session_gap
            return False
            
        elif self.window_type == WindowType.GLOBAL:
            return True  # Always ready
            
        return False
        
    def get_data(self) -> List[SensorMessage]:
        """Get current window data."""
        return list(self.buffer)
        
    def clear(self) -> None:
        """Clear the window (for tumbling windows)."""
        if self.window_type == WindowType.TUMBLING:
            self.buffer.clear()
            self.window_start_time = None
        elif self.window_type == WindowType.SESSION:
            self.buffer.clear()
            self.window_start_time = self.last_message_time


class StreamingBuffer:
    """Thread-safe buffer for streaming data."""
    
    def __init__(self, max_size: int = 10000, enable_backpressure: bool = True):
        self.max_size = max_size
        self.enable_backpressure = enable_backpressure
        self.queue = queue.Queue(maxsize=max_size if enable_backpressure else 0)
        self.dropped_count = 0
        self._lock = threading.Lock()
        
    def put(self, message: SensorMessage, timeout: float = 1.0) -> bool:
        """
        Put a message in the buffer.
        
        Returns:
            True if successful, False if dropped
        """
        try:
            self.queue.put(message, timeout=timeout)
            return True
        except queue.Full:
            with self._lock:
                self.dropped_count += 1
            return False
            
    def get(self, timeout: float = 1.0) -> Optional[SensorMessage]:
        """Get a message from the buffer."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_batch(self, batch_size: int, timeout: float = 1.0) -> List[SensorMessage]:
        """Get a batch of messages."""
        messages = []
        deadline = time.time() + timeout
        
        while len(messages) < batch_size and time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.get(timeout=max(0.001, remaining))
            if msg:
                messages.append(msg)
            else:
                break
                
        return messages
        
    @property
    def size(self) -> int:
        return self.queue.qsize()
        
    @property
    def is_full(self) -> bool:
        return self.queue.full()


class IncrementalFusion:
    """Incremental fusion algorithm for streaming data."""
    
    def __init__(self, fusion_algorithm: SensorFusionAlgorithm,
                 state_dim: int = 100):
        self.fusion_algorithm = fusion_algorithm
        self.state_dim = state_dim
        
        # Kalman filter for incremental updates
        config = KalmanConfig(
            state_dim=state_dim,
            measurement_dim=state_dim,
            process_noise=0.01,
            measurement_noise=0.1
        )
        self.kalman_filter = StandardKalmanFilter(config)
        
        # Running statistics
        self.running_mean: Optional[np.ndarray] = None
        self.running_var: Optional[np.ndarray] = None
        self.count = 0
        
        # Sensor-specific buffers
        self.sensor_buffers: Dict[str, deque] = {}
        self.sensor_weights: Dict[str, float] = {}
        
    def initialize(self, initial_state: np.ndarray) -> None:
        """Initialize the incremental fusion state."""
        state = initial_state.flatten()[:self.state_dim]
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
            
        self.kalman_filter.initialize(state)
        self.running_mean = state.copy()
        self.running_var = np.ones(self.state_dim)
        self.count = 1
        
    def update(self, message: SensorMessage) -> np.ndarray:
        """
        Update fusion state with new sensor message.
        
        Args:
            message: New sensor message
            
        Returns:
            Updated fused state
        """
        # Flatten and pad/truncate data
        data = message.data.flatten()[:self.state_dim]
        if len(data) < self.state_dim:
            data = np.pad(data, (0, self.state_dim - len(data)))
            
        # Initialize if needed
        if self.running_mean is None:
            self.initialize(data)
            return data
            
        # Get sensor weight
        weight = self.sensor_weights.get(message.sensor_id, 1.0)
        
        # Update Kalman filter
        self.kalman_filter.predict()
        R = np.eye(self.state_dim) * (0.1 / weight)
        state = self.kalman_filter.update(data, R)
        
        # Update running statistics (Welford's algorithm)
        self.count += 1
        delta = data - self.running_mean
        self.running_mean += delta / self.count
        delta2 = data - self.running_mean
        self.running_var += delta * delta2
        
        return state.mean
        
    def update_batch(self, messages: List[SensorMessage]) -> np.ndarray:
        """
        Update fusion state with a batch of messages.
        
        Args:
            messages: List of sensor messages
            
        Returns:
            Updated fused state
        """
        for message in messages:
            result = self.update(message)
        return result
        
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current fused state and uncertainty."""
        if self.kalman_filter.state is None:
            return np.zeros(self.state_dim), np.ones(self.state_dim)
            
        state = self.kalman_filter.get_state()
        return state.mean, np.diag(state.covariance)
        
    def set_sensor_weight(self, sensor_id: str, weight: float) -> None:
        """Set weight for a specific sensor."""
        self.sensor_weights[sensor_id] = weight


class StreamingFusionPipeline:
    """
    Complete streaming fusion pipeline with async processing.
    """
    
    def __init__(self, config: StreamConfig,
                 fusion_algorithm: Optional[SensorFusionAlgorithm] = None):
        self.config = config
        self.fusion_algorithm = fusion_algorithm
        
        # Buffers and windows per sensor
        self.input_buffer = StreamingBuffer(
            max_size=config.max_queue_size,
            enable_backpressure=config.enable_backpressure
        )
        self.sensor_windows: Dict[str, StreamingWindow] = {}
        
        # Incremental fusion
        self.incremental_fusion = IncrementalFusion(
            fusion_algorithm=fusion_algorithm,
            state_dim=100
        )
        
        # Output buffer
        self.output_buffer = StreamingBuffer(max_size=config.buffer_size)
        
        # Metrics
        self.metrics = StreamingMetrics()
        
        # Control
        self._running = False
        self._workers: List[threading.Thread] = []
        self._callbacks: List[Callable] = []
        
        # Checkpointing
        self._checkpoint_counter = 0
        self._last_checkpoint: Optional[Dict] = None
        
    def register_callback(self, callback: Callable[[np.ndarray, datetime], None]) -> None:
        """Register a callback for fusion results."""
        self._callbacks.append(callback)
        
    def unregister_callback(self, callback: Callable) -> None:
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            
    def ingest(self, message: SensorMessage) -> bool:
        """
        Ingest a sensor message into the pipeline.
        
        Returns:
            True if message was accepted
        """
        success = self.input_buffer.put(message)
        
        if success:
            self.metrics.messages_processed += 1
            latency = message.latency_ms
            self.metrics.total_latency_ms += latency
            self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency)
            self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency)
        else:
            self.metrics.messages_dropped += 1
            
        return success
        
    def ingest_batch(self, messages: List[SensorMessage]) -> int:
        """
        Ingest a batch of messages.
        
        Returns:
            Number of messages accepted
        """
        accepted = 0
        for message in messages:
            if self.ingest(message):
                accepted += 1
        return accepted
        
    def _get_or_create_window(self, sensor_id: str) -> StreamingWindow:
        """Get or create a window for a sensor."""
        if sensor_id not in self.sensor_windows:
            self.sensor_windows[sensor_id] = StreamingWindow(
                window_type=self.config.window_type,
                window_size=self.config.window_size,
                slide_interval=self.config.slide_interval,
                session_gap=self.config.session_gap
            )
        return self.sensor_windows[sensor_id]
        
    def _process_message(self, message: SensorMessage) -> Optional[np.ndarray]:
        """Process a single message."""
        try:
            # Add to sensor window
            window = self._get_or_create_window(message.sensor_id)
            window_ready = window.add(message)
            
            # Update incremental fusion
            result = self.incremental_fusion.update(message)
            
            # If window is ready, perform full fusion
            if window_ready:
                self.metrics.fusion_count += 1
                window.clear()
                
            # Checkpointing
            self._checkpoint_counter += 1
            if self._checkpoint_counter >= self.config.checkpoint_interval:
                self._create_checkpoint()
                self._checkpoint_counter = 0
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics.errors += 1
            return None
            
    def _worker_loop(self) -> None:
        """Worker thread loop for processing messages."""
        while self._running:
            message = self.input_buffer.get(timeout=0.1)
            if message is None:
                continue
                
            result = self._process_message(message)
            
            if result is not None:
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(result, message.timestamp)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
                # Put result in output buffer
                output_msg = SensorMessage(
                    sensor_id='fused',
                    sensor_type=SensorType.CUSTOM,
                    data=result,
                    timestamp=message.timestamp
                )
                self.output_buffer.put(output_msg, timeout=0.1)
                
    def start(self) -> None:
        """Start the streaming pipeline."""
        if self._running:
            return
            
        self._running = True
        self.metrics = StreamingMetrics()
        
        # Start worker threads
        for i in range(self.config.num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)
            
        logger.info(f"Started streaming pipeline with {self.config.num_workers} workers")
        
    def stop(self) -> None:
        """Stop the streaming pipeline."""
        self._running = False
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
            
        self._workers.clear()
        logger.info("Stopped streaming pipeline")
        
    def get_result(self, timeout: float = 1.0) -> Optional[SensorMessage]:
        """Get a fusion result from the output buffer."""
        return self.output_buffer.get(timeout=timeout)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.to_dict()
        
    def _create_checkpoint(self) -> None:
        """Create a checkpoint of the current state."""
        state, uncertainty = self.incremental_fusion.get_state()
        self._last_checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'state': state.tolist(),
            'uncertainty': uncertainty.tolist(),
            'metrics': self.metrics.to_dict(),
            'sensor_weights': self.incremental_fusion.sensor_weights.copy()
        }
        
    def get_checkpoint(self) -> Optional[Dict]:
        """Get the last checkpoint."""
        return self._last_checkpoint
        
    def restore_checkpoint(self, checkpoint: Dict) -> None:
        """Restore from a checkpoint."""
        state = np.array(checkpoint['state'])
        self.incremental_fusion.initialize(state)
        self.incremental_fusion.sensor_weights = checkpoint.get('sensor_weights', {})


class AsyncStreamingFusion:
    """
    Async streaming fusion using asyncio for high-performance processing.
    """
    
    def __init__(self, config: StreamConfig,
                 fusion_algorithm: Optional[SensorFusionAlgorithm] = None):
        self.config = config
        self.fusion_algorithm = fusion_algorithm
        
        # Async queues
        self.input_queue: Optional[asyncio.Queue] = None
        self.output_queue: Optional[asyncio.Queue] = None
        
        # Incremental fusion
        self.incremental_fusion = IncrementalFusion(
            fusion_algorithm=fusion_algorithm,
            state_dim=100
        )
        
        # Metrics
        self.metrics = StreamingMetrics()
        
        # Control
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._callbacks: List[Callable] = []
        
    async def initialize(self) -> None:
        """Initialize async resources."""
        self.input_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.output_queue = asyncio.Queue(maxsize=self.config.buffer_size)
        
    async def ingest(self, message: SensorMessage) -> bool:
        """Ingest a message asynchronously."""
        try:
            await asyncio.wait_for(
                self.input_queue.put(message),
                timeout=self.config.max_latency_ms / 1000
            )
            self.metrics.messages_processed += 1
            return True
        except asyncio.TimeoutError:
            self.metrics.messages_dropped += 1
            return False
            
    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.input_queue.get(),
                    timeout=0.1
                )
                
                # Process message
                result = self.incremental_fusion.update(message)
                self.metrics.fusion_count += 1
                
                # Create output message
                output_msg = SensorMessage(
                    sensor_id='fused',
                    sensor_type=SensorType.CUSTOM,
                    data=result,
                    timestamp=message.timestamp
                )
                
                # Put in output queue
                try:
                    self.output_queue.put_nowait(output_msg)
                except asyncio.QueueFull:
                    pass
                    
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result, message.timestamp)
                        else:
                            callback(result, message.timestamp)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
                self.metrics.errors += 1
                
    async def start(self) -> None:
        """Start async processing."""
        if self._running:
            return
            
        await self.initialize()
        self._running = True
        self.metrics = StreamingMetrics()
        
        # Start processing tasks
        for _ in range(self.config.num_workers):
            task = asyncio.create_task(self._process_loop())
            self._tasks.append(task)
            
        logger.info(f"Started async streaming with {self.config.num_workers} workers")
        
    async def stop(self) -> None:
        """Stop async processing."""
        self._running = False
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("Stopped async streaming")
        
    async def get_result(self, timeout: float = 1.0) -> Optional[SensorMessage]:
        """Get a result asynchronously."""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
            
    def register_callback(self, callback: Callable) -> None:
        """Register a callback for results."""
        self._callbacks.append(callback)


class StreamingFusionSource:
    """
    Source connector for streaming data from various sources.
    """
    
    def __init__(self, source_type: str, config: Dict[str, Any]):
        self.source_type = source_type
        self.config = config
        self._running = False
        self._callback: Optional[Callable] = None
        
    def set_callback(self, callback: Callable[[SensorMessage], None]) -> None:
        """Set callback for received messages."""
        self._callback = callback
        
    async def connect(self) -> None:
        """Connect to the data source."""
        self._running = True
        
        if self.source_type == 'websocket':
            await self._connect_websocket()
        elif self.source_type == 'kafka':
            await self._connect_kafka()
        elif self.source_type == 'mqtt':
            await self._connect_mqtt()
        elif self.source_type == 'file':
            await self._connect_file()
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")
            
    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        self._running = False
        
    async def _connect_websocket(self) -> None:
        """Connect to WebSocket source."""
        import websockets
        
        uri = self.config.get('uri', 'ws://localhost:8765')
        
        async with websockets.connect(uri) as websocket:
            while self._running:
                try:
                    data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message = SensorMessage.from_dict(json.loads(data))
                    
                    if self._callback:
                        self._callback(message)
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    
    async def _connect_kafka(self) -> None:
        """Connect to Kafka source (placeholder for aiokafka)."""
        topic = self.config.get('topic', 'sensor-data')
        bootstrap_servers = self.config.get('bootstrap_servers', 'localhost:9092')
        
        logger.info(f"Kafka consumer would connect to {bootstrap_servers}, topic: {topic}")
        
        # Simulated Kafka consumer
        while self._running:
            await asyncio.sleep(0.1)
            
    async def _connect_mqtt(self) -> None:
        """Connect to MQTT source (placeholder for aiomqtt)."""
        broker = self.config.get('broker', 'localhost')
        port = self.config.get('port', 1883)
        topic = self.config.get('topic', 'sensors/#')
        
        logger.info(f"MQTT client would connect to {broker}:{port}, topic: {topic}")
        
        while self._running:
            await asyncio.sleep(0.1)
            
    async def _connect_file(self) -> None:
        """Stream from file source."""
        file_path = self.config.get('file_path')
        delay = self.config.get('delay', 0.1)
        
        if not file_path:
            raise ValueError("file_path required for file source")
            
        with open(file_path, 'r') as f:
            for line in f:
                if not self._running:
                    break
                    
                try:
                    message = SensorMessage.from_dict(json.loads(line))
                    if self._callback:
                        self._callback(message)
                except Exception as e:
                    logger.error(f"File parse error: {e}")
                    
                await asyncio.sleep(delay)


class StreamingFusionSink:
    """
    Sink connector for outputting fused data.
    """
    
    def __init__(self, sink_type: str, config: Dict[str, Any]):
        self.sink_type = sink_type
        self.config = config
        self._running = False
        
    async def connect(self) -> None:
        """Connect to the sink."""
        self._running = True
        
    async def disconnect(self) -> None:
        """Disconnect from the sink."""
        self._running = False
        
    async def write(self, message: SensorMessage) -> bool:
        """Write a message to the sink."""
        if not self._running:
            return False
            
        try:
            if self.sink_type == 'websocket':
                return await self._write_websocket(message)
            elif self.sink_type == 'kafka':
                return await self._write_kafka(message)
            elif self.sink_type == 'file':
                return await self._write_file(message)
            elif self.sink_type == 'database':
                return await self._write_database(message)
            else:
                logger.warning(f"Unknown sink type: {self.sink_type}")
                return False
        except Exception as e:
            logger.error(f"Sink write error: {e}")
            return False
            
    async def _write_websocket(self, message: SensorMessage) -> bool:
        """Write to WebSocket sink."""
        # Placeholder - would use websockets library
        logger.debug(f"WebSocket write: {message.sensor_id}")
        return True
        
    async def _write_kafka(self, message: SensorMessage) -> bool:
        """Write to Kafka sink."""
        # Placeholder - would use aiokafka
        logger.debug(f"Kafka write: {message.sensor_id}")
        return True
        
    async def _write_file(self, message: SensorMessage) -> bool:
        """Write to file sink."""
        file_path = self.config.get('file_path', 'output.jsonl')
        
        with open(file_path, 'a') as f:
            f.write(json.dumps(message.to_dict()) + '\n')
            
        return True
        
    async def _write_database(self, message: SensorMessage) -> bool:
        """Write to database sink."""
        # Placeholder - would use asyncpg or similar
        logger.debug(f"Database write: {message.sensor_id}")
        return True


def create_streaming_pipeline(
    config: Optional[StreamConfig] = None,
    fusion_algorithm: Optional[SensorFusionAlgorithm] = None,
    async_mode: bool = False
) -> Union[StreamingFusionPipeline, AsyncStreamingFusion]:
    """
    Factory function to create a streaming fusion pipeline.
    
    Args:
        config: Streaming configuration
        fusion_algorithm: Fusion algorithm to use
        async_mode: Whether to use async processing
        
    Returns:
        Streaming pipeline instance
    """
    if config is None:
        config = StreamConfig()
        
    if async_mode:
        return AsyncStreamingFusion(config, fusion_algorithm)
    else:
        return StreamingFusionPipeline(config, fusion_algorithm)


def create_sensor_message(
    sensor_id: str,
    sensor_type: Union[str, SensorType],
    data: np.ndarray,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict] = None
) -> SensorMessage:
    """
    Convenience function to create a sensor message.
    
    Args:
        sensor_id: Unique sensor identifier
        sensor_type: Type of sensor
        data: Sensor data array
        timestamp: Optional timestamp
        metadata: Optional metadata
        
    Returns:
        SensorMessage instance
    """
    if isinstance(sensor_type, str):
        sensor_type = SensorType(sensor_type)
        
    return SensorMessage(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        data=data,
        timestamp=timestamp,
        metadata=metadata
    )
