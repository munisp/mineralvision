"""
Real-time Data Streaming for MineralVision Digital Twin.

This module provides real-time data streaming capabilities for the digital twin,
enabling live updates from sensors, simulations, and external data sources.
"""

import numpy as np
import asyncio
import threading
import queue
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import logging
import weakref

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams."""
    SENSOR = "sensor"
    SIMULATION = "simulation"
    TELEMETRY = "telemetry"
    EVENT = "event"
    COMMAND = "command"
    STATE = "state"


class StreamPriority(Enum):
    """Priority levels for streams."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class StreamMessage:
    """Message in a data stream."""
    stream_id: str
    stream_type: StreamType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    priority: StreamPriority = StreamPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    sequence_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stream_id': self.stream_id,
            'stream_type': self.stream_type.value,
            'data': self.data if isinstance(self.data, (dict, list, str, int, float, bool)) else str(self.data),
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'metadata': self.metadata,
            'sequence_number': self.sequence_number
        }
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamMessage':
        return cls(
            stream_id=data['stream_id'],
            stream_type=StreamType(data['stream_type']),
            data=data['data'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            priority=StreamPriority(data['priority']),
            metadata=data.get('metadata', {}),
            sequence_number=data.get('sequence_number', 0)
        )


@dataclass
class StreamConfig:
    """Configuration for a data stream."""
    stream_id: str
    stream_type: StreamType
    buffer_size: int = 1000
    max_rate_hz: float = 100.0
    enable_compression: bool = False
    enable_batching: bool = False
    batch_size: int = 10
    batch_timeout_ms: int = 100
    priority: StreamPriority = StreamPriority.NORMAL


@dataclass
class StreamMetrics:
    """Metrics for stream monitoring."""
    messages_received: int = 0
    messages_sent: int = 0
    messages_dropped: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            'messages_received': self.messages_received,
            'messages_sent': self.messages_sent,
            'messages_dropped': self.messages_dropped,
            'bytes_received': self.bytes_received,
            'bytes_sent': self.bytes_sent,
            'avg_latency_ms': self.avg_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'errors': self.errors,
            'uptime_seconds': uptime,
            'throughput_msg_per_sec': self.messages_received / uptime if uptime > 0 else 0
        }


class StreamBuffer:
    """Thread-safe buffer for stream messages."""
    
    def __init__(self, max_size: int = 1000, priority_queue: bool = False):
        self.max_size = max_size
        self.priority_queue = priority_queue
        
        if priority_queue:
            self._buffers = {p: deque(maxlen=max_size // 4) for p in StreamPriority}
        else:
            self._buffer = deque(maxlen=max_size)
            
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self.dropped_count = 0
        
    def put(self, message: StreamMessage, timeout: float = 1.0) -> bool:
        """Put a message in the buffer."""
        with self._lock:
            if self.priority_queue:
                buffer = self._buffers[message.priority]
                if len(buffer) >= self.max_size // 4:
                    self.dropped_count += 1
                    return False
                buffer.append(message)
            else:
                if len(self._buffer) >= self.max_size:
                    self.dropped_count += 1
                    return False
                self._buffer.append(message)
                
            self._not_empty.notify()
            return True
            
    def get(self, timeout: float = 1.0) -> Optional[StreamMessage]:
        """Get a message from the buffer (highest priority first)."""
        with self._not_empty:
            deadline = time.time() + timeout
            
            while True:
                if self.priority_queue:
                    # Check buffers in priority order
                    for priority in reversed(StreamPriority):
                        if self._buffers[priority]:
                            return self._buffers[priority].popleft()
                else:
                    if self._buffer:
                        return self._buffer.popleft()
                        
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                    
                self._not_empty.wait(timeout=remaining)
                
    def get_batch(self, batch_size: int, timeout: float = 1.0) -> List[StreamMessage]:
        """Get a batch of messages."""
        messages = []
        deadline = time.time() + timeout
        
        while len(messages) < batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
                
            msg = self.get(timeout=remaining)
            if msg:
                messages.append(msg)
            else:
                break
                
        return messages
        
    @property
    def size(self) -> int:
        with self._lock:
            if self.priority_queue:
                return sum(len(b) for b in self._buffers.values())
            return len(self._buffer)


class DataStream:
    """Individual data stream with buffering and rate limiting."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.buffer = StreamBuffer(config.buffer_size, priority_queue=True)
        self.metrics = StreamMetrics()
        
        # Rate limiting
        self._min_interval = 1.0 / config.max_rate_hz
        self._last_send_time = 0.0
        
        # Subscribers
        self._subscribers: List[Callable[[StreamMessage], None]] = []
        self._async_subscribers: List[Callable[[StreamMessage], Any]] = []
        
        # Sequence number
        self._sequence = 0
        self._lock = threading.Lock()
        
        # Running state
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        
    def subscribe(self, callback: Callable[[StreamMessage], None]) -> None:
        """Subscribe to stream messages."""
        self._subscribers.append(callback)
        
    def subscribe_async(self, callback: Callable[[StreamMessage], Any]) -> None:
        """Subscribe with async callback."""
        self._async_subscribers.append(callback)
        
    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from stream."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
        if callback in self._async_subscribers:
            self._async_subscribers.remove(callback)
            
    def publish(self, data: Any, metadata: Optional[Dict] = None) -> bool:
        """Publish data to the stream."""
        with self._lock:
            self._sequence += 1
            seq = self._sequence
            
        message = StreamMessage(
            stream_id=self.config.stream_id,
            stream_type=self.config.stream_type,
            data=data,
            priority=self.config.priority,
            metadata=metadata or {},
            sequence_number=seq
        )
        
        success = self.buffer.put(message)
        
        if success:
            self.metrics.messages_received += 1
            data_size = len(json.dumps(message.to_dict()))
            self.metrics.bytes_received += data_size
        else:
            self.metrics.messages_dropped += 1
            
        return success
        
    def _process_messages(self) -> None:
        """Process messages from buffer and notify subscribers."""
        while self._running:
            # Rate limiting
            now = time.time()
            elapsed = now - self._last_send_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
                
            # Get message(s)
            if self.config.enable_batching:
                messages = self.buffer.get_batch(
                    self.config.batch_size,
                    timeout=self.config.batch_timeout_ms / 1000
                )
            else:
                msg = self.buffer.get(timeout=0.1)
                messages = [msg] if msg else []
                
            if not messages:
                continue
                
            self._last_send_time = time.time()
            
            # Notify subscribers
            for message in messages:
                latency = (datetime.now() - message.timestamp).total_seconds() * 1000
                self.metrics.avg_latency_ms = (
                    self.metrics.avg_latency_ms * 0.9 + latency * 0.1
                )
                self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency)
                
                for callback in self._subscribers:
                    try:
                        callback(message)
                        self.metrics.messages_sent += 1
                    except Exception as e:
                        logger.error(f"Subscriber error: {e}")
                        self.metrics.errors += 1
                        
    def start(self) -> None:
        """Start stream processing."""
        if self._running:
            return
            
        self._running = True
        self._processor_thread = threading.Thread(
            target=self._process_messages,
            daemon=True
        )
        self._processor_thread.start()
        
    def stop(self) -> None:
        """Stop stream processing."""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=2.0)
            self._processor_thread = None
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get stream metrics."""
        return self.metrics.to_dict()


class StreamHub:
    """Central hub for managing multiple data streams."""
    
    def __init__(self):
        self.streams: Dict[str, DataStream] = {}
        self._global_subscribers: List[Callable[[StreamMessage], None]] = []
        self._type_subscribers: Dict[StreamType, List[Callable]] = {
            t: [] for t in StreamType
        }
        self._running = False
        
    def create_stream(self, config: StreamConfig) -> DataStream:
        """Create a new data stream."""
        stream = DataStream(config)
        self.streams[config.stream_id] = stream
        
        # Forward to global and type subscribers
        def forward_message(msg: StreamMessage):
            for callback in self._global_subscribers:
                try:
                    callback(msg)
                except Exception as e:
                    logger.error(f"Global subscriber error: {e}")
                    
            for callback in self._type_subscribers.get(msg.stream_type, []):
                try:
                    callback(msg)
                except Exception as e:
                    logger.error(f"Type subscriber error: {e}")
                    
        stream.subscribe(forward_message)
        
        return stream
        
    def get_stream(self, stream_id: str) -> Optional[DataStream]:
        """Get a stream by ID."""
        return self.streams.get(stream_id)
        
    def remove_stream(self, stream_id: str) -> None:
        """Remove a stream."""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
            del self.streams[stream_id]
            
    def subscribe_all(self, callback: Callable[[StreamMessage], None]) -> None:
        """Subscribe to all streams."""
        self._global_subscribers.append(callback)
        
    def subscribe_type(self, stream_type: StreamType,
                      callback: Callable[[StreamMessage], None]) -> None:
        """Subscribe to streams of a specific type."""
        self._type_subscribers[stream_type].append(callback)
        
    def publish(self, stream_id: str, data: Any,
               metadata: Optional[Dict] = None) -> bool:
        """Publish to a specific stream."""
        stream = self.streams.get(stream_id)
        if stream:
            return stream.publish(data, metadata)
        return False
        
    def broadcast(self, data: Any, stream_type: Optional[StreamType] = None,
                 metadata: Optional[Dict] = None) -> int:
        """Broadcast to multiple streams."""
        count = 0
        for stream in self.streams.values():
            if stream_type is None or stream.config.stream_type == stream_type:
                if stream.publish(data, metadata):
                    count += 1
        return count
        
    def start_all(self) -> None:
        """Start all streams."""
        self._running = True
        for stream in self.streams.values():
            stream.start()
            
    def stop_all(self) -> None:
        """Stop all streams."""
        self._running = False
        for stream in self.streams.values():
            stream.stop()
            
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all streams."""
        return {
            stream_id: stream.get_metrics()
            for stream_id, stream in self.streams.items()
        }


class DigitalTwinStreamManager:
    """
    Stream manager specifically for digital twin real-time updates.
    """
    
    def __init__(self):
        self.hub = StreamHub()
        
        # Pre-defined streams for digital twin
        self._setup_default_streams()
        
        # State tracking
        self.current_state: Dict[str, Any] = {}
        self._state_lock = threading.Lock()
        
        # State change callbacks
        self._state_callbacks: List[Callable[[str, Any, Any], None]] = []
        
    def _setup_default_streams(self) -> None:
        """Setup default streams for digital twin."""
        # Sensor data stream
        self.hub.create_stream(StreamConfig(
            stream_id='sensors',
            stream_type=StreamType.SENSOR,
            buffer_size=5000,
            max_rate_hz=100.0,
            priority=StreamPriority.HIGH
        ))
        
        # Simulation state stream
        self.hub.create_stream(StreamConfig(
            stream_id='simulation',
            stream_type=StreamType.SIMULATION,
            buffer_size=1000,
            max_rate_hz=60.0,
            priority=StreamPriority.NORMAL
        ))
        
        # Equipment telemetry stream
        self.hub.create_stream(StreamConfig(
            stream_id='telemetry',
            stream_type=StreamType.TELEMETRY,
            buffer_size=2000,
            max_rate_hz=50.0,
            enable_batching=True,
            batch_size=10,
            priority=StreamPriority.NORMAL
        ))
        
        # Event stream
        self.hub.create_stream(StreamConfig(
            stream_id='events',
            stream_type=StreamType.EVENT,
            buffer_size=500,
            max_rate_hz=10.0,
            priority=StreamPriority.CRITICAL
        ))
        
        # Command stream
        self.hub.create_stream(StreamConfig(
            stream_id='commands',
            stream_type=StreamType.COMMAND,
            buffer_size=100,
            max_rate_hz=10.0,
            priority=StreamPriority.CRITICAL
        ))
        
        # State synchronization stream
        self.hub.create_stream(StreamConfig(
            stream_id='state_sync',
            stream_type=StreamType.STATE,
            buffer_size=100,
            max_rate_hz=30.0,
            priority=StreamPriority.HIGH
        ))
        
    def register_state_callback(self, 
                               callback: Callable[[str, Any, Any], None]) -> None:
        """Register callback for state changes (key, old_value, new_value)."""
        self._state_callbacks.append(callback)
        
    def update_state(self, key: str, value: Any) -> None:
        """Update digital twin state and notify subscribers."""
        with self._state_lock:
            old_value = self.current_state.get(key)
            self.current_state[key] = value
            
        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                callback(key, old_value, value)
            except Exception as e:
                logger.error(f"State callback error: {e}")
                
        # Publish state change
        self.hub.publish('state_sync', {
            'key': key,
            'value': value,
            'previous': old_value
        })
        
    def get_state(self, key: Optional[str] = None) -> Any:
        """Get current state or specific key."""
        with self._state_lock:
            if key is None:
                return self.current_state.copy()
            return self.current_state.get(key)
            
    def publish_sensor_data(self, sensor_id: str, data: Any,
                           metadata: Optional[Dict] = None) -> bool:
        """Publish sensor data."""
        return self.hub.publish('sensors', {
            'sensor_id': sensor_id,
            'data': data
        }, metadata)
        
    def publish_simulation_state(self, state: Dict[str, Any]) -> bool:
        """Publish simulation state update."""
        return self.hub.publish('simulation', state)
        
    def publish_telemetry(self, equipment_id: str, 
                         telemetry: Dict[str, Any]) -> bool:
        """Publish equipment telemetry."""
        return self.hub.publish('telemetry', {
            'equipment_id': equipment_id,
            'telemetry': telemetry
        })
        
    def publish_event(self, event_type: str, event_data: Any,
                     severity: str = 'info') -> bool:
        """Publish an event."""
        return self.hub.publish('events', {
            'event_type': event_type,
            'data': event_data,
            'severity': severity
        })
        
    def send_command(self, target: str, command: str,
                    parameters: Optional[Dict] = None) -> bool:
        """Send a command."""
        return self.hub.publish('commands', {
            'target': target,
            'command': command,
            'parameters': parameters or {}
        })
        
    def subscribe_sensors(self, callback: Callable[[StreamMessage], None]) -> None:
        """Subscribe to sensor data."""
        stream = self.hub.get_stream('sensors')
        if stream:
            stream.subscribe(callback)
            
    def subscribe_simulation(self, callback: Callable[[StreamMessage], None]) -> None:
        """Subscribe to simulation updates."""
        stream = self.hub.get_stream('simulation')
        if stream:
            stream.subscribe(callback)
            
    def subscribe_telemetry(self, callback: Callable[[StreamMessage], None]) -> None:
        """Subscribe to telemetry data."""
        stream = self.hub.get_stream('telemetry')
        if stream:
            stream.subscribe(callback)
            
    def subscribe_events(self, callback: Callable[[StreamMessage], None]) -> None:
        """Subscribe to events."""
        stream = self.hub.get_stream('events')
        if stream:
            stream.subscribe(callback)
            
    def subscribe_commands(self, callback: Callable[[StreamMessage], None]) -> None:
        """Subscribe to commands."""
        stream = self.hub.get_stream('commands')
        if stream:
            stream.subscribe(callback)
            
    def start(self) -> None:
        """Start all streams."""
        self.hub.start_all()
        logger.info("Digital twin stream manager started")
        
    def stop(self) -> None:
        """Stop all streams."""
        self.hub.stop_all()
        logger.info("Digital twin stream manager stopped")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all stream metrics."""
        return self.hub.get_all_metrics()


class AsyncDigitalTwinStreamer:
    """
    Async streaming for digital twin with WebSocket support.
    """
    
    def __init__(self):
        self.manager = DigitalTwinStreamManager()
        self._websocket_clients: Set[Any] = set()
        self._running = False
        
    async def start(self) -> None:
        """Start async streaming."""
        self._running = True
        self.manager.start()
        
        # Setup forwarding to WebSocket clients
        def forward_to_websockets(msg: StreamMessage):
            asyncio.create_task(self._broadcast_to_websockets(msg))
            
        self.manager.hub.subscribe_all(forward_to_websockets)
        
    async def stop(self) -> None:
        """Stop async streaming."""
        self._running = False
        self.manager.stop()
        
        # Close all WebSocket connections
        for client in list(self._websocket_clients):
            try:
                await client.close()
            except Exception:
                pass
        self._websocket_clients.clear()
        
    async def _broadcast_to_websockets(self, msg: StreamMessage) -> None:
        """Broadcast message to all WebSocket clients."""
        if not self._websocket_clients:
            return
            
        message_json = msg.to_json()
        
        disconnected = set()
        for client in self._websocket_clients:
            try:
                await client.send(message_json)
            except Exception:
                disconnected.add(client)
                
        self._websocket_clients -= disconnected
        
    def register_websocket(self, websocket: Any) -> None:
        """Register a WebSocket client."""
        self._websocket_clients.add(websocket)
        
    def unregister_websocket(self, websocket: Any) -> None:
        """Unregister a WebSocket client."""
        self._websocket_clients.discard(websocket)
        
    async def handle_websocket(self, websocket: Any) -> None:
        """Handle WebSocket connection."""
        self.register_websocket(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from WebSocket: {message}")
        finally:
            self.unregister_websocket(websocket)
            
    async def _handle_websocket_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        msg_type = data.get('type')
        
        if msg_type == 'subscribe':
            # Handle subscription request
            stream_id = data.get('stream_id')
            logger.info(f"WebSocket subscription to {stream_id}")
            
        elif msg_type == 'command':
            # Handle command
            self.manager.send_command(
                data.get('target', ''),
                data.get('command', ''),
                data.get('parameters')
            )
            
        elif msg_type == 'state_update':
            # Handle state update
            key = data.get('key')
            value = data.get('value')
            if key:
                self.manager.update_state(key, value)


class SimulationStreamBridge:
    """
    Bridge between simulation and streaming system.
    """
    
    def __init__(self, stream_manager: DigitalTwinStreamManager):
        self.stream_manager = stream_manager
        self._simulation_running = False
        
    def connect_simulation(self, simulation: Any) -> None:
        """Connect a simulation to the streaming system."""
        # Register progress callback
        if hasattr(simulation, 'register_progress_callback'):
            simulation.register_progress_callback(self._on_progress)
            
        # Register state callback
        if hasattr(simulation, 'register_state_callback'):
            simulation.register_state_callback(self._on_state_update)
            
    def _on_progress(self, progress: float, message: str) -> None:
        """Handle simulation progress update."""
        self.stream_manager.publish_event('simulation_progress', {
            'progress': progress,
            'message': message
        })
        
    def _on_state_update(self, state: Dict[str, Any]) -> None:
        """Handle simulation state update."""
        self.stream_manager.publish_simulation_state(state)
        
        # Update equipment positions if present
        if 'equipment' in state:
            for eq_id, eq_state in state['equipment'].items():
                self.stream_manager.publish_telemetry(eq_id, eq_state)
                
    def start_streaming(self) -> None:
        """Start streaming simulation data."""
        self._simulation_running = True
        self.stream_manager.start()
        
    def stop_streaming(self) -> None:
        """Stop streaming simulation data."""
        self._simulation_running = False
        self.stream_manager.stop()


def create_stream_manager() -> DigitalTwinStreamManager:
    """Factory function to create a stream manager."""
    return DigitalTwinStreamManager()


def create_async_streamer() -> AsyncDigitalTwinStreamer:
    """Factory function to create an async streamer."""
    return AsyncDigitalTwinStreamer()


def create_simulation_bridge(stream_manager: DigitalTwinStreamManager) -> SimulationStreamBridge:
    """Factory function to create a simulation bridge."""
    return SimulationStreamBridge(stream_manager)
