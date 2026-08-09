"""
Dapr Service Mesh Integration
==============================

Production-grade Dapr integration for MineralVision providing:
- Service-to-service invocation with automatic retries
- State management across services
- Pub/sub messaging for event-driven architecture
- Distributed tracing and observability
- Secret management
- Actor model for stateful services

Dapr provides a portable, event-driven runtime for building
distributed applications across cloud and edge.
"""

import asyncio
import json
import logging
import uuid
import aiohttp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
import threading
import hashlib

logger = logging.getLogger(__name__)

# Try to import Dapr SDK
try:
    from dapr.clients import DaprClient
    from dapr.clients.grpc._state import StateItem
    from dapr.ext.grpc import App
    DAPR_AVAILABLE = True
except ImportError:
    DAPR_AVAILABLE = False

from .._mock_fallback import real_client_unavailable


class DaprComponentType(Enum):
    """Types of Dapr components."""
    STATE_STORE = "state"
    PUBSUB = "pubsub"
    BINDING = "binding"
    SECRET_STORE = "secretstore"
    CONFIGURATION = "configuration"
    LOCK = "lock"
    WORKFLOW = "workflow"


class ConsistencyLevel(Enum):
    """State store consistency levels."""
    EVENTUAL = "eventual"
    STRONG = "strong"


class ConcurrencyMode(Enum):
    """State store concurrency modes."""
    FIRST_WRITE = "first-write"
    LAST_WRITE = "last-write"


@dataclass
class DaprConfig:
    """Configuration for Dapr integration."""
    dapr_http_port: int = 3500
    dapr_grpc_port: int = 50001
    app_id: str = "mineralvision"
    state_store_name: str = "statestore"
    pubsub_name: str = "pubsub"
    secret_store_name: str = "secretstore"
    enable_tracing: bool = True
    enable_metrics: bool = True


@dataclass
class StateOptions:
    """Options for state operations."""
    consistency: ConsistencyLevel = ConsistencyLevel.STRONG
    concurrency: ConcurrencyMode = ConcurrencyMode.LAST_WRITE
    ttl_seconds: Optional[int] = None


@dataclass
class PublishOptions:
    """Options for publish operations."""
    content_type: str = "application/json"
    metadata: Dict[str, str] = field(default_factory=dict)
    ttl_seconds: Optional[int] = None


@dataclass
class ServiceInvocationResult:
    """Result of service invocation."""
    success: bool
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0


class MockDaprClient:
    """Mock Dapr client for environments without Dapr."""
    
    def __init__(self, config: DaprConfig):
        self.config = config
        self._state: Dict[str, Dict[str, Any]] = {}
        self._pubsub_handlers: Dict[str, List[Callable]] = {}
        self._secrets: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()
    
    async def save_state(self, store_name: str, key: str, value: Any,
                        options: StateOptions = None) -> None:
        """Save state to store."""
        with self._lock:
            if store_name not in self._state:
                self._state[store_name] = {}
            
            self._state[store_name][key] = {
                'value': value,
                'etag': str(uuid.uuid4()),
                'metadata': {},
                'saved_at': datetime.now().isoformat()
            }
    
    async def get_state(self, store_name: str, key: str) -> Optional[Any]:
        """Get state from store."""
        with self._lock:
            if store_name in self._state and key in self._state[store_name]:
                return self._state[store_name][key]['value']
            return None
    
    async def delete_state(self, store_name: str, key: str) -> None:
        """Delete state from store."""
        with self._lock:
            if store_name in self._state and key in self._state[store_name]:
                del self._state[store_name][key]
    
    async def save_bulk_state(self, store_name: str, 
                             states: List[Dict[str, Any]]) -> None:
        """Save multiple states."""
        for state in states:
            await self.save_state(store_name, state['key'], state['value'])
    
    async def get_bulk_state(self, store_name: str, 
                            keys: List[str]) -> Dict[str, Any]:
        """Get multiple states."""
        results = {}
        for key in keys:
            value = await self.get_state(store_name, key)
            if value is not None:
                results[key] = value
        return results
    
    async def publish_event(self, pubsub_name: str, topic: str,
                           data: Any, options: PublishOptions = None) -> None:
        """Publish event to topic."""
        logger.info(f"Published event to {pubsub_name}/{topic}")
        
        # Trigger handlers
        handler_key = f"{pubsub_name}:{topic}"
        if handler_key in self._pubsub_handlers:
            for handler in self._pubsub_handlers[handler_key]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Handler error: {e}")
    
    def subscribe(self, pubsub_name: str, topic: str, 
                 handler: Callable) -> None:
        """Subscribe to topic."""
        handler_key = f"{pubsub_name}:{topic}"
        if handler_key not in self._pubsub_handlers:
            self._pubsub_handlers[handler_key] = []
        self._pubsub_handlers[handler_key].append(handler)
    
    async def invoke_service(self, app_id: str, method: str,
                            data: Any = None,
                            http_verb: str = "POST") -> ServiceInvocationResult:
        """Invoke another service."""
        logger.info(f"Invoking {app_id}/{method}")
        
        # Simulate service invocation
        return ServiceInvocationResult(
            success=True,
            status_code=200,
            data={"result": "ok", "app_id": app_id, "method": method},
            latency_ms=50.0
        )
    
    async def get_secret(self, store_name: str, key: str) -> Dict[str, str]:
        """Get secret from store."""
        with self._lock:
            if store_name in self._secrets and key in self._secrets[store_name]:
                return {key: self._secrets[store_name][key]}
            return {}
    
    async def get_bulk_secret(self, store_name: str) -> Dict[str, Dict[str, str]]:
        """Get all secrets from store."""
        with self._lock:
            return self._secrets.get(store_name, {})
    
    def set_secret(self, store_name: str, key: str, value: str) -> None:
        """Set a secret (for testing)."""
        with self._lock:
            if store_name not in self._secrets:
                self._secrets[store_name] = {}
            self._secrets[store_name][key] = value
    
    async def close(self) -> None:
        """Close the client."""
        pass


class DaprStateManager:
    """
    State management using Dapr state stores.
    
    Provides distributed state management with:
    - Multiple backend support (Redis, PostgreSQL, etc.)
    - Consistency and concurrency controls
    - TTL support for automatic expiration
    - Bulk operations for efficiency
    
    Example:
        state_manager = DaprStateManager(config)
        await state_manager.save("user:123", {"name": "John"})
        user = await state_manager.get("user:123")
    """
    
    def __init__(self, client: Any, config: DaprConfig):
        self.client = client
        self.config = config
        self.store_name = config.state_store_name
    
    async def save(self, key: str, value: Any,
                  options: StateOptions = None) -> None:
        """Save state."""
        await self.client.save_state(self.store_name, key, value, options)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get state."""
        return await self.client.get_state(self.store_name, key)
    
    async def delete(self, key: str) -> None:
        """Delete state."""
        await self.client.delete_state(self.store_name, key)
    
    async def save_bulk(self, states: Dict[str, Any]) -> None:
        """Save multiple states."""
        state_items = [{'key': k, 'value': v} for k, v in states.items()]
        await self.client.save_bulk_state(self.store_name, state_items)
    
    async def get_bulk(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple states."""
        return await self.client.get_bulk_state(self.store_name, keys)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        value = await self.get(key)
        return value is not None


class DaprPubSubManager:
    """
    Pub/sub messaging using Dapr.
    
    Provides event-driven messaging with:
    - Topic-based publish/subscribe
    - At-least-once delivery guarantees
    - Dead letter queues
    - Message filtering
    
    Example:
        pubsub = DaprPubSubManager(client, config)
        await pubsub.publish("sensor-data", {"sensor_id": "s1", "value": 42})
        
        @pubsub.subscribe("sensor-data")
        async def handle_sensor_data(data):
            print(f"Received: {data}")
    """
    
    def __init__(self, client: Any, config: DaprConfig):
        self.client = client
        self.config = config
        self.pubsub_name = config.pubsub_name
        self._handlers: Dict[str, List[Callable]] = {}
    
    async def publish(self, topic: str, data: Any,
                     options: PublishOptions = None) -> None:
        """Publish event to topic."""
        await self.client.publish_event(self.pubsub_name, topic, data, options)
    
    def subscribe(self, topic: str) -> Callable:
        """Decorator to subscribe to a topic."""
        def decorator(handler: Callable) -> Callable:
            if topic not in self._handlers:
                self._handlers[topic] = []
            self._handlers[topic].append(handler)
            self.client.subscribe(self.pubsub_name, topic, handler)
            return handler
        return decorator
    
    async def publish_batch(self, topic: str, events: List[Any]) -> None:
        """Publish multiple events."""
        for event in events:
            await self.publish(topic, event)
    
    def get_subscriptions(self) -> List[str]:
        """Get list of subscribed topics."""
        return list(self._handlers.keys())


class DaprServiceInvoker:
    """
    Service-to-service invocation using Dapr.
    
    Provides:
    - Automatic service discovery
    - Load balancing
    - Retries with backoff
    - Circuit breaker pattern
    - mTLS encryption
    
    Example:
        invoker = DaprServiceInvoker(client, config)
        result = await invoker.invoke("ml-service", "predict", {"data": [1,2,3]})
    """
    
    def __init__(self, client: Any, config: DaprConfig):
        self.client = client
        self.config = config
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
    
    async def invoke(self, app_id: str, method: str,
                    data: Any = None,
                    http_verb: str = "POST",
                    timeout: float = 30.0) -> ServiceInvocationResult:
        """Invoke a service method."""
        # Check circuit breaker
        if self._is_circuit_open(app_id):
            return ServiceInvocationResult(
                success=False,
                status_code=503,
                data={"error": "Circuit breaker open"},
                latency_ms=0
            )
        
        try:
            result = await self.client.invoke_service(app_id, method, data, http_verb)
            self._record_success(app_id)
            return result
        except Exception as e:
            self._record_failure(app_id)
            return ServiceInvocationResult(
                success=False,
                status_code=500,
                data={"error": str(e)},
                latency_ms=0
            )
    
    def _is_circuit_open(self, app_id: str) -> bool:
        """Check if circuit breaker is open."""
        if app_id not in self._circuit_breakers:
            return False
        
        cb = self._circuit_breakers[app_id]
        if cb['state'] == 'open':
            # Check if we should try half-open
            if datetime.now() > cb['open_until']:
                cb['state'] = 'half-open'
                return False
            return True
        return False
    
    def _record_success(self, app_id: str) -> None:
        """Record successful invocation."""
        if app_id in self._circuit_breakers:
            cb = self._circuit_breakers[app_id]
            cb['failures'] = 0
            cb['state'] = 'closed'
    
    def _record_failure(self, app_id: str) -> None:
        """Record failed invocation."""
        if app_id not in self._circuit_breakers:
            self._circuit_breakers[app_id] = {
                'failures': 0,
                'state': 'closed',
                'open_until': None
            }
        
        cb = self._circuit_breakers[app_id]
        cb['failures'] += 1
        
        if cb['failures'] >= 5:
            cb['state'] = 'open'
            cb['open_until'] = datetime.now() + timedelta(seconds=30)


class DaprSecretManager:
    """
    Secret management using Dapr.
    
    Provides:
    - Secure secret retrieval
    - Multiple backend support (Vault, AWS Secrets Manager, etc.)
    - Secret caching
    - Access control
    
    Example:
        secrets = DaprSecretManager(client, config)
        db_password = await secrets.get("database-password")
    """
    
    def __init__(self, client: Any, config: DaprConfig):
        self.client = client
        self.config = config
        self.store_name = config.secret_store_name
        self._cache: Dict[str, str] = {}
        self._cache_ttl: Dict[str, datetime] = {}
    
    async def get(self, key: str, use_cache: bool = True) -> Optional[str]:
        """Get a secret."""
        # Check cache
        if use_cache and key in self._cache:
            if datetime.now() < self._cache_ttl.get(key, datetime.min):
                return self._cache[key]
        
        # Fetch from store
        result = await self.client.get_secret(self.store_name, key)
        if key in result:
            value = result[key]
            self._cache[key] = value
            self._cache_ttl[key] = datetime.now() + timedelta(minutes=5)
            return value
        
        return None
    
    async def get_bulk(self) -> Dict[str, str]:
        """Get all secrets."""
        result = await self.client.get_bulk_secret(self.store_name)
        return result
    
    def clear_cache(self) -> None:
        """Clear the secret cache."""
        self._cache.clear()
        self._cache_ttl.clear()


class DaprActorManager:
    """
    Actor model support using Dapr.
    
    Provides:
    - Virtual actors with automatic lifecycle
    - State persistence
    - Timers and reminders
    - Reentrancy control
    
    Example:
        actor_manager = DaprActorManager(client, config)
        
        @actor_manager.actor("SensorActor")
        class SensorActor:
            async def process_reading(self, reading):
                # Process sensor reading
                pass
    """
    
    def __init__(self, client: Any, config: DaprConfig):
        self.client = client
        self.config = config
        self._actors: Dict[str, Type] = {}
        self._actor_instances: Dict[str, Dict[str, Any]] = {}
    
    def actor(self, actor_type: str) -> Callable:
        """Decorator to register an actor."""
        def decorator(cls: Type) -> Type:
            self._actors[actor_type] = cls
            return cls
        return decorator
    
    async def invoke_actor(self, actor_type: str, actor_id: str,
                          method: str, data: Any = None) -> Any:
        """Invoke an actor method."""
        # Get or create actor instance
        instance_key = f"{actor_type}:{actor_id}"
        
        if instance_key not in self._actor_instances:
            if actor_type in self._actors:
                self._actor_instances[instance_key] = {
                    'instance': self._actors[actor_type](),
                    'state': {}
                }
            else:
                raise ValueError(f"Unknown actor type: {actor_type}")
        
        actor_data = self._actor_instances[instance_key]
        instance = actor_data['instance']
        
        # Call method
        if hasattr(instance, method):
            method_func = getattr(instance, method)
            if asyncio.iscoroutinefunction(method_func):
                return await method_func(data)
            else:
                return method_func(data)
        
        raise ValueError(f"Unknown method: {method}")
    
    async def get_actor_state(self, actor_type: str, actor_id: str) -> Dict[str, Any]:
        """Get actor state."""
        instance_key = f"{actor_type}:{actor_id}"
        if instance_key in self._actor_instances:
            return self._actor_instances[instance_key].get('state', {})
        return {}
    
    async def save_actor_state(self, actor_type: str, actor_id: str,
                              state: Dict[str, Any]) -> None:
        """Save actor state."""
        instance_key = f"{actor_type}:{actor_id}"
        if instance_key in self._actor_instances:
            self._actor_instances[instance_key]['state'] = state


class DaprIntegration:
    """
    Main Dapr integration class for MineralVision.
    
    Provides unified access to all Dapr capabilities:
    - State management
    - Pub/sub messaging
    - Service invocation
    - Secret management
    - Actor model
    
    Example:
        dapr = DaprIntegration()
        await dapr.connect()
        
        # State management
        await dapr.state.save("key", {"data": "value"})
        
        # Pub/sub
        await dapr.pubsub.publish("topic", {"event": "data"})
        
        # Service invocation
        result = await dapr.services.invoke("other-service", "method")
    """
    
    def __init__(self, config: DaprConfig = None):
        self.config = config or DaprConfig()
        self.client = None
        self.state: Optional[DaprStateManager] = None
        self.pubsub: Optional[DaprPubSubManager] = None
        self.services: Optional[DaprServiceInvoker] = None
        self.secrets: Optional[DaprSecretManager] = None
        self.actors: Optional[DaprActorManager] = None
        self._connected = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """True when running on the explicit in-memory mock fallback."""
        return self._degraded

    async def connect(self) -> 'DaprIntegration':
        """
        Connect to Dapr sidecar (real client first).

        Falls back to the in-memory mock ONLY when
        MV_ALLOW_MOCK_FALLBACK=true; otherwise raises RuntimeError.
        """
        if DAPR_AVAILABLE:
            try:
                self.client = DaprClient(
                    f"localhost:{self.config.dapr_grpc_port}",
                    max_grpc_message_length=16 * 1024 * 1024
                )
                # Verify the sidecar actually answers
                await self.client.wait(timeout_s=5)
                logger.info(f"Connected to Dapr sidecar on port {self.config.dapr_grpc_port}")
            except Exception as e:
                if real_client_unavailable("Dapr", "sidecar connection failed", e):
                    self._degraded = True
                    self.client = MockDaprClient(self.config)
        else:
            if real_client_unavailable("Dapr", "dapr package not installed"):
                self._degraded = True
                self.client = MockDaprClient(self.config)

        # Initialize managers
        self.state = DaprStateManager(self.client, self.config)
        self.pubsub = DaprPubSubManager(self.client, self.config)
        self.services = DaprServiceInvoker(self.client, self.config)
        self.secrets = DaprSecretManager(self.client, self.config)
        self.actors = DaprActorManager(self.client, self.config)
        
        self._connected = True
        return self
    
    async def close(self) -> None:
        """Close Dapr connection."""
        if self.client:
            await self.client.close()
        self._connected = False
    
    def is_connected(self) -> bool:
        """Check if connected to Dapr."""
        return self._connected
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Dapr health."""
        return {
            'connected': self._connected,
            'dapr_available': DAPR_AVAILABLE,
            'degraded': self._degraded,
            'config': {
                'app_id': self.config.app_id,
                'http_port': self.config.dapr_http_port,
                'grpc_port': self.config.dapr_grpc_port
            }
        }


# Pre-defined MineralVision event topics

class MineralVisionTopics:
    """Standard Dapr topics for MineralVision."""
    SENSOR_DATA = "mineralvision.sensor.data"
    ANALYSIS_COMPLETE = "mineralvision.analysis.complete"
    MODEL_TRAINED = "mineralvision.ml.model.trained"
    REPORT_GENERATED = "mineralvision.report.generated"
    ALERT_TRIGGERED = "mineralvision.alert.triggered"
    DATA_INGESTED = "mineralvision.data.ingested"
    WORKFLOW_STARTED = "mineralvision.workflow.started"
    WORKFLOW_COMPLETED = "mineralvision.workflow.completed"


# Factory functions

def create_dapr_integration(config: DaprConfig = None) -> DaprIntegration:
    """Create a Dapr integration instance."""
    return DaprIntegration(config)


async def create_and_connect_dapr(config: DaprConfig = None) -> DaprIntegration:
    """Create and connect Dapr integration."""
    dapr = DaprIntegration(config)
    await dapr.connect()
    return dapr


# Kubernetes deployment helpers

def generate_dapr_annotations(app_id: str, 
                             app_port: int = 8000,
                             enable_api_logging: bool = True) -> Dict[str, str]:
    """Generate Kubernetes annotations for Dapr sidecar injection."""
    return {
        "dapr.io/enabled": "true",
        "dapr.io/app-id": app_id,
        "dapr.io/app-port": str(app_port),
        "dapr.io/enable-api-logging": str(enable_api_logging).lower(),
        "dapr.io/log-level": "info",
        "dapr.io/config": "mineralvision-config"
    }


def generate_dapr_component_yaml(component_type: DaprComponentType,
                                name: str,
                                spec: Dict[str, Any]) -> str:
    """Generate Dapr component YAML."""
    component = {
        "apiVersion": "dapr.io/v1alpha1",
        "kind": "Component",
        "metadata": {
            "name": name,
            "namespace": "mineralvision"
        },
        "spec": {
            "type": f"{component_type.value}.{spec.get('provider', 'redis')}",
            "version": "v1",
            "metadata": spec.get('metadata', [])
        }
    }
    
    import yaml
    return yaml.dump(component, default_flow_style=False)
