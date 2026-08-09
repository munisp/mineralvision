"""
Apache APISIX API Gateway Integration
======================================

Production-grade API gateway integration for MineralVision providing:
- Dynamic routing and load balancing
- Rate limiting and throttling
- Authentication and authorization
- Request/response transformation
- Circuit breaker and health checks
- Observability with Prometheus metrics
- Plugin system for extensibility

APISIX provides high-performance API management with
cloud-native architecture and extensive plugin ecosystem.
"""

import asyncio
import json
import logging
import uuid
# NOTE: aiohttp was imported here but never used; removed. A real APISIX
# Admin API HTTP client (when implemented) should import it lazily.
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import re

logger = logging.getLogger(__name__)

from .._mock_fallback import probe_url, real_client_unavailable


class LoadBalancerType(Enum):
    """Load balancer algorithms."""
    ROUND_ROBIN = "roundrobin"
    WEIGHTED = "chash"
    LEAST_CONN = "least_conn"
    EWMA = "ewma"
    IP_HASH = "ip_hash"


class HealthCheckType(Enum):
    """Health check types."""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"


class AuthType(Enum):
    """Authentication types."""
    KEY_AUTH = "key-auth"
    JWT_AUTH = "jwt-auth"
    BASIC_AUTH = "basic-auth"
    OAUTH2 = "authz-keycloak"
    LDAP = "ldap-auth"
    OIDC = "openid-connect"


class RateLimitType(Enum):
    """Rate limiting strategies."""
    LOCAL = "limit-count"
    CLUSTER = "limit-count"
    CONN = "limit-conn"
    REQ = "limit-req"


@dataclass
class UpstreamNode:
    """Upstream service node."""
    host: str
    port: int
    weight: int = 1
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check configuration."""
    active: bool = True
    check_type: HealthCheckType = HealthCheckType.HTTP
    http_path: str = "/health"
    interval: int = 5
    timeout: int = 3
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    healthy_http_statuses: List[int] = field(default_factory=lambda: [200, 302])


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    limit_type: RateLimitType = RateLimitType.LOCAL
    count: int = 100
    time_window: int = 60
    key: str = "remote_addr"
    rejected_code: int = 429
    rejected_msg: str = "Rate limit exceeded"


@dataclass
class AuthConfig:
    """Authentication configuration."""
    auth_type: AuthType = AuthType.KEY_AUTH
    key_header: str = "X-API-Key"
    hide_credentials: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteConfig:
    """Route configuration."""
    name: str
    uri: str
    methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    upstream_id: Optional[str] = None
    service_id: Optional[str] = None
    plugins: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    labels: Dict[str, str] = field(default_factory=dict)
    enable_websocket: bool = False
    timeout: Dict[str, int] = field(default_factory=lambda: {"connect": 6, "send": 6, "read": 6})


@dataclass
class UpstreamConfig:
    """Upstream configuration."""
    name: str
    nodes: List[UpstreamNode]
    type: LoadBalancerType = LoadBalancerType.ROUND_ROBIN
    retries: int = 3
    retry_timeout: int = 6
    health_check: Optional[HealthCheck] = None
    pass_host: str = "pass"
    scheme: str = "http"


@dataclass
class ServiceConfig:
    """Service configuration."""
    name: str
    upstream_id: str
    plugins: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    enable_websocket: bool = False


class MockApisixClient:
    """Mock APISIX Admin API client."""
    
    def __init__(self, admin_url: str, api_key: str):
        self.admin_url = admin_url
        self.api_key = api_key
        self._routes: Dict[str, Dict[str, Any]] = {}
        self._upstreams: Dict[str, Dict[str, Any]] = {}
        self._services: Dict[str, Dict[str, Any]] = {}
        self._consumers: Dict[str, Dict[str, Any]] = {}
        self._plugins: Dict[str, Dict[str, Any]] = {}
    
    async def create_route(self, route_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a route."""
        self._routes[route_id] = {
            'id': route_id,
            'create_time': datetime.now().timestamp(),
            'update_time': datetime.now().timestamp(),
            **config
        }
        return self._routes[route_id]
    
    async def get_route(self, route_id: str) -> Optional[Dict[str, Any]]:
        """Get a route."""
        return self._routes.get(route_id)
    
    async def update_route(self, route_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update a route."""
        if route_id in self._routes:
            self._routes[route_id].update(config)
            self._routes[route_id]['update_time'] = datetime.now().timestamp()
        return self._routes.get(route_id, {})
    
    async def delete_route(self, route_id: str) -> bool:
        """Delete a route."""
        if route_id in self._routes:
            del self._routes[route_id]
            return True
        return False
    
    async def list_routes(self) -> List[Dict[str, Any]]:
        """List all routes."""
        return list(self._routes.values())
    
    async def create_upstream(self, upstream_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an upstream."""
        self._upstreams[upstream_id] = {
            'id': upstream_id,
            'create_time': datetime.now().timestamp(),
            **config
        }
        return self._upstreams[upstream_id]
    
    async def get_upstream(self, upstream_id: str) -> Optional[Dict[str, Any]]:
        """Get an upstream."""
        return self._upstreams.get(upstream_id)
    
    async def update_upstream(self, upstream_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update an upstream."""
        if upstream_id in self._upstreams:
            self._upstreams[upstream_id].update(config)
        return self._upstreams.get(upstream_id, {})
    
    async def delete_upstream(self, upstream_id: str) -> bool:
        """Delete an upstream."""
        if upstream_id in self._upstreams:
            del self._upstreams[upstream_id]
            return True
        return False
    
    async def list_upstreams(self) -> List[Dict[str, Any]]:
        """List all upstreams."""
        return list(self._upstreams.values())
    
    async def create_service(self, service_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a service."""
        self._services[service_id] = {
            'id': service_id,
            'create_time': datetime.now().timestamp(),
            **config
        }
        return self._services[service_id]
    
    async def get_service(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get a service."""
        return self._services.get(service_id)
    
    async def delete_service(self, service_id: str) -> bool:
        """Delete a service."""
        if service_id in self._services:
            del self._services[service_id]
            return True
        return False
    
    async def create_consumer(self, username: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a consumer."""
        self._consumers[username] = {
            'username': username,
            'create_time': datetime.now().timestamp(),
            **config
        }
        return self._consumers[username]
    
    async def get_consumer(self, username: str) -> Optional[Dict[str, Any]]:
        """Get a consumer."""
        return self._consumers.get(username)
    
    async def delete_consumer(self, username: str) -> bool:
        """Delete a consumer."""
        if username in self._consumers:
            del self._consumers[username]
            return True
        return False
    
    async def list_consumers(self) -> List[Dict[str, Any]]:
        """List all consumers."""
        return list(self._consumers.values())
    
    async def enable_plugin(self, plugin_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enable a global plugin."""
        self._plugins[plugin_name] = config
        return config
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a global plugin."""
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            return True
        return False
    
    async def get_schema(self, resource_type: str) -> Dict[str, Any]:
        """Get resource schema."""
        return {"type": "object", "properties": {}}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check APISIX health."""
        return {"status": "healthy", "routes": len(self._routes)}


class ApisixRouteManager:
    """
    Route management for APISIX.
    
    Provides:
    - Dynamic route creation and updates
    - URI pattern matching
    - Method-based routing
    - Plugin configuration per route
    
    Example:
        route_manager = ApisixRouteManager(client)
        await route_manager.create_route(RouteConfig(
            name="api-v1",
            uri="/api/v1/*",
            methods=["GET", "POST"]
        ))
    """
    
    def __init__(self, client: MockApisixClient):
        self.client = client
    
    async def create_route(self, config: RouteConfig, 
                          route_id: str = None) -> Dict[str, Any]:
        """Create a new route."""
        route_id = route_id or str(uuid.uuid4())[:8]
        
        route_config = {
            'name': config.name,
            'uri': config.uri,
            'methods': config.methods,
            'priority': config.priority,
            'labels': config.labels,
            'enable_websocket': config.enable_websocket,
            'timeout': config.timeout,
            'plugins': config.plugins
        }
        
        if config.upstream_id:
            route_config['upstream_id'] = config.upstream_id
        if config.service_id:
            route_config['service_id'] = config.service_id
        
        return await self.client.create_route(route_id, route_config)
    
    async def get_route(self, route_id: str) -> Optional[Dict[str, Any]]:
        """Get route by ID."""
        return await self.client.get_route(route_id)
    
    async def update_route(self, route_id: str, 
                          config: RouteConfig) -> Dict[str, Any]:
        """Update an existing route."""
        route_config = {
            'name': config.name,
            'uri': config.uri,
            'methods': config.methods,
            'plugins': config.plugins
        }
        return await self.client.update_route(route_id, route_config)
    
    async def delete_route(self, route_id: str) -> bool:
        """Delete a route."""
        return await self.client.delete_route(route_id)
    
    async def list_routes(self) -> List[Dict[str, Any]]:
        """List all routes."""
        return await self.client.list_routes()
    
    async def add_plugin_to_route(self, route_id: str, 
                                  plugin_name: str,
                                  plugin_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add a plugin to a route."""
        route = await self.get_route(route_id)
        if route:
            plugins = route.get('plugins', {})
            plugins[plugin_name] = plugin_config
            return await self.client.update_route(route_id, {'plugins': plugins})
        return {}
    
    async def remove_plugin_from_route(self, route_id: str,
                                       plugin_name: str) -> Dict[str, Any]:
        """Remove a plugin from a route."""
        route = await self.get_route(route_id)
        if route:
            plugins = route.get('plugins', {})
            if plugin_name in plugins:
                del plugins[plugin_name]
                return await self.client.update_route(route_id, {'plugins': plugins})
        return {}


class ApisixUpstreamManager:
    """
    Upstream management for APISIX.
    
    Provides:
    - Service discovery integration
    - Load balancing configuration
    - Health check management
    - Node management
    
    Example:
        upstream_manager = ApisixUpstreamManager(client)
        await upstream_manager.create_upstream(UpstreamConfig(
            name="ml-service",
            nodes=[UpstreamNode(host="ml-1", port=8000, weight=10)]
        ))
    """
    
    def __init__(self, client: MockApisixClient):
        self.client = client
    
    async def create_upstream(self, config: UpstreamConfig,
                             upstream_id: str = None) -> Dict[str, Any]:
        """Create a new upstream."""
        upstream_id = upstream_id or str(uuid.uuid4())[:8]
        
        nodes = {}
        for node in config.nodes:
            key = f"{node.host}:{node.port}"
            nodes[key] = node.weight
        
        upstream_config = {
            'name': config.name,
            'nodes': nodes,
            'type': config.type.value,
            'retries': config.retries,
            'retry_timeout': config.retry_timeout,
            'pass_host': config.pass_host,
            'scheme': config.scheme
        }
        
        if config.health_check:
            upstream_config['checks'] = {
                'active': {
                    'type': config.health_check.check_type.value,
                    'http_path': config.health_check.http_path,
                    'timeout': config.health_check.timeout,
                    'healthy': {
                        'interval': config.health_check.interval,
                        'successes': config.health_check.healthy_threshold,
                        'http_statuses': config.health_check.healthy_http_statuses
                    },
                    'unhealthy': {
                        'interval': config.health_check.interval,
                        'http_failures': config.health_check.unhealthy_threshold
                    }
                }
            }
        
        return await self.client.create_upstream(upstream_id, upstream_config)
    
    async def get_upstream(self, upstream_id: str) -> Optional[Dict[str, Any]]:
        """Get upstream by ID."""
        return await self.client.get_upstream(upstream_id)
    
    async def update_upstream(self, upstream_id: str,
                             config: UpstreamConfig) -> Dict[str, Any]:
        """Update an existing upstream."""
        nodes = {}
        for node in config.nodes:
            key = f"{node.host}:{node.port}"
            nodes[key] = node.weight
        
        return await self.client.update_upstream(upstream_id, {'nodes': nodes})
    
    async def delete_upstream(self, upstream_id: str) -> bool:
        """Delete an upstream."""
        return await self.client.delete_upstream(upstream_id)
    
    async def list_upstreams(self) -> List[Dict[str, Any]]:
        """List all upstreams."""
        return await self.client.list_upstreams()
    
    async def add_node(self, upstream_id: str, node: UpstreamNode) -> Dict[str, Any]:
        """Add a node to upstream."""
        upstream = await self.get_upstream(upstream_id)
        if upstream:
            nodes = upstream.get('nodes', {})
            key = f"{node.host}:{node.port}"
            nodes[key] = node.weight
            return await self.client.update_upstream(upstream_id, {'nodes': nodes})
        return {}
    
    async def remove_node(self, upstream_id: str, host: str, port: int) -> Dict[str, Any]:
        """Remove a node from upstream."""
        upstream = await self.get_upstream(upstream_id)
        if upstream:
            nodes = upstream.get('nodes', {})
            key = f"{host}:{port}"
            if key in nodes:
                del nodes[key]
                return await self.client.update_upstream(upstream_id, {'nodes': nodes})
        return {}


class ApisixPluginManager:
    """
    Plugin management for APISIX.
    
    Provides configuration for:
    - Authentication plugins
    - Rate limiting plugins
    - Transformation plugins
    - Observability plugins
    - Security plugins
    """
    
    def __init__(self, client: MockApisixClient):
        self.client = client
    
    def create_rate_limit_config(self, config: RateLimitConfig) -> Dict[str, Any]:
        """Create rate limiting plugin configuration."""
        return {
            config.limit_type.value: {
                'count': config.count,
                'time_window': config.time_window,
                'key': config.key,
                'rejected_code': config.rejected_code,
                'rejected_msg': config.rejected_msg
            }
        }
    
    def create_auth_config(self, config: AuthConfig) -> Dict[str, Any]:
        """Create authentication plugin configuration."""
        auth_config = {
            config.auth_type.value: {
                'hide_credentials': config.hide_credentials,
                **config.config
            }
        }
        
        if config.auth_type == AuthType.KEY_AUTH:
            auth_config[config.auth_type.value]['header'] = config.key_header
        
        return auth_config
    
    def create_cors_config(self, allow_origins: List[str] = None,
                          allow_methods: List[str] = None,
                          allow_headers: List[str] = None,
                          max_age: int = 3600) -> Dict[str, Any]:
        """Create CORS plugin configuration."""
        return {
            'cors': {
                'allow_origins': ','.join(allow_origins or ['*']),
                'allow_methods': ','.join(allow_methods or ['*']),
                'allow_headers': ','.join(allow_headers or ['*']),
                'max_age': max_age,
                'allow_credential': True
            }
        }
    
    def create_prometheus_config(self, prefer_name: bool = True) -> Dict[str, Any]:
        """Create Prometheus metrics plugin configuration."""
        return {
            'prometheus': {
                'prefer_name': prefer_name
            }
        }
    
    def create_request_transform_config(self, 
                                        add_headers: Dict[str, str] = None,
                                        remove_headers: List[str] = None,
                                        rename_headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Create request transformation plugin configuration."""
        config = {'proxy-rewrite': {}}
        
        if add_headers:
            config['proxy-rewrite']['headers'] = {
                'add': add_headers
            }
        if remove_headers:
            config['proxy-rewrite']['headers'] = config['proxy-rewrite'].get('headers', {})
            config['proxy-rewrite']['headers']['remove'] = remove_headers
        
        return config
    
    def create_response_transform_config(self,
                                         add_headers: Dict[str, str] = None,
                                         remove_headers: List[str] = None) -> Dict[str, Any]:
        """Create response transformation plugin configuration."""
        return {
            'response-rewrite': {
                'headers': {
                    'add': add_headers or {},
                    'remove': remove_headers or []
                }
            }
        }
    
    def create_circuit_breaker_config(self,
                                      break_response_code: int = 502,
                                      max_breaker_sec: int = 300,
                                      unhealthy_statuses: List[int] = None,
                                      healthy_statuses: List[int] = None) -> Dict[str, Any]:
        """Create circuit breaker plugin configuration."""
        return {
            'api-breaker': {
                'break_response_code': break_response_code,
                'max_breaker_sec': max_breaker_sec,
                'unhealthy': {
                    'http_statuses': unhealthy_statuses or [500, 502, 503]
                },
                'healthy': {
                    'http_statuses': healthy_statuses or [200]
                }
            }
        }


class ApisixConsumerManager:
    """
    Consumer management for APISIX.
    
    Manages API consumers with:
    - API key management
    - JWT token configuration
    - Rate limit per consumer
    - Plugin configuration per consumer
    """
    
    def __init__(self, client: MockApisixClient):
        self.client = client
    
    async def create_consumer(self, username: str,
                             plugins: Dict[str, Any] = None,
                             labels: Dict[str, str] = None) -> Dict[str, Any]:
        """Create a new consumer."""
        config = {
            'plugins': plugins or {},
            'labels': labels or {}
        }
        return await self.client.create_consumer(username, config)
    
    async def create_consumer_with_key(self, username: str,
                                       api_key: str = None) -> Dict[str, Any]:
        """Create consumer with API key authentication."""
        api_key = api_key or str(uuid.uuid4())
        
        plugins = {
            'key-auth': {
                'key': api_key
            }
        }
        
        result = await self.create_consumer(username, plugins)
        result['api_key'] = api_key
        return result
    
    async def create_consumer_with_jwt(self, username: str,
                                       key: str = None,
                                       secret: str = None,
                                       algorithm: str = "HS256") -> Dict[str, Any]:
        """Create consumer with JWT authentication."""
        key = key or str(uuid.uuid4())
        secret = secret or str(uuid.uuid4())
        
        plugins = {
            'jwt-auth': {
                'key': key,
                'secret': secret,
                'algorithm': algorithm
            }
        }
        
        result = await self.create_consumer(username, plugins)
        result['jwt_key'] = key
        result['jwt_secret'] = secret
        return result
    
    async def get_consumer(self, username: str) -> Optional[Dict[str, Any]]:
        """Get consumer by username."""
        return await self.client.get_consumer(username)
    
    async def delete_consumer(self, username: str) -> bool:
        """Delete a consumer."""
        return await self.client.delete_consumer(username)
    
    async def list_consumers(self) -> List[Dict[str, Any]]:
        """List all consumers."""
        return await self.client.list_consumers()
    
    async def set_consumer_rate_limit(self, username: str,
                                      config: RateLimitConfig) -> Dict[str, Any]:
        """Set rate limit for a consumer."""
        consumer = await self.get_consumer(username)
        if consumer:
            plugins = consumer.get('plugins', {})
            plugins[config.limit_type.value] = {
                'count': config.count,
                'time_window': config.time_window,
                'rejected_code': config.rejected_code
            }
            return await self.client.create_consumer(username, {'plugins': plugins})
        return {}


class ApisixGateway:
    """
    Main APISIX gateway integration for MineralVision.
    
    Provides unified API gateway management:
    - Route management
    - Upstream management
    - Consumer management
    - Plugin configuration
    
    Example:
        gateway = ApisixGateway()
        await gateway.connect()
        
        # Create upstream
        await gateway.upstreams.create_upstream(UpstreamConfig(...))
        
        # Create route
        await gateway.routes.create_route(RouteConfig(...))
        
        # Create consumer
        await gateway.consumers.create_consumer_with_key("user1")
    """
    
    def __init__(self, admin_url: str = "http://localhost:9180",
                 api_key: str = "mineralvision-admin-key"):
        self.admin_url = admin_url
        self.api_key = api_key
        self.client: Optional[MockApisixClient] = None
        self.routes: Optional[ApisixRouteManager] = None
        self.upstreams: Optional[ApisixUpstreamManager] = None
        self.plugins: Optional[ApisixPluginManager] = None
        self.consumers: Optional[ApisixConsumerManager] = None
        self._connected = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """True when running on the explicit in-memory mock fallback."""
        return self._degraded
    
    async def connect(self) -> 'ApisixGateway':
        """
        Connect to APISIX Admin API (real connection first).

        A real HTTP client implementation is not available yet, so this
        falls back to the in-memory mock ONLY when
        MV_ALLOW_MOCK_FALLBACK=true; otherwise raises RuntimeError.
        """
        reachable = probe_url(self.admin_url, timeout=2.0)
        reason = (
            f"server reachable at {self.admin_url} but real HTTP client not implemented"
            if reachable else f"no APISIX server reachable at {self.admin_url}"
        )
        if real_client_unavailable("APISIX Gateway", reason):
            self._degraded = True
            self.client = MockApisixClient(self.admin_url, self.api_key)
        
        self.routes = ApisixRouteManager(self.client)
        self.upstreams = ApisixUpstreamManager(self.client)
        self.plugins = ApisixPluginManager(self.client)
        self.consumers = ApisixConsumerManager(self.client)
        
        self._connected = True
        logger.info(f"Connected to APISIX at {self.admin_url}")
        return self
    
    async def health_check(self) -> Dict[str, Any]:
        """Check APISIX health."""
        if self.client:
            health = await self.client.health_check()
            health['degraded'] = self._degraded
            return health
        return {"status": "disconnected", "degraded": self._degraded}
    
    async def setup_mineralvision_routes(self) -> Dict[str, Any]:
        """Setup default MineralVision API routes."""
        results = {'upstreams': [], 'routes': []}
        
        # Create upstreams for MineralVision services
        services = [
            ('api-service', [UpstreamNode('api', 8000)]),
            ('ml-service', [UpstreamNode('ml', 8001)]),
            ('sensor-service', [UpstreamNode('sensor', 8002)]),
            ('waldo-service', [UpstreamNode('waldo', 8003)])
        ]
        
        for name, nodes in services:
            upstream = await self.upstreams.create_upstream(UpstreamConfig(
                name=name,
                nodes=nodes,
                health_check=HealthCheck()
            ))
            results['upstreams'].append(upstream)
        
        # Create routes
        routes = [
            RouteConfig(name='api-v1', uri='/api/v1/*', upstream_id='api-service'),
            RouteConfig(name='ml-api', uri='/ml/*', upstream_id='ml-service'),
            RouteConfig(name='sensor-api', uri='/sensors/*', upstream_id='sensor-service'),
            RouteConfig(name='waldo-api', uri='/waldo/*', upstream_id='waldo-service')
        ]
        
        for route_config in routes:
            # Add common plugins
            route_config.plugins = {
                **self.plugins.create_rate_limit_config(RateLimitConfig(count=1000)),
                **self.plugins.create_cors_config(),
                **self.plugins.create_prometheus_config()
            }
            
            route = await self.routes.create_route(route_config)
            results['routes'].append(route)
        
        return results
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Factory functions

def create_apisix_gateway(admin_url: str = None,
                         api_key: str = None) -> ApisixGateway:
    """Create an APISIX gateway instance."""
    return ApisixGateway(
        admin_url=admin_url or "http://localhost:9180",
        api_key=api_key or "mineralvision-admin-key"
    )


async def create_and_connect_gateway(admin_url: str = None,
                                    api_key: str = None) -> ApisixGateway:
    """Create and connect APISIX gateway."""
    gateway = create_apisix_gateway(admin_url, api_key)
    await gateway.connect()
    return gateway
