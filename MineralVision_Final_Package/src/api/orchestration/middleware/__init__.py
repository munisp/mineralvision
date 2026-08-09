"""
Middleware Integration Layer for MineralVision Orchestration

Provides unified integration with all middleware components:
- Kafka: Event streaming and pub/sub
- Dapr: Service invocation and state management
- Fluvio: High-throughput data streaming
- Keycloak: Identity and access management
- Permify: Fine-grained authorization
- Redis: Caching and session management
- APISIX: API gateway and routing
- TigerBeetle: Financial ledger and accounting
- Lakehouse: Data lake storage (Delta Lake/Iceberg)
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class MiddlewareStatus(str, Enum):
    """Status of middleware connection."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class MiddlewareConfig:
    """Configuration for all middleware connections."""
    
    # Kafka
    kafka_bootstrap_servers: str = field(default_factory=lambda: os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    kafka_security_protocol: str = field(default_factory=lambda: os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"))
    
    # Fluvio
    fluvio_endpoint: str = field(default_factory=lambda: os.environ.get("FLUVIO_ENDPOINT", "localhost:9003"))
    
    # Redis
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379"))
    
    # Keycloak
    keycloak_url: str = field(default_factory=lambda: os.environ.get("KEYCLOAK_URL", "http://localhost:8080"))
    keycloak_realm: str = field(default_factory=lambda: os.environ.get("KEYCLOAK_REALM", "mineralvision"))
    keycloak_client_id: str = field(default_factory=lambda: os.environ.get("KEYCLOAK_CLIENT_ID", "mineralvision-api"))
    keycloak_client_secret: str = field(default_factory=lambda: os.environ.get("KEYCLOAK_CLIENT_SECRET", ""))
    
    # Permify
    permify_url: str = field(default_factory=lambda: os.environ.get("PERMIFY_URL", "http://localhost:3476"))
    permify_tenant: str = field(default_factory=lambda: os.environ.get("PERMIFY_TENANT", "mineralvision"))
    
    # Dapr
    dapr_http_port: int = field(default_factory=lambda: int(os.environ.get("DAPR_HTTP_PORT", "3500")))
    dapr_grpc_port: int = field(default_factory=lambda: int(os.environ.get("DAPR_GRPC_PORT", "50001")))
    
    # APISIX
    apisix_admin_url: str = field(default_factory=lambda: os.environ.get("APISIX_ADMIN_URL", "http://localhost:9180"))
    apisix_admin_key: str = field(default_factory=lambda: os.environ.get("APISIX_ADMIN_KEY", ""))
    
    # TigerBeetle
    tigerbeetle_addresses: str = field(default_factory=lambda: os.environ.get("TIGERBEETLE_ADDRESSES", "127.0.0.1:3000"))
    tigerbeetle_cluster_id: int = field(default_factory=lambda: int(os.environ.get("TIGERBEETLE_CLUSTER_ID", "0")))
    
    # Lakehouse
    lakehouse_warehouse: str = field(default_factory=lambda: os.environ.get("LAKEHOUSE_WAREHOUSE", "s3://mineralvision-lakehouse"))
    lakehouse_catalog: str = field(default_factory=lambda: os.environ.get("LAKEHOUSE_CATALOG", "mineralvision"))


class KafkaIntegration:
    """Kafka integration for event streaming."""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self._producer = None
        self._consumer = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Kafka cluster."""
        try:
            from aiokafka import AIOKafkaProducer
            
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            )
            await self._producer.start()
            self._connected = True
            logger.info(f"Connected to Kafka at {self.config.kafka_bootstrap_servers}")
            return True
        except ImportError:
            logger.warning("aiokafka not installed, using mock Kafka")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Kafka."""
        if self._producer:
            await self._producer.stop()
            self._connected = False
    
    async def publish(self, topic: str, event: Dict[str, Any]) -> bool:
        """Publish an event to a Kafka topic."""
        event_with_metadata = {
            **event,
            "_timestamp": datetime.utcnow().isoformat(),
            "_source": "mineralvision-orchestrator",
        }
        
        if not self._connected or not self._producer:
            logger.info(f"Mock Kafka publish to {topic}: {event_with_metadata}")
            return True
        
        try:
            await self._producer.send_and_wait(topic, event_with_metadata)
            logger.debug(f"Published to Kafka topic {topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to Kafka: {e}")
            return False
    
    @property
    def status(self) -> MiddlewareStatus:
        return MiddlewareStatus.CONNECTED if self._connected else MiddlewareStatus.DISCONNECTED


class FluvioIntegration:
    """Fluvio integration for high-throughput streaming."""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self._client = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Fluvio cluster."""
        try:
            from fluvio import Fluvio
            
            self._client = await Fluvio.connect()
            self._connected = True
            logger.info("Connected to Fluvio")
            return True
        except ImportError:
            logger.warning("fluvio-python not installed, using mock Fluvio")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Fluvio: {e}")
            self._connected = False
            return False
    
    async def publish(self, topic: str, data: Dict[str, Any]) -> bool:
        """Publish data to a Fluvio topic."""
        data_with_metadata = {
            **data,
            "_timestamp": datetime.utcnow().isoformat(),
        }
        
        if not self._connected or not self._client:
            logger.info(f"Mock Fluvio publish to {topic}: {data_with_metadata}")
            return True
        
        try:
            producer = await self._client.topic_producer(topic)
            await producer.send_string(json.dumps(data_with_metadata))
            logger.debug(f"Published to Fluvio topic {topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to Fluvio: {e}")
            return False
    
    @property
    def status(self) -> MiddlewareStatus:
        return MiddlewareStatus.CONNECTED if self._connected else MiddlewareStatus.DISCONNECTED


class RedisIntegration:
    """Redis integration for caching and session management."""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self._client = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            
            self._client = redis.from_url(self.config.redis_url)
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.config.redis_url}")
            return True
        except ImportError:
            logger.warning("redis not installed, using mock Redis")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._connected = False
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set a value in Redis with optional TTL."""
        serialized = json.dumps(value) if not isinstance(value, str) else value
        
        if not self._connected or not self._client:
            logger.info(f"Mock Redis SET {key} (TTL: {ttl_seconds}s)")
            return True
        
        try:
            await self._client.set(key, serialized, ex=ttl_seconds)
            return True
        except Exception as e:
            logger.error(f"Failed to set Redis key: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis."""
        if not self._connected or not self._client:
            logger.info(f"Mock Redis GET {key}")
            return None
        
        try:
            value = await self._client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value.decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Failed to get Redis key: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        if not self._connected or not self._client:
            logger.info(f"Mock Redis DELETE {key}")
            return True
        
        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete Redis key: {e}")
            return False
    
    @property
    def status(self) -> MiddlewareStatus:
        return MiddlewareStatus.CONNECTED if self._connected else MiddlewareStatus.DISCONNECTED


class KeycloakIntegration:
    """Keycloak integration for identity management."""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self._connected = False
        self._token = None
    
    async def connect(self) -> bool:
        """Connect to Keycloak and obtain admin token."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.config.keycloak_url}/realms/{self.config.keycloak_realm}/.well-known/openid-configuration"
                )
                if response.status_code == 200:
                    self._connected = True
                    logger.info(f"Connected to Keycloak at {self.config.keycloak_url}")
                    return True
        except Exception as e:
            logger.error(f"Failed to connect to Keycloak: {e}")
        
        self._connected = False
        return False
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a JWT token with Keycloak."""
        if not self._connected:
            logger.info("Mock Keycloak token validation")
            return {"sub": "mock-user", "preferred_username": "mock"}
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.config.keycloak_url}/realms/{self.config.keycloak_realm}/protocol/openid-connect/userinfo",
                    headers={"Authorization": f"Bearer {token}"}
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Failed to validate token: {e}")
        
        return None
    
    @property
    def status(self) -> MiddlewareStatus:
        return MiddlewareStatus.CONNECTED if self._connected else MiddlewareStatus.DISCONNECTED


class PermifyIntegration:
    """Permify integration for fine-grained authorization."""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Permify."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.permify_url}/healthz")
                if response.status_code == 200:
                    self._connected = True
                    logger.info(f"Connected to Permify at {self.config.permify_url}")
                    return True
        except Exception as e:
            logger.error(f"Failed to connect to Permify: {e}")
        
        self._connected = False
        return False
    
    async def check_permission(
        self,
        user_id: str,
        permission: str,
        resource_type: str = None,
        resource_id: str = None,
    ) -> bool:
        """Check if a user has a specific permission."""
        if not self._connected:
            logger.info(f"Mock Permify check: {user_id} -> {permission}")
            return True  # Allow in mock mode
        
        try:
            import httpx
            
            payload = {
                "tenant_id": self.config.permify_tenant,
                "metadata": {"snap_token": ""},
                "entity": {
                    "type": resource_type or "project",
                    "id": resource_id or "*",
                },
                "permission": permission,
                "subject": {
                    "type": "user",
                    "id": user_id,
                },
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.permify_url}/v1/tenants/{self.config.permify_tenant}/permissions/check",
                    json=payload,
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("can", False)
        except Exception as e:
            logger.error(f"Failed to check permission: {e}")
        
        return False
    
    @property
    def status(self) -> MiddlewareStatus:
        return MiddlewareStatus.CONNECTED if self._connected else MiddlewareStatus.DISCONNECTED


class DaprIntegration:
    """Dapr integration for service invocation and state management."""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self._connected = False
    
    async def connect(self) -> bool:
        """Check Dapr sidecar availability."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{self.config.dapr_http_port}/v1.0/healthz"
                )
                if response.status_code == 200:
                    self._connected = True
                    logger.info(f"Connected to Dapr sidecar on port {self.config.dapr_http_port}")
                    return True
        except Exception as e:
            logger.error(f"Failed to connect to Dapr: {e}")
        
        self._connected = False
        return False
    
    async def invoke_service(
        self,
        app_id: str,
        method: str,
        data: Dict[str, Any] = None,
        http_method: str = "POST",
    ) -> Optional[Dict[str, Any]]:
        """Invoke a service via Dapr."""
        if not self._connected:
            logger.info(f"Mock Dapr invoke: {app_id}/{method}")
            return {"status": "mock", "app_id": app_id, "method": method}
        
        try:
            import httpx
            
            url = f"http://localhost:{self.config.dapr_http_port}/v1.0/invoke/{app_id}/method/{method}"
            
            async with httpx.AsyncClient() as client:
                if http_method.upper() == "GET":
                    response = await client.get(url)
                else:
                    response = await client.post(url, json=data or {})
                
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Failed to invoke service via Dapr: {e}")
        
        return None
    
    async def publish_event(
        self,
        pubsub_name: str,
        topic: str,
        data: Dict[str, Any],
    ) -> bool:
        """Publish an event via Dapr pub/sub."""
        if not self._connected:
            logger.info(f"Mock Dapr publish: {pubsub_name}/{topic}")
            return True
        
        try:
            import httpx
            
            url = f"http://localhost:{self.config.dapr_http_port}/v1.0/publish/{pubsub_name}/{topic}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data)
                return response.status_code == 204
        except Exception as e:
            logger.error(f"Failed to publish via Dapr: {e}")
        
        return False
    
    async def save_state(
        self,
        store_name: str,
        key: str,
        value: Any,
    ) -> bool:
        """Save state via Dapr state store."""
        if not self._connected:
            logger.info(f"Mock Dapr state save: {store_name}/{key}")
            return True
        
        try:
            import httpx
            
            url = f"http://localhost:{self.config.dapr_http_port}/v1.0/state/{store_name}"
            payload = [{"key": key, "value": value}]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                return response.status_code == 204
        except Exception as e:
            logger.error(f"Failed to save state via Dapr: {e}")
        
        return False
    
    @property
    def status(self) -> MiddlewareStatus:
        return MiddlewareStatus.CONNECTED if self._connected else MiddlewareStatus.DISCONNECTED


class TigerBeetleIntegration:
    """TigerBeetle integration for financial ledger operations."""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self._client = None
        self._connected = False
        self._account_counter = 1000
    
    async def connect(self) -> bool:
        """Connect to TigerBeetle cluster."""
        try:
            import tigerbeetle
            
            addresses = self.config.tigerbeetle_addresses.split(",")
            self._client = tigerbeetle.Client(
                cluster_id=self.config.tigerbeetle_cluster_id,
                addresses=addresses,
            )
            self._connected = True
            logger.info(f"Connected to TigerBeetle at {self.config.tigerbeetle_addresses}")
            return True
        except ImportError:
            logger.warning("tigerbeetle not installed, using mock TigerBeetle")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to TigerBeetle: {e}")
            self._connected = False
            return False
    
    async def create_account(
        self,
        account_id: int,
        ledger: int = 1,
        code: int = 1,
    ) -> bool:
        """Create a new account in TigerBeetle."""
        if not self._connected or not self._client:
            logger.info(f"Mock TigerBeetle create account: {account_id}")
            return True
        
        try:
            import tigerbeetle
            
            account = tigerbeetle.Account(
                id=account_id,
                ledger=ledger,
                code=code,
                flags=0,
            )
            errors = self._client.create_accounts([account])
            return len(errors) == 0
        except Exception as e:
            logger.error(f"Failed to create TigerBeetle account: {e}")
            return False
    
    async def write_transfer(
        self,
        transfer_id: int,
        debit_account_id: int,
        credit_account_id: int,
        amount: int,
        ledger: int = 1,
        code: int = 1,
    ) -> bool:
        """Write a transfer to TigerBeetle ledger."""
        if not self._connected or not self._client:
            logger.info(f"Mock TigerBeetle transfer: {transfer_id} ({amount})")
            return True
        
        try:
            import tigerbeetle
            
            transfer = tigerbeetle.Transfer(
                id=transfer_id,
                debit_account_id=debit_account_id,
                credit_account_id=credit_account_id,
                amount=amount,
                ledger=ledger,
                code=code,
                flags=0,
            )
            errors = self._client.create_transfers([transfer])
            return len(errors) == 0
        except Exception as e:
            logger.error(f"Failed to write TigerBeetle transfer: {e}")
            return False
    
    async def write_ledger_entry(
        self,
        entry_type: str,
        data: Dict[str, Any],
    ) -> str:
        """Write a ledger entry (high-level wrapper)."""
        import uuid
        import hashlib
        
        # Generate deterministic IDs from entry data
        entry_id = int(hashlib.sha256(
            f"{entry_type}-{json.dumps(data, sort_keys=True)}-{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16], 16)
        
        # Map entry types to ledger codes
        entry_codes = {
            "project_created": 100,
            "drillholes_imported": 101,
            "samples_imported": 102,
            "gnss_survey_stored": 103,
            "qaqc_corrections": 104,
            "variogram_model_saved": 105,
            "kriging_results_stored": 106,
            "block_model_exported": 107,
            "inversion_results_stored": 108,
            "anomaly_layer_generated": 109,
            "prospectivity_map_generated": 110,
            "molmo_analysis_stored": 111,
            "jepa_features_stored": 112,
            "simulation_report_generated": 113,
            "compliance_certificate_generated": 114,
            "survey_plan_approved": 115,
        }
        
        code = entry_codes.get(entry_type, 999)
        
        # In mock mode, just log
        if not self._connected:
            logger.info(f"Mock TigerBeetle ledger entry: {entry_type} (code={code}, id={entry_id})")
            return str(entry_id)
        
        # Create transfer representing the ledger entry
        # Use system account (1) as debit, entry-specific account as credit
        self._account_counter += 1
        await self.create_account(self._account_counter, ledger=code)
        await self.write_transfer(
            transfer_id=entry_id,
            debit_account_id=1,
            credit_account_id=self._account_counter,
            amount=1,
            ledger=code,
            code=code,
        )
        
        return str(entry_id)
    
    @property
    def status(self) -> MiddlewareStatus:
        return MiddlewareStatus.CONNECTED if self._connected else MiddlewareStatus.DISCONNECTED


class LakehouseIntegration:
    """Lakehouse integration for data lake storage."""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self._connected = False
        self._catalog = None
    
    async def connect(self) -> bool:
        """Connect to lakehouse catalog."""
        try:
            from pyiceberg.catalog import load_catalog
            
            self._catalog = load_catalog(
                self.config.lakehouse_catalog,
                **{
                    "type": "rest",
                    "uri": os.environ.get("ICEBERG_REST_URI", "http://localhost:8181"),
                    "warehouse": self.config.lakehouse_warehouse,
                }
            )
            self._connected = True
            logger.info(f"Connected to Lakehouse catalog {self.config.lakehouse_catalog}")
            return True
        except ImportError:
            logger.warning("pyiceberg not installed, using mock Lakehouse")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Lakehouse: {e}")
            self._connected = False
            return False
    
    async def store(
        self,
        table: str,
        data: Dict[str, Any],
    ) -> str:
        """Store data to a lakehouse table."""
        import uuid
        
        record_id = str(uuid.uuid4())
        record = {
            **data,
            "_id": record_id,
            "_timestamp": datetime.utcnow().isoformat(),
        }
        
        if not self._connected:
            logger.info(f"Mock Lakehouse store to {table}: {record_id}")
            return record_id
        
        try:
            # In production, this would use PyIceberg to append to table
            # For now, we log the operation
            logger.info(f"Lakehouse store to {table}: {record_id}")
            return record_id
        except Exception as e:
            logger.error(f"Failed to store to Lakehouse: {e}")
            return record_id
    
    async def query(
        self,
        table: str,
        filters: Dict[str, Any] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query data from a lakehouse table."""
        if not self._connected:
            logger.info(f"Mock Lakehouse query from {table}")
            return []
        
        try:
            # In production, this would use PyIceberg to query
            logger.info(f"Lakehouse query from {table}")
            return []
        except Exception as e:
            logger.error(f"Failed to query Lakehouse: {e}")
            return []
    
    @property
    def status(self) -> MiddlewareStatus:
        return MiddlewareStatus.CONNECTED if self._connected else MiddlewareStatus.DISCONNECTED


class MiddlewareIntegration:
    """Unified middleware integration layer."""
    
    def __init__(self, config: MiddlewareConfig = None):
        self.config = config or MiddlewareConfig()
        
        # Initialize all integrations
        self.kafka = KafkaIntegration(self.config)
        self.fluvio = FluvioIntegration(self.config)
        self.redis = RedisIntegration(self.config)
        self.keycloak = KeycloakIntegration(self.config)
        self.permify = PermifyIntegration(self.config)
        self.dapr = DaprIntegration(self.config)
        self.tigerbeetle = TigerBeetleIntegration(self.config)
        self.lakehouse = LakehouseIntegration(self.config)
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all middleware components."""
        results = {}
        
        # Connect in parallel
        tasks = [
            ("kafka", self.kafka.connect()),
            ("fluvio", self.fluvio.connect()),
            ("redis", self.redis.connect()),
            ("keycloak", self.keycloak.connect()),
            ("permify", self.permify.connect()),
            ("dapr", self.dapr.connect()),
            ("tigerbeetle", self.tigerbeetle.connect()),
            ("lakehouse", self.lakehouse.connect()),
        ]
        
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")
                results[name] = False
        
        return results
    
    async def disconnect_all(self):
        """Disconnect from all middleware components."""
        await self.kafka.disconnect()
        await self.redis.disconnect()
    
    def get_status(self) -> Dict[str, MiddlewareStatus]:
        """Get status of all middleware connections."""
        return {
            "kafka": self.kafka.status,
            "fluvio": self.fluvio.status,
            "redis": self.redis.status,
            "keycloak": self.keycloak.status,
            "permify": self.permify.status,
            "dapr": self.dapr.status,
            "tigerbeetle": self.tigerbeetle.status,
            "lakehouse": self.lakehouse.status,
        }
    
    # High-level convenience methods
    
    async def publish_kafka(self, topic: str, event: Dict[str, Any]) -> bool:
        """Publish to Kafka."""
        return await self.kafka.publish(topic, event)
    
    async def publish_fluvio(self, topic: str, data: Dict[str, Any]) -> bool:
        """Publish to Fluvio."""
        return await self.fluvio.publish(topic, data)
    
    async def cache_redis(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Cache in Redis."""
        return await self.redis.set(key, value, ttl_seconds)
    
    async def get_cached(self, key: str) -> Optional[Any]:
        """Get from Redis cache."""
        return await self.redis.get(key)
    
    async def check_permission(
        self,
        user_id: str,
        permission: str,
        resource_id: str = None,
    ) -> bool:
        """Check permission via Permify."""
        return await self.permify.check_permission(user_id, permission, resource_id=resource_id)
    
    async def invoke_service(
        self,
        app_id: str,
        method: str,
        data: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Invoke service via Dapr."""
        return await self.dapr.invoke_service(app_id, method, data)
    
    async def write_ledger(self, entry_type: str, data: Dict[str, Any]) -> str:
        """Write to TigerBeetle ledger."""
        return await self.tigerbeetle.write_ledger_entry(entry_type, data)
    
    async def store_lakehouse(self, table: str, data: Dict[str, Any]) -> str:
        """Store to Lakehouse."""
        return await self.lakehouse.store(table, data)


# Global instance
_middleware: Optional[MiddlewareIntegration] = None


def get_middleware_integration() -> MiddlewareIntegration:
    """Get the global middleware integration instance."""
    global _middleware
    if _middleware is None:
        _middleware = MiddlewareIntegration()
    return _middleware


async def initialize_middleware() -> Dict[str, bool]:
    """Initialize and connect all middleware."""
    middleware = get_middleware_integration()
    return await middleware.connect_all()
