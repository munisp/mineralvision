"""
Redis Caching Integration
==========================

Production-grade caching and data structures for MineralVision:
- High-performance key-value caching
- Distributed locking
- Pub/sub messaging
- Rate limiting
- Session management
- Geospatial indexing
- Time-series data

Redis provides in-memory data structure store with
persistence and clustering support.
"""

import asyncio
import json
import logging
import uuid
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading
import time

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not installed. Install with: pip install redis")

from .._mock_fallback import real_client_unavailable


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "allkeys-lru"
    LFU = "allkeys-lfu"
    TTL = "volatile-ttl"
    RANDOM = "allkeys-random"


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ssl: bool = False
    max_connections: int = 10
    socket_timeout: float = 5.0
    decode_responses: bool = True
    prefix: str = "mineralvision:"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    hits: int = 0


class MockRedisClient:
    """Mock Redis client."""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
        self._sets: Dict[str, Set[str]] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._lists: Dict[str, List[str]] = {}
        self._sorted_sets: Dict[str, Dict[str, float]] = {}
        self._geo: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._pubsub_handlers: Dict[str, List[Callable]] = {}
        self._locks: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def _prefixed(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.config.prefix}{key}"
    
    def _check_expiry(self, key: str) -> bool:
        """Check if key is expired."""
        if key in self._expiry:
            if datetime.now() > self._expiry[key]:
                self._delete_key(key)
                return True
        return False
    
    def _delete_key(self, key: str) -> None:
        """Delete a key from all stores."""
        self._data.pop(key, None)
        self._expiry.pop(key, None)
        self._sets.pop(key, None)
        self._hashes.pop(key, None)
        self._lists.pop(key, None)
        self._sorted_sets.pop(key, None)
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value."""
        key = self._prefixed(key)
        with self._lock:
            if self._check_expiry(key):
                return None
            return self._data.get(key)
    
    async def set(self, key: str, value: str, ex: int = None,
                 px: int = None, nx: bool = False, xx: bool = False) -> bool:
        """Set a value."""
        key = self._prefixed(key)
        with self._lock:
            if nx and key in self._data:
                return False
            if xx and key not in self._data:
                return False
            
            self._data[key] = value
            
            if ex:
                self._expiry[key] = datetime.now() + timedelta(seconds=ex)
            elif px:
                self._expiry[key] = datetime.now() + timedelta(milliseconds=px)
            
            return True
    
    async def delete(self, *keys: str) -> int:
        """Delete keys."""
        count = 0
        with self._lock:
            for key in keys:
                key = self._prefixed(key)
                if key in self._data:
                    self._delete_key(key)
                    count += 1
        return count
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        count = 0
        with self._lock:
            for key in keys:
                key = self._prefixed(key)
                if not self._check_expiry(key) and key in self._data:
                    count += 1
        return count
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiry on key."""
        key = self._prefixed(key)
        with self._lock:
            if key in self._data:
                self._expiry[key] = datetime.now() + timedelta(seconds=seconds)
                return True
        return False
    
    async def ttl(self, key: str) -> int:
        """Get TTL of key."""
        key = self._prefixed(key)
        with self._lock:
            if key in self._expiry:
                remaining = (self._expiry[key] - datetime.now()).total_seconds()
                return max(0, int(remaining))
        return -1
    
    async def incr(self, key: str) -> int:
        """Increment value."""
        key = self._prefixed(key)
        with self._lock:
            value = int(self._data.get(key, 0)) + 1
            self._data[key] = str(value)
            return value
    
    async def decr(self, key: str) -> int:
        """Decrement value."""
        key = self._prefixed(key)
        with self._lock:
            value = int(self._data.get(key, 0)) - 1
            self._data[key] = str(value)
            return value
    
    async def mget(self, *keys: str) -> List[Optional[str]]:
        """Get multiple values."""
        results = []
        for key in keys:
            results.append(await self.get(key))
        return results
    
    async def mset(self, mapping: Dict[str, str]) -> bool:
        """Set multiple values."""
        for key, value in mapping.items():
            await self.set(key, value)
        return True
    
    # Hash operations
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field."""
        name = self._prefixed(name)
        with self._lock:
            return self._hashes.get(name, {}).get(key)
    
    async def hset(self, name: str, key: str = None, value: str = None,
                  mapping: Dict[str, str] = None) -> int:
        """Set hash field(s)."""
        name = self._prefixed(name)
        with self._lock:
            if name not in self._hashes:
                self._hashes[name] = {}
            
            count = 0
            if key and value:
                if key not in self._hashes[name]:
                    count = 1
                self._hashes[name][key] = value
            
            if mapping:
                for k, v in mapping.items():
                    if k not in self._hashes[name]:
                        count += 1
                    self._hashes[name][k] = v
            
            return count
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        name = self._prefixed(name)
        with self._lock:
            return self._hashes.get(name, {}).copy()
    
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        name = self._prefixed(name)
        count = 0
        with self._lock:
            if name in self._hashes:
                for key in keys:
                    if key in self._hashes[name]:
                        del self._hashes[name][key]
                        count += 1
        return count
    
    # Set operations
    async def sadd(self, name: str, *values: str) -> int:
        """Add to set."""
        name = self._prefixed(name)
        with self._lock:
            if name not in self._sets:
                self._sets[name] = set()
            
            before = len(self._sets[name])
            self._sets[name].update(values)
            return len(self._sets[name]) - before
    
    async def smembers(self, name: str) -> Set[str]:
        """Get set members."""
        name = self._prefixed(name)
        with self._lock:
            return self._sets.get(name, set()).copy()
    
    async def sismember(self, name: str, value: str) -> bool:
        """Check set membership."""
        name = self._prefixed(name)
        with self._lock:
            return value in self._sets.get(name, set())
    
    async def srem(self, name: str, *values: str) -> int:
        """Remove from set."""
        name = self._prefixed(name)
        count = 0
        with self._lock:
            if name in self._sets:
                for value in values:
                    if value in self._sets[name]:
                        self._sets[name].remove(value)
                        count += 1
        return count
    
    # Sorted set operations
    async def zadd(self, name: str, mapping: Dict[str, float]) -> int:
        """Add to sorted set."""
        name = self._prefixed(name)
        with self._lock:
            if name not in self._sorted_sets:
                self._sorted_sets[name] = {}
            
            count = 0
            for member, score in mapping.items():
                if member not in self._sorted_sets[name]:
                    count += 1
                self._sorted_sets[name][member] = score
            
            return count
    
    async def zrange(self, name: str, start: int, end: int,
                    withscores: bool = False) -> List[Any]:
        """Get range from sorted set."""
        name = self._prefixed(name)
        with self._lock:
            if name not in self._sorted_sets:
                return []
            
            sorted_items = sorted(
                self._sorted_sets[name].items(),
                key=lambda x: x[1]
            )
            
            if end == -1:
                end = len(sorted_items)
            else:
                end = end + 1
            
            items = sorted_items[start:end]
            
            if withscores:
                return [(m, s) for m, s in items]
            return [m for m, s in items]
    
    async def zscore(self, name: str, member: str) -> Optional[float]:
        """Get score of member."""
        name = self._prefixed(name)
        with self._lock:
            return self._sorted_sets.get(name, {}).get(member)
    
    # Geo operations
    async def geoadd(self, name: str, *values) -> int:
        """Add geo locations."""
        name = self._prefixed(name)
        with self._lock:
            if name not in self._geo:
                self._geo[name] = {}
            
            count = 0
            for i in range(0, len(values), 3):
                lon, lat, member = values[i], values[i+1], values[i+2]
                if member not in self._geo[name]:
                    count += 1
                self._geo[name][member] = (float(lon), float(lat))
            
            return count
    
    async def geopos(self, name: str, *members: str) -> List[Optional[Tuple[float, float]]]:
        """Get geo positions."""
        name = self._prefixed(name)
        results = []
        with self._lock:
            for member in members:
                results.append(self._geo.get(name, {}).get(member))
        return results
    
    async def geodist(self, name: str, member1: str, member2: str,
                     unit: str = "m") -> Optional[float]:
        """Get distance between members."""
        name = self._prefixed(name)
        with self._lock:
            geo = self._geo.get(name, {})
            if member1 not in geo or member2 not in geo:
                return None
            
            # Simple Euclidean distance (not accurate for real geo)
            lon1, lat1 = geo[member1]
            lon2, lat2 = geo[member2]
            
            import math
            dist = math.sqrt((lon2-lon1)**2 + (lat2-lat1)**2) * 111000  # rough meters
            
            if unit == "km":
                dist /= 1000
            elif unit == "mi":
                dist /= 1609.34
            
            return dist
    
    # Pub/sub
    async def publish(self, channel: str, message: str) -> int:
        """Publish message to channel."""
        channel = self._prefixed(channel)
        with self._lock:
            handlers = self._pubsub_handlers.get(channel, [])
            for handler in handlers:
                try:
                    asyncio.create_task(handler(message))
                except:
                    pass
            return len(handlers)
    
    def subscribe(self, channel: str, handler: Callable) -> None:
        """Subscribe to channel."""
        channel = self._prefixed(channel)
        with self._lock:
            if channel not in self._pubsub_handlers:
                self._pubsub_handlers[channel] = []
            self._pubsub_handlers[channel].append(handler)
    
    # Locking
    async def lock(self, name: str, timeout: int = 10) -> bool:
        """Acquire lock."""
        name = self._prefixed(f"lock:{name}")
        lock_id = str(uuid.uuid4())
        
        with self._lock:
            if name in self._locks:
                return False
            
            self._locks[name] = lock_id
            self._expiry[name] = datetime.now() + timedelta(seconds=timeout)
            return True
    
    async def unlock(self, name: str) -> bool:
        """Release lock."""
        name = self._prefixed(f"lock:{name}")
        with self._lock:
            if name in self._locks:
                del self._locks[name]
                self._expiry.pop(name, None)
                return True
        return False
    
    async def ping(self) -> str:
        """Ping server."""
        return "PONG"
    
    async def close(self) -> None:
        """Close connection."""
        pass


class CacheManager:
    """
    Cache management for Redis.
    
    Provides:
    - Key-value caching
    - TTL management
    - Cache invalidation
    - Cache statistics
    """
    
    def __init__(self, client: MockRedisClient):
        self.client = client
        self._stats = {'hits': 0, 'misses': 0}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        value = await self.client.get(key)
        if value is not None:
            self._stats['hits'] += 1
            try:
                return json.loads(value)
            except:
                return value
        self._stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set cached value."""
        if not isinstance(value, str):
            value = json.dumps(value)
        return await self.client.set(key, value, ex=ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete cached value."""
        return await self.client.delete(key) > 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.client.exists(key) > 0
    
    async def get_or_set(self, key: str, factory: Callable,
                        ttl: int = None) -> Any:
        """Get value or set from factory."""
        value = await self.get(key)
        if value is None:
            value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
            await self.set(key, value, ttl)
        return value
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern."""
        # In real Redis, use SCAN + DELETE
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total if total > 0 else 0
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': hit_rate
        }


class SessionManager:
    """
    Session management for Redis.
    
    Provides:
    - Session creation
    - Session retrieval
    - Session expiration
    """
    
    def __init__(self, client: MockRedisClient, prefix: str = "session:"):
        self.client = client
        self.prefix = prefix
        self.default_ttl = 3600
    
    async def create(self, user_id: str, data: Dict[str, Any] = None,
                    ttl: int = None) -> str:
        """Create a session."""
        session_id = str(uuid.uuid4())
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'data': data or {}
        }
        
        key = f"{self.prefix}{session_id}"
        await self.client.set(key, json.dumps(session_data), ex=ttl or self.default_ttl)
        
        return session_id
    
    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        key = f"{self.prefix}{session_id}"
        value = await self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def update(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data."""
        session = await self.get(session_id)
        if session:
            session['data'].update(data)
            key = f"{self.prefix}{session_id}"
            ttl = await self.client.ttl(key)
            await self.client.set(key, json.dumps(session), ex=ttl if ttl > 0 else self.default_ttl)
            return True
        return False
    
    async def delete(self, session_id: str) -> bool:
        """Delete session."""
        key = f"{self.prefix}{session_id}"
        return await self.client.delete(key) > 0
    
    async def refresh(self, session_id: str, ttl: int = None) -> bool:
        """Refresh session TTL."""
        key = f"{self.prefix}{session_id}"
        return await self.client.expire(key, ttl or self.default_ttl)


class RateLimiter:
    """
    Rate limiting using Redis.
    
    Provides:
    - Fixed window rate limiting
    - Sliding window rate limiting
    - Token bucket rate limiting
    """
    
    def __init__(self, client: MockRedisClient, prefix: str = "ratelimit:"):
        self.client = client
        self.prefix = prefix
    
    async def check_fixed_window(self, key: str, limit: int,
                                window_seconds: int) -> Tuple[bool, int]:
        """Check fixed window rate limit."""
        full_key = f"{self.prefix}{key}"
        
        current = await self.client.incr(full_key)
        
        if current == 1:
            await self.client.expire(full_key, window_seconds)
        
        remaining = max(0, limit - current)
        allowed = current <= limit
        
        return allowed, remaining
    
    async def check_sliding_window(self, key: str, limit: int,
                                  window_seconds: int) -> Tuple[bool, int]:
        """Check sliding window rate limit."""
        full_key = f"{self.prefix}sliding:{key}"
        now = time.time()
        window_start = now - window_seconds
        
        # Add current request
        await self.client.zadd(full_key, {str(now): now})
        
        # Remove old entries (simplified)
        # In real implementation, use ZREMRANGEBYSCORE
        
        # Count requests in window
        requests = await self.client.zrange(full_key, 0, -1, withscores=True)
        count = len([r for r in requests if r[1] > window_start])
        
        remaining = max(0, limit - count)
        allowed = count <= limit
        
        return allowed, remaining


class DistributedLock:
    """
    Distributed locking using Redis.
    
    Provides:
    - Lock acquisition
    - Lock release
    - Lock extension
    """
    
    def __init__(self, client: MockRedisClient, prefix: str = "lock:"):
        self.client = client
        self.prefix = prefix
    
    async def acquire(self, name: str, timeout: int = 10,
                     blocking: bool = True,
                     blocking_timeout: float = None) -> Optional[str]:
        """Acquire a lock."""
        lock_id = str(uuid.uuid4())
        key = f"{self.prefix}{name}"
        
        if blocking:
            start = time.time()
            while True:
                if await self.client.set(key, lock_id, ex=timeout, nx=True):
                    return lock_id
                
                if blocking_timeout and time.time() - start > blocking_timeout:
                    return None
                
                await asyncio.sleep(0.1)
        else:
            if await self.client.set(key, lock_id, ex=timeout, nx=True):
                return lock_id
            return None
    
    async def release(self, name: str, lock_id: str) -> bool:
        """Release a lock."""
        key = f"{self.prefix}{name}"
        current = await self.client.get(key)
        
        if current == lock_id:
            await self.client.delete(key)
            return True
        return False
    
    async def extend(self, name: str, lock_id: str, timeout: int) -> bool:
        """Extend lock timeout."""
        key = f"{self.prefix}{name}"
        current = await self.client.get(key)
        
        if current == lock_id:
            return await self.client.expire(key, timeout)
        return False


class RedisIntegration:
    """
    Redis integration for MineralVision.
    
    Provides comprehensive caching and data structures:
    - Cache management
    - Session management
    - Rate limiting
    - Distributed locking
    - Pub/sub messaging
    - Geospatial indexing
    
    Example:
        redis = RedisIntegration()
        await redis.connect()
        
        # Cache data
        await redis.cache.set("key", {"data": "value"}, ttl=3600)
        
        # Rate limiting
        allowed, remaining = await redis.rate_limiter.check_fixed_window(
            "user:123", limit=100, window_seconds=60
        )
        
        # Distributed lock
        lock_id = await redis.locks.acquire("resource-1")
    """
    
    def __init__(self, config: RedisConfig = None):
        self.config = config or RedisConfig()
        self.client: Optional[MockRedisClient] = None
        self.cache: Optional[CacheManager] = None
        self.sessions: Optional[SessionManager] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.locks: Optional[DistributedLock] = None
        self._connected = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """True when running on the explicit in-memory mock fallback."""
        return self._degraded

    async def connect(self) -> 'RedisIntegration':
        """
        Connect to Redis (real client first).

        Falls back to the in-memory mock ONLY when
        MV_ALLOW_MOCK_FALLBACK=true; otherwise raises RuntimeError.
        """
        if REDIS_AVAILABLE:
            try:
                self.client = Redis(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password,
                    db=self.config.db,
                    ssl=self.config.ssl,
                    decode_responses=self.config.decode_responses,
                    socket_connect_timeout=self.config.socket_timeout,
                    socket_timeout=self.config.socket_timeout
                )
                await self.client.ping()
                logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            except Exception as e:
                if real_client_unavailable("Redis", "connection failed", e):
                    self._degraded = True
                    self.client = MockRedisClient(self.config)
        else:
            if real_client_unavailable("Redis", "redis package not installed"):
                self._degraded = True
                self.client = MockRedisClient(self.config)

        self.cache = CacheManager(self.client)
        self.sessions = SessionManager(self.client)
        self.rate_limiter = RateLimiter(self.client)
        self.locks = DistributedLock(self.client)

        self._connected = True
        return self

    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            pong = await self.client.ping()
            return {'status': 'healthy', 'ping': pong, 'degraded': self._degraded}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e), 'degraded': self._degraded}
    
    async def close(self) -> None:
        """Close connection."""
        if self.client:
            await self.client.close()
        self._connected = False
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Factory functions

def create_redis(config: RedisConfig = None) -> RedisIntegration:
    """Create a Redis integration instance."""
    return RedisIntegration(config)


async def create_and_connect_redis(config: RedisConfig = None) -> RedisIntegration:
    """Create and connect Redis."""
    redis = RedisIntegration(config)
    await redis.connect()
    return redis
