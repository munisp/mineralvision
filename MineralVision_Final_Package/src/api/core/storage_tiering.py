"""
Storage Tiering Architecture for MineralVision.

This module provides:
- Clear storage tiering strategy (TileDB vs Parquet/Delta vs Object blobs)
- Intelligent data placement based on access patterns
- Caching layer for frequently accessed data
- Performance benchmarking and regression testing
- Automatic tiering based on data characteristics

Optimized for 10GB-1000GB survey datasets.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import json
import hashlib
import time
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


class StorageTier(Enum):
    """Storage tier types."""
    HOT = "hot"           # Frequently accessed, low latency (SSD/memory)
    WARM = "warm"         # Moderate access, balanced (SSD)
    COLD = "cold"         # Infrequent access, high capacity (HDD/object storage)
    ARCHIVE = "archive"   # Rare access, lowest cost (glacier/tape)


class DataFormat(Enum):
    """Data storage formats."""
    TILEDB = "tiledb"           # Dense/sparse arrays, random access
    PARQUET = "parquet"         # Columnar, analytics
    DELTA = "delta"             # Versioned tables, ACID
    ICEBERG = "iceberg"         # Large tables, schema evolution
    ZARR = "zarr"               # N-dimensional arrays
    GEOTIFF = "geotiff"         # Raster imagery
    LAS = "las"                 # Point clouds
    SEGY = "segy"               # Seismic data
    OBJECT_BLOB = "blob"        # Raw binary


class AccessPattern(Enum):
    """Data access patterns."""
    RANDOM = "random"           # Random tile/slice access
    SEQUENTIAL = "sequential"   # Sequential scan
    ANALYTICAL = "analytical"   # Aggregations, filters
    STREAMING = "streaming"     # Real-time ingestion
    BATCH = "batch"            # Bulk processing


@dataclass
class DataCharacteristics:
    """Characteristics of a dataset for tiering decisions."""
    size_bytes: int
    n_dimensions: int
    is_sparse: bool
    access_pattern: AccessPattern
    update_frequency: str  # 'never', 'daily', 'hourly', 'realtime'
    query_types: List[str]  # 'slice', 'point', 'range', 'aggregate'
    compression_ratio: float = 1.0
    
    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)


@dataclass
class TieringDecision:
    """Result of tiering decision."""
    recommended_format: DataFormat
    recommended_tier: StorageTier
    chunk_size: Tuple[int, ...]
    compression: str
    cache_policy: str
    rationale: List[str]
    estimated_query_latency_ms: float
    estimated_storage_cost_per_gb: float


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds


class LRUCache:
    """
    LRU cache for frequently accessed data.
    
    Thread-safe implementation with size limits.
    """
    
    def __init__(self, max_size_bytes: int = 1024 * 1024 * 1024):  # 1GB default
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
                
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired:
                self._remove(key)
                self._stats['misses'] += 1
                return None
                
            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._stats['hits'] += 1
            return entry.data
            
    def put(self, key: str, data: Any, size_bytes: int, ttl_seconds: Optional[int] = None):
        """Put item in cache."""
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove(key)
                
            # Evict if necessary
            while self.current_size_bytes + size_bytes > self.max_size_bytes and self._cache:
                self._evict_lru()
                
            # Add new entry
            entry = CacheEntry(
                key=key,
                data=data,
                size_bytes=size_bytes,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl_seconds
            )
            self._cache[key] = entry
            self.current_size_bytes += size_bytes
            
    def _remove(self, key: str):
        """Remove entry from cache."""
        if key in self._cache:
            self.current_size_bytes -= self._cache[key].size_bytes
            del self._cache[key]
            
    def _evict_lru(self):
        """Evict least recently used entry."""
        if self._cache:
            key = next(iter(self._cache))
            self._remove(key)
            self._stats['evictions'] += 1
            
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.current_size_bytes = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total if total > 0 else 0
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'hit_rate': hit_rate,
                'size_bytes': self.current_size_bytes,
                'max_size_bytes': self.max_size_bytes,
                'utilization': self.current_size_bytes / self.max_size_bytes,
                'n_entries': len(self._cache)
            }


class TieringEngine:
    """
    Intelligent data tiering engine.
    
    Determines optimal storage format and tier based on data characteristics.
    """
    
    def __init__(self):
        self.format_scores = self._init_format_scores()
        
    def _init_format_scores(self) -> Dict[DataFormat, Dict[str, float]]:
        """Initialize format scoring matrix."""
        return {
            DataFormat.TILEDB: {
                'random_access': 0.95,
                'sequential': 0.70,
                'compression': 0.85,
                'sparse_support': 0.95,
                'cloud_native': 0.90,
                'query_speed': 0.90
            },
            DataFormat.PARQUET: {
                'random_access': 0.60,
                'sequential': 0.95,
                'compression': 0.90,
                'sparse_support': 0.70,
                'cloud_native': 0.95,
                'query_speed': 0.85
            },
            DataFormat.DELTA: {
                'random_access': 0.65,
                'sequential': 0.90,
                'compression': 0.85,
                'sparse_support': 0.75,
                'cloud_native': 0.95,
                'query_speed': 0.80,
                'versioning': 0.95
            },
            DataFormat.ZARR: {
                'random_access': 0.90,
                'sequential': 0.80,
                'compression': 0.90,
                'sparse_support': 0.60,
                'cloud_native': 0.85,
                'query_speed': 0.85
            },
            DataFormat.GEOTIFF: {
                'random_access': 0.70,
                'sequential': 0.85,
                'compression': 0.75,
                'sparse_support': 0.40,
                'cloud_native': 0.70,
                'query_speed': 0.75
            }
        }
        
    def decide_tiering(self, characteristics: DataCharacteristics) -> TieringDecision:
        """
        Decide optimal storage format and tier.
        
        Args:
            characteristics: Data characteristics
            
        Returns:
            TieringDecision with recommendations
        """
        rationale = []
        
        # Determine format based on access pattern and data type
        format_scores = {}
        for fmt, scores in self.format_scores.items():
            score = 0.0
            
            # Access pattern scoring
            if characteristics.access_pattern == AccessPattern.RANDOM:
                score += scores.get('random_access', 0.5) * 0.4
                rationale.append(f"Random access favors {fmt.value}")
            elif characteristics.access_pattern == AccessPattern.SEQUENTIAL:
                score += scores.get('sequential', 0.5) * 0.4
            elif characteristics.access_pattern == AccessPattern.ANALYTICAL:
                score += scores.get('query_speed', 0.5) * 0.4
                
            # Sparse data scoring
            if characteristics.is_sparse:
                score += scores.get('sparse_support', 0.5) * 0.2
                
            # Size-based scoring
            if characteristics.size_gb > 100:
                score += scores.get('cloud_native', 0.5) * 0.2
                
            # Compression scoring
            score += scores.get('compression', 0.5) * 0.2
            
            format_scores[fmt] = score
            
        # Select best format
        recommended_format = max(format_scores, key=format_scores.get)
        
        # Determine tier based on access frequency and size
        if characteristics.update_frequency == 'realtime':
            recommended_tier = StorageTier.HOT
            rationale.append("Real-time updates require hot storage")
        elif characteristics.size_gb < 10 and characteristics.access_pattern == AccessPattern.RANDOM:
            recommended_tier = StorageTier.HOT
            rationale.append("Small dataset with random access placed in hot tier")
        elif characteristics.size_gb < 100:
            recommended_tier = StorageTier.WARM
            rationale.append("Medium dataset placed in warm tier")
        elif characteristics.update_frequency == 'never':
            recommended_tier = StorageTier.COLD
            rationale.append("Static dataset placed in cold tier")
        else:
            recommended_tier = StorageTier.WARM
            
        # Determine chunk size
        chunk_size = self._calculate_chunk_size(characteristics, recommended_format)
        
        # Determine compression
        compression = self._select_compression(characteristics, recommended_format)
        
        # Determine cache policy
        cache_policy = self._select_cache_policy(characteristics, recommended_tier)
        
        # Estimate performance
        latency_ms = self._estimate_latency(characteristics, recommended_format, recommended_tier)
        cost_per_gb = self._estimate_cost(recommended_tier)
        
        return TieringDecision(
            recommended_format=recommended_format,
            recommended_tier=recommended_tier,
            chunk_size=chunk_size,
            compression=compression,
            cache_policy=cache_policy,
            rationale=rationale,
            estimated_query_latency_ms=latency_ms,
            estimated_storage_cost_per_gb=cost_per_gb
        )
        
    def _calculate_chunk_size(self, characteristics: DataCharacteristics, 
                             fmt: DataFormat) -> Tuple[int, ...]:
        """Calculate optimal chunk size."""
        # Target chunk size: 64MB for cloud, 16MB for local
        target_bytes = 64 * 1024 * 1024
        
        if characteristics.n_dimensions == 2:
            # 2D data (e.g., grids)
            side = int(np.sqrt(target_bytes / 4))  # Assume float32
            return (side, side)
        elif characteristics.n_dimensions == 3:
            # 3D data (e.g., seismic cubes)
            side = int(np.cbrt(target_bytes / 4))
            return (side, side, side)
        else:
            # 1D or higher
            return (target_bytes // 4,)
            
    def _select_compression(self, characteristics: DataCharacteristics,
                           fmt: DataFormat) -> str:
        """Select compression algorithm."""
        if characteristics.access_pattern == AccessPattern.RANDOM:
            return "lz4"  # Fast decompression
        elif characteristics.size_gb > 100:
            return "zstd"  # Good compression ratio
        else:
            return "blosc"  # Balanced
            
    def _select_cache_policy(self, characteristics: DataCharacteristics,
                            tier: StorageTier) -> str:
        """Select cache policy."""
        if tier == StorageTier.HOT:
            return "write-through"
        elif characteristics.access_pattern == AccessPattern.RANDOM:
            return "lru"
        else:
            return "read-ahead"
            
    def _estimate_latency(self, characteristics: DataCharacteristics,
                         fmt: DataFormat, tier: StorageTier) -> float:
        """Estimate query latency in milliseconds."""
        base_latency = {
            StorageTier.HOT: 1.0,
            StorageTier.WARM: 10.0,
            StorageTier.COLD: 100.0,
            StorageTier.ARCHIVE: 1000.0
        }
        
        format_factor = {
            DataFormat.TILEDB: 1.0,
            DataFormat.PARQUET: 1.5,
            DataFormat.DELTA: 1.8,
            DataFormat.ZARR: 1.2,
            DataFormat.GEOTIFF: 2.0
        }
        
        return base_latency[tier] * format_factor.get(fmt, 1.5)
        
    def _estimate_cost(self, tier: StorageTier) -> float:
        """Estimate storage cost per GB per month."""
        costs = {
            StorageTier.HOT: 0.10,
            StorageTier.WARM: 0.05,
            StorageTier.COLD: 0.02,
            StorageTier.ARCHIVE: 0.004
        }
        return costs[tier]


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    operation: str
    data_size_bytes: int
    duration_ms: float
    throughput_mbps: float
    latency_p50_ms: float
    latency_p99_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'data_size_mb': self.data_size_bytes / (1024 * 1024),
            'duration_ms': self.duration_ms,
            'throughput_mbps': self.throughput_mbps,
            'latency_p50_ms': self.latency_p50_ms,
            'latency_p99_ms': self.latency_p99_ms,
            'timestamp': self.timestamp.isoformat()
        }


class PerformanceBenchmark:
    """
    Performance benchmarking for storage operations.
    
    Provides automated regression testing.
    """
    
    def __init__(self):
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        self.history: List[BenchmarkResult] = []
        
    def benchmark_read(self, read_func: Callable, data_size_bytes: int,
                      n_iterations: int = 10) -> BenchmarkResult:
        """Benchmark read operation."""
        latencies = []
        
        for _ in range(n_iterations):
            start = time.perf_counter()
            read_func()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
            
        latencies = np.array(latencies)
        total_duration = np.sum(latencies)
        throughput = (data_size_bytes * n_iterations) / (total_duration / 1000) / (1024 * 1024)
        
        result = BenchmarkResult(
            operation='read',
            data_size_bytes=data_size_bytes,
            duration_ms=total_duration,
            throughput_mbps=throughput,
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p99_ms=float(np.percentile(latencies, 99))
        )
        
        self.history.append(result)
        return result
        
    def benchmark_write(self, write_func: Callable, data_size_bytes: int,
                       n_iterations: int = 10) -> BenchmarkResult:
        """Benchmark write operation."""
        latencies = []
        
        for _ in range(n_iterations):
            start = time.perf_counter()
            write_func()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            
        latencies = np.array(latencies)
        total_duration = np.sum(latencies)
        throughput = (data_size_bytes * n_iterations) / (total_duration / 1000) / (1024 * 1024)
        
        result = BenchmarkResult(
            operation='write',
            data_size_bytes=data_size_bytes,
            duration_ms=total_duration,
            throughput_mbps=throughput,
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p99_ms=float(np.percentile(latencies, 99))
        )
        
        self.history.append(result)
        return result
        
    def set_baseline(self, operation: str, result: BenchmarkResult):
        """Set baseline for regression testing."""
        self.baseline_results[operation] = result
        
    def check_regression(self, operation: str, result: BenchmarkResult,
                        threshold_percent: float = 20.0) -> Dict[str, Any]:
        """Check for performance regression."""
        if operation not in self.baseline_results:
            return {'regression': False, 'message': 'No baseline set'}
            
        baseline = self.baseline_results[operation]
        
        # Check throughput regression
        throughput_change = ((baseline.throughput_mbps - result.throughput_mbps) / 
                            baseline.throughput_mbps * 100)
        
        # Check latency regression
        latency_change = ((result.latency_p99_ms - baseline.latency_p99_ms) / 
                         baseline.latency_p99_ms * 100)
        
        regression = throughput_change > threshold_percent or latency_change > threshold_percent
        
        return {
            'regression': regression,
            'throughput_change_percent': throughput_change,
            'latency_change_percent': latency_change,
            'threshold_percent': threshold_percent,
            'message': 'Performance regression detected' if regression else 'Performance within threshold'
        }
        
    def get_report(self) -> Dict[str, Any]:
        """Generate benchmark report."""
        if not self.history:
            return {'message': 'No benchmarks run'}
            
        return {
            'n_benchmarks': len(self.history),
            'operations': list(set(r.operation for r in self.history)),
            'latest_results': [r.to_dict() for r in self.history[-5:]],
            'baselines': {k: v.to_dict() for k, v in self.baseline_results.items()}
        }


class StorageManager:
    """
    Unified storage manager for MineralVision.
    
    Manages data placement, caching, and access across tiers.
    """
    
    def __init__(self, cache_size_bytes: int = 1024 * 1024 * 1024):
        self.tiering_engine = TieringEngine()
        self.cache = LRUCache(cache_size_bytes)
        self.benchmark = PerformanceBenchmark()
        self._datasets: Dict[str, Dict[str, Any]] = {}
        
    def register_dataset(self, dataset_id: str, 
                        characteristics: DataCharacteristics,
                        path: str) -> TieringDecision:
        """
        Register a dataset and get tiering recommendation.
        
        Args:
            dataset_id: Unique dataset identifier
            characteristics: Data characteristics
            path: Storage path
            
        Returns:
            TieringDecision with recommendations
        """
        decision = self.tiering_engine.decide_tiering(characteristics)
        
        self._datasets[dataset_id] = {
            'characteristics': characteristics,
            'path': path,
            'decision': decision,
            'registered_at': datetime.now()
        }
        
        logger.info(f"Registered dataset {dataset_id}: {decision.recommended_format.value} "
                   f"on {decision.recommended_tier.value} tier")
        
        return decision
        
    def get_cached(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        return self.cache.get(key)
        
    def put_cached(self, key: str, data: Any, size_bytes: int, 
                  ttl_seconds: Optional[int] = None):
        """Put data in cache."""
        self.cache.put(key, data, size_bytes, ttl_seconds)
        
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset information."""
        return self._datasets.get(dataset_id)
        
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self._datasets.keys())
        
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        tier_counts = {tier: 0 for tier in StorageTier}
        format_counts = {fmt: 0 for fmt in DataFormat}
        total_size = 0
        
        for ds in self._datasets.values():
            decision = ds['decision']
            tier_counts[decision.recommended_tier] += 1
            format_counts[decision.recommended_format] += 1
            total_size += ds['characteristics'].size_bytes
            
        return {
            'n_datasets': len(self._datasets),
            'total_size_gb': total_size / (1024 ** 3),
            'tier_distribution': {t.value: c for t, c in tier_counts.items()},
            'format_distribution': {f.value: c for f, c in format_counts.items()},
            'cache_stats': self.cache.get_stats()
        }


# Factory functions
def create_storage_manager(cache_size_gb: float = 1.0) -> StorageManager:
    """Create storage manager with specified cache size."""
    cache_bytes = int(cache_size_gb * 1024 * 1024 * 1024)
    return StorageManager(cache_bytes)


def analyze_dataset_for_tiering(size_gb: float, 
                               n_dimensions: int,
                               is_sparse: bool,
                               access_pattern: str,
                               update_frequency: str) -> TieringDecision:
    """
    Analyze dataset and return tiering recommendation.
    
    Args:
        size_gb: Dataset size in GB
        n_dimensions: Number of dimensions
        is_sparse: Whether data is sparse
        access_pattern: 'random', 'sequential', 'analytical', 'streaming', 'batch'
        update_frequency: 'never', 'daily', 'hourly', 'realtime'
        
    Returns:
        TieringDecision with recommendations
    """
    characteristics = DataCharacteristics(
        size_bytes=int(size_gb * 1024 * 1024 * 1024),
        n_dimensions=n_dimensions,
        is_sparse=is_sparse,
        access_pattern=AccessPattern(access_pattern),
        update_frequency=update_frequency,
        query_types=['slice', 'point', 'range']
    )
    
    engine = TieringEngine()
    return engine.decide_tiering(characteristics)
