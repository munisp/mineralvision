"""
Production Operations Hardening Module
======================================

Production-grade operations with:
- Prometheus metrics (latency, FPS, GPU memory)
- Structured logging with correlation IDs
- Rate limiting and request size limits
- Health checks and readiness probes
- Circuit breaker pattern
- Demo mode separation
"""

import os
import time
import logging
import threading
import functools
import json
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
from contextlib import contextmanager
import traceback

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    """Operation modes."""
    PRODUCTION = "production"
    DEMO = "demo"
    DEBUG = "debug"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    enabled: bool = True
    prometheus_port: int = 9090
    collection_interval_s: float = 1.0
    histogram_buckets: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    ])
    gpu_metrics_enabled: bool = True


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    enabled: bool = True
    requests_per_second: float = 100.0
    burst_size: int = 200
    per_client: bool = True


@dataclass
class RequestLimitConfig:
    """Configuration for request limits."""
    max_image_size_mb: float = 50.0
    max_batch_size: int = 10
    max_request_timeout_s: float = 60.0


class PrometheusMetrics:
    """
    Prometheus metrics collector for WALDO.
    """
    
    def __init__(self, config: MetricsConfig = None):
        self.config = config or MetricsConfig()
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._initialized = False
        
        if self.config.enabled:
            self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""
        try:
            from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
            
            # Counters
            self._metrics['requests_total'] = Counter(
                'waldo_requests_total',
                'Total number of detection requests',
                ['method', 'status']
            )
            
            self._metrics['detections_total'] = Counter(
                'waldo_detections_total',
                'Total number of objects detected',
                ['class_name']
            )
            
            self._metrics['errors_total'] = Counter(
                'waldo_errors_total',
                'Total number of errors',
                ['error_type']
            )
            
            # Histograms
            self._metrics['request_latency'] = Histogram(
                'waldo_request_latency_seconds',
                'Request latency in seconds',
                ['method'],
                buckets=self.config.histogram_buckets
            )
            
            self._metrics['inference_latency'] = Histogram(
                'waldo_inference_latency_seconds',
                'Model inference latency in seconds',
                buckets=self.config.histogram_buckets
            )
            
            self._metrics['preprocessing_latency'] = Histogram(
                'waldo_preprocessing_latency_seconds',
                'Image preprocessing latency in seconds',
                buckets=self.config.histogram_buckets
            )
            
            # Gauges
            self._metrics['active_requests'] = Gauge(
                'waldo_active_requests',
                'Number of active requests'
            )
            
            self._metrics['queue_size'] = Gauge(
                'waldo_queue_size',
                'Size of the inference queue'
            )
            
            self._metrics['fps'] = Gauge(
                'waldo_fps',
                'Current frames per second'
            )
            
            if self.config.gpu_metrics_enabled:
                self._metrics['gpu_memory_used'] = Gauge(
                    'waldo_gpu_memory_used_bytes',
                    'GPU memory used in bytes',
                    ['device']
                )
                
                self._metrics['gpu_utilization'] = Gauge(
                    'waldo_gpu_utilization_percent',
                    'GPU utilization percentage',
                    ['device']
                )
            
            # Info
            self._metrics['info'] = Info(
                'waldo',
                'WALDO service information'
            )
            
            # Start metrics server
            start_http_server(self.config.prometheus_port)
            
            self._initialized = True
            logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
            
        except ImportError:
            logger.warning("prometheus_client not available, metrics disabled")
            self._initialized = False
    
    def set_info(self, version: str, model_name: str, class_count: int):
        """Set service info."""
        if not self._initialized:
            return
        
        self._metrics['info'].info({
            'version': version,
            'model_name': model_name,
            'class_count': str(class_count)
        })
    
    def inc_requests(self, method: str, status: str):
        """Increment request counter."""
        if not self._initialized:
            return
        self._metrics['requests_total'].labels(method=method, status=status).inc()
    
    def inc_detections(self, class_name: str, count: int = 1):
        """Increment detection counter."""
        if not self._initialized:
            return
        self._metrics['detections_total'].labels(class_name=class_name).inc(count)
    
    def inc_errors(self, error_type: str):
        """Increment error counter."""
        if not self._initialized:
            return
        self._metrics['errors_total'].labels(error_type=error_type).inc()
    
    def observe_request_latency(self, method: str, latency_s: float):
        """Record request latency."""
        if not self._initialized:
            return
        self._metrics['request_latency'].labels(method=method).observe(latency_s)
    
    def observe_inference_latency(self, latency_s: float):
        """Record inference latency."""
        if not self._initialized:
            return
        self._metrics['inference_latency'].observe(latency_s)
    
    def observe_preprocessing_latency(self, latency_s: float):
        """Record preprocessing latency."""
        if not self._initialized:
            return
        self._metrics['preprocessing_latency'].observe(latency_s)
    
    def set_active_requests(self, count: int):
        """Set active request count."""
        if not self._initialized:
            return
        self._metrics['active_requests'].set(count)
    
    def set_queue_size(self, size: int):
        """Set queue size."""
        if not self._initialized:
            return
        self._metrics['queue_size'].set(size)
    
    def set_fps(self, fps: float):
        """Set current FPS."""
        if not self._initialized:
            return
        self._metrics['fps'].set(fps)
    
    def update_gpu_metrics(self):
        """Update GPU metrics."""
        if not self._initialized or not self.config.gpu_metrics_enabled:
            return
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._metrics['gpu_memory_used'].labels(device=str(i)).set(mem_info.used)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._metrics['gpu_utilization'].labels(device=str(i)).set(util.gpu)
            
            pynvml.nvmlShutdown()
            
        except Exception as e:
            logger.debug(f"Failed to update GPU metrics: {e}")


class StructuredLogger:
    """
    Structured logging with correlation IDs and context.
    """
    
    def __init__(self, service_name: str = "waldo"):
        self.service_name = service_name
        self._context = threading.local()
    
    def set_correlation_id(self, correlation_id: str = None):
        """Set correlation ID for current thread."""
        self._context.correlation_id = correlation_id or str(uuid.uuid4())
    
    def get_correlation_id(self) -> str:
        """Get correlation ID for current thread."""
        return getattr(self._context, 'correlation_id', None) or str(uuid.uuid4())
    
    def set_context(self, **kwargs):
        """Set additional context."""
        if not hasattr(self._context, 'extra'):
            self._context.extra = {}
        self._context.extra.update(kwargs)
    
    def clear_context(self):
        """Clear context."""
        self._context.correlation_id = None
        self._context.extra = {}
    
    def _format_log(self, level: str, message: str, **kwargs) -> str:
        """Format log message as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'service': self.service_name,
            'correlation_id': self.get_correlation_id(),
            'message': message
        }
        
        # Add context
        if hasattr(self._context, 'extra'):
            log_entry.update(self._context.extra)
        
        # Add additional kwargs
        log_entry.update(kwargs)
        
        return json.dumps(log_entry)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        logger.debug(self._format_log('DEBUG', message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        logger.info(self._format_log('INFO', message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        logger.warning(self._format_log('WARNING', message, **kwargs))
    
    def error(self, message: str, exception: Exception = None, **kwargs):
        """Log error message."""
        if exception:
            kwargs['exception'] = str(exception)
            kwargs['traceback'] = traceback.format_exc()
        logger.error(self._format_log('ERROR', message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        logger.critical(self._format_log('CRITICAL', message, **kwargs))
    
    @contextmanager
    def request_context(self, correlation_id: str = None, **kwargs):
        """Context manager for request logging."""
        self.set_correlation_id(correlation_id)
        self.set_context(**kwargs)
        try:
            yield self.get_correlation_id()
        finally:
            self.clear_context()


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def _get_bucket(self, key: str) -> Dict:
        """Get or create bucket for key."""
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = {
                    'tokens': self.config.burst_size,
                    'last_update': time.time()
                }
            return self._buckets[key]
    
    def _refill(self, bucket: Dict):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - bucket['last_update']
        tokens_to_add = elapsed * self.config.requests_per_second
        bucket['tokens'] = min(self.config.burst_size, bucket['tokens'] + tokens_to_add)
        bucket['last_update'] = now
    
    def allow(self, client_id: str = "default") -> bool:
        """
        Check if request is allowed.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed
        """
        if not self.config.enabled:
            return True
        
        key = client_id if self.config.per_client else "global"
        bucket = self._get_bucket(key)
        
        with self._lock:
            self._refill(bucket)
            
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            
            return False
    
    def get_wait_time(self, client_id: str = "default") -> float:
        """Get time to wait before next request is allowed."""
        if not self.config.enabled:
            return 0.0
        
        key = client_id if self.config.per_client else "global"
        bucket = self._get_bucket(key)
        
        with self._lock:
            self._refill(bucket)
            
            if bucket['tokens'] >= 1:
                return 0.0
            
            tokens_needed = 1 - bucket['tokens']
            return tokens_needed / self.config.requests_per_second


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    """
    
    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout_s: float = 30.0,
                 half_open_requests: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_requests = half_open_requests
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
    
    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time and \
                   time.time() - self.last_failure_time >= self.recovery_timeout_s:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return True
                return False
            
            if self.state == CircuitState.HALF_OPEN:
                return self.success_count < self.half_open_requests
            
            return False
    
    def record_success(self):
        """Record successful request."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_requests:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def record_failure(self):
        """Record failed request."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time
        }


class HealthChecker:
    """
    Health and readiness checks.
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], bool]] = {}
        self._ready = False
        self._healthy = True
    
    def register_check(self, name: str, check_fn: Callable[[], bool]):
        """Register a health check."""
        self._checks[name] = check_fn
    
    def set_ready(self, ready: bool):
        """Set readiness status."""
        self._ready = ready
    
    def set_healthy(self, healthy: bool):
        """Set health status."""
        self._healthy = healthy
    
    def is_healthy(self) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if service is healthy.
        
        Returns:
            Tuple of (overall_healthy, check_results)
        """
        results = {}
        overall = self._healthy
        
        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                results[name] = result
                if not result:
                    overall = False
            except Exception as e:
                results[name] = False
                overall = False
                logger.error(f"Health check '{name}' failed: {e}")
        
        return overall, results
    
    def is_ready(self) -> bool:
        """Check if service is ready to accept traffic."""
        return self._ready and self._healthy
    
    def get_status(self) -> Dict[str, Any]:
        """Get full health status."""
        healthy, checks = self.is_healthy()
        return {
            'healthy': healthy,
            'ready': self.is_ready(),
            'checks': checks
        }


class RequestValidator:
    """
    Request validation and limits.
    """
    
    def __init__(self, config: RequestLimitConfig = None):
        self.config = config or RequestLimitConfig()
    
    def validate_image_size(self, size_bytes: int) -> bool:
        """Validate image size."""
        max_bytes = self.config.max_image_size_mb * 1024 * 1024
        return size_bytes <= max_bytes
    
    def validate_batch_size(self, batch_size: int) -> bool:
        """Validate batch size."""
        return batch_size <= self.config.max_batch_size
    
    def validate_request(self, image_size: int = 0, batch_size: int = 1) -> Tuple[bool, str]:
        """
        Validate request.
        
        Returns:
            Tuple of (valid, error_message)
        """
        if not self.validate_image_size(image_size):
            return False, f"Image size exceeds limit of {self.config.max_image_size_mb}MB"
        
        if not self.validate_batch_size(batch_size):
            return False, f"Batch size exceeds limit of {self.config.max_batch_size}"
        
        return True, ""


class DemoModeManager:
    """
    Manages demo mode vs production mode.
    """
    
    def __init__(self, mode: OperationMode = OperationMode.PRODUCTION):
        self.mode = mode
        self._demo_responses: Dict[str, Any] = {}
    
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode."""
        return self.mode == OperationMode.DEMO
    
    def is_production_mode(self) -> bool:
        """Check if running in production mode."""
        return self.mode == OperationMode.PRODUCTION
    
    def set_mode(self, mode: OperationMode):
        """Set operation mode."""
        self.mode = mode
        logger.info(f"Operation mode set to: {mode.value}")
    
    def register_demo_response(self, endpoint: str, response: Any):
        """Register a demo response for an endpoint."""
        self._demo_responses[endpoint] = response
    
    def get_demo_response(self, endpoint: str) -> Optional[Any]:
        """Get demo response for endpoint."""
        return self._demo_responses.get(endpoint)
    
    def should_use_demo_response(self, endpoint: str) -> bool:
        """Check if demo response should be used."""
        return self.is_demo_mode() and endpoint in self._demo_responses


class OpsHardeningManager:
    """
    Complete operations hardening manager.
    """
    
    def __init__(self, service_name: str = "waldo",
                 mode: OperationMode = OperationMode.PRODUCTION):
        self.service_name = service_name
        self.mode = mode
        
        # Initialize components
        self.metrics = PrometheusMetrics()
        self.logger = StructuredLogger(service_name)
        self.rate_limiter = TokenBucketRateLimiter()
        self.circuit_breaker = CircuitBreaker()
        self.health_checker = HealthChecker()
        self.request_validator = RequestValidator()
        self.demo_manager = DemoModeManager(mode)
        
        # Request tracking
        self._active_requests = 0
        self._request_lock = threading.Lock()
        
        # FPS tracking
        self._frame_times: deque = deque(maxlen=100)
    
    def configure(self, metrics_config: MetricsConfig = None,
                 rate_limit_config: RateLimitConfig = None,
                 request_limit_config: RequestLimitConfig = None):
        """Configure components."""
        if metrics_config:
            self.metrics = PrometheusMetrics(metrics_config)
        if rate_limit_config:
            self.rate_limiter = TokenBucketRateLimiter(rate_limit_config)
        if request_limit_config:
            self.request_validator = RequestValidator(request_limit_config)
    
    @contextmanager
    def request_context(self, method: str, client_id: str = "default",
                       image_size: int = 0, batch_size: int = 1):
        """
        Context manager for request handling with all ops features.
        
        Args:
            method: Request method/endpoint
            client_id: Client identifier
            image_size: Image size in bytes
            batch_size: Batch size
            
        Yields:
            Correlation ID
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        with self.logger.request_context(correlation_id, method=method, client_id=client_id):
            # Check rate limit
            if not self.rate_limiter.allow(client_id):
                self.metrics.inc_errors('rate_limited')
                self.logger.warning("Rate limit exceeded", client_id=client_id)
                raise RateLimitExceeded(f"Rate limit exceeded for {client_id}")
            
            # Check circuit breaker
            if not self.circuit_breaker.allow_request():
                self.metrics.inc_errors('circuit_open')
                self.logger.warning("Circuit breaker open")
                raise CircuitBreakerOpen("Service temporarily unavailable")
            
            # Validate request
            valid, error = self.request_validator.validate_request(image_size, batch_size)
            if not valid:
                self.metrics.inc_errors('validation_failed')
                self.logger.warning("Request validation failed", error=error)
                raise RequestValidationError(error)
            
            # Track active requests
            with self._request_lock:
                self._active_requests += 1
                self.metrics.set_active_requests(self._active_requests)
            
            try:
                self.logger.info("Request started")
                yield correlation_id
                
                # Record success
                self.circuit_breaker.record_success()
                self.metrics.inc_requests(method, 'success')
                
            except Exception as e:
                # Record failure
                self.circuit_breaker.record_failure()
                self.metrics.inc_requests(method, 'error')
                self.metrics.inc_errors(type(e).__name__)
                self.logger.error("Request failed", exception=e)
                raise
            
            finally:
                # Track request completion
                with self._request_lock:
                    self._active_requests -= 1
                    self.metrics.set_active_requests(self._active_requests)
                
                # Record latency
                latency = time.time() - start_time
                self.metrics.observe_request_latency(method, latency)
                
                # Track FPS
                self._frame_times.append(time.time())
                if len(self._frame_times) > 1:
                    fps = len(self._frame_times) / (self._frame_times[-1] - self._frame_times[0])
                    self.metrics.set_fps(fps)
                
                self.logger.info("Request completed", latency_ms=latency * 1000)
    
    def record_inference(self, latency_s: float, detections: List[Dict]):
        """Record inference metrics."""
        self.metrics.observe_inference_latency(latency_s)
        
        for det in detections:
            class_name = det.get('class_name', 'unknown')
            self.metrics.inc_detections(class_name)
    
    def record_preprocessing(self, latency_s: float):
        """Record preprocessing latency."""
        self.metrics.observe_preprocessing_latency(latency_s)
    
    def update_gpu_metrics(self):
        """Update GPU metrics."""
        self.metrics.update_gpu_metrics()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        return self.health_checker.get_status()
    
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self.health_checker.is_ready()
    
    def set_ready(self, ready: bool):
        """Set readiness status."""
        self.health_checker.set_ready(ready)
    
    def register_health_check(self, name: str, check_fn: Callable[[], bool]):
        """Register a health check."""
        self.health_checker.register_check(name, check_fn)


# Custom exceptions
class RateLimitExceeded(Exception):
    """Rate limit exceeded."""
    pass


class CircuitBreakerOpen(Exception):
    """Circuit breaker is open."""
    pass


class RequestValidationError(Exception):
    """Request validation failed."""
    pass


def create_ops_manager(service_name: str = "waldo",
                      mode: str = "production",
                      config: Optional[Dict] = None) -> OpsHardeningManager:
    """Factory function to create ops manager."""
    op_mode = OperationMode(mode)
    manager = OpsHardeningManager(service_name, op_mode)
    
    if config:
        metrics_config = MetricsConfig(
            enabled=config.get('metrics_enabled', True),
            prometheus_port=config.get('prometheus_port', 9090),
            gpu_metrics_enabled=config.get('gpu_metrics_enabled', True)
        )
        
        rate_limit_config = RateLimitConfig(
            enabled=config.get('rate_limit_enabled', True),
            requests_per_second=config.get('requests_per_second', 100.0),
            burst_size=config.get('burst_size', 200)
        )
        
        request_limit_config = RequestLimitConfig(
            max_image_size_mb=config.get('max_image_size_mb', 50.0),
            max_batch_size=config.get('max_batch_size', 10)
        )
        
        manager.configure(metrics_config, rate_limit_config, request_limit_config)
    
    return manager
