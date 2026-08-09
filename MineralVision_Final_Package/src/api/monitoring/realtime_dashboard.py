"""
Real-Time Monitoring Dashboard for MineralVision.

Provides centralized observability with:
- Prometheus metrics integration
- Grafana dashboard configuration
- Model inference latency tracking
- Data pipeline throughput monitoring
- Storage utilization alerts
- Drift detection visualization
- Custom metric collectors
"""

import time
import threading
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from collections import deque
import hashlib

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved"
    PENDING = "pending"
    SILENCED = "silenced"


class DashboardPanel(Enum):
    """Dashboard panel types."""
    GRAPH = "graph"
    STAT = "stat"
    GAUGE = "gauge"
    TABLE = "table"
    HEATMAP = "heatmap"
    LOGS = "logs"
    ALERT_LIST = "alertlist"


@dataclass
class MetricLabel:
    """Metric label for dimensional data."""
    name: str
    value: str


@dataclass
class MetricSample:
    """Single metric sample."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HistogramBucket:
    """Histogram bucket."""
    le: float  # less than or equal
    count: int


@dataclass
class HistogramMetric:
    """Histogram metric with buckets."""
    name: str
    buckets: List[HistogramBucket]
    sum_value: float
    count: int
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    name: str
    expression: str  # PromQL expression
    duration: timedelta
    severity: AlertSeverity
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    def to_prometheus_rule(self) -> Dict[str, Any]:
        """Convert to Prometheus alerting rule format."""
        return {
            'alert': self.name,
            'expr': self.expression,
            'for': f"{int(self.duration.total_seconds())}s",
            'labels': {**self.labels, 'severity': self.severity.value},
            'annotations': self.annotations
        }


@dataclass
class Alert:
    """Active alert instance."""
    alert_id: str
    rule: AlertRule
    status: AlertStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'name': self.rule.name,
            'status': self.status.value,
            'severity': self.rule.severity.value,
            'started_at': self.started_at.isoformat(),
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'value': self.value,
            'labels': self.labels,
            'annotations': self.annotations
        }


@dataclass
class GrafanaPanel:
    """Grafana dashboard panel configuration."""
    panel_id: int
    title: str
    panel_type: DashboardPanel
    targets: List[Dict[str, Any]]  # PromQL queries
    grid_pos: Dict[str, int]  # x, y, w, h
    options: Dict[str, Any] = field(default_factory=dict)
    
    def to_grafana_json(self) -> Dict[str, Any]:
        """Convert to Grafana panel JSON."""
        return {
            'id': self.panel_id,
            'title': self.title,
            'type': self.panel_type.value,
            'targets': self.targets,
            'gridPos': self.grid_pos,
            **self.options
        }


@dataclass
class GrafanaDashboard:
    """Grafana dashboard configuration."""
    uid: str
    title: str
    panels: List[GrafanaPanel]
    tags: List[str] = field(default_factory=list)
    refresh: str = "5s"
    time_from: str = "now-1h"
    time_to: str = "now"
    
    def to_grafana_json(self) -> Dict[str, Any]:
        """Convert to Grafana dashboard JSON."""
        return {
            'uid': self.uid,
            'title': self.title,
            'tags': self.tags,
            'refresh': self.refresh,
            'time': {
                'from': self.time_from,
                'to': self.time_to
            },
            'panels': [p.to_grafana_json() for p in self.panels]
        }


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    @abstractmethod
    def collect(self) -> List[MetricSample]:
        """Collect metrics."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get collector name."""
        pass


class CounterMetric:
    """Thread-safe counter metric."""
    
    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def inc(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment counter."""
        key = self._label_key(labels or {})
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value
            
    def get(self, labels: Dict[str, str] = None) -> float:
        """Get counter value."""
        key = self._label_key(labels or {})
        with self._lock:
            return self._values.get(key, 0)
            
    def _label_key(self, labels: Dict[str, str]) -> str:
        """Generate key from labels."""
        return json.dumps(labels, sort_keys=True)
        
    def collect(self) -> List[MetricSample]:
        """Collect all counter values."""
        samples = []
        with self._lock:
            for key, value in self._values.items():
                labels = json.loads(key) if key else {}
                samples.append(MetricSample(
                    name=self.name,
                    value=value,
                    timestamp=datetime.utcnow(),
                    labels=labels,
                    metric_type=MetricType.COUNTER
                ))
        return samples


class GaugeMetric:
    """Thread-safe gauge metric."""
    
    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def set(self, value: float, labels: Dict[str, str] = None) -> None:
        """Set gauge value."""
        key = self._label_key(labels or {})
        with self._lock:
            self._values[key] = value
            
    def inc(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment gauge."""
        key = self._label_key(labels or {})
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value
            
    def dec(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Decrement gauge."""
        key = self._label_key(labels or {})
        with self._lock:
            self._values[key] = self._values.get(key, 0) - value
            
    def get(self, labels: Dict[str, str] = None) -> float:
        """Get gauge value."""
        key = self._label_key(labels or {})
        with self._lock:
            return self._values.get(key, 0)
            
    def _label_key(self, labels: Dict[str, str]) -> str:
        return json.dumps(labels, sort_keys=True)
        
    def collect(self) -> List[MetricSample]:
        """Collect all gauge values."""
        samples = []
        with self._lock:
            for key, value in self._values.items():
                labels = json.loads(key) if key else {}
                samples.append(MetricSample(
                    name=self.name,
                    value=value,
                    timestamp=datetime.utcnow(),
                    labels=labels,
                    metric_type=MetricType.GAUGE
                ))
        return samples


class HistogramMetricCollector:
    """Thread-safe histogram metric."""
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    def __init__(self, name: str, description: str, 
                 buckets: List[float] = None, labels: List[str] = None):
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self.label_names = labels or []
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
    def observe(self, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a value."""
        key = self._label_key(labels or {})
        with self._lock:
            if key not in self._data:
                self._data[key] = {
                    'buckets': {b: 0 for b in self.buckets},
                    'sum': 0.0,
                    'count': 0
                }
            
            data = self._data[key]
            data['sum'] += value
            data['count'] += 1
            
            for bucket in self.buckets:
                if value <= bucket:
                    data['buckets'][bucket] += 1
                    
    def _label_key(self, labels: Dict[str, str]) -> str:
        return json.dumps(labels, sort_keys=True)
        
    def collect(self) -> List[HistogramMetric]:
        """Collect histogram data."""
        histograms = []
        with self._lock:
            for key, data in self._data.items():
                labels = json.loads(key) if key else {}
                buckets = [
                    HistogramBucket(le=b, count=c)
                    for b, c in sorted(data['buckets'].items())
                ]
                histograms.append(HistogramMetric(
                    name=self.name,
                    buckets=buckets,
                    sum_value=data['sum'],
                    count=data['count'],
                    labels=labels
                ))
        return histograms


class InferenceLatencyCollector(MetricCollector):
    """Collect model inference latency metrics."""
    
    def __init__(self):
        self.histogram = HistogramMetricCollector(
            name='mineralvision_inference_latency_seconds',
            description='Model inference latency in seconds',
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            labels=['model_name', 'model_version', 'endpoint']
        )
        self.counter = CounterMetric(
            name='mineralvision_inference_total',
            description='Total inference requests',
            labels=['model_name', 'model_version', 'status']
        )
        
    def record_inference(self, model_name: str, model_version: str,
                        latency_seconds: float, endpoint: str = 'predict',
                        success: bool = True) -> None:
        """Record an inference event."""
        labels = {
            'model_name': model_name,
            'model_version': model_version,
            'endpoint': endpoint
        }
        self.histogram.observe(latency_seconds, labels)
        
        status_labels = {
            'model_name': model_name,
            'model_version': model_version,
            'status': 'success' if success else 'error'
        }
        self.counter.inc(labels=status_labels)
        
    def get_name(self) -> str:
        return 'inference_latency'
        
    def collect(self) -> List[MetricSample]:
        """Collect latency metrics as samples."""
        samples = []
        for hist in self.histogram.collect():
            for bucket in hist.buckets:
                samples.append(MetricSample(
                    name=f'{hist.name}_bucket',
                    value=bucket.count,
                    timestamp=datetime.utcnow(),
                    labels={**hist.labels, 'le': str(bucket.le)},
                    metric_type=MetricType.HISTOGRAM
                ))
            samples.append(MetricSample(
                name=f'{hist.name}_sum',
                value=hist.sum_value,
                timestamp=datetime.utcnow(),
                labels=hist.labels,
                metric_type=MetricType.HISTOGRAM
            ))
            samples.append(MetricSample(
                name=f'{hist.name}_count',
                value=hist.count,
                timestamp=datetime.utcnow(),
                labels=hist.labels,
                metric_type=MetricType.HISTOGRAM
            ))
        samples.extend(self.counter.collect())
        return samples


class PipelineThroughputCollector(MetricCollector):
    """Collect data pipeline throughput metrics."""
    
    def __init__(self):
        self.bytes_processed = CounterMetric(
            name='mineralvision_pipeline_bytes_total',
            description='Total bytes processed by pipeline',
            labels=['pipeline_name', 'stage']
        )
        self.records_processed = CounterMetric(
            name='mineralvision_pipeline_records_total',
            description='Total records processed by pipeline',
            labels=['pipeline_name', 'stage']
        )
        self.processing_time = HistogramMetricCollector(
            name='mineralvision_pipeline_duration_seconds',
            description='Pipeline stage processing duration',
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            labels=['pipeline_name', 'stage']
        )
        self.active_pipelines = GaugeMetric(
            name='mineralvision_pipeline_active',
            description='Number of active pipeline instances',
            labels=['pipeline_name']
        )
        
    def record_stage_completion(self, pipeline_name: str, stage: str,
                               bytes_count: int, records_count: int,
                               duration_seconds: float) -> None:
        """Record pipeline stage completion."""
        labels = {'pipeline_name': pipeline_name, 'stage': stage}
        self.bytes_processed.inc(bytes_count, labels)
        self.records_processed.inc(records_count, labels)
        self.processing_time.observe(duration_seconds, labels)
        
    def set_active_pipelines(self, pipeline_name: str, count: int) -> None:
        """Set number of active pipeline instances."""
        self.active_pipelines.set(count, {'pipeline_name': pipeline_name})
        
    def get_name(self) -> str:
        return 'pipeline_throughput'
        
    def collect(self) -> List[MetricSample]:
        samples = []
        samples.extend(self.bytes_processed.collect())
        samples.extend(self.records_processed.collect())
        samples.extend(self.active_pipelines.collect())
        return samples


class StorageUtilizationCollector(MetricCollector):
    """Collect storage utilization metrics."""
    
    def __init__(self):
        self.storage_bytes = GaugeMetric(
            name='mineralvision_storage_bytes',
            description='Storage utilization in bytes',
            labels=['storage_type', 'tier', 'tenant_id']
        )
        self.storage_objects = GaugeMetric(
            name='mineralvision_storage_objects',
            description='Number of objects in storage',
            labels=['storage_type', 'tier', 'tenant_id']
        )
        self.storage_quota = GaugeMetric(
            name='mineralvision_storage_quota_bytes',
            description='Storage quota in bytes',
            labels=['storage_type', 'tenant_id']
        )
        
    def update_storage(self, storage_type: str, tier: str, tenant_id: str,
                      bytes_used: int, object_count: int) -> None:
        """Update storage metrics."""
        labels = {
            'storage_type': storage_type,
            'tier': tier,
            'tenant_id': tenant_id
        }
        self.storage_bytes.set(bytes_used, labels)
        self.storage_objects.set(object_count, labels)
        
    def set_quota(self, storage_type: str, tenant_id: str, quota_bytes: int) -> None:
        """Set storage quota."""
        labels = {'storage_type': storage_type, 'tenant_id': tenant_id}
        self.storage_quota.set(quota_bytes, labels)
        
    def get_name(self) -> str:
        return 'storage_utilization'
        
    def collect(self) -> List[MetricSample]:
        samples = []
        samples.extend(self.storage_bytes.collect())
        samples.extend(self.storage_objects.collect())
        samples.extend(self.storage_quota.collect())
        return samples


class DriftMetricsCollector(MetricCollector):
    """Collect model drift metrics."""
    
    def __init__(self):
        self.drift_score = GaugeMetric(
            name='mineralvision_drift_score',
            description='Model drift score (0-1)',
            labels=['model_name', 'drift_type', 'feature']
        )
        self.drift_alerts = CounterMetric(
            name='mineralvision_drift_alerts_total',
            description='Total drift alerts triggered',
            labels=['model_name', 'drift_type', 'severity']
        )
        self.feature_distribution = GaugeMetric(
            name='mineralvision_feature_distribution',
            description='Feature distribution statistics',
            labels=['model_name', 'feature', 'statistic']
        )
        
    def record_drift(self, model_name: str, drift_type: str,
                    feature: str, score: float) -> None:
        """Record drift score."""
        labels = {
            'model_name': model_name,
            'drift_type': drift_type,
            'feature': feature
        }
        self.drift_score.set(score, labels)
        
    def record_drift_alert(self, model_name: str, drift_type: str,
                          severity: str) -> None:
        """Record drift alert."""
        labels = {
            'model_name': model_name,
            'drift_type': drift_type,
            'severity': severity
        }
        self.drift_alerts.inc(labels=labels)
        
    def record_feature_stats(self, model_name: str, feature: str,
                            mean: float, std: float, min_val: float,
                            max_val: float) -> None:
        """Record feature distribution statistics."""
        base_labels = {'model_name': model_name, 'feature': feature}
        self.feature_distribution.set(mean, {**base_labels, 'statistic': 'mean'})
        self.feature_distribution.set(std, {**base_labels, 'statistic': 'std'})
        self.feature_distribution.set(min_val, {**base_labels, 'statistic': 'min'})
        self.feature_distribution.set(max_val, {**base_labels, 'statistic': 'max'})
        
    def get_name(self) -> str:
        return 'drift_metrics'
        
    def collect(self) -> List[MetricSample]:
        samples = []
        samples.extend(self.drift_score.collect())
        samples.extend(self.drift_alerts.collect())
        samples.extend(self.feature_distribution.collect())
        return samples


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[Alert], None]] = []
        
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        with self._lock:
            self.rules[rule.rule_id] = rule
            
    def remove_rule(self, rule_id: str) -> None:
        """Remove alert rule."""
        with self._lock:
            self.rules.pop(rule_id, None)
            
    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register alert callback."""
        self._callbacks.append(callback)
        
    def fire_alert(self, rule_id: str, value: float,
                  labels: Dict[str, str] = None) -> Alert:
        """Fire an alert."""
        with self._lock:
            rule = self.rules.get(rule_id)
            if not rule:
                raise ValueError(f"Unknown rule: {rule_id}")
                
            alert_id = hashlib.md5(
                f"{rule_id}:{json.dumps(labels or {}, sort_keys=True)}".encode()
            ).hexdigest()[:16]
            
            if alert_id in self.active_alerts:
                return self.active_alerts[alert_id]
                
            alert = Alert(
                alert_id=alert_id,
                rule=rule,
                status=AlertStatus.FIRING,
                started_at=datetime.utcnow(),
                value=value,
                labels=labels or {},
                annotations=rule.annotations
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
                
        return alert
        
    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """Resolve an alert."""
        with self._lock:
            alert = self.active_alerts.pop(alert_id, None)
            if alert:
                alert.status = AlertStatus.RESOLVED
                alert.ended_at = datetime.utcnow()
                self.alert_history.append(alert)
        return alert
        
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts."""
        with self._lock:
            alerts = list(self.active_alerts.values())
            if severity:
                alerts = [a for a in alerts if a.rule.severity == severity]
            return alerts
            
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            return list(self.alert_history)[-limit:]


class MetricsRegistry:
    """Central registry for all metrics."""
    
    def __init__(self):
        self.collectors: Dict[str, MetricCollector] = {}
        self._lock = threading.Lock()
        
    def register(self, collector: MetricCollector) -> None:
        """Register a metric collector."""
        with self._lock:
            self.collectors[collector.get_name()] = collector
            
    def unregister(self, name: str) -> None:
        """Unregister a metric collector."""
        with self._lock:
            self.collectors.pop(name, None)
            
    def collect_all(self) -> List[MetricSample]:
        """Collect all metrics."""
        samples = []
        with self._lock:
            for collector in self.collectors.values():
                try:
                    samples.extend(collector.collect())
                except Exception as e:
                    logger.error(f"Error collecting metrics from {collector.get_name()}: {e}")
        return samples
        
    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        samples = self.collect_all()
        
        for sample in samples:
            labels_str = ''
            if sample.labels:
                label_pairs = [f'{k}="{v}"' for k, v in sample.labels.items()]
                labels_str = '{' + ','.join(label_pairs) + '}'
            
            lines.append(f'{sample.name}{labels_str} {sample.value}')
            
        return '\n'.join(lines)


class DashboardBuilder:
    """Build Grafana dashboards for MineralVision."""
    
    def __init__(self):
        self._panel_id = 0
        
    def _next_panel_id(self) -> int:
        self._panel_id += 1
        return self._panel_id
        
    def build_overview_dashboard(self) -> GrafanaDashboard:
        """Build main overview dashboard."""
        panels = []
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Inference Requests/sec',
            panel_type=DashboardPanel.GRAPH,
            targets=[{
                'expr': 'rate(mineralvision_inference_total[5m])',
                'legendFormat': '{{model_name}} - {{status}}'
            }],
            grid_pos={'x': 0, 'y': 0, 'w': 12, 'h': 8}
        ))
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='P99 Inference Latency',
            panel_type=DashboardPanel.GRAPH,
            targets=[{
                'expr': 'histogram_quantile(0.99, rate(mineralvision_inference_latency_seconds_bucket[5m]))',
                'legendFormat': '{{model_name}}'
            }],
            grid_pos={'x': 12, 'y': 0, 'w': 12, 'h': 8}
        ))
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Pipeline Throughput (records/sec)',
            panel_type=DashboardPanel.GRAPH,
            targets=[{
                'expr': 'rate(mineralvision_pipeline_records_total[5m])',
                'legendFormat': '{{pipeline_name}} - {{stage}}'
            }],
            grid_pos={'x': 0, 'y': 8, 'w': 12, 'h': 8}
        ))
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Storage Utilization',
            panel_type=DashboardPanel.GAUGE,
            targets=[{
                'expr': 'mineralvision_storage_bytes / mineralvision_storage_quota_bytes * 100',
                'legendFormat': '{{storage_type}}'
            }],
            grid_pos={'x': 12, 'y': 8, 'w': 12, 'h': 8},
            options={'max': 100, 'thresholds': [70, 90]}
        ))
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Model Drift Scores',
            panel_type=DashboardPanel.HEATMAP,
            targets=[{
                'expr': 'mineralvision_drift_score',
                'legendFormat': '{{model_name}} - {{feature}}'
            }],
            grid_pos={'x': 0, 'y': 16, 'w': 24, 'h': 8}
        ))
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Active Alerts',
            panel_type=DashboardPanel.ALERT_LIST,
            targets=[],
            grid_pos={'x': 0, 'y': 24, 'w': 24, 'h': 6}
        ))
        
        return GrafanaDashboard(
            uid='mineralvision-overview',
            title='MineralVision Overview',
            panels=panels,
            tags=['mineralvision', 'overview'],
            refresh='10s'
        )
        
    def build_model_dashboard(self) -> GrafanaDashboard:
        """Build model performance dashboard."""
        panels = []
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Model Inference Rate',
            panel_type=DashboardPanel.GRAPH,
            targets=[{
                'expr': 'sum(rate(mineralvision_inference_total{status="success"}[5m])) by (model_name)',
                'legendFormat': '{{model_name}}'
            }],
            grid_pos={'x': 0, 'y': 0, 'w': 12, 'h': 8}
        ))
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Error Rate',
            panel_type=DashboardPanel.GRAPH,
            targets=[{
                'expr': 'sum(rate(mineralvision_inference_total{status="error"}[5m])) by (model_name) / sum(rate(mineralvision_inference_total[5m])) by (model_name) * 100',
                'legendFormat': '{{model_name}}'
            }],
            grid_pos={'x': 12, 'y': 0, 'w': 12, 'h': 8}
        ))
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Latency Distribution',
            panel_type=DashboardPanel.HEATMAP,
            targets=[{
                'expr': 'rate(mineralvision_inference_latency_seconds_bucket[5m])',
                'legendFormat': '{{le}}'
            }],
            grid_pos={'x': 0, 'y': 8, 'w': 24, 'h': 8}
        ))
        
        return GrafanaDashboard(
            uid='mineralvision-models',
            title='MineralVision Model Performance',
            panels=panels,
            tags=['mineralvision', 'models'],
            refresh='10s'
        )
        
    def build_drift_dashboard(self) -> GrafanaDashboard:
        """Build drift monitoring dashboard."""
        panels = []
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Data Drift by Feature',
            panel_type=DashboardPanel.HEATMAP,
            targets=[{
                'expr': 'mineralvision_drift_score{drift_type="data_drift"}',
                'legendFormat': '{{model_name}} - {{feature}}'
            }],
            grid_pos={'x': 0, 'y': 0, 'w': 24, 'h': 10}
        ))
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Concept Drift',
            panel_type=DashboardPanel.GRAPH,
            targets=[{
                'expr': 'mineralvision_drift_score{drift_type="concept_drift"}',
                'legendFormat': '{{model_name}}'
            }],
            grid_pos={'x': 0, 'y': 10, 'w': 12, 'h': 8}
        ))
        
        panels.append(GrafanaPanel(
            panel_id=self._next_panel_id(),
            title='Drift Alerts',
            panel_type=DashboardPanel.STAT,
            targets=[{
                'expr': 'sum(increase(mineralvision_drift_alerts_total[24h])) by (severity)',
                'legendFormat': '{{severity}}'
            }],
            grid_pos={'x': 12, 'y': 10, 'w': 12, 'h': 8}
        ))
        
        return GrafanaDashboard(
            uid='mineralvision-drift',
            title='MineralVision Drift Monitoring',
            panels=panels,
            tags=['mineralvision', 'drift', 'ml'],
            refresh='30s'
        )


class MonitoringService:
    """Main monitoring service."""
    
    def __init__(self):
        self.registry = MetricsRegistry()
        self.alert_manager = AlertManager()
        self.dashboard_builder = DashboardBuilder()
        
        self.inference_collector = InferenceLatencyCollector()
        self.pipeline_collector = PipelineThroughputCollector()
        self.storage_collector = StorageUtilizationCollector()
        self.drift_collector = DriftMetricsCollector()
        
        self.registry.register(self.inference_collector)
        self.registry.register(self.pipeline_collector)
        self.registry.register(self.storage_collector)
        self.registry.register(self.drift_collector)
        
        self._setup_default_alerts()
        
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules."""
        self.alert_manager.add_rule(AlertRule(
            rule_id='high_latency',
            name='HighInferenceLatency',
            expression='histogram_quantile(0.99, rate(mineralvision_inference_latency_seconds_bucket[5m])) > 5',
            duration=timedelta(minutes=5),
            severity=AlertSeverity.WARNING,
            annotations={
                'summary': 'High inference latency detected',
                'description': 'P99 latency is above 5 seconds'
            }
        ))
        
        self.alert_manager.add_rule(AlertRule(
            rule_id='high_error_rate',
            name='HighErrorRate',
            expression='sum(rate(mineralvision_inference_total{status="error"}[5m])) / sum(rate(mineralvision_inference_total[5m])) > 0.05',
            duration=timedelta(minutes=5),
            severity=AlertSeverity.CRITICAL,
            annotations={
                'summary': 'High error rate detected',
                'description': 'Error rate is above 5%'
            }
        ))
        
        self.alert_manager.add_rule(AlertRule(
            rule_id='storage_full',
            name='StorageNearlyFull',
            expression='mineralvision_storage_bytes / mineralvision_storage_quota_bytes > 0.9',
            duration=timedelta(minutes=10),
            severity=AlertSeverity.WARNING,
            annotations={
                'summary': 'Storage nearly full',
                'description': 'Storage utilization is above 90%'
            }
        ))
        
        self.alert_manager.add_rule(AlertRule(
            rule_id='drift_detected',
            name='ModelDriftDetected',
            expression='mineralvision_drift_score > 0.7',
            duration=timedelta(minutes=30),
            severity=AlertSeverity.WARNING,
            annotations={
                'summary': 'Model drift detected',
                'description': 'Drift score is above 0.7'
            }
        ))
        
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return self.registry.to_prometheus_format()
        
    def get_dashboards(self) -> List[GrafanaDashboard]:
        """Get all Grafana dashboards."""
        return [
            self.dashboard_builder.build_overview_dashboard(),
            self.dashboard_builder.build_model_dashboard(),
            self.dashboard_builder.build_drift_dashboard()
        ]
        
    def export_dashboards_json(self) -> Dict[str, Any]:
        """Export all dashboards as JSON."""
        return {
            'dashboards': [d.to_grafana_json() for d in self.get_dashboards()]
        }


def create_monitoring_service() -> MonitoringService:
    """Factory function to create monitoring service."""
    return MonitoringService()


def create_alert_manager() -> AlertManager:
    """Factory function to create alert manager."""
    return AlertManager()


def create_metrics_registry() -> MetricsRegistry:
    """Factory function to create metrics registry."""
    return MetricsRegistry()
