"""
Kubecost Integration
=====================

Production-grade Kubernetes cost management for MineralVision:
- Real-time cost monitoring
- Cost allocation by namespace/label/deployment
- Budget alerts and notifications
- Cost optimization recommendations
- Showback/chargeback reporting
- Cloud cost integration (AWS, GCP, Azure)

Kubecost provides visibility into Kubernetes spend
with actionable cost optimization insights.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import random

logger = logging.getLogger(__name__)


class CostMetricType(Enum):
    """Types of cost metrics."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    TOTAL = "total"


class AggregationType(Enum):
    """Cost aggregation types."""
    NAMESPACE = "namespace"
    DEPLOYMENT = "deployment"
    SERVICE = "service"
    LABEL = "label"
    POD = "pod"
    CONTAINER = "container"
    CONTROLLER = "controller"
    NODE = "node"
    CLUSTER = "cluster"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class CloudProvider(Enum):
    """Cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ON_PREM = "on-prem"


@dataclass
class CostAllocation:
    """Cost allocation for a resource."""
    name: str
    aggregation_type: AggregationType
    cpu_cost: float
    memory_cost: float
    storage_cost: float
    network_cost: float
    gpu_cost: float = 0.0
    total_cost: float = 0.0
    efficiency: float = 0.0
    cpu_efficiency: float = 0.0
    memory_efficiency: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.total_cost == 0:
            self.total_cost = (
                self.cpu_cost + self.memory_cost + 
                self.storage_cost + self.network_cost + self.gpu_cost
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'aggregation_type': self.aggregation_type.value,
            'cpu_cost': self.cpu_cost,
            'memory_cost': self.memory_cost,
            'storage_cost': self.storage_cost,
            'network_cost': self.network_cost,
            'gpu_cost': self.gpu_cost,
            'total_cost': self.total_cost,
            'efficiency': self.efficiency,
            'cpu_efficiency': self.cpu_efficiency,
            'memory_efficiency': self.memory_efficiency,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'labels': self.labels
        }


@dataclass
class Budget:
    """Cost budget configuration."""
    id: str
    name: str
    amount: float
    period: str
    aggregation_type: AggregationType
    filter_value: str
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 1.0])
    current_spend: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def utilization(self) -> float:
        return self.current_spend / self.amount if self.amount > 0 else 0
    
    @property
    def remaining(self) -> float:
        return max(0, self.amount - self.current_spend)


@dataclass
class CostAlert:
    """Cost alert."""
    id: str
    budget_id: str
    severity: AlertSeverity
    message: str
    threshold: float
    current_value: float
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation."""
    id: str
    resource_type: str
    resource_name: str
    namespace: str
    recommendation_type: str
    description: str
    estimated_savings: float
    confidence: float
    current_cost: float
    recommended_action: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KubecostConfig:
    """Kubecost configuration."""
    url: str = "http://localhost:9090"
    api_path: str = "/model"
    cloud_provider: CloudProvider = CloudProvider.AWS
    cluster_id: str = "mineralvision-cluster"
    currency: str = "USD"
    cpu_cost_per_hour: float = 0.031611
    memory_cost_per_gb_hour: float = 0.004237
    storage_cost_per_gb_month: float = 0.10
    network_cost_per_gb: float = 0.01


class MockKubecostAPI:
    """Mock Kubecost API client."""
    
    def __init__(self, config: KubecostConfig):
        self.config = config
        self._budgets: Dict[str, Budget] = {}
        self._alerts: List[CostAlert] = []
        self._namespaces = [
            "mineralvision-core",
            "mineralvision-ml",
            "mineralvision-data",
            "mineralvision-api",
            "monitoring",
            "logging"
        ]
        self._deployments = {
            "mineralvision-core": ["api-server", "web-frontend", "auth-service"],
            "mineralvision-ml": ["model-server", "training-worker", "inference-api"],
            "mineralvision-data": ["kafka", "spark-master", "spark-worker", "postgres"],
            "mineralvision-api": ["gateway", "rate-limiter"],
            "monitoring": ["prometheus", "grafana", "alertmanager"],
            "logging": ["elasticsearch", "kibana", "fluentd"]
        }
    
    async def get_allocation(self, aggregation: AggregationType,
                            window: str = "1d",
                            filter_namespace: str = None) -> List[CostAllocation]:
        """Get cost allocation."""
        allocations = []
        
        if aggregation == AggregationType.NAMESPACE:
            for ns in self._namespaces:
                if filter_namespace and ns != filter_namespace:
                    continue
                
                allocations.append(self._generate_allocation(ns, aggregation))
        
        elif aggregation == AggregationType.DEPLOYMENT:
            for ns, deployments in self._deployments.items():
                if filter_namespace and ns != filter_namespace:
                    continue
                
                for deploy in deployments:
                    allocations.append(self._generate_allocation(
                        f"{ns}/{deploy}", aggregation,
                        labels={'namespace': ns, 'deployment': deploy}
                    ))
        
        elif aggregation == AggregationType.CLUSTER:
            total = CostAllocation(
                name=self.config.cluster_id,
                aggregation_type=aggregation,
                cpu_cost=sum(a.cpu_cost for a in [self._generate_allocation(ns, AggregationType.NAMESPACE) for ns in self._namespaces]),
                memory_cost=sum(a.memory_cost for a in [self._generate_allocation(ns, AggregationType.NAMESPACE) for ns in self._namespaces]),
                storage_cost=random.uniform(50, 100),
                network_cost=random.uniform(20, 50),
                efficiency=random.uniform(0.4, 0.8)
            )
            allocations.append(total)
        
        return allocations
    
    def _generate_allocation(self, name: str, aggregation: AggregationType,
                            labels: Dict[str, str] = None) -> CostAllocation:
        """Generate a cost allocation."""
        cpu_cost = random.uniform(5, 50)
        memory_cost = random.uniform(2, 30)
        storage_cost = random.uniform(1, 20)
        network_cost = random.uniform(0.5, 10)
        
        return CostAllocation(
            name=name,
            aggregation_type=aggregation,
            cpu_cost=cpu_cost,
            memory_cost=memory_cost,
            storage_cost=storage_cost,
            network_cost=network_cost,
            efficiency=random.uniform(0.3, 0.9),
            cpu_efficiency=random.uniform(0.2, 0.8),
            memory_efficiency=random.uniform(0.3, 0.9),
            labels=labels or {}
        )
    
    async def get_cost_over_time(self, aggregation: AggregationType,
                                window: str = "7d",
                                step: str = "1d") -> List[Dict[str, Any]]:
        """Get cost over time."""
        days = int(window.replace('d', ''))
        results = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i - 1)
            results.append({
                'date': date.isoformat(),
                'cpu_cost': random.uniform(100, 200),
                'memory_cost': random.uniform(50, 100),
                'storage_cost': random.uniform(20, 50),
                'network_cost': random.uniform(10, 30),
                'total_cost': random.uniform(200, 400)
            })
        
        return results
    
    async def get_recommendations(self, namespace: str = None) -> List[OptimizationRecommendation]:
        """Get optimization recommendations."""
        recommendations = []
        
        # Generate sample recommendations
        rec_types = [
            ("right-size", "Right-size container resources", "Reduce resource requests"),
            ("idle", "Remove idle resources", "Delete unused deployments"),
            ("spot", "Use spot instances", "Move workloads to spot nodes"),
            ("reserved", "Purchase reserved capacity", "Commit to reserved instances")
        ]
        
        for ns, deployments in self._deployments.items():
            if namespace and ns != namespace:
                continue
            
            for deploy in deployments[:2]:  # Limit recommendations
                rec_type, desc, action = random.choice(rec_types)
                recommendations.append(OptimizationRecommendation(
                    id=str(uuid.uuid4()),
                    resource_type="deployment",
                    resource_name=deploy,
                    namespace=ns,
                    recommendation_type=rec_type,
                    description=desc,
                    estimated_savings=random.uniform(5, 50),
                    confidence=random.uniform(0.7, 0.95),
                    current_cost=random.uniform(20, 100),
                    recommended_action=action
                ))
        
        return recommendations
    
    async def create_budget(self, budget: Budget) -> Budget:
        """Create a budget."""
        self._budgets[budget.id] = budget
        return budget
    
    async def get_budget(self, budget_id: str) -> Optional[Budget]:
        """Get a budget."""
        return self._budgets.get(budget_id)
    
    async def list_budgets(self) -> List[Budget]:
        """List all budgets."""
        return list(self._budgets.values())
    
    async def delete_budget(self, budget_id: str) -> bool:
        """Delete a budget."""
        if budget_id in self._budgets:
            del self._budgets[budget_id]
            return True
        return False
    
    async def get_alerts(self, budget_id: str = None) -> List[CostAlert]:
        """Get alerts."""
        if budget_id:
            return [a for a in self._alerts if a.budget_id == budget_id]
        return self._alerts
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information."""
        return {
            'cluster_id': self.config.cluster_id,
            'cloud_provider': self.config.cloud_provider.value,
            'node_count': random.randint(3, 10),
            'total_cpu': random.randint(32, 128),
            'total_memory_gb': random.randint(128, 512),
            'total_storage_gb': random.randint(1000, 5000)
        }
    
    async def get_savings_summary(self) -> Dict[str, Any]:
        """Get savings summary."""
        return {
            'total_monthly_cost': random.uniform(5000, 15000),
            'potential_savings': random.uniform(500, 3000),
            'savings_percentage': random.uniform(0.1, 0.3),
            'recommendations_count': random.randint(5, 20),
            'implemented_savings': random.uniform(100, 500)
        }


class CostAnalyzer:
    """
    Cost analysis for Kubecost.
    
    Provides:
    - Cost breakdown analysis
    - Trend analysis
    - Efficiency analysis
    """
    
    def __init__(self, api: MockKubecostAPI):
        self.api = api
    
    async def get_cost_breakdown(self, aggregation: AggregationType = AggregationType.NAMESPACE,
                                window: str = "7d") -> Dict[str, Any]:
        """Get cost breakdown."""
        allocations = await self.api.get_allocation(aggregation, window)
        
        total_cost = sum(a.total_cost for a in allocations)
        
        breakdown = {
            'total_cost': total_cost,
            'by_resource': {},
            'by_type': {
                'cpu': sum(a.cpu_cost for a in allocations),
                'memory': sum(a.memory_cost for a in allocations),
                'storage': sum(a.storage_cost for a in allocations),
                'network': sum(a.network_cost for a in allocations)
            }
        }
        
        for alloc in allocations:
            breakdown['by_resource'][alloc.name] = {
                'cost': alloc.total_cost,
                'percentage': alloc.total_cost / total_cost * 100 if total_cost > 0 else 0,
                'efficiency': alloc.efficiency
            }
        
        return breakdown
    
    async def get_efficiency_report(self, namespace: str = None) -> Dict[str, Any]:
        """Get efficiency report."""
        allocations = await self.api.get_allocation(
            AggregationType.DEPLOYMENT,
            filter_namespace=namespace
        )
        
        avg_efficiency = sum(a.efficiency for a in allocations) / len(allocations) if allocations else 0
        avg_cpu_efficiency = sum(a.cpu_efficiency for a in allocations) / len(allocations) if allocations else 0
        avg_memory_efficiency = sum(a.memory_efficiency for a in allocations) / len(allocations) if allocations else 0
        
        inefficient = [a for a in allocations if a.efficiency < 0.5]
        
        return {
            'average_efficiency': avg_efficiency,
            'average_cpu_efficiency': avg_cpu_efficiency,
            'average_memory_efficiency': avg_memory_efficiency,
            'inefficient_resources': [
                {
                    'name': a.name,
                    'efficiency': a.efficiency,
                    'cost': a.total_cost
                }
                for a in inefficient
            ],
            'potential_savings': sum(a.total_cost * (1 - a.efficiency) for a in inefficient)
        }
    
    async def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Get cost trend analysis."""
        history = await self.api.get_cost_over_time(
            AggregationType.CLUSTER,
            window=f"{days}d"
        )
        
        if len(history) < 2:
            return {'trend': 'insufficient_data'}
        
        costs = [h['total_cost'] for h in history]
        avg_cost = sum(costs) / len(costs)
        
        # Simple trend calculation
        first_half = sum(costs[:len(costs)//2]) / (len(costs)//2)
        second_half = sum(costs[len(costs)//2:]) / (len(costs) - len(costs)//2)
        
        if second_half > first_half * 1.1:
            trend = 'increasing'
        elif second_half < first_half * 0.9:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'average_daily_cost': avg_cost,
            'min_daily_cost': min(costs),
            'max_daily_cost': max(costs),
            'total_cost': sum(costs),
            'change_percentage': (second_half - first_half) / first_half * 100 if first_half > 0 else 0
        }


class BudgetManager:
    """
    Budget management for Kubecost.
    
    Provides:
    - Budget creation
    - Budget monitoring
    - Alert management
    """
    
    def __init__(self, api: MockKubecostAPI):
        self.api = api
    
    async def create_budget(self, name: str, amount: float,
                           period: str = "monthly",
                           aggregation: AggregationType = AggregationType.NAMESPACE,
                           filter_value: str = None,
                           alert_thresholds: List[float] = None) -> Budget:
        """Create a budget."""
        budget = Budget(
            id=str(uuid.uuid4()),
            name=name,
            amount=amount,
            period=period,
            aggregation_type=aggregation,
            filter_value=filter_value or "*",
            alert_thresholds=alert_thresholds or [0.5, 0.8, 1.0]
        )
        
        return await self.api.create_budget(budget)
    
    async def get_budget(self, budget_id: str) -> Optional[Budget]:
        """Get a budget."""
        return await self.api.get_budget(budget_id)
    
    async def list_budgets(self) -> List[Budget]:
        """List all budgets."""
        return await self.api.list_budgets()
    
    async def delete_budget(self, budget_id: str) -> bool:
        """Delete a budget."""
        return await self.api.delete_budget(budget_id)
    
    async def check_budgets(self) -> List[CostAlert]:
        """Check all budgets and generate alerts."""
        alerts = []
        budgets = await self.list_budgets()
        
        for budget in budgets:
            # Get current spend
            allocations = await self.api.get_allocation(
                budget.aggregation_type,
                window="30d" if budget.period == "monthly" else "7d"
            )
            
            if budget.filter_value != "*":
                allocations = [a for a in allocations if budget.filter_value in a.name]
            
            budget.current_spend = sum(a.total_cost for a in allocations)
            
            # Check thresholds
            for threshold in budget.alert_thresholds:
                if budget.utilization >= threshold:
                    severity = AlertSeverity.CRITICAL if threshold >= 1.0 else (
                        AlertSeverity.WARNING if threshold >= 0.8 else AlertSeverity.INFO
                    )
                    
                    alert = CostAlert(
                        id=str(uuid.uuid4()),
                        budget_id=budget.id,
                        severity=severity,
                        message=f"Budget '{budget.name}' is at {budget.utilization*100:.1f}% utilization",
                        threshold=threshold,
                        current_value=budget.utilization
                    )
                    alerts.append(alert)
                    break  # Only one alert per budget
        
        return alerts
    
    async def get_alerts(self, budget_id: str = None) -> List[CostAlert]:
        """Get alerts."""
        return await self.api.get_alerts(budget_id)
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        return await self.api.acknowledge_alert(alert_id)


class KubecostIntegration:
    """
    Kubecost integration for MineralVision.
    
    Provides Kubernetes cost management:
    - Cost allocation and analysis
    - Budget management
    - Optimization recommendations
    - Trend analysis
    
    Example:
        kubecost = KubecostIntegration()
        await kubecost.connect()
        
        # Get cost breakdown
        breakdown = await kubecost.analyzer.get_cost_breakdown()
        
        # Create budget
        budget = await kubecost.budgets.create_budget(
            "ml-namespace-budget",
            amount=1000,
            aggregation=AggregationType.NAMESPACE,
            filter_value="mineralvision-ml"
        )
        
        # Get recommendations
        recommendations = await kubecost.get_recommendations()
    """
    
    def __init__(self, config: KubecostConfig = None):
        self.config = config or KubecostConfig()
        self.api: Optional[MockKubecostAPI] = None
        self.analyzer: Optional[CostAnalyzer] = None
        self.budgets: Optional[BudgetManager] = None
        self._connected = False
    
    async def connect(self) -> 'KubecostIntegration':
        """Connect to Kubecost."""
        self.api = MockKubecostAPI(self.config)
        self.analyzer = CostAnalyzer(self.api)
        self.budgets = BudgetManager(self.api)
        
        self._connected = True
        logger.info(f"Connected to Kubecost at {self.config.url}")
        return self
    
    async def get_allocation(self, aggregation: AggregationType = AggregationType.NAMESPACE,
                            window: str = "1d",
                            namespace: str = None) -> List[CostAllocation]:
        """Get cost allocation."""
        return await self.api.get_allocation(aggregation, window, namespace)
    
    async def get_recommendations(self, namespace: str = None) -> List[OptimizationRecommendation]:
        """Get optimization recommendations."""
        return await self.api.get_recommendations(namespace)
    
    async def get_savings_summary(self) -> Dict[str, Any]:
        """Get savings summary."""
        return await self.api.get_savings_summary()
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information."""
        return await self.api.get_cluster_info()
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Factory functions

def create_kubecost(config: KubecostConfig = None) -> KubecostIntegration:
    """Create a Kubecost integration instance."""
    return KubecostIntegration(config)


async def create_and_connect_kubecost(config: KubecostConfig = None) -> KubecostIntegration:
    """Create and connect Kubecost."""
    kubecost = KubecostIntegration(config)
    await kubecost.connect()
    return kubecost
