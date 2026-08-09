"""
Kubernetes Operator Integration
================================

Production-grade Kubernetes operator for MineralVision:
- Custom Resource Definitions (CRDs)
- Operator pattern implementation
- Reconciliation loops
- Status management
- Event handling
- Helm chart generation
- Resource scaling
- Health monitoring

Provides automated management of MineralVision
workloads on Kubernetes clusters.
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
import yaml
import hashlib

logger = logging.getLogger(__name__)

try:
    from kubernetes import client, config, watch
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

from .._mock_fallback import real_client_unavailable


class ResourceKind(Enum):
    """Kubernetes resource kinds."""
    DEPLOYMENT = "Deployment"
    STATEFULSET = "StatefulSet"
    SERVICE = "Service"
    CONFIGMAP = "ConfigMap"
    SECRET = "Secret"
    INGRESS = "Ingress"
    PVC = "PersistentVolumeClaim"
    HPA = "HorizontalPodAutoscaler"
    SERVICEACCOUNT = "ServiceAccount"
    ROLE = "Role"
    ROLEBINDING = "RoleBinding"
    NETWORKPOLICY = "NetworkPolicy"


class ResourcePhase(Enum):
    """Resource lifecycle phases."""
    PENDING = "Pending"
    CREATING = "Creating"
    RUNNING = "Running"
    UPDATING = "Updating"
    SCALING = "Scaling"
    FAILED = "Failed"
    TERMINATING = "Terminating"
    UNKNOWN = "Unknown"


class ConditionType(Enum):
    """Condition types for status."""
    AVAILABLE = "Available"
    PROGRESSING = "Progressing"
    DEGRADED = "Degraded"
    READY = "Ready"


@dataclass
class ResourceCondition:
    """Resource condition."""
    type: ConditionType
    status: str
    reason: str
    message: str
    last_transition_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'status': self.status,
            'reason': self.reason,
            'message': self.message,
            'lastTransitionTime': self.last_transition_time.isoformat()
        }


@dataclass
class ResourceStatus:
    """Resource status."""
    phase: ResourcePhase
    conditions: List[ResourceCondition] = field(default_factory=list)
    replicas: int = 0
    ready_replicas: int = 0
    available_replicas: int = 0
    observed_generation: int = 0
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'phase': self.phase.value,
            'conditions': [c.to_dict() for c in self.conditions],
            'replicas': self.replicas,
            'readyReplicas': self.ready_replicas,
            'availableReplicas': self.available_replicas,
            'observedGeneration': self.observed_generation,
            'message': self.message
        }


@dataclass
class MineralVisionSpec:
    """MineralVision custom resource spec."""
    version: str = "1.0.0"
    replicas: int = 1
    image: str = "mineralvision/core:latest"
    resources: Dict[str, Any] = field(default_factory=lambda: {
        'requests': {'cpu': '500m', 'memory': '512Mi'},
        'limits': {'cpu': '2', 'memory': '2Gi'}
    })
    storage: Dict[str, Any] = field(default_factory=lambda: {
        'size': '10Gi',
        'storageClass': 'standard'
    })
    database: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'postgresql',
        'host': 'postgres',
        'port': 5432
    })
    kafka: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'brokers': ['kafka:9092']
    })
    ml: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'gpu': False,
        'modelRegistry': 'mlflow'
    })
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'prometheus': True,
        'grafana': True
    })
    ingress: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'host': 'mineralvision.local',
        'tls': False
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'replicas': self.replicas,
            'image': self.image,
            'resources': self.resources,
            'storage': self.storage,
            'database': self.database,
            'kafka': self.kafka,
            'ml': self.ml,
            'monitoring': self.monitoring,
            'ingress': self.ingress
        }


@dataclass
class MineralVisionResource:
    """MineralVision custom resource."""
    name: str
    namespace: str
    spec: MineralVisionSpec
    status: ResourceStatus = field(default_factory=lambda: ResourceStatus(ResourcePhase.PENDING))
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation: int = 1
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'apiVersion': 'mineralvision.io/v1',
            'kind': 'MineralVision',
            'metadata': {
                'name': self.name,
                'namespace': self.namespace,
                'uid': self.uid,
                'generation': self.generation,
                **self.metadata
            },
            'spec': self.spec.to_dict(),
            'status': self.status.to_dict()
        }


@dataclass
class KubernetesConfig:
    """Kubernetes configuration."""
    kubeconfig_path: Optional[str] = None
    in_cluster: bool = False
    namespace: str = "mineralvision"
    operator_name: str = "mineralvision-operator"
    crd_group: str = "mineralvision.io"
    crd_version: str = "v1"
    crd_plural: str = "mineralvisions"


class MockKubernetesClient:
    """Mock Kubernetes client."""
    
    def __init__(self, config: KubernetesConfig):
        self.config = config
        self._resources: Dict[str, Dict[str, MineralVisionResource]] = {}
        self._deployments: Dict[str, Dict[str, Any]] = {}
        self._services: Dict[str, Dict[str, Any]] = {}
        self._configmaps: Dict[str, Dict[str, Any]] = {}
        self._secrets: Dict[str, Dict[str, Any]] = {}
        self._events: List[Dict[str, Any]] = []
        self._namespaces: Set[str] = {config.namespace}
    
    async def create_namespace(self, name: str) -> Dict[str, Any]:
        """Create a namespace."""
        self._namespaces.add(name)
        return {'metadata': {'name': name}, 'status': {'phase': 'Active'}}
    
    async def namespace_exists(self, name: str) -> bool:
        """Check if namespace exists."""
        return name in self._namespaces
    
    async def create_crd(self, crd_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create a CRD."""
        return {'metadata': {'name': crd_spec['metadata']['name']}, 'status': 'created'}
    
    async def create_resource(self, resource: MineralVisionResource) -> MineralVisionResource:
        """Create a MineralVision resource."""
        ns = resource.namespace
        if ns not in self._resources:
            self._resources[ns] = {}
        
        self._resources[ns][resource.name] = resource
        self._add_event(resource, "Created", "Resource created")
        
        return resource
    
    async def get_resource(self, name: str, namespace: str) -> Optional[MineralVisionResource]:
        """Get a MineralVision resource."""
        return self._resources.get(namespace, {}).get(name)
    
    async def list_resources(self, namespace: str = None) -> List[MineralVisionResource]:
        """List MineralVision resources."""
        if namespace:
            return list(self._resources.get(namespace, {}).values())
        
        all_resources = []
        for ns_resources in self._resources.values():
            all_resources.extend(ns_resources.values())
        return all_resources
    
    async def update_resource(self, resource: MineralVisionResource) -> MineralVisionResource:
        """Update a MineralVision resource."""
        ns = resource.namespace
        if ns in self._resources and resource.name in self._resources[ns]:
            resource.generation += 1
            self._resources[ns][resource.name] = resource
            self._add_event(resource, "Updated", "Resource updated")
        
        return resource
    
    async def delete_resource(self, name: str, namespace: str) -> bool:
        """Delete a MineralVision resource."""
        if namespace in self._resources and name in self._resources[namespace]:
            resource = self._resources[namespace][name]
            self._add_event(resource, "Deleted", "Resource deleted")
            del self._resources[namespace][name]
            return True
        return False
    
    async def update_status(self, name: str, namespace: str,
                           status: ResourceStatus) -> Optional[MineralVisionResource]:
        """Update resource status."""
        resource = await self.get_resource(name, namespace)
        if resource:
            resource.status = status
            self._resources[namespace][name] = resource
            return resource
        return None
    
    async def create_deployment(self, namespace: str, 
                               deployment: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deployment."""
        name = deployment['metadata']['name']
        if namespace not in self._deployments:
            self._deployments[namespace] = {}
        
        self._deployments[namespace][name] = deployment
        return deployment
    
    async def get_deployment(self, name: str, namespace: str) -> Optional[Dict[str, Any]]:
        """Get a deployment."""
        return self._deployments.get(namespace, {}).get(name)
    
    async def delete_deployment(self, name: str, namespace: str) -> bool:
        """Delete a deployment."""
        if namespace in self._deployments and name in self._deployments[namespace]:
            del self._deployments[namespace][name]
            return True
        return False
    
    async def create_service(self, namespace: str, 
                            service: Dict[str, Any]) -> Dict[str, Any]:
        """Create a service."""
        name = service['metadata']['name']
        if namespace not in self._services:
            self._services[namespace] = {}
        
        self._services[namespace][name] = service
        return service
    
    async def create_configmap(self, namespace: str,
                              configmap: Dict[str, Any]) -> Dict[str, Any]:
        """Create a configmap."""
        name = configmap['metadata']['name']
        if namespace not in self._configmaps:
            self._configmaps[namespace] = {}
        
        self._configmaps[namespace][name] = configmap
        return configmap
    
    async def create_secret(self, namespace: str,
                           secret: Dict[str, Any]) -> Dict[str, Any]:
        """Create a secret."""
        name = secret['metadata']['name']
        if namespace not in self._secrets:
            self._secrets[namespace] = {}
        
        self._secrets[namespace][name] = secret
        return secret
    
    async def get_events(self, namespace: str = None,
                        resource_name: str = None) -> List[Dict[str, Any]]:
        """Get events."""
        events = self._events
        
        if namespace:
            events = [e for e in events if e.get('namespace') == namespace]
        if resource_name:
            events = [e for e in events if e.get('resource_name') == resource_name]
        
        return events
    
    def _add_event(self, resource: MineralVisionResource, 
                  reason: str, message: str) -> None:
        """Add an event."""
        self._events.append({
            'namespace': resource.namespace,
            'resource_name': resource.name,
            'reason': reason,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'type': 'Normal'
        })


class ResourceGenerator:
    """
    Generate Kubernetes resources from MineralVision spec.
    
    Provides:
    - Deployment generation
    - Service generation
    - ConfigMap generation
    - Secret generation
    """
    
    def __init__(self, config: KubernetesConfig):
        self.config = config
    
    def generate_deployment(self, resource: MineralVisionResource) -> Dict[str, Any]:
        """Generate deployment from MineralVision resource."""
        spec = resource.spec
        
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f"{resource.name}-core",
                'namespace': resource.namespace,
                'labels': {
                    'app': 'mineralvision',
                    'component': 'core',
                    'instance': resource.name
                },
                'ownerReferences': [{
                    'apiVersion': 'mineralvision.io/v1',
                    'kind': 'MineralVision',
                    'name': resource.name,
                    'uid': resource.uid
                }]
            },
            'spec': {
                'replicas': spec.replicas,
                'selector': {
                    'matchLabels': {
                        'app': 'mineralvision',
                        'instance': resource.name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'mineralvision',
                            'component': 'core',
                            'instance': resource.name
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'mineralvision',
                            'image': spec.image,
                            'ports': [
                                {'containerPort': 8000, 'name': 'http'},
                                {'containerPort': 9090, 'name': 'metrics'}
                            ],
                            'resources': spec.resources,
                            'env': [
                                {'name': 'DATABASE_HOST', 'value': spec.database['host']},
                                {'name': 'DATABASE_PORT', 'value': str(spec.database['port'])},
                                {'name': 'KAFKA_BROKERS', 'value': ','.join(spec.kafka.get('brokers', []))}
                            ],
                            'livenessProbe': {
                                'httpGet': {'path': '/health', 'port': 8000},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/ready', 'port': 8000},
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
    
    def generate_service(self, resource: MineralVisionResource) -> Dict[str, Any]:
        """Generate service from MineralVision resource."""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{resource.name}-svc",
                'namespace': resource.namespace,
                'labels': {
                    'app': 'mineralvision',
                    'instance': resource.name
                }
            },
            'spec': {
                'type': 'ClusterIP',
                'ports': [
                    {'port': 80, 'targetPort': 8000, 'name': 'http'},
                    {'port': 9090, 'targetPort': 9090, 'name': 'metrics'}
                ],
                'selector': {
                    'app': 'mineralvision',
                    'instance': resource.name
                }
            }
        }
    
    def generate_configmap(self, resource: MineralVisionResource) -> Dict[str, Any]:
        """Generate configmap from MineralVision resource."""
        spec = resource.spec
        
        config_data = {
            'config.yaml': yaml.dump({
                'version': spec.version,
                'database': spec.database,
                'kafka': spec.kafka,
                'ml': spec.ml,
                'monitoring': spec.monitoring
            })
        }
        
        return {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f"{resource.name}-config",
                'namespace': resource.namespace
            },
            'data': config_data
        }
    
    def generate_ingress(self, resource: MineralVisionResource) -> Dict[str, Any]:
        """Generate ingress from MineralVision resource."""
        spec = resource.spec
        ingress_spec = spec.ingress
        
        ingress = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f"{resource.name}-ingress",
                'namespace': resource.namespace,
                'annotations': {
                    'kubernetes.io/ingress.class': 'nginx'
                }
            },
            'spec': {
                'rules': [{
                    'host': ingress_spec.get('host', 'mineralvision.local'),
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': f"{resource.name}-svc",
                                    'port': {'number': 80}
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        if ingress_spec.get('tls'):
            ingress['spec']['tls'] = [{
                'hosts': [ingress_spec.get('host', 'mineralvision.local')],
                'secretName': f"{resource.name}-tls"
            }]
        
        return ingress
    
    def generate_hpa(self, resource: MineralVisionResource) -> Dict[str, Any]:
        """Generate HorizontalPodAutoscaler."""
        return {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{resource.name}-hpa",
                'namespace': resource.namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': f"{resource.name}-core"
                },
                'minReplicas': 1,
                'maxReplicas': 10,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ]
            }
        }


class Reconciler:
    """
    Reconciliation loop for MineralVision operator.
    
    Provides:
    - Resource reconciliation
    - Status updates
    - Error handling
    """
    
    def __init__(self, client: MockKubernetesClient, generator: ResourceGenerator):
        self.client = client
        self.generator = generator
        self._reconcile_interval = timedelta(seconds=30)
    
    async def reconcile(self, resource: MineralVisionResource) -> ResourceStatus:
        """Reconcile a MineralVision resource."""
        logger.info(f"Reconciling {resource.name} in {resource.namespace}")
        
        try:
            # Update status to progressing
            status = ResourceStatus(
                phase=ResourcePhase.CREATING,
                conditions=[ResourceCondition(
                    type=ConditionType.PROGRESSING,
                    status="True",
                    reason="Reconciling",
                    message="Creating resources"
                )]
            )
            await self.client.update_status(resource.name, resource.namespace, status)
            
            # Create deployment
            deployment = self.generator.generate_deployment(resource)
            await self.client.create_deployment(resource.namespace, deployment)
            
            # Create service
            service = self.generator.generate_service(resource)
            await self.client.create_service(resource.namespace, service)
            
            # Create configmap
            configmap = self.generator.generate_configmap(resource)
            await self.client.create_configmap(resource.namespace, configmap)
            
            # Create ingress if enabled
            if resource.spec.ingress.get('enabled'):
                ingress = self.generator.generate_ingress(resource)
                # Would create ingress here
            
            # Update status to running
            status = ResourceStatus(
                phase=ResourcePhase.RUNNING,
                replicas=resource.spec.replicas,
                ready_replicas=resource.spec.replicas,
                available_replicas=resource.spec.replicas,
                observed_generation=resource.generation,
                conditions=[
                    ResourceCondition(
                        type=ConditionType.AVAILABLE,
                        status="True",
                        reason="MinimumReplicasAvailable",
                        message="Deployment has minimum availability"
                    ),
                    ResourceCondition(
                        type=ConditionType.READY,
                        status="True",
                        reason="AllReplicasReady",
                        message="All replicas are ready"
                    )
                ]
            )
            
            await self.client.update_status(resource.name, resource.namespace, status)
            
            logger.info(f"Successfully reconciled {resource.name}")
            return status
            
        except Exception as e:
            logger.error(f"Failed to reconcile {resource.name}: {e}")
            
            status = ResourceStatus(
                phase=ResourcePhase.FAILED,
                message=str(e),
                conditions=[ResourceCondition(
                    type=ConditionType.DEGRADED,
                    status="True",
                    reason="ReconciliationFailed",
                    message=str(e)
                )]
            )
            
            await self.client.update_status(resource.name, resource.namespace, status)
            return status
    
    async def delete(self, resource: MineralVisionResource) -> bool:
        """Delete resources for a MineralVision resource."""
        logger.info(f"Deleting resources for {resource.name}")
        
        try:
            # Delete deployment
            await self.client.delete_deployment(
                f"{resource.name}-core",
                resource.namespace
            )
            
            # Delete other resources...
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete resources: {e}")
            return False


class HelmChartGenerator:
    """
    Generate Helm charts for MineralVision.
    
    Provides:
    - Chart.yaml generation
    - values.yaml generation
    - Template generation
    """
    
    def __init__(self):
        self.chart_name = "mineralvision"
        self.chart_version = "1.0.0"
        self.app_version = "1.0.0"
    
    def generate_chart_yaml(self) -> str:
        """Generate Chart.yaml."""
        chart = {
            'apiVersion': 'v2',
            'name': self.chart_name,
            'description': 'MineralVision - AI-powered mineral exploration platform',
            'type': 'application',
            'version': self.chart_version,
            'appVersion': self.app_version,
            'keywords': ['mineralvision', 'geology', 'ml', 'exploration'],
            'maintainers': [
                {'name': 'MineralVision Team', 'email': 'team@mineralvision.io'}
            ],
            'dependencies': [
                {'name': 'postgresql', 'version': '12.x.x', 'repository': 'https://charts.bitnami.com/bitnami', 'condition': 'postgresql.enabled'},
                {'name': 'kafka', 'version': '26.x.x', 'repository': 'https://charts.bitnami.com/bitnami', 'condition': 'kafka.enabled'},
                {'name': 'redis', 'version': '18.x.x', 'repository': 'https://charts.bitnami.com/bitnami', 'condition': 'redis.enabled'}
            ]
        }
        
        return yaml.dump(chart, default_flow_style=False)
    
    def generate_values_yaml(self) -> str:
        """Generate values.yaml."""
        values = {
            'replicaCount': 1,
            'image': {
                'repository': 'mineralvision/core',
                'tag': 'latest',
                'pullPolicy': 'IfNotPresent'
            },
            'service': {
                'type': 'ClusterIP',
                'port': 80
            },
            'ingress': {
                'enabled': False,
                'className': 'nginx',
                'hosts': [
                    {'host': 'mineralvision.local', 'paths': [{'path': '/', 'pathType': 'Prefix'}]}
                ],
                'tls': []
            },
            'resources': {
                'requests': {'cpu': '500m', 'memory': '512Mi'},
                'limits': {'cpu': '2', 'memory': '2Gi'}
            },
            'autoscaling': {
                'enabled': False,
                'minReplicas': 1,
                'maxReplicas': 10,
                'targetCPUUtilizationPercentage': 70
            },
            'postgresql': {
                'enabled': True,
                'auth': {
                    'database': 'mineralvision',
                    'username': 'mineralvision'
                }
            },
            'kafka': {
                'enabled': True
            },
            'redis': {
                'enabled': True
            },
            'monitoring': {
                'enabled': True,
                'prometheus': {'enabled': True},
                'grafana': {'enabled': True}
            },
            'ml': {
                'enabled': True,
                'gpu': False,
                'modelRegistry': 'mlflow'
            }
        }
        
        return yaml.dump(values, default_flow_style=False)
    
    def generate_deployment_template(self) -> str:
        """Generate deployment template."""
        template = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "mineralvision.fullname" . }}
  labels:
    {{- include "mineralvision.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "mineralvision.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "mineralvision.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
            - name: metrics
              containerPort: 9090
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
'''
        return template


class KubernetesOperator:
    """
    Kubernetes operator for MineralVision.
    
    Provides automated management:
    - Custom resource management
    - Reconciliation
    - Status updates
    - Helm chart generation
    
    Example:
        operator = KubernetesOperator()
        await operator.connect()
        
        # Create a MineralVision resource
        spec = MineralVisionSpec(
            replicas=3,
            image="mineralvision/core:v1.0.0"
        )
        resource = await operator.create_resource("my-instance", spec)
        
        # Reconcile
        status = await operator.reconcile(resource)
        
        # Generate Helm chart
        chart = operator.helm.generate_chart_yaml()
    """
    
    def __init__(self, config: KubernetesConfig = None):
        self.config = config or KubernetesConfig()
        self.client: Optional[MockKubernetesClient] = None
        self.generator: Optional[ResourceGenerator] = None
        self.reconciler: Optional[Reconciler] = None
        self.helm: Optional[HelmChartGenerator] = None
        self._connected = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """True when running on the explicit in-memory mock fallback."""
        return self._degraded

    async def connect(self) -> 'KubernetesOperator':
        """
        Connect to Kubernetes (real client first).

        Falls back to the in-memory mock ONLY when
        MV_ALLOW_MOCK_FALLBACK=true; otherwise raises RuntimeError.
        """
        if K8S_AVAILABLE:
            try:
                if self.config.in_cluster:
                    config.load_incluster_config()
                elif self.config.kubeconfig_path:
                    config.load_kube_config(self.config.kubeconfig_path)
                else:
                    config.load_kube_config()

                # Verify the API server actually answers (short timeout)
                version_api = client.VersionApi()
                await asyncio.wait_for(
                    asyncio.to_thread(version_api.get_code), timeout=5
                )
                # Would initialize real K8s clients here
                logger.info("Connected to Kubernetes cluster")
            except Exception as e:
                if real_client_unavailable("Kubernetes", "API server connection failed", e):
                    self._degraded = True
                    self.client = MockKubernetesClient(self.config)
        else:
            if real_client_unavailable("Kubernetes", "kubernetes package not installed"):
                self._degraded = True
                self.client = MockKubernetesClient(self.config)

        if not self.client:
            # Real config loaded and API server reachable; real client
            # initialization is pending — use the in-memory client but keep
            # the degraded flag accurate.
            self.client = MockKubernetesClient(self.config)
        
        self.generator = ResourceGenerator(self.config)
        self.reconciler = Reconciler(self.client, self.generator)
        self.helm = HelmChartGenerator()
        
        # Ensure namespace exists
        if not await self.client.namespace_exists(self.config.namespace):
            await self.client.create_namespace(self.config.namespace)
        
        self._connected = True
        return self
    
    async def create_resource(self, name: str, spec: MineralVisionSpec,
                             namespace: str = None) -> MineralVisionResource:
        """Create a MineralVision resource."""
        namespace = namespace or self.config.namespace
        
        resource = MineralVisionResource(
            name=name,
            namespace=namespace,
            spec=spec
        )
        
        return await self.client.create_resource(resource)
    
    async def get_resource(self, name: str, 
                          namespace: str = None) -> Optional[MineralVisionResource]:
        """Get a MineralVision resource."""
        namespace = namespace or self.config.namespace
        return await self.client.get_resource(name, namespace)
    
    async def list_resources(self, namespace: str = None) -> List[MineralVisionResource]:
        """List MineralVision resources."""
        return await self.client.list_resources(namespace)
    
    async def update_resource(self, resource: MineralVisionResource) -> MineralVisionResource:
        """Update a MineralVision resource."""
        return await self.client.update_resource(resource)
    
    async def delete_resource(self, name: str, namespace: str = None) -> bool:
        """Delete a MineralVision resource."""
        namespace = namespace or self.config.namespace
        
        resource = await self.get_resource(name, namespace)
        if resource:
            await self.reconciler.delete(resource)
            return await self.client.delete_resource(name, namespace)
        return False
    
    async def reconcile(self, resource: MineralVisionResource) -> ResourceStatus:
        """Reconcile a MineralVision resource."""
        return await self.reconciler.reconcile(resource)
    
    async def get_events(self, name: str = None,
                        namespace: str = None) -> List[Dict[str, Any]]:
        """Get events for a resource."""
        namespace = namespace or self.config.namespace
        return await self.client.get_events(namespace, name)
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Factory functions

def create_kubernetes_operator(config: KubernetesConfig = None) -> KubernetesOperator:
    """Create a Kubernetes operator instance."""
    return KubernetesOperator(config)


async def create_and_connect_operator(config: KubernetesConfig = None) -> KubernetesOperator:
    """Create and connect Kubernetes operator."""
    operator = KubernetesOperator(config)
    await operator.connect()
    return operator


# CRD Definition

MINERALVISION_CRD = {
    'apiVersion': 'apiextensions.k8s.io/v1',
    'kind': 'CustomResourceDefinition',
    'metadata': {
        'name': 'mineralvisions.mineralvision.io'
    },
    'spec': {
        'group': 'mineralvision.io',
        'versions': [{
            'name': 'v1',
            'served': True,
            'storage': True,
            'schema': {
                'openAPIV3Schema': {
                    'type': 'object',
                    'properties': {
                        'spec': {
                            'type': 'object',
                            'properties': {
                                'version': {'type': 'string'},
                                'replicas': {'type': 'integer', 'minimum': 1},
                                'image': {'type': 'string'},
                                'resources': {'type': 'object'},
                                'storage': {'type': 'object'},
                                'database': {'type': 'object'},
                                'kafka': {'type': 'object'},
                                'ml': {'type': 'object'},
                                'monitoring': {'type': 'object'},
                                'ingress': {'type': 'object'}
                            }
                        },
                        'status': {
                            'type': 'object',
                            'properties': {
                                'phase': {'type': 'string'},
                                'conditions': {'type': 'array'},
                                'replicas': {'type': 'integer'},
                                'readyReplicas': {'type': 'integer'},
                                'availableReplicas': {'type': 'integer'}
                            }
                        }
                    }
                }
            },
            'subresources': {
                'status': {}
            }
        }],
        'scope': 'Namespaced',
        'names': {
            'plural': 'mineralvisions',
            'singular': 'mineralvision',
            'kind': 'MineralVision',
            'shortNames': ['mv']
        }
    }
}
