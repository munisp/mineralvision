# MineralVision Infrastructure - High Availability Configurations

This directory contains production-ready HA configurations for all MineralVision infrastructure services.

## Quick Start

```bash
# Deploy all services
./ha-configs/deploy-all.sh all

# Deploy individual services
./ha-configs/deploy-all.sh kafka
./ha-configs/deploy-all.sh redis
./ha-configs/deploy-all.sh temporal
```

## Services

| Service | Replicas | Config File |
|---------|----------|-------------|
| Kafka (Strimzi) | 3 brokers, 3 ZK | `kafka/kafka-ha.yaml` |
| Redis Sentinel | 1 master, 2 replicas, 3 sentinels | `redis/redis-ha.yaml` |
| Temporal | 3 frontend, 3 history, 3 matching | `temporal/temporal-ha.yaml` |
| Keycloak | 3 replicas (Infinispan cluster) | `keycloak/keycloak-ha.yaml` |
| APISIX | 3 gateways, 3 etcd | `apisix/apisix-ha.yaml` |
| Dapr | 3 placement, 3 sentry | `dapr/dapr-ha.yaml` |
| Fluvio | 3 SC, 3 SPU | `fluvio/fluvio-ha.yaml` |
| Permify | 3 replicas | `permify/permify-ha.yaml` |
| TigerBeetle | 3 replicas | `tigerbeetle/tigerbeetle-ha.yaml` |
| Lakehouse (MinIO/Trino) | 4 MinIO, 3 Trino workers | `lakehouse/lakehouse-ha.yaml` |
| OpenAppSec | 3 learning, 3 storage | `openappsec/openappsec-ha.yaml` |

## Prerequisites

- Kubernetes cluster with 3+ nodes
- kubectl configured
- Helm 3.x installed
- Storage class `fast-ssd` available
- NVIDIA GPU support (optional, for ML workloads)

## Service Endpoints

After deployment, services are available at:

- Kafka: `mineralvision-kafka-kafka-bootstrap.mineralvision:9092`
- Redis: `redis-master.mineralvision:6379`
- Temporal: `temporal-frontend.temporal:7233`
- Keycloak: `keycloak.keycloak:8080`
- APISIX: `apisix-gateway.apisix:80`
- Permify: `permify.permify:3476`
- TigerBeetle: `tigerbeetle.tigerbeetle:3000`
- MinIO: `minio.lakehouse:9000`
- Trino: `trino-coordinator.lakehouse:8080`

## Security Notes

Update default passwords in secret files before production deployment:
- `temporal/temporal-ha.yaml` - PostgreSQL credentials
- `keycloak/keycloak-ha.yaml` - Admin credentials
- `lakehouse/lakehouse-ha.yaml` - MinIO credentials
- `apisix/apisix-ha.yaml` - Admin API keys
