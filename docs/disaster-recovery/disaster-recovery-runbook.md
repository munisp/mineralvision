# MineralVision Disaster Recovery Runbook

## Overview

This runbook provides step-by-step procedures for disaster recovery scenarios affecting the MineralVision platform.

## Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

| Service | RTO | RPO | Priority |
|---------|-----|-----|----------|
| API Gateway (APISIX) | 5 min | 0 | Critical |
| Authentication (Keycloak) | 10 min | 1 hour | Critical |
| Message Queue (Kafka) | 15 min | 0 | Critical |
| Cache (Redis) | 5 min | 5 min | High |
| Workflow (Temporal) | 15 min | 1 hour | High |
| Database (PostgreSQL) | 30 min | 5 min | Critical |
| Object Storage (MinIO) | 30 min | 1 hour | High |
| Query Engine (Trino) | 15 min | N/A | Medium |

## Backup Procedures

### Automated Backups

All critical data is backed up automatically:

```bash
# Verify backup status
kubectl get cronjobs -n backup

# List recent backups
kubectl exec -n backup backup-manager-0 -- ls -la /backups/

# Check backup integrity
kubectl exec -n backup backup-manager-0 -- /scripts/verify-backups.sh
```

### Manual Backup Commands

#### PostgreSQL (Keycloak, Temporal, Permify)

```bash
# Backup all PostgreSQL databases
kubectl exec -n postgresql postgresql-0 -- pg_dumpall -U postgres > /backups/postgresql-$(date +%Y%m%d-%H%M%S).sql

# Backup specific database
kubectl exec -n postgresql postgresql-0 -- pg_dump -U postgres keycloak > /backups/keycloak-$(date +%Y%m%d-%H%M%S).sql
```

#### Redis

```bash
# Trigger Redis backup
kubectl exec -n redis redis-master-0 -- redis-cli BGSAVE

# Copy RDB file
kubectl cp redis/redis-master-0:/data/dump.rdb /backups/redis-$(date +%Y%m%d-%H%M%S).rdb
```

#### Kafka

```bash
# Backup Kafka topics configuration
kubectl exec -n kafka kafka-0 -- kafka-topics.sh --bootstrap-server localhost:9092 --describe > /backups/kafka-topics-$(date +%Y%m%d-%H%M%S).txt

# Backup consumer group offsets
kubectl exec -n kafka kafka-0 -- kafka-consumer-groups.sh --bootstrap-server localhost:9092 --all-groups --describe > /backups/kafka-offsets-$(date +%Y%m%d-%H%M%S).txt
```

#### MinIO (Lakehouse Data)

```bash
# Sync to backup location
mc mirror minio/lakehouse s3/backup-bucket/lakehouse-$(date +%Y%m%d)/

# Verify backup
mc diff minio/lakehouse s3/backup-bucket/lakehouse-$(date +%Y%m%d)/
```

#### Vault

```bash
# Create Vault snapshot
kubectl exec -n vault vault-0 -- vault operator raft snapshot save /vault/data/snapshot-$(date +%Y%m%d-%H%M%S).snap

# Copy snapshot
kubectl cp vault/vault-0:/vault/data/snapshot-*.snap /backups/
```

## Disaster Recovery Scenarios

### Scenario 1: Single Pod Failure

**Symptoms:** Single pod in CrashLoopBackOff or not responding

**Recovery Steps:**

```bash
# 1. Identify failed pod
kubectl get pods -A | grep -v Running

# 2. Check pod logs
kubectl logs -n <namespace> <pod-name> --previous

# 3. Delete pod to trigger restart
kubectl delete pod -n <namespace> <pod-name>

# 4. Verify recovery
kubectl get pods -n <namespace> -w
```

### Scenario 2: Node Failure

**Symptoms:** Multiple pods unavailable, node NotReady

**Recovery Steps:**

```bash
# 1. Identify failed node
kubectl get nodes

# 2. Cordon the node
kubectl cordon <node-name>

# 3. Drain workloads (if possible)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# 4. Verify pods rescheduled
kubectl get pods -A -o wide | grep <node-name>

# 5. Once node recovered, uncordon
kubectl uncordon <node-name>
```

### Scenario 3: Kafka Cluster Failure

**Symptoms:** Message processing stopped, producers/consumers failing

**Recovery Steps:**

```bash
# 1. Check Kafka cluster status
kubectl get kafka -n kafka

# 2. Check broker status
kubectl exec -n kafka kafka-0 -- kafka-broker-api-versions.sh --bootstrap-server localhost:9092

# 3. If brokers down, check ZooKeeper
kubectl exec -n kafka kafka-zookeeper-0 -- zkCli.sh -server localhost:2181 stat

# 4. Force leader election if needed
kubectl exec -n kafka kafka-0 -- kafka-leader-election.sh --bootstrap-server localhost:9092 --election-type PREFERRED --all-topic-partitions

# 5. Verify topic health
kubectl exec -n kafka kafka-0 -- kafka-topics.sh --bootstrap-server localhost:9092 --describe --under-replicated-partitions
```

### Scenario 4: Redis Sentinel Failover

**Symptoms:** Redis connection errors, cache misses

**Recovery Steps:**

```bash
# 1. Check Sentinel status
kubectl exec -n redis redis-sentinel-0 -- redis-cli -p 26379 SENTINEL masters

# 2. Identify current master
kubectl exec -n redis redis-sentinel-0 -- redis-cli -p 26379 SENTINEL get-master-addr-by-name mymaster

# 3. Force failover if needed
kubectl exec -n redis redis-sentinel-0 -- redis-cli -p 26379 SENTINEL failover mymaster

# 4. Verify replication
kubectl exec -n redis redis-master-0 -- redis-cli INFO replication
```

### Scenario 5: Keycloak Authentication Failure

**Symptoms:** Users cannot authenticate, 401/403 errors

**Recovery Steps:**

```bash
# 1. Check Keycloak pods
kubectl get pods -n keycloak

# 2. Check Keycloak logs
kubectl logs -n keycloak keycloak-0 --tail=100

# 3. Verify database connectivity
kubectl exec -n keycloak keycloak-0 -- curl -s http://postgresql.postgresql:5432

# 4. Restart Keycloak if needed
kubectl rollout restart statefulset -n keycloak keycloak

# 5. Verify Keycloak health
kubectl exec -n keycloak keycloak-0 -- curl -s http://localhost:8080/health
```

### Scenario 6: Complete Cluster Failure

**Symptoms:** All services unavailable, cluster unreachable

**Recovery Steps:**

```bash
# 1. Verify infrastructure (OpenStack/AWS)
openstack server list
# or
aws ec2 describe-instances --filters "Name=tag:Cluster,Values=mineralvision"

# 2. If infrastructure OK, check Kubernetes control plane
ssh master-1 "sudo systemctl status kubelet"
ssh master-1 "sudo crictl ps"

# 3. Restore etcd from backup if needed
ssh master-1 "sudo ETCDCTL_API=3 etcdctl snapshot restore /backups/etcd-latest.db"

# 4. Restart control plane components
ssh master-1 "sudo systemctl restart kubelet"

# 5. Verify cluster health
kubectl get nodes
kubectl get pods -n kube-system
```

### Scenario 7: Data Corruption

**Symptoms:** Inconsistent data, application errors

**Recovery Steps:**

```bash
# 1. Identify affected service
kubectl logs -n mineralvision <service-pod> | grep -i error

# 2. Stop affected service
kubectl scale deployment -n mineralvision <service> --replicas=0

# 3. Restore from backup
# For PostgreSQL:
kubectl exec -n postgresql postgresql-0 -- psql -U postgres -d <database> < /backups/<database>-<timestamp>.sql

# For MinIO:
mc mirror s3/backup-bucket/lakehouse-<date>/ minio/lakehouse/

# 4. Restart service
kubectl scale deployment -n mineralvision <service> --replicas=3

# 5. Verify data integrity
kubectl exec -n mineralvision <service-pod> -- /scripts/verify-data.sh
```

## Post-Recovery Verification

After any recovery procedure, verify system health:

```bash
# 1. Check all pods running
kubectl get pods -A | grep -v Running | grep -v Completed

# 2. Run health checks
curl -s https://api.mineralvision.local/health | jq

# 3. Verify Kafka topics
kubectl exec -n kafka kafka-0 -- kafka-topics.sh --bootstrap-server localhost:9092 --list

# 4. Check Redis connectivity
kubectl exec -n redis redis-master-0 -- redis-cli PING

# 5. Verify Keycloak
curl -s https://keycloak.mineralvision.local/health | jq

# 6. Run integration tests
kubectl create job -n mineralvision integration-test --from=cronjob/integration-test

# 7. Check Grafana dashboards for anomalies
# Navigate to https://grafana.mineralvision.local
```

## Escalation Contacts

| Level | Contact | Response Time |
|-------|---------|---------------|
| L1 | On-call Engineer | 15 min |
| L2 | Platform Team Lead | 30 min |
| L3 | Infrastructure Architect | 1 hour |
| L4 | CTO | 2 hours |

## Appendix: Useful Commands

```bash
# Quick cluster health check
kubectl get nodes && kubectl get pods -A | grep -v Running | grep -v Completed

# Check resource usage
kubectl top nodes && kubectl top pods -A

# View recent events
kubectl get events -A --sort-by='.lastTimestamp' | tail -20

# Check PVC status
kubectl get pvc -A

# View service endpoints
kubectl get endpoints -A
```
