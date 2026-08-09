# MineralVision Backup and Restore Procedures

## Backup Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Backup Infrastructure                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  PostgreSQL  │    │    Redis     │    │    Kafka     │      │
│  │   Backups    │    │   Backups    │    │   Backups    │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Backup Manager (CronJob)                    │   │
│  │  - Scheduled backups every 6 hours                       │   │
│  │  - Retention: 7 days local, 30 days remote               │   │
│  │  - Encryption: AES-256                                   │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Remote Storage (S3/MinIO)                   │   │
│  │  - Cross-region replication                              │   │
│  │  - Versioning enabled                                    │   │
│  │  - Lifecycle policies                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Backup Schedule

| Component | Frequency | Retention | Type |
|-----------|-----------|-----------|------|
| PostgreSQL | Every 6 hours | 7 days | Full dump |
| PostgreSQL WAL | Continuous | 24 hours | Incremental |
| Redis | Every hour | 24 hours | RDB snapshot |
| Kafka | Every 6 hours | 7 days | Topic configs |
| MinIO | Daily | 30 days | Incremental sync |
| Vault | Every 6 hours | 30 days | Raft snapshot |
| etcd | Every hour | 7 days | Snapshot |

## Kubernetes Backup Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: backup-scripts
  namespace: backup
data:
  backup-all.sh: |
    #!/bin/bash
    set -e
    
    BACKUP_DATE=$(date +%Y%m%d-%H%M%S)
    BACKUP_DIR="/backups/${BACKUP_DATE}"
    mkdir -p ${BACKUP_DIR}
    
    echo "Starting backup at ${BACKUP_DATE}"
    
    # PostgreSQL backup
    echo "Backing up PostgreSQL..."
    PGPASSWORD=${POSTGRES_PASSWORD} pg_dumpall -h postgresql.postgresql -U postgres | \
      gzip > ${BACKUP_DIR}/postgresql.sql.gz
    
    # Redis backup
    echo "Backing up Redis..."
    redis-cli -h redis-master.redis BGSAVE
    sleep 5
    kubectl cp redis/redis-master-0:/data/dump.rdb ${BACKUP_DIR}/redis.rdb
    
    # Vault backup
    echo "Backing up Vault..."
    kubectl exec -n vault vault-0 -- vault operator raft snapshot save /tmp/vault.snap
    kubectl cp vault/vault-0:/tmp/vault.snap ${BACKUP_DIR}/vault.snap
    
    # Encrypt backups
    echo "Encrypting backups..."
    for file in ${BACKUP_DIR}/*; do
      openssl enc -aes-256-cbc -salt -pbkdf2 -in ${file} -out ${file}.enc -pass env:BACKUP_ENCRYPTION_KEY
      rm ${file}
    done
    
    # Upload to remote storage
    echo "Uploading to remote storage..."
    mc cp -r ${BACKUP_DIR} backup/mineralvision-backups/
    
    # Cleanup old local backups
    find /backups -type d -mtime +7 -exec rm -rf {} +
    
    echo "Backup completed successfully"
    
  restore-postgresql.sh: |
    #!/bin/bash
    set -e
    
    BACKUP_FILE=$1
    
    if [ -z "$BACKUP_FILE" ]; then
      echo "Usage: restore-postgresql.sh <backup-file>"
      exit 1
    fi
    
    echo "Restoring PostgreSQL from ${BACKUP_FILE}"
    
    # Decrypt if encrypted
    if [[ ${BACKUP_FILE} == *.enc ]]; then
      openssl enc -aes-256-cbc -d -pbkdf2 -in ${BACKUP_FILE} -out /tmp/restore.sql.gz -pass env:BACKUP_ENCRYPTION_KEY
      BACKUP_FILE=/tmp/restore.sql.gz
    fi
    
    # Decompress if compressed
    if [[ ${BACKUP_FILE} == *.gz ]]; then
      gunzip -c ${BACKUP_FILE} > /tmp/restore.sql
      BACKUP_FILE=/tmp/restore.sql
    fi
    
    # Restore
    PGPASSWORD=${POSTGRES_PASSWORD} psql -h postgresql.postgresql -U postgres < ${BACKUP_FILE}
    
    echo "PostgreSQL restore completed"
    
  restore-redis.sh: |
    #!/bin/bash
    set -e
    
    BACKUP_FILE=$1
    
    if [ -z "$BACKUP_FILE" ]; then
      echo "Usage: restore-redis.sh <backup-file>"
      exit 1
    fi
    
    echo "Restoring Redis from ${BACKUP_FILE}"
    
    # Stop Redis
    kubectl scale statefulset -n redis redis-master --replicas=0
    sleep 10
    
    # Copy backup file
    kubectl cp ${BACKUP_FILE} redis/redis-master-0:/data/dump.rdb
    
    # Start Redis
    kubectl scale statefulset -n redis redis-master --replicas=1
    
    echo "Redis restore completed"
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scheduled-backup
  namespace: backup
spec:
  schedule: "0 */6 * * *"
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: backup-sa
          containers:
            - name: backup
              image: mineralvision/backup-tools:latest
              command: ["/scripts/backup-all.sh"]
              env:
                - name: POSTGRES_PASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: postgresql-secrets
                      key: password
                - name: BACKUP_ENCRYPTION_KEY
                  valueFrom:
                    secretKeyRef:
                      name: backup-secrets
                      key: encryption-key
              volumeMounts:
                - name: backup-scripts
                  mountPath: /scripts
                - name: backup-storage
                  mountPath: /backups
          volumes:
            - name: backup-scripts
              configMap:
                name: backup-scripts
                defaultMode: 0755
            - name: backup-storage
              persistentVolumeClaim:
                claimName: backup-pvc
          restartPolicy: OnFailure
```

## Point-in-Time Recovery (PITR)

### PostgreSQL PITR

```bash
# 1. Stop the application
kubectl scale deployment -n mineralvision --all --replicas=0

# 2. Identify target recovery time
TARGET_TIME="2024-01-15 14:30:00"

# 3. Restore base backup
kubectl exec -n postgresql postgresql-0 -- pg_restore \
  --clean --if-exists \
  -d mineralvision \
  /backups/base/postgresql-20240115.dump

# 4. Apply WAL logs up to target time
kubectl exec -n postgresql postgresql-0 -- psql -c "
  SELECT pg_wal_replay_resume();
  SELECT pg_create_restore_point('recovery-${TARGET_TIME}');
"

# 5. Verify recovery
kubectl exec -n postgresql postgresql-0 -- psql -c "
  SELECT max(created_at) FROM sensor_data;
"

# 6. Restart application
kubectl scale deployment -n mineralvision --all --replicas=3
```

## Disaster Recovery Testing

### Monthly DR Test Procedure

```bash
#!/bin/bash
# dr-test.sh - Monthly disaster recovery test

echo "=== MineralVision DR Test ==="
echo "Date: $(date)"

# 1. Create test namespace
kubectl create namespace dr-test

# 2. Deploy test instance from backups
echo "Deploying test instance..."
./restore-to-namespace.sh dr-test

# 3. Run verification tests
echo "Running verification tests..."
kubectl run -n dr-test verification-test \
  --image=mineralvision/test-runner:latest \
  --restart=Never \
  --command -- /tests/verify-data-integrity.sh

# 4. Wait for tests
kubectl wait -n dr-test --for=condition=complete job/verification-test --timeout=600s

# 5. Get test results
kubectl logs -n dr-test verification-test

# 6. Cleanup
kubectl delete namespace dr-test

echo "=== DR Test Complete ==="
```

## Recovery Verification Checklist

- [ ] All pods running and healthy
- [ ] Database connections working
- [ ] Authentication functional
- [ ] API endpoints responding
- [ ] Message queues processing
- [ ] Cache populated
- [ ] Data integrity verified
- [ ] Integration tests passing
- [ ] Monitoring alerts cleared
- [ ] User acceptance verified
