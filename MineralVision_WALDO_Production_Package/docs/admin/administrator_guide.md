# MineralVision WALDO Integration - Administrator Guide

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Installation and Configuration](#installation-and-configuration)
4. [User Management](#user-management)
5. [System Monitoring](#system-monitoring)
6. [Backup and Recovery](#backup-and-recovery)
7. [Performance Tuning](#performance-tuning)
8. [Security](#security)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance](#maintenance)

## Introduction

This administrator guide provides detailed information for IT administrators responsible for deploying, configuring, and maintaining the MineralVision WALDO integration. The guide covers all aspects of system administration across cloud, on-premise, and edge deployments.

## System Architecture

### Components Overview

The MineralVision WALDO integration consists of the following core components:

1. **WALDO Detection Service**: Handles object detection using WALDO models
2. **API Service**: Provides RESTful API endpoints for application integration
3. **Web UI**: User interface for interacting with the system
4. **Database**: Stores detection results, tracking data, and system configuration
5. **Message Broker**: Handles asynchronous communication between components
6. **ArcGIS Integration**: Connects with ArcGIS for spatial data visualization

### Deployment Architectures

#### Cloud Architecture
- Containerized microservices running on Kubernetes
- Horizontally scalable components
- Cloud-native storage and database services
- Load balancing and auto-scaling

#### On-Premise Architecture
- Services running as systemd units
- PostgreSQL database for data storage
- RabbitMQ for message brokering
- Nginx for web serving and reverse proxy

#### Edge Architecture
- Lightweight services optimized for edge devices
- SQLite database for local storage
- Periodic synchronization with central server
- Optimized models for resource-constrained environments

### Data Flow

1. Input data (images/video) is processed by the WALDO Detection Service
2. Detection results are stored in the database
3. The API Service provides access to detection data
4. The Web UI visualizes results for users
5. The ArcGIS Integration synchronizes data with ArcGIS
6. Edge devices synchronize with the central server

## Installation and Configuration

Refer to the [Installation Guide](../installation/installation_guide.md) for detailed installation instructions for each deployment environment.

### Configuration Files

#### Main Configuration File

Location: `/etc/mineralvision/waldo/config.yaml`

```yaml
# Main configuration file
system:
  environment: production  # production, development, testing
  log_level: info  # debug, info, warning, error
  temp_dir: /tmp/mineralvision

database:
  type: postgresql  # postgresql, sqlite
  host: localhost
  port: 5432
  name: waldo_detections
  user: mineralvision
  password: ${DB_PASSWORD}  # Set via environment variable

messaging:
  type: rabbitmq  # rabbitmq, redis
  host: localhost
  port: 5672
  user: guest
  password: ${RABBITMQ_PASSWORD}  # Set via environment variable
  vhost: /

waldo:
  model_path: /var/lib/mineralvision/waldo/models/waldo_v3.pt
  confidence_threshold: 0.25
  device: cuda  # cuda, cpu
  precision: fp16  # fp32, fp16, int8
  batch_size: 4

tracking:
  max_age: 10
  min_hits: 3
  iou_threshold: 0.3

arcgis:
  enabled: true
  url: ${ARCGIS_URL}  # Set via environment variable
  username: ${ARCGIS_USERNAME}  # Set via environment variable
  password: ${ARCGIS_PASSWORD}  # Set via environment variable
  sync_interval: 3600  # seconds
```

#### Environment-Specific Configuration

For cloud deployments, environment variables are set in Kubernetes secrets or Docker Compose environment files.

For on-premise deployments, environment variables are set in systemd service files.

For edge deployments, environment variables are set in the edge-specific configuration file.

### Advanced Configuration

#### Scaling Configuration

For cloud deployments, configure resource limits and requests in Kubernetes manifests:

```yaml
resources:
  limits:
    cpu: 4
    memory: 8Gi
    nvidia.com/gpu: 1
  requests:
    cpu: 2
    memory: 4Gi
```

#### Network Configuration

Configure network settings in the appropriate files:

- Cloud: Kubernetes Service and Ingress resources
- On-premise: Nginx configuration
- Edge: Edge-specific network settings

## User Management

### User Roles

The system supports the following user roles:

1. **Administrator**: Full system access
2. **Manager**: Access to all features except system configuration
3. **Analyst**: Access to detection and analysis features
4. **Viewer**: Read-only access to results and reports

### Managing Users

#### Adding Users

1. Navigate to "Settings" > "User Management" in the web UI
2. Click "Add User"
3. Enter user details:
   - Username
   - Email
   - Password
   - Role
4. Click "Save"

#### Modifying Users

1. Navigate to "Settings" > "User Management" in the web UI
2. Find the user in the list
3. Click "Edit"
4. Modify user details
5. Click "Save"

#### Deleting Users

1. Navigate to "Settings" > "User Management" in the web UI
2. Find the user in the list
3. Click "Delete"
4. Confirm deletion

### Authentication Configuration

#### LDAP/Active Directory Integration

Configure LDAP integration in the authentication configuration file:

```yaml
authentication:
  type: ldap  # local, ldap, oauth
  ldap:
    server: ldap://ldap.example.com
    bind_dn: cn=admin,dc=example,dc=com
    bind_password: ${LDAP_PASSWORD}
    search_base: ou=users,dc=example,dc=com
    search_filter: (uid=%s)
    group_search_base: ou=groups,dc=example,dc=com
    group_search_filter: (memberUid=%s)
    admin_group: cn=admins,ou=groups,dc=example,dc=com
```

#### OAuth Integration

Configure OAuth integration in the authentication configuration file:

```yaml
authentication:
  type: oauth  # local, ldap, oauth
  oauth:
    provider: google  # google, github, azure
    client_id: ${OAUTH_CLIENT_ID}
    client_secret: ${OAUTH_CLIENT_SECRET}
    redirect_uri: https://waldo.mineralvision.com/auth/callback
    scopes: email,profile
```

## System Monitoring

### Monitoring Tools

The system includes built-in monitoring capabilities:

1. **System Dashboard**: Available in the web UI under "Settings" > "System"
2. **Prometheus Metrics**: Exposed on `/metrics` endpoint
3. **Log Files**: Detailed logs for all components

### Key Metrics

Monitor the following key metrics:

1. **CPU Usage**: Overall system CPU utilization
2. **Memory Usage**: RAM utilization
3. **GPU Usage**: GPU utilization (if applicable)
4. **Storage Usage**: Disk space utilization
5. **Database Performance**: Query times and connection pool status
6. **API Response Times**: Latency for API endpoints
7. **Detection Processing Time**: Time to process detections
8. **Queue Length**: Number of jobs in processing queue

### Log Files

Log files are stored in the following locations:

- Cloud: Container logs, accessible via `kubectl logs`
- On-premise: `/var/log/mineralvision/waldo/`
- Edge: `/var/log/mineralvision/waldo/`

### Alerting

Configure alerts for critical system events:

1. Navigate to "Settings" > "Alerts" in the web UI
2. Click "Add Alert Rule"
3. Configure alert conditions:
   - Metric
   - Threshold
   - Duration
   - Notification method (email, webhook, etc.)
4. Click "Save"

## Backup and Recovery

### Database Backup

#### Automated Backups

For PostgreSQL databases:

```bash
# Add to crontab
0 2 * * * pg_dump -U mineralvision waldo_detections | gzip > /backup/waldo_$(date +\%Y\%m\%d).sql.gz
```

For SQLite databases (edge devices):

```bash
# Add to crontab
0 2 * * * sqlite3 /var/lib/mineralvision/waldo/database/waldo.db .dump | gzip > /backup/waldo_$(date +\%Y\%m\%d).sql.gz
```

#### Manual Backups

For PostgreSQL databases:

```bash
pg_dump -U mineralvision waldo_detections > waldo_backup.sql
```

For SQLite databases:

```bash
sqlite3 /var/lib/mineralvision/waldo/database/waldo.db .dump > waldo_backup.sql
```

### Configuration Backup

Backup configuration files:

```bash
tar -czf waldo_config_backup.tar.gz /etc/mineralvision/waldo/
```

### Model Backup

Backup model files:

```bash
tar -czf waldo_models_backup.tar.gz /var/lib/mineralvision/waldo/models/
```

### Recovery Procedures

#### Database Recovery

For PostgreSQL databases:

```bash
psql -U mineralvision waldo_detections < waldo_backup.sql
```

For SQLite databases:

```bash
sqlite3 /var/lib/mineralvision/waldo/database/waldo.db < waldo_backup.sql
```

#### Configuration Recovery

Restore configuration files:

```bash
tar -xzf waldo_config_backup.tar.gz -C /
```

#### Complete System Recovery

1. Reinstall the system following the installation guide
2. Restore configuration files
3. Restore database
4. Restore models
5. Restart all services

## Performance Tuning

### Hardware Recommendations

#### Cloud Deployment
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: NVIDIA T4 or better
- Storage: 500+ GB SSD

#### On-Premise Deployment
- CPU: 8+ cores
- RAM: 32+ GB
- GPU: NVIDIA RTX 3080 or better
- Storage: 1+ TB SSD

#### Edge Deployment
- Jetson Xavier NX or better
- Raspberry Pi 4 with 8GB RAM or better
- 128+ GB storage

### Model Optimization

Optimize models for different environments:

1. **Cloud**: Use FP16 precision for optimal performance/accuracy balance
2. **On-Premise**: Use FP16 or FP32 depending on GPU capabilities
3. **Edge**: Use INT8 quantized models for resource-constrained devices

### Database Optimization

Optimize database performance:

1. **Indexing**: Ensure proper indexes on frequently queried fields
2. **Connection Pooling**: Configure appropriate connection pool size
3. **Query Optimization**: Review and optimize slow queries
4. **Partitioning**: Partition large tables by time for improved performance

### Scaling Guidelines

#### Vertical Scaling
- Increase CPU/RAM/GPU resources for single-instance deployments
- Upgrade to more powerful hardware

#### Horizontal Scaling (Cloud)
- Add more detection service replicas for increased throughput
- Configure auto-scaling based on CPU/GPU utilization
- Distribute database load using read replicas

## Security

### Network Security

1. **Firewall Configuration**: Restrict access to necessary ports only
2. **TLS/SSL**: Enable HTTPS for all web interfaces
3. **VPN**: Consider VPN for remote access to on-premise deployments
4. **Network Segmentation**: Isolate system components in separate network segments

### Authentication Security

1. **Password Policy**: Enforce strong password requirements
2. **MFA**: Enable multi-factor authentication
3. **Session Management**: Configure appropriate session timeouts
4. **API Keys**: Rotate API keys regularly

### Data Security

1. **Encryption at Rest**: Enable database encryption
2. **Encryption in Transit**: Use TLS for all communications
3. **Data Retention**: Implement data retention policies
4. **Access Control**: Restrict data access based on user roles

### Audit Logging

Enable comprehensive audit logging:

```yaml
audit:
  enabled: true
  log_path: /var/log/mineralvision/waldo/audit.log
  events:
    - user.login
    - user.logout
    - user.create
    - user.modify
    - user.delete
    - detection.create
    - detection.delete
    - system.config_change
```

## Troubleshooting

### Common Issues

#### Detection Service Not Starting

1. Check GPU availability: `nvidia-smi`
2. Verify model file exists
3. Check log files for errors
4. Verify database connection

#### Database Connection Issues

1. Verify database is running
2. Check connection string in configuration
3. Verify network connectivity
4. Check database logs

#### API Errors

1. Check API service logs
2. Verify API service is running
3. Check database connection
4. Verify authentication configuration

#### Edge Device Synchronization Issues

1. Check network connectivity
2. Verify central server URL
3. Check authentication credentials
4. Verify local database is not corrupted

### Diagnostic Tools

#### System Diagnostics

Run the built-in diagnostic tool:

```bash
/opt/mineralvision/waldo/bin/diagnose.sh
```

#### Log Analysis

Use the log analysis tool to identify issues:

```bash
/opt/mineralvision/waldo/bin/analyze_logs.sh
```

#### Database Verification

Verify database integrity:

```bash
/opt/mineralvision/waldo/bin/verify_db.sh
```

## Maintenance

### Routine Maintenance Tasks

1. **Database Maintenance**: Run vacuum and analyze operations
2. **Log Rotation**: Ensure logs are properly rotated
3. **Backup Verification**: Test backups regularly
4. **Storage Cleanup**: Remove temporary files and old logs
5. **Security Updates**: Apply security patches

### Upgrade Procedures

#### Cloud Deployment

1. Update Kubernetes manifests or Docker Compose files
2. Apply updates: `kubectl apply -f updated-manifests.yaml`
3. Monitor rollout: `kubectl rollout status deployment/waldo-detection`

#### On-Premise Deployment

1. Stop services: `systemctl stop waldo-detection waldo-api waldo-arcgis`
2. Backup configuration and database
3. Update software: `./upgrade.sh`
4. Start services: `systemctl start waldo-detection waldo-api waldo-arcgis`
5. Verify upgrade: `systemctl status waldo-detection`

#### Edge Deployment

1. Stop services: `systemctl stop waldo-edge waldo-sync`
2. Backup configuration and database
3. Update software: `./upgrade_edge.sh`
4. Start services: `systemctl start waldo-edge waldo-sync`
5. Verify upgrade: `systemctl status waldo-edge`

### Version Compatibility

Ensure compatibility between components:

| Component | Version | Compatible With |
|-----------|---------|-----------------|
| WALDO Detection | 1.2.x | API 1.2.x, Web UI 1.2.x |
| API Service | 1.2.x | WALDO 1.2.x, Web UI 1.2.x |
| Web UI | 1.2.x | API 1.2.x, WALDO 1.2.x |
| Database Schema | 1.2.x | WALDO 1.2.x, API 1.2.x |

### Maintenance Windows

Schedule maintenance during low-usage periods:

1. Notify users in advance
2. Set up maintenance mode in the web UI
3. Perform maintenance tasks
4. Verify system functionality
5. Disable maintenance mode
6. Notify users of completion
