#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="mineralvision"

echo "=============================================="
echo "MineralVision HA Infrastructure Deployment"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

wait_for_pods() {
    local namespace=$1
    local label=$2
    local timeout=${3:-300}
    
    log_info "Waiting for pods with label '$label' in namespace '$namespace'..."
    kubectl wait --for=condition=ready pod -l "$label" -n "$namespace" --timeout="${timeout}s" || {
        log_warn "Some pods may not be ready yet. Continuing..."
    }
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

create_namespaces() {
    log_info "Creating namespaces..."
    
    namespaces=(
        "mineralvision"
        "kafka"
        "redis"
        "temporal"
        "keycloak"
        "apisix"
        "dapr-system"
        "fluvio"
        "permify"
        "tigerbeetle"
        "lakehouse"
        "openappsec"
        "observability"
    )
    
    for ns in "${namespaces[@]}"; do
        kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f -
    done
}

install_strimzi_operator() {
    log_info "Installing Strimzi Kafka Operator..."
    
    helm repo add strimzi https://strimzi.io/charts/ || true
    helm repo update
    
    helm upgrade --install strimzi-kafka-operator strimzi/strimzi-kafka-operator \
        --namespace kafka \
        --set watchNamespaces="{mineralvision}" \
        --wait --timeout 5m || log_warn "Strimzi operator installation may have issues"
}

deploy_kafka() {
    log_info "Deploying Kafka HA cluster..."
    kubectl apply -f "$SCRIPT_DIR/kafka/kafka-ha.yaml"
    wait_for_pods "mineralvision" "strimzi.io/name=mineralvision-kafka-kafka" 600
}

deploy_redis() {
    log_info "Deploying Redis HA cluster..."
    kubectl apply -f "$SCRIPT_DIR/redis/redis-ha.yaml"
    wait_for_pods "mineralvision" "app=redis" 300
}

deploy_temporal() {
    log_info "Deploying Temporal HA cluster..."
    kubectl apply -f "$SCRIPT_DIR/temporal/temporal-ha.yaml"
    wait_for_pods "temporal" "app=temporal" 300
}

deploy_keycloak() {
    log_info "Deploying Keycloak HA cluster..."
    kubectl apply -f "$SCRIPT_DIR/keycloak/keycloak-ha.yaml"
    wait_for_pods "keycloak" "app=keycloak" 300
}

deploy_apisix() {
    log_info "Deploying APISIX HA cluster..."
    kubectl apply -f "$SCRIPT_DIR/apisix/apisix-ha.yaml"
    wait_for_pods "apisix" "app=apisix" 300
    wait_for_pods "apisix" "app=etcd" 300
}

install_dapr() {
    log_info "Installing Dapr..."
    
    helm repo add dapr https://dapr.github.io/helm-charts/ || true
    helm repo update
    
    helm upgrade --install dapr dapr/dapr \
        --namespace dapr-system \
        --set global.ha.enabled=true \
        --set global.ha.replicaCount=3 \
        --wait --timeout 5m || log_warn "Dapr installation may have issues"
    
    kubectl apply -f "$SCRIPT_DIR/dapr/dapr-ha.yaml"
}

deploy_fluvio() {
    log_info "Deploying Fluvio HA cluster..."
    kubectl apply -f "$SCRIPT_DIR/fluvio/fluvio-ha.yaml"
    wait_for_pods "fluvio" "app=fluvio-sc" 300
    wait_for_pods "fluvio" "app=fluvio-spu" 300
}

deploy_permify() {
    log_info "Deploying Permify HA cluster..."
    kubectl apply -f "$SCRIPT_DIR/permify/permify-ha.yaml"
    wait_for_pods "permify" "app=permify" 300
}

deploy_tigerbeetle() {
    log_info "Deploying TigerBeetle HA cluster..."
    kubectl apply -f "$SCRIPT_DIR/tigerbeetle/tigerbeetle-ha.yaml"
    wait_for_pods "tigerbeetle" "app=tigerbeetle" 300
}

deploy_lakehouse() {
    log_info "Deploying Lakehouse HA infrastructure..."
    kubectl apply -f "$SCRIPT_DIR/lakehouse/lakehouse-ha.yaml"
    wait_for_pods "lakehouse" "app=minio" 300
    wait_for_pods "lakehouse" "app=trino" 300
}

deploy_openappsec() {
    log_info "Deploying OpenAppSec HA cluster..."
    kubectl apply -f "$SCRIPT_DIR/openappsec/openappsec-ha.yaml"
    wait_for_pods "openappsec" "app=openappsec-learning" 300
}

deploy_kubernetes_configs() {
    log_info "Applying Kubernetes HA configurations..."
    kubectl apply -f "$SCRIPT_DIR/kubernetes/kubernetes-ha.yaml"
}

verify_deployment() {
    log_info "Verifying deployment status..."
    
    echo ""
    echo "=============================================="
    echo "Deployment Status Summary"
    echo "=============================================="
    
    namespaces=(
        "mineralvision"
        "kafka"
        "temporal"
        "keycloak"
        "apisix"
        "dapr-system"
        "fluvio"
        "permify"
        "tigerbeetle"
        "lakehouse"
        "openappsec"
    )
    
    for ns in "${namespaces[@]}"; do
        echo ""
        echo "--- Namespace: $ns ---"
        kubectl get pods -n "$ns" --no-headers 2>/dev/null | head -10 || echo "No pods found"
    done
    
    echo ""
    echo "=============================================="
    echo "Services"
    echo "=============================================="
    kubectl get svc -A | grep -E "(LoadBalancer|NodePort)" | head -20
    
    echo ""
    log_info "Deployment verification complete"
}

print_endpoints() {
    echo ""
    echo "=============================================="
    echo "Service Endpoints"
    echo "=============================================="
    echo ""
    echo "Kafka Bootstrap: mineralvision-kafka-kafka-bootstrap.mineralvision:9092"
    echo "Redis Master: redis-master.mineralvision:6379"
    echo "Redis Sentinel: redis-sentinel.mineralvision:26379"
    echo "Temporal Frontend: temporal-frontend.temporal:7233"
    echo "Temporal UI: temporal-ui.temporal:8080"
    echo "Keycloak: keycloak.keycloak:8080"
    echo "APISIX Gateway: apisix-gateway.apisix:80"
    echo "APISIX Admin: apisix-admin.apisix:9180"
    echo "Fluvio SC: fluvio-sc.fluvio:9003"
    echo "Permify: permify.permify:3476"
    echo "TigerBeetle: tigerbeetle.tigerbeetle:3000"
    echo "MinIO: minio.lakehouse:9000"
    echo "MinIO Console: minio.lakehouse:9001"
    echo "Trino: trino-coordinator.lakehouse:8080"
    echo "Spark History: spark-history-server.lakehouse:18080"
    echo ""
}

# Main deployment flow
main() {
    local component=${1:-all}
    
    check_prerequisites
    create_namespaces
    
    case $component in
        all)
            install_strimzi_operator
            deploy_kafka
            deploy_redis
            deploy_temporal
            deploy_keycloak
            deploy_apisix
            install_dapr
            deploy_fluvio
            deploy_permify
            deploy_tigerbeetle
            deploy_lakehouse
            deploy_openappsec
            deploy_kubernetes_configs
            ;;
        kafka)
            install_strimzi_operator
            deploy_kafka
            ;;
        redis)
            deploy_redis
            ;;
        temporal)
            deploy_temporal
            ;;
        keycloak)
            deploy_keycloak
            ;;
        apisix)
            deploy_apisix
            ;;
        dapr)
            install_dapr
            ;;
        fluvio)
            deploy_fluvio
            ;;
        permify)
            deploy_permify
            ;;
        tigerbeetle)
            deploy_tigerbeetle
            ;;
        lakehouse)
            deploy_lakehouse
            ;;
        openappsec)
            deploy_openappsec
            ;;
        kubernetes)
            deploy_kubernetes_configs
            ;;
        *)
            log_error "Unknown component: $component"
            echo "Usage: $0 [all|kafka|redis|temporal|keycloak|apisix|dapr|fluvio|permify|tigerbeetle|lakehouse|openappsec|kubernetes]"
            exit 1
            ;;
    esac
    
    verify_deployment
    print_endpoints
    
    echo ""
    log_info "MineralVision HA Infrastructure deployment complete!"
}

main "$@"
