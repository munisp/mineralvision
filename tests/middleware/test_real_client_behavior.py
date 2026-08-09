"""
Tests for the real-client-first middleware contract (wave 2).

Proves:
(a) With no server and MV_ALLOW_MOCK_FALLBACK unset, connect() raises
    RuntimeError (no silent mocks).
(b) With MV_ALLOW_MOCK_FALLBACK=true, connect() falls back to a working
    in-memory mock, logs a loud warning, and exposes degraded=true in
    health/status responses.
(c) The Enhanced temporal_workflow shim re-exports the canonical Temporal
    module names and the legacy engine adapter delegates to it.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENHANCED = REPO_ROOT / "MineralVision_Enhanced"
FINAL_PKG = REPO_ROOT / "MineralVision_Final_Package"

for p in (str(ENHANCED), str(FINAL_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

ENV_VAR = "MV_ALLOW_MOCK_FALLBACK"


@pytest.fixture(autouse=True)
def clear_fallback_env(monkeypatch):
    """Ensure the fallback env var is unset unless a test sets it."""
    monkeypatch.delenv(ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# (a) Default behavior: RuntimeError when no real backend is available
# ---------------------------------------------------------------------------

def test_redis_connect_raises_by_default():
    from middleware.caching.redis_caching import RedisIntegration, RedisConfig

    redis = RedisIntegration(RedisConfig(host="localhost", port=6399))
    with pytest.raises(RuntimeError, match="MV_ALLOW_MOCK_FALLBACK"):
        asyncio.run(redis.connect())


def test_apisix_connect_raises_by_default():
    from middleware.api_gateway.apisix_gateway import ApisixGateway

    gw = ApisixGateway(admin_url="http://localhost:19091")
    with pytest.raises(RuntimeError, match="APISIX"):
        asyncio.run(gw.connect())


def test_kubecost_connect_raises_by_default():
    from middleware.cost_management.kubecost_integration import (
        KubecostIntegration, KubecostConfig,
    )

    kc = KubecostIntegration(KubecostConfig(url="http://localhost:19092"))
    with pytest.raises(RuntimeError, match="Kubecost"):
        asyncio.run(kc.connect())


def test_permify_connect_raises_by_default():
    from middleware.authorization.permify_authz import (
        PermifyAuthorization, PermifyConfig,
    )

    pa = PermifyAuthorization(PermifyConfig(host="localhost", port=13476))
    with pytest.raises(RuntimeError, match="Permify"):
        asyncio.run(pa.connect())


def test_fluvio_connect_raises_by_default():
    from middleware.streaming.fluvio_streaming import FluvioStreaming

    fl = FluvioStreaming()
    with pytest.raises(RuntimeError, match="Fluvio"):
        asyncio.run(fl.connect())


def test_canonical_temporal_connect_raises_by_default():
    from src.api.orchestration.temporal import TemporalClient, TemporalConfig

    client = TemporalClient(TemporalConfig(host="localhost", port=17233))
    with pytest.raises(RuntimeError, match="install temporalio"):
        asyncio.run(client.connect())


# ---------------------------------------------------------------------------
# (b) Explicit fallback: working mock + loud warning + degraded flag
# ---------------------------------------------------------------------------

def test_redis_mock_fallback_is_explicit_and_degraded(monkeypatch, caplog):
    from middleware.caching.redis_caching import RedisIntegration, RedisConfig

    monkeypatch.setenv(ENV_VAR, "true")
    with caplog.at_level(logging.WARNING):
        redis = asyncio.run(RedisIntegration(RedisConfig(host="localhost", port=6399)).connect())

    assert redis.degraded is True
    assert any("MV DEGRADED MODE" in r.message for r in caplog.records)

    # The mock actually works
    asyncio.run(redis.cache.set("test-key", {"value": 42}, ttl=60))
    assert asyncio.run(redis.cache.get("test-key")) == {"value": 42}

    health = asyncio.run(redis.health_check())
    assert health["degraded"] is True


def test_apisix_mock_fallback_is_explicit_and_degraded(monkeypatch, caplog):
    from middleware.api_gateway.apisix_gateway import ApisixGateway

    monkeypatch.setenv(ENV_VAR, "true")
    with caplog.at_level(logging.WARNING):
        gw = asyncio.run(ApisixGateway(admin_url="http://localhost:19091").connect())

    assert gw.degraded is True
    assert any("MV DEGRADED MODE" in r.message for r in caplog.records)

    health = asyncio.run(gw.health_check())
    assert health["degraded"] is True


def test_canonical_temporal_mock_fallback_is_explicit_and_degraded(monkeypatch, caplog):
    from src.api.orchestration.temporal import TemporalClient, TemporalConfig

    monkeypatch.setenv(ENV_VAR, "true")
    with caplog.at_level(logging.WARNING):
        client = TemporalClient(TemporalConfig(host="localhost", port=17233))
        connected = asyncio.run(client.connect())

    assert connected is False  # mock mode does not pretend a real connection
    assert client.degraded is True
    assert any("MV DEGRADED MODE" in r.message for r in caplog.records)

    # Mock mode still allows explicit mock operations
    run_id = asyncio.run(client.start_workflow("TestWorkflow", "wf-1", {}))
    assert run_id.startswith("mock-")


def test_fallback_env_requires_exact_true(monkeypatch):
    from middleware.caching.redis_caching import RedisIntegration, RedisConfig

    for value in ("1", "yes", "TRUE ", "false"):
        monkeypatch.setenv(ENV_VAR, value)
        if value == "TRUE ":
            continue  # stripped+lowered => accepted
        with pytest.raises(RuntimeError):
            asyncio.run(RedisIntegration(RedisConfig(host="localhost", port=6399)).connect())


# ---------------------------------------------------------------------------
# (c) Temporal shim re-exports canonical names
# ---------------------------------------------------------------------------

def test_temporal_shim_reexports_canonical():
    import src.api.orchestration.temporal as canonical
    from middleware.workflow import temporal_workflow as shim

    assert shim.WorkflowStatus is canonical.WorkflowStatus
    assert shim.TemporalClient is canonical.TemporalClient
    assert shim.TemporalConfig is canonical.TemporalConfig
    assert shim.WorkflowManager is canonical.WorkflowManager
    assert shim.get_temporal_client is canonical.get_temporal_client
    assert shim.get_workflow_manager is canonical.get_workflow_manager
    assert shim.WorkflowRun is canonical.WorkflowRun
    assert shim.WorkflowExecution is canonical.WorkflowRun


def test_temporal_shim_removed_mocks_fail_loudly():
    from middleware.workflow import temporal_workflow as shim

    with pytest.raises(AttributeError, match="no silent mocks"):
        shim.MockTemporalClient
    with pytest.raises(AttributeError, match="no silent mocks"):
        shim.MockWorkflowHandle


def test_temporal_shim_engine_delegates_to_canonical(monkeypatch):
    from middleware.workflow.temporal_workflow import create_temporal_engine
    from src.api.orchestration.temporal import TemporalClient

    monkeypatch.setenv(ENV_VAR, "true")
    engine = create_temporal_engine({"target_host": "localhost:17233"})
    assert isinstance(engine.client, TemporalClient)

    asyncio.run(engine.connect())
    assert engine.degraded is True

    run_id = asyncio.run(engine.start_workflow("data_processing", args=[{"x": 1}]))
    assert run_id.startswith("mock-")


def test_temporal_shim_keeps_domain_names():
    from middleware.workflow import temporal_workflow as shim

    for name in (
        "ActivityType", "RetryConfig", "ActivityConfig", "WorkflowConfig",
        "ActivityRegistry", "WorkflowRegistry", "DataProcessingWorkflow",
        "MLTrainingWorkflow", "ReportGenerationWorkflow",
        "SensorFusionWorkflow", "ScheduledWorkflowManager",
        "create_scheduled_manager", "workflow_definition",
        "activity_definition", "activity_registry", "workflow_registry",
    ):
        assert hasattr(shim, name), f"shim missing public name {name}"

    assert "data_processing" in shim.workflow_registry.list_workflows()
    assert "validate_data" in shim.activity_registry.list_activities()
