"""
MineralVision Test Configuration

This module provides pytest fixtures and configuration for the MineralVision test suite.
"""

import os
import shutil
import sys
import tempfile
from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Add source directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MineralVision_Enhanced', 'lakehouse_architecture'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MineralVision_Final_Package', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MineralVision_WALDO_Production_Package', 'src'))


# Test configuration
TEST_CONFIG = {
    "database_url": "postgresql://test:test@localhost:5432/test_mineralvision",
    "redis_url": "redis://localhost:6379/0",
    "kafka_bootstrap_servers": "localhost:9092",
    # Object storage (RustFS/S3-compatible)
    "object_store_backend": "local",  # Use local for testing
    "s3_endpoint": "localhost:9000",
    "s3_access_key": "mineralvision",
    "s3_secret_key": "mineralvision_test",
    # Legacy MinIO config (deprecated)
    "minio_endpoint": "localhost:9000",
    "minio_access_key": "minioadmin",
    "minio_secret_key": "minioadmin",
}


@pytest.fixture(scope="session")
def test_config() -> dict[str, Any]:
    """Provide test configuration."""
    return TEST_CONFIG.copy()


@pytest.fixture(scope="session")
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp(prefix="mineralvision_test_")
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_sensor_data() -> dict[str, Any]:
    """Provide sample sensor data for testing."""
    return {
        "sensor_type": "hyperspectral",
        "data": {
            "values": [float(i) * 0.1 for i in range(100)],
            "coordinates": {
                "lat": -23.5,
                "lon": 119.5,
            },
            "timestamp": datetime.utcnow().isoformat(),
        },
        "metadata": {
            "source": "test-sensor",
            "quality": 0.95,
        },
    }


@pytest.fixture
def sample_region() -> dict[str, float]:
    """Provide sample region for testing."""
    return {
        "min_lat": -24.0,
        "max_lat": -23.0,
        "min_lon": 119.0,
        "max_lon": 120.0,
    }


@pytest.fixture
def sample_mineral_deposit() -> dict[str, Any]:
    """Provide sample mineral deposit data."""
    return {
        "name": "Test Deposit",
        "mineral_type": "gold",
        "probability": 0.85,
        "latitude": -23.5,
        "longitude": 119.5,
        "estimated_tonnage": 1000000,
        "grade": 2.5,
        "metadata": {
            "discovery_date": "2024-01-15",
            "confidence_level": "high",
        },
    }


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for testing."""
    with patch('kafka.KafkaProducer') as mock:
        producer = MagicMock()
        mock.return_value = producer
        yield producer


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    with patch('redis.Redis') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_minio_client():
    """Mock MinIO client for testing (legacy)."""
    with patch('minio.Minio') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_object_store():
    """Mock object store client for testing."""
    with patch('boto3.client') as mock_client, \
         patch('boto3.resource') as mock_resource:
        client = MagicMock()
        resource = MagicMock()
        mock_client.return_value = client
        mock_resource.return_value = resource
        yield client


@pytest.fixture
def local_object_store(temp_dir):
    """Create a local object store for testing."""
    from api.core.object_storage import LocalStorageClient, ObjectStoreConfig, StorageBackendType
    config = ObjectStoreConfig(backend=StorageBackendType.LOCAL)
    return LocalStorageClient(config, base_path=temp_dir)


@pytest.fixture
def mock_postgresql():
    """Mock PostgreSQL connection for testing."""
    with patch('psycopg2.connect') as mock:
        conn = MagicMock()
        mock.return_value = conn
        yield conn


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "performance: Performance tests")


# Test reporting (requires pytest-html plugin)
# def pytest_html_report_title(report):
#     """Set HTML report title."""
#     report.title = "MineralVision Test Report"


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Add extra information to test reports."""
    if call.when == "call":
        # Add test duration
        item.user_properties.append(("duration", call.duration))
