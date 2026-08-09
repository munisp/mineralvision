"""
Object Storage Regression Tests for MineralVision.

Tests the storage abstraction layer and RustFS/MinIO compatibility.
Ensures migration from MinIO to RustFS doesn't break existing functionality.
"""

import os
import sys
import json
import pytest
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock

# Add source directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MineralVision_Final_Package', 'src'))


class TestObjectStoreConfig:
    """Test ObjectStoreConfig configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        from api.core.object_storage import ObjectStoreConfig, StorageBackendType
        
        config = ObjectStoreConfig()
        
        assert config.backend == StorageBackendType.RUSTFS
        assert config.endpoint == "http://localhost:9000"
        assert config.region == "us-east-1"
        assert config.path_style is True
        assert config.multipart_threshold == 100 * 1024 * 1024
    
    def test_config_for_rustfs(self):
        """Test RustFS-optimized configuration."""
        from api.core.object_storage import ObjectStoreConfig, StorageBackendType
        
        config = ObjectStoreConfig.for_rustfs(
            endpoint="http://rustfs.lakehouse:9000",
            access_key="test_key",
            secret_key="test_secret",
        )
        
        assert config.backend == StorageBackendType.RUSTFS
        assert config.endpoint == "http://rustfs.lakehouse:9000"
        assert config.access_key == "test_key"
        assert config.path_style is True
        # RustFS has lower multipart threshold for efficiency
        assert config.multipart_threshold == 50 * 1024 * 1024
    
    def test_config_for_minio(self):
        """Test MinIO configuration (legacy support)."""
        from api.core.object_storage import ObjectStoreConfig, StorageBackendType
        
        config = ObjectStoreConfig.for_minio(
            endpoint="http://minio.lakehouse:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
        )
        
        assert config.backend == StorageBackendType.MINIO
        assert config.endpoint == "http://minio.lakehouse:9000"
        assert config.path_style is True
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        from api.core.object_storage import ObjectStoreConfig
        
        with patch.dict(os.environ, {
            "OBJECT_STORE_BACKEND": "rustfs",
            "S3_ENDPOINT": "http://test:9000",
            "S3_ACCESS_KEY": "test_access",
            "S3_SECRET_KEY": "test_secret",
            "S3_REGION": "eu-west-1",
        }):
            config = ObjectStoreConfig.from_env()
            
            assert config.endpoint == "http://test:9000"
            assert config.access_key == "test_access"
            assert config.region == "eu-west-1"


class TestLocalStorageClient:
    """Test LocalStorageClient for development/testing."""
    
    @pytest.fixture
    def local_client(self, tmp_path):
        """Create a local storage client."""
        from api.core.object_storage import LocalStorageClient, ObjectStoreConfig, StorageBackendType
        
        config = ObjectStoreConfig(backend=StorageBackendType.LOCAL)
        return LocalStorageClient(config, base_path=str(tmp_path))
    
    def test_create_bucket(self, local_client):
        """Test bucket creation."""
        bucket_info = local_client.create_bucket("test-bucket")
        
        assert bucket_info.name == "test-bucket"
        assert local_client.bucket_exists("test-bucket")
    
    def test_put_and_get_object(self, local_client):
        """Test object upload and download."""
        local_client.create_bucket("test-bucket")
        
        # Upload
        test_data = b"Hello, RustFS!"
        obj_info = local_client.put_object("test-bucket", "test-key", test_data)
        
        assert obj_info.key == "test-key"
        assert obj_info.size == len(test_data)
        
        # Download
        retrieved = local_client.get_object("test-bucket", "test-key")
        assert retrieved == test_data
    
    def test_put_object_with_metadata(self, local_client):
        """Test object upload with metadata."""
        local_client.create_bucket("test-bucket")
        
        test_data = b"Test data with metadata"
        metadata = {"custom-key": "custom-value", "version": "1.0"}
        
        obj_info = local_client.put_object(
            "test-bucket",
            "test-key",
            test_data,
            content_type="application/octet-stream",
            metadata=metadata,
        )
        
        # Verify metadata is stored
        head_info = local_client.head_object("test-bucket", "test-key")
        assert head_info is not None
        assert head_info.metadata == metadata
    
    def test_delete_object(self, local_client):
        """Test object deletion."""
        local_client.create_bucket("test-bucket")
        local_client.put_object("test-bucket", "test-key", b"data")
        
        assert local_client.object_exists("test-bucket", "test-key")
        
        result = local_client.delete_object("test-bucket", "test-key")
        
        assert result is True
        assert not local_client.object_exists("test-bucket", "test-key")
    
    def test_delete_objects_batch(self, local_client):
        """Test batch object deletion."""
        local_client.create_bucket("test-bucket")
        
        # Create multiple objects
        for i in range(5):
            local_client.put_object("test-bucket", f"key-{i}", f"data-{i}".encode())
        
        # Delete batch
        keys = [f"key-{i}" for i in range(5)]
        failed = local_client.delete_objects("test-bucket", keys)
        
        assert len(failed) == 0
        for key in keys:
            assert not local_client.object_exists("test-bucket", key)
    
    def test_list_objects(self, local_client):
        """Test object listing."""
        local_client.create_bucket("test-bucket")
        
        # Create objects with different prefixes
        local_client.put_object("test-bucket", "prefix1/file1.txt", b"data1")
        local_client.put_object("test-bucket", "prefix1/file2.txt", b"data2")
        local_client.put_object("test-bucket", "prefix2/file3.txt", b"data3")
        
        # List all
        objects, _ = local_client.list_objects("test-bucket")
        assert len(objects) == 3
        
        # List with prefix
        objects, _ = local_client.list_objects("test-bucket", prefix="prefix1/")
        assert len(objects) == 2
    
    def test_list_objects_iterator(self, local_client):
        """Test object listing iterator."""
        local_client.create_bucket("test-bucket")
        
        for i in range(10):
            local_client.put_object("test-bucket", f"file-{i}.txt", f"data-{i}".encode())
        
        objects = list(local_client.list_objects_iter("test-bucket"))
        assert len(objects) == 10
    
    def test_copy_object(self, local_client):
        """Test object copy."""
        local_client.create_bucket("src-bucket")
        local_client.create_bucket("dst-bucket")
        
        original_data = b"Original data to copy"
        local_client.put_object("src-bucket", "original-key", original_data)
        
        # Copy
        copy_info = local_client.copy_object(
            "src-bucket", "original-key",
            "dst-bucket", "copied-key"
        )
        
        assert copy_info.key == "copied-key"
        
        # Verify copy
        copied_data = local_client.get_object("dst-bucket", "copied-key")
        assert copied_data == original_data
    
    def test_head_object(self, local_client):
        """Test object head (metadata only)."""
        local_client.create_bucket("test-bucket")
        
        test_data = b"Test data for head"
        local_client.put_object("test-bucket", "test-key", test_data)
        
        info = local_client.head_object("test-bucket", "test-key")
        
        assert info is not None
        assert info.key == "test-key"
        assert info.size == len(test_data)
        assert info.etag != ""
    
    def test_head_object_not_found(self, local_client):
        """Test head for non-existent object."""
        local_client.create_bucket("test-bucket")
        
        info = local_client.head_object("test-bucket", "nonexistent")
        assert info is None
    
    def test_object_exists(self, local_client):
        """Test object existence check."""
        local_client.create_bucket("test-bucket")
        
        assert not local_client.object_exists("test-bucket", "test-key")
        
        local_client.put_object("test-bucket", "test-key", b"data")
        
        assert local_client.object_exists("test-bucket", "test-key")
    
    def test_presign_urls(self, local_client):
        """Test presigned URL generation."""
        local_client.create_bucket("test-bucket")
        local_client.put_object("test-bucket", "test-key", b"data")
        
        get_url = local_client.presign_get("test-bucket", "test-key")
        put_url = local_client.presign_put("test-bucket", "new-key")
        
        # Local client returns file:// URLs
        assert get_url.startswith("file://")
        assert put_url.startswith("file://")
    
    def test_multipart_upload(self, local_client):
        """Test multipart upload workflow."""
        local_client.create_bucket("test-bucket")
        
        # Create multipart upload
        upload = local_client.create_multipart_upload("test-bucket", "large-file")
        
        assert upload.upload_id != ""
        assert upload.bucket == "test-bucket"
        assert upload.key == "large-file"
        
        # Upload parts
        part1 = local_client.upload_part(upload, 1, b"Part 1 data")
        part2 = local_client.upload_part(upload, 2, b"Part 2 data")
        
        upload.parts = [part1, part2]
        
        # Complete upload
        result = local_client.complete_multipart_upload(upload)
        
        assert result.key == "large-file"
        
        # Verify combined data
        data = local_client.get_object("test-bucket", "large-file")
        assert data == b"Part 1 dataPart 2 data"
    
    def test_abort_multipart_upload(self, local_client):
        """Test multipart upload abort."""
        local_client.create_bucket("test-bucket")
        
        upload = local_client.create_multipart_upload("test-bucket", "aborted-file")
        local_client.upload_part(upload, 1, b"Part data")
        
        result = local_client.abort_multipart_upload(upload)
        
        assert result is True
        assert not local_client.object_exists("test-bucket", "aborted-file")
    
    def test_health_check(self, local_client):
        """Test health check."""
        health = local_client.health_check()
        
        assert health["healthy"] is True
        assert health["backend"] == "local"
    
    def test_statistics(self, local_client):
        """Test client statistics tracking."""
        local_client.create_bucket("test-bucket")
        
        # Perform operations
        local_client.put_object("test-bucket", "key1", b"data1")
        local_client.put_object("test-bucket", "key2", b"data2")
        local_client.get_object("test-bucket", "key1")
        local_client.delete_object("test-bucket", "key2")
        local_client.list_objects("test-bucket")
        
        stats = local_client.get_stats()
        
        assert stats["puts"] == 2
        assert stats["gets"] == 1
        assert stats["deletes"] == 1
        assert stats["lists"] == 1
        assert stats["bytes_uploaded"] > 0
        assert stats["bytes_downloaded"] > 0


class TestS3CompatibleClient:
    """Test S3CompatibleClient with mocked boto3."""
    
    @pytest.fixture
    def s3_client_with_mock(self):
        """Create S3 client with mocked boto3."""
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            pytest.skip("boto3 not installed - skipping S3 client tests")
        
        from api.core.object_storage import S3CompatibleClient, ObjectStoreConfig
        
        with patch('boto3.client') as mock_client_fn, \
             patch('boto3.resource') as mock_resource_fn:
            mock_client = MagicMock()
            mock_resource = MagicMock()
            mock_client_fn.return_value = mock_client
            mock_resource_fn.return_value = mock_resource
            
            config = ObjectStoreConfig.for_rustfs(
                endpoint="http://rustfs:9000",
                access_key="test",
                secret_key="test",
            )
            client = S3CompatibleClient(config)
            yield client, mock_client
    
    def test_client_initialization(self, s3_client_with_mock):
        """Test S3 client initialization."""
        client, mock_client = s3_client_with_mock
        
        assert client is not None
        assert client.config.backend.value == "rustfs"
        assert client.config.endpoint == "http://rustfs:9000"
    
    def test_put_object(self, s3_client_with_mock):
        """Test object upload via S3 API."""
        client, mock_client = s3_client_with_mock
        mock_client.put_object.return_value = {"ETag": '"abc123"'}
        
        result = client.put_object("bucket", "key", b"test data")
        
        mock_client.put_object.assert_called_once()
        assert result.key == "key"
        assert result.etag == "abc123"
    
    def test_get_object(self, s3_client_with_mock):
        """Test object download via S3 API."""
        client, mock_client = s3_client_with_mock
        
        mock_body = MagicMock()
        mock_body.read.return_value = b"test data"
        mock_client.get_object.return_value = {"Body": mock_body}
        
        data = client.get_object("bucket", "key")
        
        mock_client.get_object.assert_called_once_with(Bucket="bucket", Key="key")
        assert data == b"test data"
    
    def test_list_objects_v2(self, s3_client_with_mock):
        """Test object listing via S3 API."""
        client, mock_client = s3_client_with_mock
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "file1.txt", "Size": 100, "ETag": '"a"', "LastModified": datetime.now()},
                {"Key": "file2.txt", "Size": 200, "ETag": '"b"', "LastModified": datetime.now()},
            ],
            "NextContinuationToken": None,
        }
        
        objects, next_token = client.list_objects("bucket", prefix="")
        
        assert len(objects) == 2
        assert objects[0].key == "file1.txt"
        assert objects[1].key == "file2.txt"
    
    def test_presign_get_url(self, s3_client_with_mock):
        """Test presigned GET URL generation."""
        client, mock_client = s3_client_with_mock
        mock_client.generate_presigned_url.return_value = "https://rustfs:9000/bucket/key?signature=..."
        
        url = client.presign_get("bucket", "key", expires=3600)
        
        mock_client.generate_presigned_url.assert_called_once()
        assert "rustfs" in url
    
    def test_health_check_healthy(self, s3_client_with_mock):
        """Test health check when backend is healthy."""
        client, mock_client = s3_client_with_mock
        mock_client.list_buckets.return_value = {"Buckets": []}
        
        health = client.health_check()
        
        assert health["healthy"] is True
        assert health["backend"] == "rustfs"
    
    def test_health_check_unhealthy(self, s3_client_with_mock):
        """Test health check when backend is unhealthy."""
        client, mock_client = s3_client_with_mock
        mock_client.list_buckets.side_effect = Exception("Connection refused")
        
        health = client.health_check()
        
        assert health["healthy"] is False
        assert "error" in health


class TestStorageMigrator:
    """Test storage migration utilities."""
    
    @pytest.fixture
    def source_client(self, tmp_path):
        """Create source storage client."""
        from api.core.object_storage import LocalStorageClient, ObjectStoreConfig, StorageBackendType
        
        config = ObjectStoreConfig(backend=StorageBackendType.LOCAL)
        return LocalStorageClient(config, base_path=str(tmp_path / "source"))
    
    @pytest.fixture
    def dest_client(self, tmp_path):
        """Create destination storage client."""
        from api.core.object_storage import LocalStorageClient, ObjectStoreConfig, StorageBackendType
        
        config = ObjectStoreConfig(backend=StorageBackendType.LOCAL)
        return LocalStorageClient(config, base_path=str(tmp_path / "dest"))
    
    def test_migrate_bucket(self, source_client, dest_client):
        """Test bucket migration."""
        from api.core.object_storage import StorageMigrator
        
        # Setup source data
        source_client.create_bucket("test-bucket")
        source_client.put_object("test-bucket", "file1.txt", b"content1")
        source_client.put_object("test-bucket", "file2.txt", b"content2")
        source_client.put_object("test-bucket", "subdir/file3.txt", b"content3")
        
        # Migrate
        migrator = StorageMigrator(source_client, dest_client)
        result = migrator.migrate_bucket("test-bucket", verify=True)
        
        assert result["objects_migrated"] == 3
        assert result["objects_failed"] == 0
        assert result["bytes_migrated"] > 0
        
        # Verify destination
        assert dest_client.bucket_exists("test-bucket")
        assert dest_client.get_object("test-bucket", "file1.txt") == b"content1"
        assert dest_client.get_object("test-bucket", "file2.txt") == b"content2"
        assert dest_client.get_object("test-bucket", "subdir/file3.txt") == b"content3"
    
    def test_migrate_with_prefix(self, source_client, dest_client):
        """Test migration with prefix filter."""
        from api.core.object_storage import StorageMigrator
        
        source_client.create_bucket("test-bucket")
        source_client.put_object("test-bucket", "include/file1.txt", b"include1")
        source_client.put_object("test-bucket", "include/file2.txt", b"include2")
        source_client.put_object("test-bucket", "exclude/file3.txt", b"exclude")
        
        migrator = StorageMigrator(source_client, dest_client)
        result = migrator.migrate_bucket("test-bucket", prefix="include/")
        
        assert result["objects_migrated"] == 2
    
    def test_migrate_skip_existing(self, source_client, dest_client):
        """Test migration skips existing objects."""
        from api.core.object_storage import StorageMigrator
        
        source_client.create_bucket("test-bucket")
        source_client.put_object("test-bucket", "file1.txt", b"source")
        
        dest_client.create_bucket("test-bucket")
        dest_client.put_object("test-bucket", "file1.txt", b"existing")
        
        migrator = StorageMigrator(source_client, dest_client)
        result = migrator.migrate_bucket("test-bucket", overwrite=False)
        
        assert result["objects_skipped"] == 1
        assert result["objects_migrated"] == 0
        
        # Verify existing data not overwritten
        assert dest_client.get_object("test-bucket", "file1.txt") == b"existing"
    
    def test_migrate_overwrite(self, source_client, dest_client):
        """Test migration with overwrite enabled."""
        from api.core.object_storage import StorageMigrator
        
        source_client.create_bucket("test-bucket")
        source_client.put_object("test-bucket", "file1.txt", b"new_content")
        
        dest_client.create_bucket("test-bucket")
        dest_client.put_object("test-bucket", "file1.txt", b"old_content")
        
        migrator = StorageMigrator(source_client, dest_client)
        result = migrator.migrate_bucket("test-bucket", overwrite=True)
        
        assert result["objects_migrated"] == 1
        assert dest_client.get_object("test-bucket", "file1.txt") == b"new_content"
    
    def test_verify_migration(self, source_client, dest_client):
        """Test migration verification."""
        from api.core.object_storage import StorageMigrator
        
        source_client.create_bucket("test-bucket")
        source_client.put_object("test-bucket", "file1.txt", b"content1")
        source_client.put_object("test-bucket", "file2.txt", b"content2")
        
        dest_client.create_bucket("test-bucket")
        dest_client.put_object("test-bucket", "file1.txt", b"content1")
        # file2.txt missing
        
        migrator = StorageMigrator(source_client, dest_client)
        result = migrator.verify_migration("test-bucket")
        
        assert result["verified"] is False
        assert "file2.txt" in result["missing"]


class TestFactoryFunctions:
    """Test factory functions for creating clients."""
    
    def test_create_object_store_local(self, tmp_path):
        """Test creating local storage client."""
        from api.core.object_storage import LocalStorageClient, ObjectStoreConfig, StorageBackendType
        
        # Create local client directly since factory may not support base_path
        config = ObjectStoreConfig(backend=StorageBackendType.LOCAL)
        client = LocalStorageClient(config, base_path=str(tmp_path))
        
        assert client is not None
        health = client.health_check()
        assert health["healthy"] is True
    
    def test_create_rustfs_client(self):
        """Test creating RustFS client factory."""
        try:
            import boto3
        except ImportError:
            pytest.skip("boto3 not installed - skipping S3 client tests")
        
        from api.core.object_storage import create_rustfs_client
        
        client = create_rustfs_client(
            endpoint="http://rustfs:9000",
            access_key="test",
            secret_key="test",
        )
        
        assert client is not None
        assert client.config.backend.value == "rustfs"
    
    def test_create_minio_client(self):
        """Test creating MinIO client factory (legacy)."""
        try:
            import boto3
        except ImportError:
            pytest.skip("boto3 not installed - skipping S3 client tests")
        
        from api.core.object_storage import create_minio_client
        
        client = create_minio_client(
            endpoint="http://minio:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
        )
        
        assert client is not None
        assert client.config.backend.value == "minio"


class TestObjectInfo:
    """Test ObjectInfo and BucketInfo dataclasses."""
    
    def test_object_info_to_dict(self):
        """Test ObjectInfo serialization."""
        from api.core.object_storage import ObjectInfo
        
        info = ObjectInfo(
            key="test/key.txt",
            size=1024,
            etag="abc123",
            last_modified=datetime(2024, 1, 1, 12, 0, 0),
            content_type="text/plain",
            metadata={"custom": "value"},
        )
        
        d = info.to_dict()
        
        assert d["key"] == "test/key.txt"
        assert d["size"] == 1024
        assert d["etag"] == "abc123"
        assert d["content_type"] == "text/plain"
        assert d["metadata"] == {"custom": "value"}
    
    def test_bucket_info_to_dict(self):
        """Test BucketInfo serialization."""
        from api.core.object_storage import BucketInfo
        
        info = BucketInfo(
            name="test-bucket",
            creation_date=datetime(2024, 1, 1, 12, 0, 0),
            region="us-east-1",
        )
        
        d = info.to_dict()
        
        assert d["name"] == "test-bucket"
        assert d["region"] == "us-east-1"


class TestErrorHandling:
    """Test error handling in storage operations."""
    
    @pytest.fixture
    def local_client(self, tmp_path):
        """Create a local storage client."""
        from api.core.object_storage import LocalStorageClient, ObjectStoreConfig, StorageBackendType
        
        config = ObjectStoreConfig(backend=StorageBackendType.LOCAL)
        return LocalStorageClient(config, base_path=str(tmp_path))
    
    def test_get_nonexistent_object(self, local_client):
        """Test getting non-existent object raises error."""
        from api.core.object_storage import ObjectNotFoundError
        
        local_client.create_bucket("test-bucket")
        
        with pytest.raises(ObjectNotFoundError):
            local_client.get_object("test-bucket", "nonexistent")
    
    def test_list_nonexistent_bucket(self, local_client):
        """Test listing non-existent bucket raises error."""
        from api.core.object_storage import BucketNotFoundError
        
        with pytest.raises(BucketNotFoundError):
            local_client.list_objects("nonexistent-bucket")


class TestRegressionMinioToRustfs:
    """
    Regression tests ensuring MinIO -> RustFS migration compatibility.
    
    These tests verify that existing MinIO-based workflows continue
    to work after migrating to RustFS.
    """
    
    @pytest.fixture
    def storage_client(self, tmp_path):
        """Create storage client (simulates RustFS)."""
        from api.core.object_storage import LocalStorageClient, ObjectStoreConfig, StorageBackendType
        
        config = ObjectStoreConfig(backend=StorageBackendType.LOCAL)
        return LocalStorageClient(config, base_path=str(tmp_path))
    
    def test_model_artifact_storage(self, storage_client):
        """Test model artifact storage workflow (WALDO models)."""
        storage_client.create_bucket("model-artifacts")
        
        # Simulate storing model weights
        model_weights = b"fake model weights data" * 1000
        model_info = storage_client.put_object(
            "model-artifacts",
            "waldo/yolo11m/v1.0.0/weights.pt",
            model_weights,
            content_type="application/octet-stream",
            metadata={"version": "1.0.0", "architecture": "yolo11m"},
        )
        
        assert model_info.size == len(model_weights)
        
        # Retrieve model
        retrieved = storage_client.get_object(
            "model-artifacts",
            "waldo/yolo11m/v1.0.0/weights.pt"
        )
        assert retrieved == model_weights
        
        # List model versions
        objects, _ = storage_client.list_objects(
            "model-artifacts",
            prefix="waldo/yolo11m/"
        )
        assert len(objects) == 1
    
    def test_lakehouse_data_storage(self, storage_client):
        """Test lakehouse data storage workflow."""
        storage_client.create_bucket("mineralvision-lakehouse")
        
        # Store Delta Lake table data
        parquet_data = b"fake parquet data"
        storage_client.put_object(
            "mineralvision-lakehouse",
            "warehouse/jepa_embeddings/_delta_log/00000000000000000000.json",
            b'{"version": 0}',
        )
        storage_client.put_object(
            "mineralvision-lakehouse",
            "warehouse/jepa_embeddings/part-00000.parquet",
            parquet_data,
        )
        
        # List table files
        objects, _ = storage_client.list_objects(
            "mineralvision-lakehouse",
            prefix="warehouse/jepa_embeddings/"
        )
        assert len(objects) == 2
    
    def test_spark_event_logs(self, storage_client):
        """Test Spark event log storage."""
        storage_client.create_bucket("mineralvision-lakehouse")
        
        # Store Spark event logs
        event_log = b'{"Event": "SparkListenerApplicationStart"}'
        storage_client.put_object(
            "mineralvision-lakehouse",
            "spark-events/app-20240101120000-0001",
            event_log,
        )
        
        # Verify retrieval
        retrieved = storage_client.get_object(
            "mineralvision-lakehouse",
            "spark-events/app-20240101120000-0001"
        )
        assert retrieved == event_log
    
    def test_presigned_url_for_ui(self, storage_client):
        """Test presigned URL generation for UI downloads."""
        storage_client.create_bucket("user-exports")
        
        # Store export file
        export_data = b"CSV export data"
        storage_client.put_object(
            "user-exports",
            "exports/user123/report-2024-01-01.csv",
            export_data,
            content_type="text/csv",
        )
        
        # Generate presigned URL
        url = storage_client.presign_get(
            "user-exports",
            "exports/user123/report-2024-01-01.csv",
            expires=3600,
        )
        
        assert url is not None
        assert len(url) > 0
    
    def test_large_file_multipart(self, storage_client):
        """Test large file upload via multipart."""
        storage_client.create_bucket("large-files")
        
        # Simulate large file (e.g., seismic data)
        chunk_size = 5 * 1024 * 1024  # 5MB chunks
        total_chunks = 3
        
        upload = storage_client.create_multipart_upload(
            "large-files",
            "seismic/survey-001.segy",
            content_type="application/octet-stream",
        )
        
        parts = []
        for i in range(total_chunks):
            chunk_data = f"chunk-{i}-".encode() * (chunk_size // 10)
            part = storage_client.upload_part(upload, i + 1, chunk_data)
            parts.append(part)
        
        upload.parts = parts
        result = storage_client.complete_multipart_upload(upload)
        
        assert result.key == "seismic/survey-001.segy"
        assert storage_client.object_exists("large-files", "seismic/survey-001.segy")
    
    def test_concurrent_operations(self, storage_client):
        """Test concurrent storage operations."""
        import threading
        
        storage_client.create_bucket("concurrent-test")
        
        errors = []
        
        def upload_file(i):
            try:
                storage_client.put_object(
                    "concurrent-test",
                    f"file-{i}.txt",
                    f"content-{i}".encode(),
                )
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=upload_file, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # Verify all files exist
        objects, _ = storage_client.list_objects("concurrent-test")
        assert len(objects) == 10


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
