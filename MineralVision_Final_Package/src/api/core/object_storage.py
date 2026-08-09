"""
Object Storage Abstraction Layer for MineralVision.

This module provides a unified interface for object storage backends,
supporting both MinIO and RustFS (and other S3-compatible stores).

Key Features:
- Backend-agnostic API for PUT/GET/DELETE/LIST operations
- Presigned URL generation for direct client access
- Multipart upload support for large files
- Connection pooling and retry logic
- Health checks and metrics
- Seamless migration between MinIO and RustFS

Usage:
    from api.core.object_storage import (
        create_object_store,
        ObjectStoreConfig,
        StorageBackendType,
    )
    
    # Create RustFS-backed store
    store = create_object_store(
        backend=StorageBackendType.RUSTFS,
        endpoint="http://rustfs.lakehouse:9000",
        access_key="...",
        secret_key="...",
    )
    
    # Upload object
    store.put_object("bucket", "key", data)
    
    # Download object
    data = store.get_object("bucket", "key")
    
    # Generate presigned URL
    url = store.presign_get("bucket", "key", expires=3600)
"""

import io
import os
import json
import time
import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlencode, urlparse
import hmac
import base64

logger = logging.getLogger(__name__)


class StorageBackendType(Enum):
    """Supported storage backend types."""
    MINIO = "minio"
    RUSTFS = "rustfs"
    AWS_S3 = "aws_s3"
    LOCAL = "local"  # For testing


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class BucketNotFoundError(StorageError):
    """Bucket does not exist."""
    pass


class ObjectNotFoundError(StorageError):
    """Object does not exist."""
    pass


class AccessDeniedError(StorageError):
    """Access denied to resource."""
    pass


class ConnectionError(StorageError):
    """Failed to connect to storage backend."""
    pass


@dataclass
class ObjectStoreConfig:
    """Configuration for object storage connection."""
    backend: StorageBackendType = StorageBackendType.RUSTFS
    endpoint: str = "http://localhost:9000"
    access_key: str = ""
    secret_key: str = ""
    region: str = "us-east-1"
    secure: bool = False
    
    # Connection settings
    connect_timeout: int = 10
    read_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Pool settings
    pool_size: int = 10
    pool_timeout: int = 30
    
    # Path style vs virtual host style
    path_style: bool = True  # RustFS/MinIO typically use path style
    
    # Multipart settings
    multipart_threshold: int = 100 * 1024 * 1024  # 100MB
    multipart_chunksize: int = 16 * 1024 * 1024   # 16MB
    
    @classmethod
    def from_env(cls) -> "ObjectStoreConfig":
        """Create config from environment variables."""
        backend_str = os.getenv("OBJECT_STORE_BACKEND", "rustfs").lower()
        backend = StorageBackendType(backend_str) if backend_str in [b.value for b in StorageBackendType] else StorageBackendType.RUSTFS
        
        return cls(
            backend=backend,
            endpoint=os.getenv("S3_ENDPOINT", os.getenv("RUSTFS_ENDPOINT", "http://localhost:9000")),
            access_key=os.getenv("S3_ACCESS_KEY", os.getenv("RUSTFS_ACCESS_KEY", "")),
            secret_key=os.getenv("S3_SECRET_KEY", os.getenv("RUSTFS_SECRET_KEY", "")),
            region=os.getenv("S3_REGION", "us-east-1"),
            secure=os.getenv("S3_SECURE", "false").lower() == "true",
            path_style=os.getenv("S3_PATH_STYLE", "true").lower() == "true",
        )
    
    @classmethod
    def for_rustfs(
        cls,
        endpoint: str = "http://rustfs.lakehouse:9000",
        access_key: str = "",
        secret_key: str = "",
    ) -> "ObjectStoreConfig":
        """Create config optimized for RustFS."""
        return cls(
            backend=StorageBackendType.RUSTFS,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            path_style=True,
            # RustFS handles small objects very efficiently
            multipart_threshold=50 * 1024 * 1024,  # 50MB
        )
    
    @classmethod
    def for_minio(
        cls,
        endpoint: str = "http://minio.lakehouse:9000",
        access_key: str = "",
        secret_key: str = "",
    ) -> "ObjectStoreConfig":
        """Create config for MinIO (legacy support)."""
        return cls(
            backend=StorageBackendType.MINIO,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            path_style=True,
        )


@dataclass
class ObjectInfo:
    """Information about a stored object."""
    key: str
    size: int
    etag: str
    last_modified: datetime
    content_type: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    storage_class: str = "STANDARD"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "size": self.size,
            "etag": self.etag,
            "last_modified": self.last_modified.isoformat(),
            "content_type": self.content_type,
            "metadata": self.metadata,
            "storage_class": self.storage_class,
        }


@dataclass
class BucketInfo:
    """Information about a bucket."""
    name: str
    creation_date: datetime
    region: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "creation_date": self.creation_date.isoformat(),
            "region": self.region,
        }


@dataclass
class MultipartUpload:
    """Multipart upload state."""
    upload_id: str
    bucket: str
    key: str
    parts: List[Dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)


class ObjectStoreClient(ABC):
    """Abstract base class for object storage clients."""
    
    def __init__(self, config: ObjectStoreConfig):
        self.config = config
        self._stats = {
            "puts": 0,
            "gets": 0,
            "deletes": 0,
            "lists": 0,
            "bytes_uploaded": 0,
            "bytes_downloaded": 0,
            "errors": 0,
        }
        self._lock = threading.Lock()
    
    @abstractmethod
    def put_object(
        self,
        bucket: str,
        key: str,
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectInfo:
        """Upload an object to storage."""
        pass
    
    @abstractmethod
    def get_object(self, bucket: str, key: str) -> bytes:
        """Download an object from storage."""
        pass
    
    @abstractmethod
    def get_object_stream(self, bucket: str, key: str) -> BinaryIO:
        """Get a streaming handle to an object."""
        pass
    
    @abstractmethod
    def delete_object(self, bucket: str, key: str) -> bool:
        """Delete an object from storage."""
        pass
    
    @abstractmethod
    def delete_objects(self, bucket: str, keys: List[str]) -> List[str]:
        """Delete multiple objects. Returns list of failed keys."""
        pass
    
    @abstractmethod
    def head_object(self, bucket: str, key: str) -> Optional[ObjectInfo]:
        """Get object metadata without downloading."""
        pass
    
    @abstractmethod
    def object_exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists."""
        pass
    
    @abstractmethod
    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        delimiter: str = "",
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> Tuple[List[ObjectInfo], Optional[str]]:
        """List objects in a bucket. Returns (objects, next_token)."""
        pass
    
    @abstractmethod
    def list_objects_iter(
        self,
        bucket: str,
        prefix: str = "",
    ) -> Iterator[ObjectInfo]:
        """Iterate over all objects with given prefix."""
        pass
    
    @abstractmethod
    def copy_object(
        self,
        src_bucket: str,
        src_key: str,
        dst_bucket: str,
        dst_key: str,
    ) -> ObjectInfo:
        """Copy an object within storage."""
        pass
    
    # Bucket operations
    
    @abstractmethod
    def create_bucket(self, bucket: str, region: Optional[str] = None) -> BucketInfo:
        """Create a new bucket."""
        pass
    
    @abstractmethod
    def delete_bucket(self, bucket: str) -> bool:
        """Delete an empty bucket."""
        pass
    
    @abstractmethod
    def bucket_exists(self, bucket: str) -> bool:
        """Check if a bucket exists."""
        pass
    
    @abstractmethod
    def list_buckets(self) -> List[BucketInfo]:
        """List all buckets."""
        pass
    
    # Presigned URLs
    
    @abstractmethod
    def presign_get(
        self,
        bucket: str,
        key: str,
        expires: int = 3600,
    ) -> str:
        """Generate a presigned URL for GET."""
        pass
    
    @abstractmethod
    def presign_put(
        self,
        bucket: str,
        key: str,
        expires: int = 3600,
        content_type: Optional[str] = None,
    ) -> str:
        """Generate a presigned URL for PUT."""
        pass
    
    # Multipart upload
    
    @abstractmethod
    def create_multipart_upload(
        self,
        bucket: str,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> MultipartUpload:
        """Initiate a multipart upload."""
        pass
    
    @abstractmethod
    def upload_part(
        self,
        upload: MultipartUpload,
        part_number: int,
        data: bytes,
    ) -> Dict[str, Any]:
        """Upload a part of a multipart upload."""
        pass
    
    @abstractmethod
    def complete_multipart_upload(
        self,
        upload: MultipartUpload,
    ) -> ObjectInfo:
        """Complete a multipart upload."""
        pass
    
    @abstractmethod
    def abort_multipart_upload(self, upload: MultipartUpload) -> bool:
        """Abort a multipart upload."""
        pass
    
    # Health and metrics
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check storage backend health."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        with self._lock:
            return self._stats.copy()
    
    def _record_stat(self, stat: str, value: int = 1):
        """Record a statistic."""
        with self._lock:
            self._stats[stat] = self._stats.get(stat, 0) + value


class S3CompatibleClient(ObjectStoreClient):
    """
    S3-compatible client that works with RustFS, MinIO, and AWS S3.
    
    Uses boto3 under the hood but provides a cleaner interface.
    """
    
    def __init__(self, config: ObjectStoreConfig):
        super().__init__(config)
        self._client = None
        self._resource = None
        self._init_client()
    
    def _init_client(self):
        """Initialize boto3 client."""
        try:
            import boto3
            from botocore.config import Config as BotoConfig
            
            boto_config = BotoConfig(
                connect_timeout=self.config.connect_timeout,
                read_timeout=self.config.read_timeout,
                retries={
                    "max_attempts": self.config.max_retries,
                    "mode": "adaptive",
                },
                max_pool_connections=self.config.pool_size,
                s3={
                    "addressing_style": "path" if self.config.path_style else "virtual",
                },
            )
            
            self._client = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region,
                use_ssl=self.config.secure,
                config=boto_config,
            )
            
            self._resource = boto3.resource(
                "s3",
                endpoint_url=self.config.endpoint,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region,
                use_ssl=self.config.secure,
                config=boto_config,
            )
            
            logger.info(f"Initialized S3 client for {self.config.backend.value} at {self.config.endpoint}")
            
        except ImportError:
            raise ImportError("boto3 is required for S3-compatible storage. Install with: pip install boto3")
    
    def put_object(
        self,
        bucket: str,
        key: str,
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectInfo:
        """Upload an object to storage."""
        try:
            # Convert to bytes if needed
            if hasattr(data, "read"):
                body = data.read()
            else:
                body = data
            
            # Check if multipart upload is needed
            if len(body) > self.config.multipart_threshold:
                return self._put_object_multipart(bucket, key, body, content_type, metadata)
            
            kwargs = {
                "Bucket": bucket,
                "Key": key,
                "Body": body,
            }
            
            if content_type:
                kwargs["ContentType"] = content_type
            if metadata:
                kwargs["Metadata"] = metadata
            
            response = self._client.put_object(**kwargs)
            
            self._record_stat("puts")
            self._record_stat("bytes_uploaded", len(body))
            
            return ObjectInfo(
                key=key,
                size=len(body),
                etag=response.get("ETag", "").strip('"'),
                last_modified=datetime.utcnow(),
                content_type=content_type,
                metadata=metadata or {},
            )
            
        except Exception as e:
            self._record_stat("errors")
            logger.error(f"Failed to put object {bucket}/{key}: {e}")
            raise StorageError(f"Failed to upload object: {e}")
    
    def _put_object_multipart(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectInfo:
        """Upload large object using multipart upload."""
        upload = self.create_multipart_upload(bucket, key, content_type, metadata)
        
        try:
            chunk_size = self.config.multipart_chunksize
            parts = []
            
            for i, offset in enumerate(range(0, len(data), chunk_size)):
                part_number = i + 1
                chunk = data[offset:offset + chunk_size]
                part = self.upload_part(upload, part_number, chunk)
                parts.append(part)
                upload.parts = parts
            
            return self.complete_multipart_upload(upload)
            
        except Exception as e:
            self.abort_multipart_upload(upload)
            raise StorageError(f"Multipart upload failed: {e}")
    
    def get_object(self, bucket: str, key: str) -> bytes:
        """Download an object from storage."""
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            data = response["Body"].read()
            
            self._record_stat("gets")
            self._record_stat("bytes_downloaded", len(data))
            
            return data
            
        except self._client.exceptions.NoSuchKey:
            raise ObjectNotFoundError(f"Object not found: {bucket}/{key}")
        except self._client.exceptions.NoSuchBucket:
            raise BucketNotFoundError(f"Bucket not found: {bucket}")
        except Exception as e:
            self._record_stat("errors")
            logger.error(f"Failed to get object {bucket}/{key}: {e}")
            raise StorageError(f"Failed to download object: {e}")
    
    def get_object_stream(self, bucket: str, key: str) -> BinaryIO:
        """Get a streaming handle to an object."""
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            self._record_stat("gets")
            return response["Body"]
        except self._client.exceptions.NoSuchKey:
            raise ObjectNotFoundError(f"Object not found: {bucket}/{key}")
        except Exception as e:
            self._record_stat("errors")
            raise StorageError(f"Failed to stream object: {e}")
    
    def delete_object(self, bucket: str, key: str) -> bool:
        """Delete an object from storage."""
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
            self._record_stat("deletes")
            return True
        except Exception as e:
            self._record_stat("errors")
            logger.error(f"Failed to delete object {bucket}/{key}: {e}")
            return False
    
    def delete_objects(self, bucket: str, keys: List[str]) -> List[str]:
        """Delete multiple objects. Returns list of failed keys."""
        if not keys:
            return []
        
        try:
            # S3 delete_objects accepts up to 1000 keys
            failed = []
            for i in range(0, len(keys), 1000):
                batch = keys[i:i + 1000]
                response = self._client.delete_objects(
                    Bucket=bucket,
                    Delete={
                        "Objects": [{"Key": k} for k in batch],
                        "Quiet": False,
                    },
                )
                
                # Collect errors
                for error in response.get("Errors", []):
                    failed.append(error["Key"])
                
                self._record_stat("deletes", len(batch) - len(response.get("Errors", [])))
            
            return failed
            
        except Exception as e:
            self._record_stat("errors")
            logger.error(f"Failed to delete objects from {bucket}: {e}")
            return keys  # All failed
    
    def head_object(self, bucket: str, key: str) -> Optional[ObjectInfo]:
        """Get object metadata without downloading."""
        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
            
            return ObjectInfo(
                key=key,
                size=response["ContentLength"],
                etag=response.get("ETag", "").strip('"'),
                last_modified=response["LastModified"],
                content_type=response.get("ContentType"),
                metadata=response.get("Metadata", {}),
                storage_class=response.get("StorageClass", "STANDARD"),
            )
            
        except self._client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None
            raise StorageError(f"Failed to head object: {e}")
    
    def object_exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists."""
        return self.head_object(bucket, key) is not None
    
    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        delimiter: str = "",
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> Tuple[List[ObjectInfo], Optional[str]]:
        """List objects in a bucket. Returns (objects, next_token)."""
        try:
            kwargs = {
                "Bucket": bucket,
                "MaxKeys": max_keys,
            }
            
            if prefix:
                kwargs["Prefix"] = prefix
            if delimiter:
                kwargs["Delimiter"] = delimiter
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            
            response = self._client.list_objects_v2(**kwargs)
            
            self._record_stat("lists")
            
            objects = []
            for obj in response.get("Contents", []):
                objects.append(ObjectInfo(
                    key=obj["Key"],
                    size=obj["Size"],
                    etag=obj.get("ETag", "").strip('"'),
                    last_modified=obj["LastModified"],
                    storage_class=obj.get("StorageClass", "STANDARD"),
                ))
            
            next_token = response.get("NextContinuationToken")
            return objects, next_token
            
        except self._client.exceptions.NoSuchBucket:
            raise BucketNotFoundError(f"Bucket not found: {bucket}")
        except Exception as e:
            self._record_stat("errors")
            raise StorageError(f"Failed to list objects: {e}")
    
    def list_objects_iter(
        self,
        bucket: str,
        prefix: str = "",
    ) -> Iterator[ObjectInfo]:
        """Iterate over all objects with given prefix."""
        continuation_token = None
        
        while True:
            objects, continuation_token = self.list_objects(
                bucket,
                prefix=prefix,
                continuation_token=continuation_token,
            )
            
            for obj in objects:
                yield obj
            
            if not continuation_token:
                break
    
    def copy_object(
        self,
        src_bucket: str,
        src_key: str,
        dst_bucket: str,
        dst_key: str,
    ) -> ObjectInfo:
        """Copy an object within storage."""
        try:
            response = self._client.copy_object(
                CopySource={"Bucket": src_bucket, "Key": src_key},
                Bucket=dst_bucket,
                Key=dst_key,
            )
            
            # Get the new object info
            return self.head_object(dst_bucket, dst_key)
            
        except Exception as e:
            self._record_stat("errors")
            raise StorageError(f"Failed to copy object: {e}")
    
    # Bucket operations
    
    def create_bucket(self, bucket: str, region: Optional[str] = None) -> BucketInfo:
        """Create a new bucket."""
        try:
            kwargs = {"Bucket": bucket}
            
            region = region or self.config.region
            if region and region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": region,
                }
            
            self._client.create_bucket(**kwargs)
            
            return BucketInfo(
                name=bucket,
                creation_date=datetime.utcnow(),
                region=region,
            )
            
        except Exception as e:
            raise StorageError(f"Failed to create bucket: {e}")
    
    def delete_bucket(self, bucket: str) -> bool:
        """Delete an empty bucket."""
        try:
            self._client.delete_bucket(Bucket=bucket)
            return True
        except Exception as e:
            logger.error(f"Failed to delete bucket {bucket}: {e}")
            return False
    
    def bucket_exists(self, bucket: str) -> bool:
        """Check if a bucket exists."""
        try:
            self._client.head_bucket(Bucket=bucket)
            return True
        except:
            return False
    
    def list_buckets(self) -> List[BucketInfo]:
        """List all buckets."""
        try:
            response = self._client.list_buckets()
            
            return [
                BucketInfo(
                    name=b["Name"],
                    creation_date=b["CreationDate"],
                )
                for b in response.get("Buckets", [])
            ]
            
        except Exception as e:
            raise StorageError(f"Failed to list buckets: {e}")
    
    # Presigned URLs
    
    def presign_get(
        self,
        bucket: str,
        key: str,
        expires: int = 3600,
    ) -> str:
        """Generate a presigned URL for GET."""
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires,
            )
        except Exception as e:
            raise StorageError(f"Failed to generate presigned GET URL: {e}")
    
    def presign_put(
        self,
        bucket: str,
        key: str,
        expires: int = 3600,
        content_type: Optional[str] = None,
    ) -> str:
        """Generate a presigned URL for PUT."""
        try:
            params = {"Bucket": bucket, "Key": key}
            if content_type:
                params["ContentType"] = content_type
            
            return self._client.generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=expires,
            )
        except Exception as e:
            raise StorageError(f"Failed to generate presigned PUT URL: {e}")
    
    # Multipart upload
    
    def create_multipart_upload(
        self,
        bucket: str,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> MultipartUpload:
        """Initiate a multipart upload."""
        try:
            kwargs = {"Bucket": bucket, "Key": key}
            if content_type:
                kwargs["ContentType"] = content_type
            if metadata:
                kwargs["Metadata"] = metadata
            
            response = self._client.create_multipart_upload(**kwargs)
            
            return MultipartUpload(
                upload_id=response["UploadId"],
                bucket=bucket,
                key=key,
            )
            
        except Exception as e:
            raise StorageError(f"Failed to create multipart upload: {e}")
    
    def upload_part(
        self,
        upload: MultipartUpload,
        part_number: int,
        data: bytes,
    ) -> Dict[str, Any]:
        """Upload a part of a multipart upload."""
        try:
            response = self._client.upload_part(
                Bucket=upload.bucket,
                Key=upload.key,
                UploadId=upload.upload_id,
                PartNumber=part_number,
                Body=data,
            )
            
            self._record_stat("bytes_uploaded", len(data))
            
            return {
                "PartNumber": part_number,
                "ETag": response["ETag"],
            }
            
        except Exception as e:
            raise StorageError(f"Failed to upload part {part_number}: {e}")
    
    def complete_multipart_upload(
        self,
        upload: MultipartUpload,
    ) -> ObjectInfo:
        """Complete a multipart upload."""
        try:
            response = self._client.complete_multipart_upload(
                Bucket=upload.bucket,
                Key=upload.key,
                UploadId=upload.upload_id,
                MultipartUpload={"Parts": upload.parts},
            )
            
            self._record_stat("puts")
            
            # Get final object info
            return self.head_object(upload.bucket, upload.key)
            
        except Exception as e:
            raise StorageError(f"Failed to complete multipart upload: {e}")
    
    def abort_multipart_upload(self, upload: MultipartUpload) -> bool:
        """Abort a multipart upload."""
        try:
            self._client.abort_multipart_upload(
                Bucket=upload.bucket,
                Key=upload.key,
                UploadId=upload.upload_id,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to abort multipart upload: {e}")
            return False
    
    # Health and metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Check storage backend health."""
        start = time.time()
        
        try:
            # Try to list buckets as a health check
            self._client.list_buckets()
            latency_ms = (time.time() - start) * 1000
            
            return {
                "healthy": True,
                "backend": self.config.backend.value,
                "endpoint": self.config.endpoint,
                "latency_ms": latency_ms,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "backend": self.config.backend.value,
                "endpoint": self.config.endpoint,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }


class LocalStorageClient(ObjectStoreClient):
    """
    Local filesystem-based storage client for testing.
    
    Mimics S3 API but stores files locally.
    """
    
    def __init__(self, config: ObjectStoreConfig, base_path: str = "/tmp/mineralvision-storage"):
        super().__init__(config)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def _bucket_path(self, bucket: str) -> Path:
        return self.base_path / bucket
    
    def _object_path(self, bucket: str, key: str) -> Path:
        return self._bucket_path(bucket) / key
    
    def _metadata_key(self, bucket: str, key: str) -> str:
        return f"{bucket}/{key}"
    
    def put_object(
        self,
        bucket: str,
        key: str,
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectInfo:
        """Upload an object to local storage."""
        if hasattr(data, "read"):
            body = data.read()
        else:
            body = data
        
        path = self._object_path(bucket, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        
        etag = hashlib.md5(body).hexdigest()
        now = datetime.utcnow()
        
        self._metadata[self._metadata_key(bucket, key)] = {
            "content_type": content_type,
            "metadata": metadata or {},
            "etag": etag,
            "last_modified": now,
        }
        
        self._record_stat("puts")
        self._record_stat("bytes_uploaded", len(body))
        
        return ObjectInfo(
            key=key,
            size=len(body),
            etag=etag,
            last_modified=now,
            content_type=content_type,
            metadata=metadata or {},
        )
    
    def get_object(self, bucket: str, key: str) -> bytes:
        """Download an object from local storage."""
        path = self._object_path(bucket, key)
        
        if not path.exists():
            raise ObjectNotFoundError(f"Object not found: {bucket}/{key}")
        
        data = path.read_bytes()
        self._record_stat("gets")
        self._record_stat("bytes_downloaded", len(data))
        
        return data
    
    def get_object_stream(self, bucket: str, key: str) -> BinaryIO:
        """Get a streaming handle to an object."""
        path = self._object_path(bucket, key)
        
        if not path.exists():
            raise ObjectNotFoundError(f"Object not found: {bucket}/{key}")
        
        self._record_stat("gets")
        return open(path, "rb")
    
    def delete_object(self, bucket: str, key: str) -> bool:
        """Delete an object from local storage."""
        path = self._object_path(bucket, key)
        
        try:
            if path.exists():
                path.unlink()
            self._metadata.pop(self._metadata_key(bucket, key), None)
            self._record_stat("deletes")
            return True
        except Exception:
            return False
    
    def delete_objects(self, bucket: str, keys: List[str]) -> List[str]:
        """Delete multiple objects."""
        failed = []
        for key in keys:
            if not self.delete_object(bucket, key):
                failed.append(key)
        return failed
    
    def head_object(self, bucket: str, key: str) -> Optional[ObjectInfo]:
        """Get object metadata."""
        path = self._object_path(bucket, key)
        
        if not path.exists():
            return None
        
        meta = self._metadata.get(self._metadata_key(bucket, key), {})
        stat = path.stat()
        
        return ObjectInfo(
            key=key,
            size=stat.st_size,
            etag=meta.get("etag", ""),
            last_modified=meta.get("last_modified", datetime.fromtimestamp(stat.st_mtime)),
            content_type=meta.get("content_type"),
            metadata=meta.get("metadata", {}),
        )
    
    def object_exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists."""
        return self._object_path(bucket, key).exists()
    
    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        delimiter: str = "",
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> Tuple[List[ObjectInfo], Optional[str]]:
        """List objects in a bucket."""
        bucket_path = self._bucket_path(bucket)
        
        if not bucket_path.exists():
            raise BucketNotFoundError(f"Bucket not found: {bucket}")
        
        self._record_stat("lists")
        
        objects = []
        for path in bucket_path.rglob("*"):
            if path.is_file():
                key = str(path.relative_to(bucket_path))
                if prefix and not key.startswith(prefix):
                    continue
                
                info = self.head_object(bucket, key)
                if info:
                    objects.append(info)
                
                if len(objects) >= max_keys:
                    break
        
        return objects, None
    
    def list_objects_iter(
        self,
        bucket: str,
        prefix: str = "",
    ) -> Iterator[ObjectInfo]:
        """Iterate over all objects."""
        objects, _ = self.list_objects(bucket, prefix=prefix, max_keys=10000)
        for obj in objects:
            yield obj
    
    def copy_object(
        self,
        src_bucket: str,
        src_key: str,
        dst_bucket: str,
        dst_key: str,
    ) -> ObjectInfo:
        """Copy an object."""
        data = self.get_object(src_bucket, src_key)
        src_meta = self._metadata.get(self._metadata_key(src_bucket, src_key), {})
        return self.put_object(
            dst_bucket,
            dst_key,
            data,
            content_type=src_meta.get("content_type"),
            metadata=src_meta.get("metadata"),
        )
    
    def create_bucket(self, bucket: str, region: Optional[str] = None) -> BucketInfo:
        """Create a new bucket."""
        path = self._bucket_path(bucket)
        path.mkdir(parents=True, exist_ok=True)
        return BucketInfo(name=bucket, creation_date=datetime.utcnow(), region=region)
    
    def delete_bucket(self, bucket: str) -> bool:
        """Delete an empty bucket."""
        path = self._bucket_path(bucket)
        try:
            if path.exists():
                path.rmdir()
            return True
        except Exception:
            return False
    
    def bucket_exists(self, bucket: str) -> bool:
        """Check if a bucket exists."""
        return self._bucket_path(bucket).exists()
    
    def list_buckets(self) -> List[BucketInfo]:
        """List all buckets."""
        buckets = []
        for path in self.base_path.iterdir():
            if path.is_dir():
                buckets.append(BucketInfo(
                    name=path.name,
                    creation_date=datetime.fromtimestamp(path.stat().st_ctime),
                ))
        return buckets
    
    def presign_get(self, bucket: str, key: str, expires: int = 3600) -> str:
        """Generate a presigned URL (returns file:// URL for local)."""
        return f"file://{self._object_path(bucket, key)}"
    
    def presign_put(self, bucket: str, key: str, expires: int = 3600, content_type: Optional[str] = None) -> str:
        """Generate a presigned URL for PUT."""
        return f"file://{self._object_path(bucket, key)}"
    
    def create_multipart_upload(
        self,
        bucket: str,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> MultipartUpload:
        """Create multipart upload (simplified for local)."""
        return MultipartUpload(
            upload_id=hashlib.md5(f"{bucket}/{key}/{time.time()}".encode()).hexdigest(),
            bucket=bucket,
            key=key,
        )
    
    def upload_part(self, upload: MultipartUpload, part_number: int, data: bytes) -> Dict[str, Any]:
        """Upload part (store temporarily)."""
        part_path = self._object_path(upload.bucket, f".parts/{upload.upload_id}/{part_number}")
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(data)
        
        return {
            "PartNumber": part_number,
            "ETag": hashlib.md5(data).hexdigest(),
        }
    
    def complete_multipart_upload(self, upload: MultipartUpload) -> ObjectInfo:
        """Complete multipart upload."""
        parts_dir = self._object_path(upload.bucket, f".parts/{upload.upload_id}")
        
        # Combine all parts
        combined = b""
        for part_file in sorted(parts_dir.iterdir(), key=lambda p: int(p.name)):
            combined += part_file.read_bytes()
        
        # Write final object
        result = self.put_object(upload.bucket, upload.key, combined)
        
        # Cleanup parts
        import shutil
        shutil.rmtree(parts_dir, ignore_errors=True)
        
        return result
    
    def abort_multipart_upload(self, upload: MultipartUpload) -> bool:
        """Abort multipart upload."""
        parts_dir = self._object_path(upload.bucket, f".parts/{upload.upload_id}")
        import shutil
        shutil.rmtree(parts_dir, ignore_errors=True)
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Check local storage health."""
        return {
            "healthy": self.base_path.exists(),
            "backend": "local",
            "path": str(self.base_path),
            "timestamp": datetime.utcnow().isoformat(),
        }


# Factory functions

def create_object_store(
    backend: StorageBackendType = StorageBackendType.RUSTFS,
    endpoint: str = "http://localhost:9000",
    access_key: str = "",
    secret_key: str = "",
    **kwargs,
) -> ObjectStoreClient:
    """
    Create an object store client.
    
    Args:
        backend: Storage backend type (rustfs, minio, aws_s3, local)
        endpoint: Storage endpoint URL
        access_key: Access key / username
        secret_key: Secret key / password
        **kwargs: Additional config options
        
    Returns:
        ObjectStoreClient instance
    """
    config = ObjectStoreConfig(
        backend=backend,
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        **kwargs,
    )
    
    if backend == StorageBackendType.LOCAL:
        return LocalStorageClient(config, kwargs.get("base_path", "/tmp/mineralvision-storage"))
    else:
        return S3CompatibleClient(config)


def create_object_store_from_env() -> ObjectStoreClient:
    """Create object store client from environment variables."""
    config = ObjectStoreConfig.from_env()
    
    if config.backend == StorageBackendType.LOCAL:
        return LocalStorageClient(config)
    else:
        return S3CompatibleClient(config)


def create_rustfs_client(
    endpoint: str = "http://rustfs.lakehouse:9000",
    access_key: str = "",
    secret_key: str = "",
) -> ObjectStoreClient:
    """Create a RustFS client with optimized settings."""
    config = ObjectStoreConfig.for_rustfs(endpoint, access_key, secret_key)
    return S3CompatibleClient(config)


def create_minio_client(
    endpoint: str = "http://minio.lakehouse:9000",
    access_key: str = "",
    secret_key: str = "",
) -> ObjectStoreClient:
    """Create a MinIO client (legacy support)."""
    config = ObjectStoreConfig.for_minio(endpoint, access_key, secret_key)
    return S3CompatibleClient(config)


# Migration utilities

class StorageMigrator:
    """
    Utility for migrating data between storage backends.
    
    Supports MinIO -> RustFS migration with verification.
    """
    
    def __init__(
        self,
        source: ObjectStoreClient,
        destination: ObjectStoreClient,
    ):
        self.source = source
        self.destination = destination
        self._stats = {
            "objects_migrated": 0,
            "bytes_migrated": 0,
            "objects_failed": 0,
            "objects_skipped": 0,
        }
    
    def migrate_bucket(
        self,
        bucket: str,
        prefix: str = "",
        verify: bool = True,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Migrate all objects in a bucket.
        
        Args:
            bucket: Bucket name
            prefix: Only migrate objects with this prefix
            verify: Verify each object after migration
            overwrite: Overwrite existing objects in destination
            progress_callback: Called with (key, current, total)
            
        Returns:
            Migration statistics
        """
        # Ensure destination bucket exists
        if not self.destination.bucket_exists(bucket):
            self.destination.create_bucket(bucket)
        
        # List all source objects
        objects = list(self.source.list_objects_iter(bucket, prefix))
        total = len(objects)
        
        failed_keys = []
        
        for i, obj in enumerate(objects):
            try:
                # Check if already exists
                if not overwrite and self.destination.object_exists(bucket, obj.key):
                    self._stats["objects_skipped"] += 1
                    continue
                
                # Download from source
                data = self.source.get_object(bucket, obj.key)
                
                # Get metadata
                src_info = self.source.head_object(bucket, obj.key)
                
                # Upload to destination
                self.destination.put_object(
                    bucket,
                    obj.key,
                    data,
                    content_type=src_info.content_type if src_info else None,
                    metadata=src_info.metadata if src_info else None,
                )
                
                # Verify if requested
                if verify:
                    dst_info = self.destination.head_object(bucket, obj.key)
                    if dst_info is None or dst_info.size != obj.size:
                        raise StorageError(f"Verification failed for {obj.key}")
                
                self._stats["objects_migrated"] += 1
                self._stats["bytes_migrated"] += obj.size
                
                if progress_callback:
                    progress_callback(obj.key, i + 1, total)
                    
            except Exception as e:
                logger.error(f"Failed to migrate {obj.key}: {e}")
                self._stats["objects_failed"] += 1
                failed_keys.append(obj.key)
        
        return {
            **self._stats,
            "failed_keys": failed_keys,
            "bucket": bucket,
            "prefix": prefix,
        }
    
    def verify_migration(
        self,
        bucket: str,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Verify that all objects were migrated correctly.
        
        Returns:
            Verification results
        """
        source_objects = {
            obj.key: obj
            for obj in self.source.list_objects_iter(bucket, prefix)
        }
        
        dest_objects = {
            obj.key: obj
            for obj in self.destination.list_objects_iter(bucket, prefix)
        }
        
        missing = []
        size_mismatch = []
        
        for key, src_obj in source_objects.items():
            if key not in dest_objects:
                missing.append(key)
            elif dest_objects[key].size != src_obj.size:
                size_mismatch.append(key)
        
        return {
            "verified": len(missing) == 0 and len(size_mismatch) == 0,
            "source_count": len(source_objects),
            "dest_count": len(dest_objects),
            "missing": missing,
            "size_mismatch": size_mismatch,
        }


# Singleton for global access
_default_client: Optional[ObjectStoreClient] = None
_client_lock = threading.Lock()


def get_default_client() -> ObjectStoreClient:
    """Get or create the default object store client."""
    global _default_client
    
    with _client_lock:
        if _default_client is None:
            _default_client = create_object_store_from_env()
        return _default_client


def set_default_client(client: ObjectStoreClient):
    """Set the default object store client."""
    global _default_client
    
    with _client_lock:
        _default_client = client
