"""
Schema Registry Integration Module
==================================

Production-grade schema management with:
- Avro/JSON/Protobuf schema support
- Schema versioning and evolution
- Compatibility checking
- Confluent Schema Registry integration
- Local schema store fallback
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import threading

logger = logging.getLogger(__name__)

from .._mock_fallback import real_client_unavailable


class SchemaType(Enum):
    """Supported schema types."""
    AVRO = "AVRO"
    JSON = "JSON"
    PROTOBUF = "PROTOBUF"


class CompatibilityLevel(Enum):
    """Schema compatibility levels."""
    NONE = "NONE"
    BACKWARD = "BACKWARD"
    BACKWARD_TRANSITIVE = "BACKWARD_TRANSITIVE"
    FORWARD = "FORWARD"
    FORWARD_TRANSITIVE = "FORWARD_TRANSITIVE"
    FULL = "FULL"
    FULL_TRANSITIVE = "FULL_TRANSITIVE"


@dataclass
class SchemaVersion:
    """Represents a schema version."""
    subject: str
    version: int
    schema_id: int
    schema_type: SchemaType
    schema: str
    fingerprint: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'subject': self.subject,
            'version': self.version,
            'schema_id': self.schema_id,
            'schema_type': self.schema_type.value,
            'schema': self.schema,
            'fingerprint': self.fingerprint,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SchemaVersion':
        """Create from dictionary."""
        return cls(
            subject=d['subject'],
            version=d['version'],
            schema_id=d['schema_id'],
            schema_type=SchemaType(d['schema_type']),
            schema=d['schema'],
            fingerprint=d['fingerprint'],
            created_at=datetime.fromisoformat(d['created_at']) if isinstance(d['created_at'], str) else d['created_at'],
            metadata=d.get('metadata', {})
        )


class SchemaStore(ABC):
    """Abstract base class for schema storage."""
    
    @abstractmethod
    def register_schema(self, subject: str, schema: str,
                       schema_type: SchemaType) -> SchemaVersion:
        """Register a new schema."""
        pass
    
    @abstractmethod
    def get_schema(self, subject: str, version: int = None) -> Optional[SchemaVersion]:
        """Get a schema by subject and version."""
        pass
    
    @abstractmethod
    def get_schema_by_id(self, schema_id: int) -> Optional[SchemaVersion]:
        """Get a schema by ID."""
        pass
    
    @abstractmethod
    def get_versions(self, subject: str) -> List[int]:
        """Get all versions for a subject."""
        pass
    
    @abstractmethod
    def get_subjects(self) -> List[str]:
        """Get all subjects."""
        pass
    
    @abstractmethod
    def delete_subject(self, subject: str) -> bool:
        """Delete a subject and all its versions."""
        pass


class LocalSchemaStore(SchemaStore):
    """
    Local file-based schema store.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self._schemas: Dict[str, List[SchemaVersion]] = {}
        self._schema_by_id: Dict[int, SchemaVersion] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        
        self._load_schemas()
    
    def _load_schemas(self):
        """Load schemas from storage."""
        index_path = os.path.join(self.storage_path, 'index.json')
        
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                
                self._next_id = data.get('next_id', 1)
                
                for subject, versions in data.get('schemas', {}).items():
                    self._schemas[subject] = []
                    for v in versions:
                        schema_version = SchemaVersion.from_dict(v)
                        self._schemas[subject].append(schema_version)
                        self._schema_by_id[schema_version.schema_id] = schema_version
                
                logger.info(f"Loaded {len(self._schemas)} subjects from local store")
                
            except Exception as e:
                logger.error(f"Error loading schemas: {e}")
    
    def _save_schemas(self):
        """Save schemas to storage."""
        index_path = os.path.join(self.storage_path, 'index.json')
        
        data = {
            'next_id': self._next_id,
            'schemas': {
                subject: [v.to_dict() for v in versions]
                for subject, versions in self._schemas.items()
            }
        }
        
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_fingerprint(self, schema: str) -> str:
        """Compute schema fingerprint."""
        return hashlib.sha256(schema.encode()).hexdigest()[:16]
    
    def register_schema(self, subject: str, schema: str,
                       schema_type: SchemaType) -> SchemaVersion:
        """Register a new schema."""
        with self._lock:
            fingerprint = self._compute_fingerprint(schema)
            
            # Check if schema already exists
            if subject in self._schemas:
                for existing in self._schemas[subject]:
                    if existing.fingerprint == fingerprint:
                        return existing
            
            # Create new version
            version = len(self._schemas.get(subject, [])) + 1
            schema_id = self._next_id
            self._next_id += 1
            
            schema_version = SchemaVersion(
                subject=subject,
                version=version,
                schema_id=schema_id,
                schema_type=schema_type,
                schema=schema,
                fingerprint=fingerprint
            )
            
            if subject not in self._schemas:
                self._schemas[subject] = []
            
            self._schemas[subject].append(schema_version)
            self._schema_by_id[schema_id] = schema_version
            
            self._save_schemas()
            
            logger.info(f"Registered schema {subject} version {version} (id={schema_id})")
            
            return schema_version
    
    def get_schema(self, subject: str, version: int = None) -> Optional[SchemaVersion]:
        """Get a schema by subject and version."""
        if subject not in self._schemas:
            return None
        
        versions = self._schemas[subject]
        
        if version is None:
            # Return latest
            return versions[-1] if versions else None
        
        for v in versions:
            if v.version == version:
                return v
        
        return None
    
    def get_schema_by_id(self, schema_id: int) -> Optional[SchemaVersion]:
        """Get a schema by ID."""
        return self._schema_by_id.get(schema_id)
    
    def get_versions(self, subject: str) -> List[int]:
        """Get all versions for a subject."""
        if subject not in self._schemas:
            return []
        return [v.version for v in self._schemas[subject]]
    
    def get_subjects(self) -> List[str]:
        """Get all subjects."""
        return list(self._schemas.keys())
    
    def delete_subject(self, subject: str) -> bool:
        """Delete a subject and all its versions."""
        with self._lock:
            if subject not in self._schemas:
                return False
            
            # Remove from schema_by_id
            for v in self._schemas[subject]:
                self._schema_by_id.pop(v.schema_id, None)
            
            del self._schemas[subject]
            self._save_schemas()
            
            logger.info(f"Deleted subject {subject}")
            return True


class ConfluentSchemaRegistry(SchemaStore):
    """
    Confluent Schema Registry client.
    """
    
    def __init__(self, url: str, auth: Optional[Tuple[str, str]] = None):
        self.url = url.rstrip('/')
        self.auth = auth
        self._session = None
        self._degraded = False
        self._initialize()
    
    def _initialize(self):
        """Initialize HTTP session."""
        try:
            import requests
            self._session = requests.Session()
            
            if self.auth:
                self._session.auth = self.auth
            
            self._session.headers.update({
                'Content-Type': 'application/vnd.schemaregistry.v1+json'
            })
            
            logger.info(f"Initialized Confluent Schema Registry client: {self.url}")
            
        except ImportError as exc:
            # Real-client-first: mock (registry disabled) only when explicitly allowed
            if real_client_unavailable("Confluent Schema Registry client", "requests package not installed", exc):
                self._degraded = True
                self._session = None
    
    def register_schema(self, subject: str, schema: str,
                       schema_type: SchemaType) -> SchemaVersion:
        """Register a new schema."""
        if self._session is None:
            raise RuntimeError("Schema Registry client not initialized")
        
        payload = {
            'schema': schema,
            'schemaType': schema_type.value
        }
        
        response = self._session.post(
            f"{self.url}/subjects/{subject}/versions",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        schema_id = result['id']
        
        # Get version info
        version_response = self._session.get(
            f"{self.url}/schemas/ids/{schema_id}"
        )
        version_response.raise_for_status()
        
        version_info = version_response.json()
        
        return SchemaVersion(
            subject=subject,
            version=version_info.get('version', 1),
            schema_id=schema_id,
            schema_type=schema_type,
            schema=schema,
            fingerprint=hashlib.sha256(schema.encode()).hexdigest()[:16]
        )
    
    def get_schema(self, subject: str, version: int = None) -> Optional[SchemaVersion]:
        """Get a schema by subject and version."""
        if self._session is None:
            return None
        
        try:
            if version is None:
                url = f"{self.url}/subjects/{subject}/versions/latest"
            else:
                url = f"{self.url}/subjects/{subject}/versions/{version}"
            
            response = self._session.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            return SchemaVersion(
                subject=data['subject'],
                version=data['version'],
                schema_id=data['id'],
                schema_type=SchemaType(data.get('schemaType', 'AVRO')),
                schema=data['schema'],
                fingerprint=hashlib.sha256(data['schema'].encode()).hexdigest()[:16]
            )
            
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return None
    
    def get_schema_by_id(self, schema_id: int) -> Optional[SchemaVersion]:
        """Get a schema by ID."""
        if self._session is None:
            return None
        
        try:
            response = self._session.get(f"{self.url}/schemas/ids/{schema_id}")
            response.raise_for_status()
            
            data = response.json()
            
            return SchemaVersion(
                subject=data.get('subject', 'unknown'),
                version=data.get('version', 1),
                schema_id=schema_id,
                schema_type=SchemaType(data.get('schemaType', 'AVRO')),
                schema=data['schema'],
                fingerprint=hashlib.sha256(data['schema'].encode()).hexdigest()[:16]
            )
            
        except Exception as e:
            logger.error(f"Error getting schema by ID: {e}")
            return None
    
    def get_versions(self, subject: str) -> List[int]:
        """Get all versions for a subject."""
        if self._session is None:
            return []
        
        try:
            response = self._session.get(f"{self.url}/subjects/{subject}/versions")
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting versions: {e}")
            return []
    
    def get_subjects(self) -> List[str]:
        """Get all subjects."""
        if self._session is None:
            return []
        
        try:
            response = self._session.get(f"{self.url}/subjects")
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting subjects: {e}")
            return []
    
    def delete_subject(self, subject: str) -> bool:
        """Delete a subject and all its versions."""
        if self._session is None:
            return False
        
        try:
            response = self._session.delete(f"{self.url}/subjects/{subject}")
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting subject: {e}")
            return False
    
    def check_compatibility(self, subject: str, schema: str,
                           schema_type: SchemaType) -> bool:
        """Check if schema is compatible with existing versions."""
        if self._session is None:
            return True
        
        try:
            payload = {
                'schema': schema,
                'schemaType': schema_type.value
            }
            
            response = self._session.post(
                f"{self.url}/compatibility/subjects/{subject}/versions/latest",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('is_compatible', False)
            
        except Exception as e:
            logger.error(f"Error checking compatibility: {e}")
            return False
    
    def set_compatibility(self, subject: str, level: CompatibilityLevel) -> bool:
        """Set compatibility level for a subject."""
        if self._session is None:
            return False
        
        try:
            payload = {'compatibility': level.value}
            
            response = self._session.put(
                f"{self.url}/config/{subject}",
                json=payload
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Error setting compatibility: {e}")
            return False


class SchemaEvolutionChecker:
    """
    Schema evolution and compatibility checker.
    """
    
    def check_avro_compatibility(self, old_schema: str, new_schema: str,
                                compatibility: CompatibilityLevel) -> Tuple[bool, List[str]]:
        """
        Check Avro schema compatibility.
        
        Args:
            old_schema: Previous schema version
            new_schema: New schema version
            compatibility: Compatibility level to check
            
        Returns:
            Tuple of (is_compatible, list of issues)
        """
        try:
            import json
            old = json.loads(old_schema)
            new = json.loads(new_schema)
            
            issues = []
            
            if compatibility in [CompatibilityLevel.BACKWARD, 
                               CompatibilityLevel.BACKWARD_TRANSITIVE,
                               CompatibilityLevel.FULL,
                               CompatibilityLevel.FULL_TRANSITIVE]:
                # Check backward compatibility (new can read old)
                issues.extend(self._check_backward(old, new))
            
            if compatibility in [CompatibilityLevel.FORWARD,
                               CompatibilityLevel.FORWARD_TRANSITIVE,
                               CompatibilityLevel.FULL,
                               CompatibilityLevel.FULL_TRANSITIVE]:
                # Check forward compatibility (old can read new)
                issues.extend(self._check_forward(old, new))
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Error parsing schemas: {e}"]
    
    def _check_backward(self, old: Dict, new: Dict) -> List[str]:
        """Check backward compatibility."""
        issues = []
        
        old_fields = {f['name']: f for f in old.get('fields', [])}
        new_fields = {f['name']: f for f in new.get('fields', [])}
        
        # Check for removed fields without defaults
        for name, field in old_fields.items():
            if name not in new_fields:
                if 'default' not in field:
                    issues.append(f"Field '{name}' removed without default value")
        
        # Check for type changes
        for name, new_field in new_fields.items():
            if name in old_fields:
                old_field = old_fields[name]
                if old_field.get('type') != new_field.get('type'):
                    # Allow promotion (int -> long, float -> double)
                    if not self._is_type_promotion(old_field['type'], new_field['type']):
                        issues.append(f"Field '{name}' type changed from {old_field['type']} to {new_field['type']}")
        
        return issues
    
    def _check_forward(self, old: Dict, new: Dict) -> List[str]:
        """Check forward compatibility."""
        issues = []
        
        old_fields = {f['name']: f for f in old.get('fields', [])}
        new_fields = {f['name']: f for f in new.get('fields', [])}
        
        # Check for new required fields
        for name, field in new_fields.items():
            if name not in old_fields:
                if 'default' not in field:
                    issues.append(f"New field '{name}' added without default value")
        
        return issues
    
    def _is_type_promotion(self, old_type: Any, new_type: Any) -> bool:
        """Check if type change is a valid promotion."""
        promotions = {
            'int': ['long', 'float', 'double'],
            'long': ['float', 'double'],
            'float': ['double'],
            'string': ['bytes'],
            'bytes': ['string']
        }
        
        if isinstance(old_type, str) and isinstance(new_type, str):
            return new_type in promotions.get(old_type, [])
        
        return False
    
    def check_json_compatibility(self, old_schema: str, new_schema: str,
                                compatibility: CompatibilityLevel) -> Tuple[bool, List[str]]:
        """Check JSON schema compatibility."""
        try:
            old = json.loads(old_schema)
            new = json.loads(new_schema)
            
            issues = []
            
            # Check required fields
            old_required = set(old.get('required', []))
            new_required = set(new.get('required', []))
            
            if compatibility in [CompatibilityLevel.BACKWARD, CompatibilityLevel.FULL]:
                # New required fields break backward compatibility
                new_required_fields = new_required - old_required
                if new_required_fields:
                    issues.append(f"New required fields added: {new_required_fields}")
            
            if compatibility in [CompatibilityLevel.FORWARD, CompatibilityLevel.FULL]:
                # Removed required fields break forward compatibility
                removed_required = old_required - new_required
                if removed_required:
                    issues.append(f"Required fields removed: {removed_required}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Error parsing schemas: {e}"]


class SchemaRegistry:
    """
    Complete schema registry manager.
    """
    
    def __init__(self, store: SchemaStore = None,
                 confluent_url: str = None,
                 local_path: str = None):
        if store:
            self.store = store
        elif confluent_url:
            self.store = ConfluentSchemaRegistry(confluent_url)
        else:
            self.store = LocalSchemaStore(local_path or '/tmp/schema_registry')
        
        self.evolution_checker = SchemaEvolutionChecker()
        self._compatibility_levels: Dict[str, CompatibilityLevel] = {}
    
    def register_schema(self, subject: str, schema: str,
                       schema_type: SchemaType = SchemaType.AVRO,
                       check_compatibility: bool = True) -> SchemaVersion:
        """
        Register a new schema version.
        
        Args:
            subject: Schema subject name
            schema: Schema definition
            schema_type: Type of schema
            check_compatibility: Whether to check compatibility
            
        Returns:
            Registered schema version
        """
        # Check compatibility if enabled
        if check_compatibility:
            existing = self.store.get_schema(subject)
            if existing:
                compatibility = self._compatibility_levels.get(
                    subject, CompatibilityLevel.BACKWARD
                )
                
                if schema_type == SchemaType.AVRO:
                    is_compatible, issues = self.evolution_checker.check_avro_compatibility(
                        existing.schema, schema, compatibility
                    )
                elif schema_type == SchemaType.JSON:
                    is_compatible, issues = self.evolution_checker.check_json_compatibility(
                        existing.schema, schema, compatibility
                    )
                else:
                    is_compatible, issues = True, []
                
                if not is_compatible:
                    raise SchemaCompatibilityError(
                        f"Schema not compatible: {', '.join(issues)}"
                    )
        
        return self.store.register_schema(subject, schema, schema_type)
    
    def get_schema(self, subject: str, version: int = None) -> Optional[SchemaVersion]:
        """Get a schema by subject and version."""
        return self.store.get_schema(subject, version)
    
    def get_schema_by_id(self, schema_id: int) -> Optional[SchemaVersion]:
        """Get a schema by ID."""
        return self.store.get_schema_by_id(schema_id)
    
    def get_versions(self, subject: str) -> List[int]:
        """Get all versions for a subject."""
        return self.store.get_versions(subject)
    
    def get_subjects(self) -> List[str]:
        """Get all subjects."""
        return self.store.get_subjects()
    
    def delete_subject(self, subject: str) -> bool:
        """Delete a subject."""
        return self.store.delete_subject(subject)
    
    def set_compatibility(self, subject: str, level: CompatibilityLevel):
        """Set compatibility level for a subject."""
        self._compatibility_levels[subject] = level
        
        if isinstance(self.store, ConfluentSchemaRegistry):
            self.store.set_compatibility(subject, level)
    
    def get_compatibility(self, subject: str) -> CompatibilityLevel:
        """Get compatibility level for a subject."""
        return self._compatibility_levels.get(subject, CompatibilityLevel.BACKWARD)
    
    def test_compatibility(self, subject: str, schema: str,
                          schema_type: SchemaType = SchemaType.AVRO) -> Tuple[bool, List[str]]:
        """
        Test if a schema is compatible without registering.
        
        Args:
            subject: Schema subject
            schema: Schema to test
            schema_type: Type of schema
            
        Returns:
            Tuple of (is_compatible, issues)
        """
        existing = self.store.get_schema(subject)
        if not existing:
            return True, []
        
        compatibility = self._compatibility_levels.get(subject, CompatibilityLevel.BACKWARD)
        
        if schema_type == SchemaType.AVRO:
            return self.evolution_checker.check_avro_compatibility(
                existing.schema, schema, compatibility
            )
        elif schema_type == SchemaType.JSON:
            return self.evolution_checker.check_json_compatibility(
                existing.schema, schema, compatibility
            )
        
        return True, []


class SchemaCompatibilityError(Exception):
    """Schema compatibility error."""
    pass


def create_schema_registry(confluent_url: str = None,
                          local_path: str = None) -> SchemaRegistry:
    """Factory function to create schema registry."""
    return SchemaRegistry(confluent_url=confluent_url, local_path=local_path)


def create_avro_schema(name: str, namespace: str,
                      fields: List[Dict[str, Any]]) -> str:
    """Helper to create Avro schema JSON."""
    schema = {
        'type': 'record',
        'name': name,
        'namespace': namespace,
        'fields': fields
    }
    return json.dumps(schema, indent=2)


def create_json_schema(title: str, properties: Dict[str, Dict],
                      required: List[str] = None) -> str:
    """Helper to create JSON schema."""
    schema = {
        '$schema': 'http://json-schema.org/draft-07/schema#',
        'title': title,
        'type': 'object',
        'properties': properties,
        'required': required or []
    }
    return json.dumps(schema, indent=2)
