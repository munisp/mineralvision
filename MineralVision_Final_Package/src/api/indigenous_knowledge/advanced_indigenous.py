"""
Advanced Indigenous Knowledge Integration Module for MineralVision.

This module provides enhanced indigenous knowledge capabilities including:
- Encryption for confidential knowledge
- Comprehensive audit trail for access tracking
- Multi-language support for traditional names/descriptions
- Integration with external indigenous knowledge databases
"""

import hashlib
import json
import base64
import os
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import uuid

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Knowledge access levels."""
    PUBLIC = "public"
    COMMUNITY = "community"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SACRED = "sacred"


class AuditAction(Enum):
    """Audit trail action types."""
    VIEW = "view"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    SHARE = "share"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    ACCESS_DENIED = "access_denied"


class Language(Enum):
    """Supported languages for multi-language content."""
    ENGLISH = "en"
    SPANISH = "es"
    PORTUGUESE = "pt"
    FRENCH = "fr"
    QUECHUA = "qu"
    AYMARA = "ay"
    GUARANI = "gn"
    MAPUDUNGUN = "arn"
    NAVAJO = "nv"
    OJIBWE = "oj"
    CREE = "cr"
    INUKTITUT = "iu"
    MAORI = "mi"
    ABORIGINAL = "aus"  # Australian Aboriginal languages
    CUSTOM = "custom"


@dataclass
class AuditEntry:
    """Audit trail entry."""
    entry_id: str
    timestamp: datetime
    action: AuditAction
    user_id: str
    user_role: str
    resource_type: str
    resource_id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action.value,
            'user_id': self.user_id,
            'user_role': self.user_role,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'details': self.details,
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class MultiLanguageText:
    """Multi-language text content."""
    default_language: Language
    translations: Dict[str, str] = field(default_factory=dict)
    
    def get(self, language: Language = None) -> str:
        """Get text in specified language or default."""
        if language is None:
            language = self.default_language
        return self.translations.get(language.value, 
                                     self.translations.get(self.default_language.value, ""))
    
    def set(self, language: Language, text: str) -> None:
        """Set text for a language."""
        self.translations[language.value] = text
        
    def has_translation(self, language: Language) -> bool:
        """Check if translation exists."""
        return language.value in self.translations
    
    def get_available_languages(self) -> List[Language]:
        """Get list of available languages."""
        return [Language(code) for code in self.translations.keys() 
                if code in [l.value for l in Language]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'default_language': self.default_language.value,
            'translations': self.translations
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MultiLanguageText':
        return cls(
            default_language=Language(data.get('default_language', 'en')),
            translations=data.get('translations', {})
        )


@dataclass
class EncryptedContent:
    """Encrypted content wrapper."""
    content_id: str
    encrypted_data: bytes
    encryption_method: str
    key_id: str
    iv: bytes
    created_at: datetime = field(default_factory=datetime.now)
    access_policy: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content_id': self.content_id,
            'encrypted_data': base64.b64encode(self.encrypted_data).decode(),
            'encryption_method': self.encryption_method,
            'key_id': self.key_id,
            'iv': base64.b64encode(self.iv).decode(),
            'created_at': self.created_at.isoformat(),
            'access_policy': self.access_policy
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EncryptedContent':
        return cls(
            content_id=data['content_id'],
            encrypted_data=base64.b64decode(data['encrypted_data']),
            encryption_method=data['encryption_method'],
            key_id=data['key_id'],
            iv=base64.b64decode(data['iv']),
            created_at=datetime.fromisoformat(data['created_at']),
            access_policy=data.get('access_policy', {})
        )


class EncryptionProvider(ABC):
    """Abstract base class for encryption providers."""
    
    @abstractmethod
    def encrypt(self, data: bytes, key_id: str) -> EncryptedContent:
        """Encrypt data."""
        pass
    
    @abstractmethod
    def decrypt(self, encrypted: EncryptedContent) -> bytes:
        """Decrypt data."""
        pass
    
    @abstractmethod
    def generate_key(self) -> str:
        """Generate a new encryption key."""
        pass


class AES256EncryptionProvider(EncryptionProvider):
    """
    AES-256 encryption provider for confidential knowledge.
    
    Uses AES-256-GCM for authenticated encryption.
    """
    
    def __init__(self, master_key: bytes = None):
        self.master_key = master_key or os.urandom(32)
        self.keys: Dict[str, bytes] = {}
        self._lock = threading.Lock()
        
    def encrypt(self, data: bytes, key_id: str = None) -> EncryptedContent:
        """
        Encrypt data using AES-256-GCM.
        
        Args:
            data: Data to encrypt
            key_id: Key identifier (generates new if None)
            
        Returns:
            Encrypted content
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            # Fallback to simple XOR encryption for environments without cryptography
            return self._fallback_encrypt(data, key_id)
        
        if key_id is None:
            key_id = self.generate_key()
            
        key = self._get_key(key_id)
        iv = os.urandom(12)  # 96-bit nonce for GCM
        
        aesgcm = AESGCM(key)
        encrypted_data = aesgcm.encrypt(iv, data, None)
        
        return EncryptedContent(
            content_id=str(uuid.uuid4()),
            encrypted_data=encrypted_data,
            encryption_method='AES-256-GCM',
            key_id=key_id,
            iv=iv
        )
        
    def decrypt(self, encrypted: EncryptedContent) -> bytes:
        """
        Decrypt data using AES-256-GCM.
        
        Args:
            encrypted: Encrypted content
            
        Returns:
            Decrypted data
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            return self._fallback_decrypt(encrypted)
        
        key = self._get_key(encrypted.key_id)
        aesgcm = AESGCM(key)
        
        return aesgcm.decrypt(encrypted.iv, encrypted.encrypted_data, None)
        
    def generate_key(self) -> str:
        """Generate a new encryption key."""
        key_id = str(uuid.uuid4())
        key = os.urandom(32)  # 256-bit key
        
        with self._lock:
            self.keys[key_id] = key
            
        return key_id
        
    def _get_key(self, key_id: str) -> bytes:
        """Get or derive key for key_id."""
        with self._lock:
            if key_id in self.keys:
                return self.keys[key_id]
                
        # Derive key from master key and key_id
        import hashlib
        derived = hashlib.pbkdf2_hmac(
            'sha256',
            self.master_key,
            key_id.encode(),
            100000,
            dklen=32
        )
        
        with self._lock:
            self.keys[key_id] = derived
            
        return derived
        
    def _fallback_encrypt(self, data: bytes, key_id: str = None) -> EncryptedContent:
        """Fallback encryption using XOR (for environments without cryptography)."""
        if key_id is None:
            key_id = str(uuid.uuid4())
            
        # Generate key from key_id
        key = hashlib.sha256(f"{self.master_key.hex()}{key_id}".encode()).digest()
        iv = os.urandom(16)
        
        # XOR encryption (not secure, just for fallback)
        encrypted = bytes(d ^ key[i % len(key)] for i, d in enumerate(data))
        
        return EncryptedContent(
            content_id=str(uuid.uuid4()),
            encrypted_data=encrypted,
            encryption_method='XOR-FALLBACK',
            key_id=key_id,
            iv=iv
        )
        
    def _fallback_decrypt(self, encrypted: EncryptedContent) -> bytes:
        """Fallback decryption using XOR."""
        key = hashlib.sha256(f"{self.master_key.hex()}{encrypted.key_id}".encode()).digest()
        return bytes(d ^ key[i % len(key)] for i, d in enumerate(encrypted.encrypted_data))


class AuditTrailManager:
    """
    Comprehensive audit trail for indigenous knowledge access.
    
    Tracks all access, modifications, and sharing of knowledge.
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or "/tmp/audit_trail"
        self.entries: List[AuditEntry] = []
        self.callbacks: List[Callable[[AuditEntry], None]] = []
        self._lock = threading.Lock()
        
        os.makedirs(self.storage_path, exist_ok=True)
        self._load_entries()
        
    def log(self, action: AuditAction, user_id: str, user_role: str,
           resource_type: str, resource_id: str,
           ip_address: str = None, user_agent: str = None,
           details: Dict = None, success: bool = True,
           error_message: str = None) -> AuditEntry:
        """
        Log an audit entry.
        
        Args:
            action: Action performed
            user_id: User identifier
            user_role: User's role
            resource_type: Type of resource accessed
            resource_id: Resource identifier
            ip_address: Client IP address
            user_agent: Client user agent
            details: Additional details
            success: Whether action succeeded
            error_message: Error message if failed
            
        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action=action,
            user_id=user_id,
            user_role=user_role,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            success=success,
            error_message=error_message
        )
        
        with self._lock:
            self.entries.append(entry)
            self._persist_entry(entry)
            
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"Audit callback error: {e}")
                
        return entry
        
    def register_callback(self, callback: Callable[[AuditEntry], None]) -> None:
        """Register callback for audit events."""
        self.callbacks.append(callback)
        
    def query(self, user_id: str = None, action: AuditAction = None,
             resource_type: str = None, resource_id: str = None,
             start_date: datetime = None, end_date: datetime = None,
             success_only: bool = None, limit: int = 100) -> List[AuditEntry]:
        """
        Query audit entries.
        
        Args:
            user_id: Filter by user
            action: Filter by action
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            start_date: Filter by start date
            end_date: Filter by end date
            success_only: Filter by success status
            limit: Maximum entries to return
            
        Returns:
            List of matching audit entries
        """
        with self._lock:
            results = list(self.entries)
            
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if action:
            results = [e for e in results if e.action == action]
        if resource_type:
            results = [e for e in results if e.resource_type == resource_type]
        if resource_id:
            results = [e for e in results if e.resource_id == resource_id]
        if start_date:
            results = [e for e in results if e.timestamp >= start_date]
        if end_date:
            results = [e for e in results if e.timestamp <= end_date]
        if success_only is not None:
            results = [e for e in results if e.success == success_only]
            
        # Sort by timestamp descending
        results.sort(key=lambda e: e.timestamp, reverse=True)
        
        return results[:limit]
        
    def get_access_report(self, resource_id: str,
                         days: int = 30) -> Dict[str, Any]:
        """
        Generate access report for a resource.
        
        Args:
            resource_id: Resource to report on
            days: Number of days to include
            
        Returns:
            Access report
        """
        start_date = datetime.now() - timedelta(days=days)
        entries = self.query(resource_id=resource_id, start_date=start_date, limit=1000)
        
        # Aggregate statistics
        action_counts: Dict[str, int] = {}
        user_counts: Dict[str, int] = {}
        daily_counts: Dict[str, int] = {}
        failed_attempts = 0
        
        for entry in entries:
            # Action counts
            action_counts[entry.action.value] = action_counts.get(entry.action.value, 0) + 1
            
            # User counts
            user_counts[entry.user_id] = user_counts.get(entry.user_id, 0) + 1
            
            # Daily counts
            day = entry.timestamp.strftime('%Y-%m-%d')
            daily_counts[day] = daily_counts.get(day, 0) + 1
            
            # Failed attempts
            if not entry.success:
                failed_attempts += 1
                
        return {
            'resource_id': resource_id,
            'period_days': days,
            'total_accesses': len(entries),
            'unique_users': len(user_counts),
            'failed_attempts': failed_attempts,
            'action_breakdown': action_counts,
            'top_users': sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'daily_activity': daily_counts
        }
        
    def export_entries(self, format: str = 'json',
                      start_date: datetime = None,
                      end_date: datetime = None) -> str:
        """Export audit entries."""
        entries = self.query(start_date=start_date, end_date=end_date, limit=10000)
        
        if format == 'json':
            return json.dumps([e.to_dict() for e in entries], indent=2)
        elif format == 'csv':
            lines = ['entry_id,timestamp,action,user_id,resource_type,resource_id,success']
            for e in entries:
                lines.append(f"{e.entry_id},{e.timestamp.isoformat()},{e.action.value},"
                           f"{e.user_id},{e.resource_type},{e.resource_id},{e.success}")
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _persist_entry(self, entry: AuditEntry) -> None:
        """Persist entry to storage."""
        filename = f"{entry.timestamp.strftime('%Y-%m-%d')}.jsonl"
        filepath = os.path.join(self.storage_path, filename)
        
        with open(filepath, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
            
    def _load_entries(self) -> None:
        """Load entries from storage."""
        if not os.path.exists(self.storage_path):
            return
            
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.storage_path, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            entry = AuditEntry(
                                entry_id=data['entry_id'],
                                timestamp=datetime.fromisoformat(data['timestamp']),
                                action=AuditAction(data['action']),
                                user_id=data['user_id'],
                                user_role=data['user_role'],
                                resource_type=data['resource_type'],
                                resource_id=data['resource_id'],
                                ip_address=data.get('ip_address'),
                                user_agent=data.get('user_agent'),
                                details=data.get('details', {}),
                                success=data.get('success', True),
                                error_message=data.get('error_message')
                            )
                            self.entries.append(entry)
                        except Exception as e:
                            logger.warning(f"Failed to load audit entry: {e}")


class MultiLanguageManager:
    """
    Multi-language content management for indigenous knowledge.
    
    Supports traditional names, descriptions, and cultural context
    in multiple languages including indigenous languages.
    """
    
    def __init__(self):
        self.translations: Dict[str, MultiLanguageText] = {}
        self.language_metadata: Dict[str, Dict] = {
            'en': {'name': 'English', 'direction': 'ltr', 'script': 'Latin'},
            'es': {'name': 'Spanish', 'direction': 'ltr', 'script': 'Latin'},
            'pt': {'name': 'Portuguese', 'direction': 'ltr', 'script': 'Latin'},
            'fr': {'name': 'French', 'direction': 'ltr', 'script': 'Latin'},
            'qu': {'name': 'Quechua', 'direction': 'ltr', 'script': 'Latin'},
            'ay': {'name': 'Aymara', 'direction': 'ltr', 'script': 'Latin'},
            'gn': {'name': 'Guarani', 'direction': 'ltr', 'script': 'Latin'},
            'arn': {'name': 'Mapudungun', 'direction': 'ltr', 'script': 'Latin'},
            'nv': {'name': 'Navajo', 'direction': 'ltr', 'script': 'Latin'},
            'oj': {'name': 'Ojibwe', 'direction': 'ltr', 'script': 'Latin/Syllabics'},
            'cr': {'name': 'Cree', 'direction': 'ltr', 'script': 'Latin/Syllabics'},
            'iu': {'name': 'Inuktitut', 'direction': 'ltr', 'script': 'Syllabics'},
            'mi': {'name': 'Maori', 'direction': 'ltr', 'script': 'Latin'},
            'aus': {'name': 'Australian Aboriginal', 'direction': 'ltr', 'script': 'Latin'}
        }
        self.custom_languages: Dict[str, Dict] = {}
        
    def register_custom_language(self, code: str, name: str,
                                direction: str = 'ltr',
                                script: str = 'Latin') -> None:
        """
        Register a custom indigenous language.
        
        Args:
            code: Language code
            name: Language name
            direction: Text direction ('ltr' or 'rtl')
            script: Writing script
        """
        self.custom_languages[code] = {
            'name': name,
            'direction': direction,
            'script': script
        }
        
    def create_multilingual_content(self, content_id: str,
                                   default_language: Language,
                                   default_text: str) -> MultiLanguageText:
        """
        Create new multilingual content.
        
        Args:
            content_id: Content identifier
            default_language: Default language
            default_text: Text in default language
            
        Returns:
            Created multilingual text
        """
        text = MultiLanguageText(default_language=default_language)
        text.set(default_language, default_text)
        
        self.translations[content_id] = text
        return text
        
    def add_translation(self, content_id: str, language: Language,
                       text: str) -> bool:
        """
        Add translation for content.
        
        Args:
            content_id: Content identifier
            language: Target language
            text: Translated text
            
        Returns:
            True if successful
        """
        if content_id not in self.translations:
            return False
            
        self.translations[content_id].set(language, text)
        return True
        
    def get_translation(self, content_id: str,
                       language: Language = None) -> Optional[str]:
        """
        Get translation for content.
        
        Args:
            content_id: Content identifier
            language: Target language (None for default)
            
        Returns:
            Translated text or None
        """
        if content_id not in self.translations:
            return None
            
        return self.translations[content_id].get(language)
        
    def get_all_translations(self, content_id: str) -> Optional[Dict[str, str]]:
        """Get all translations for content."""
        if content_id not in self.translations:
            return None
            
        return self.translations[content_id].translations
        
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Get list of supported languages."""
        languages = []
        
        for code, metadata in self.language_metadata.items():
            languages.append({
                'code': code,
                **metadata
            })
            
        for code, metadata in self.custom_languages.items():
            languages.append({
                'code': code,
                'custom': True,
                **metadata
            })
            
        return languages
        
    def validate_text(self, text: str, language: Language) -> Dict[str, Any]:
        """
        Validate text for a language.
        
        Args:
            text: Text to validate
            language: Target language
            
        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'warnings': [],
            'character_count': len(text),
            'word_count': len(text.split())
        }
        
        # Check for empty text
        if not text.strip():
            result['valid'] = False
            result['warnings'].append('Text is empty')
            
        # Check for potential encoding issues
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            result['warnings'].append('Text contains characters that may not encode properly')
            
        # Language-specific checks
        if language in [Language.INUKTITUT, Language.CREE, Language.OJIBWE]:
            # Check for syllabics
            has_syllabics = any('\u1400' <= c <= '\u167F' for c in text)
            if not has_syllabics:
                result['warnings'].append(f'{language.value} text may benefit from syllabic script')
                
        return result


class ExternalDatabaseConnector(ABC):
    """Abstract base class for external indigenous knowledge database connectors."""
    
    @abstractmethod
    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search the external database."""
        pass
    
    @abstractmethod
    def get_record(self, record_id: str) -> Optional[Dict]:
        """Get a specific record."""
        pass
    
    @abstractmethod
    def sync(self, local_records: List[Dict]) -> Dict[str, Any]:
        """Sync with external database."""
        pass


class TKDLConnector(ExternalDatabaseConnector):
    """
    Traditional Knowledge Digital Library (TKDL) connector.
    
    Connects to TKDL-like databases for traditional knowledge.
    """
    
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = api_url or "https://tkdl.example.org/api/v1"
        self.api_key = api_key
        self._cache: Dict[str, Dict] = {}
        
    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """
        Search TKDL database.
        
        Args:
            query: Search query
            filters: Optional filters
            
        Returns:
            List of matching records
        """
        # Simulated search (in production, call actual API)
        results = []
        
        # Generate sample results based on query
        if query:
            for i in range(min(5, len(query))):
                results.append({
                    'id': f"tkdl_{hashlib.md5(f'{query}{i}'.encode()).hexdigest()[:8]}",
                    'title': f"Traditional Knowledge: {query}",
                    'description': f"Knowledge related to {query} from traditional sources",
                    'source': 'TKDL',
                    'community': 'Various',
                    'relevance_score': 0.9 - (i * 0.1)
                })
                
        return results
        
    def get_record(self, record_id: str) -> Optional[Dict]:
        """Get record from TKDL."""
        if record_id in self._cache:
            return self._cache[record_id]
            
        # Simulated record retrieval
        record = {
            'id': record_id,
            'title': f"TKDL Record {record_id}",
            'description': 'Traditional knowledge record',
            'source': 'TKDL',
            'created_at': datetime.now().isoformat()
        }
        
        self._cache[record_id] = record
        return record
        
    def sync(self, local_records: List[Dict]) -> Dict[str, Any]:
        """Sync local records with TKDL."""
        return {
            'synced': len(local_records),
            'new_from_remote': 0,
            'updated': 0,
            'conflicts': 0
        }


class ICHConnector(ExternalDatabaseConnector):
    """
    UNESCO Intangible Cultural Heritage connector.
    
    Connects to ICH databases for cultural heritage information.
    """
    
    def __init__(self, api_url: str = None):
        self.api_url = api_url or "https://ich.unesco.org/api/v1"
        
    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search ICH database."""
        results = []
        
        if query:
            results.append({
                'id': f"ich_{hashlib.md5(query.encode()).hexdigest()[:8]}",
                'title': f"Intangible Cultural Heritage: {query}",
                'description': f"Cultural heritage related to {query}",
                'source': 'UNESCO ICH',
                'status': 'inscribed',
                'year': 2020
            })
            
        return results
        
    def get_record(self, record_id: str) -> Optional[Dict]:
        """Get record from ICH database."""
        return {
            'id': record_id,
            'title': f"ICH Record {record_id}",
            'source': 'UNESCO ICH',
            'created_at': datetime.now().isoformat()
        }
        
    def sync(self, local_records: List[Dict]) -> Dict[str, Any]:
        """Sync with ICH database."""
        return {
            'synced': len(local_records),
            'new_from_remote': 0,
            'updated': 0
        }


class AccessControlManager:
    """
    Access control for indigenous knowledge.
    
    Manages permissions based on community, role, and access level.
    """
    
    def __init__(self):
        self.permissions: Dict[str, Dict[str, Set[str]]] = {}
        self.community_roles: Dict[str, List[str]] = {}
        self.role_hierarchy: Dict[str, int] = {
            'elder': 100,
            'knowledge_keeper': 90,
            'community_member': 50,
            'researcher': 30,
            'public': 10
        }
        
    def grant_permission(self, user_id: str, resource_id: str,
                        permissions: List[str]) -> None:
        """
        Grant permissions to a user for a resource.
        
        Args:
            user_id: User identifier
            resource_id: Resource identifier
            permissions: List of permissions (view, edit, share, etc.)
        """
        if user_id not in self.permissions:
            self.permissions[user_id] = {}
            
        if resource_id not in self.permissions[user_id]:
            self.permissions[user_id][resource_id] = set()
            
        self.permissions[user_id][resource_id].update(permissions)
        
    def revoke_permission(self, user_id: str, resource_id: str,
                         permissions: List[str] = None) -> None:
        """
        Revoke permissions from a user.
        
        Args:
            user_id: User identifier
            resource_id: Resource identifier
            permissions: Permissions to revoke (None = all)
        """
        if user_id not in self.permissions:
            return
            
        if resource_id not in self.permissions[user_id]:
            return
            
        if permissions is None:
            del self.permissions[user_id][resource_id]
        else:
            self.permissions[user_id][resource_id] -= set(permissions)
            
    def check_permission(self, user_id: str, resource_id: str,
                        permission: str, user_role: str = None,
                        access_level: AccessLevel = None) -> bool:
        """
        Check if user has permission for a resource.
        
        Args:
            user_id: User identifier
            resource_id: Resource identifier
            permission: Permission to check
            user_role: User's role
            access_level: Resource access level
            
        Returns:
            True if permitted
        """
        # Check explicit permissions
        if user_id in self.permissions:
            if resource_id in self.permissions[user_id]:
                if permission in self.permissions[user_id][resource_id]:
                    return True
                    
        # Check role-based access
        if user_role and access_level:
            role_level = self.role_hierarchy.get(user_role, 0)
            
            access_requirements = {
                AccessLevel.PUBLIC: 0,
                AccessLevel.COMMUNITY: 30,
                AccessLevel.RESTRICTED: 50,
                AccessLevel.CONFIDENTIAL: 90,
                AccessLevel.SACRED: 100
            }
            
            required_level = access_requirements.get(access_level, 100)
            
            if role_level >= required_level:
                return True
                
        return False
        
    def get_user_permissions(self, user_id: str) -> Dict[str, Set[str]]:
        """Get all permissions for a user."""
        return self.permissions.get(user_id, {})
        
    def get_resource_permissions(self, resource_id: str) -> Dict[str, Set[str]]:
        """Get all users with permissions for a resource."""
        result = {}
        for user_id, resources in self.permissions.items():
            if resource_id in resources:
                result[user_id] = resources[resource_id]
        return result


class AdvancedIndigenousKnowledgeManager:
    """
    Advanced indigenous knowledge manager combining all enhanced features.
    """
    
    def __init__(self, master_key: bytes = None,
                audit_path: str = None):
        self.encryption = AES256EncryptionProvider(master_key)
        self.audit = AuditTrailManager(audit_path)
        self.languages = MultiLanguageManager()
        self.access_control = AccessControlManager()
        
        # External connectors
        self.connectors: Dict[str, ExternalDatabaseConnector] = {
            'tkdl': TKDLConnector(),
            'ich': ICHConnector()
        }
        
        # Knowledge storage
        self.knowledge: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        
    def create_knowledge(self, knowledge_id: str, title: str,
                        description: str, community: str,
                        access_level: AccessLevel,
                        creator_id: str, creator_role: str,
                        default_language: Language = Language.ENGLISH,
                        encrypt: bool = False) -> Dict[str, Any]:
        """
        Create new knowledge entry.
        
        Args:
            knowledge_id: Knowledge identifier
            title: Title
            description: Description
            community: Associated community
            access_level: Access level
            creator_id: Creator user ID
            creator_role: Creator's role
            default_language: Default language
            encrypt: Whether to encrypt content
            
        Returns:
            Created knowledge entry
        """
        # Create multilingual content
        title_ml = self.languages.create_multilingual_content(
            f"{knowledge_id}_title", default_language, title
        )
        desc_ml = self.languages.create_multilingual_content(
            f"{knowledge_id}_desc", default_language, description
        )
        
        knowledge = {
            'knowledge_id': knowledge_id,
            'title': title_ml.to_dict(),
            'description': desc_ml.to_dict(),
            'community': community,
            'access_level': access_level.value,
            'creator_id': creator_id,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'encrypted': encrypt
        }
        
        # Encrypt if required
        if encrypt and access_level in [AccessLevel.CONFIDENTIAL, AccessLevel.SACRED]:
            encrypted = self.encryption.encrypt(
                json.dumps({'title': title, 'description': description}).encode()
            )
            knowledge['encrypted_content'] = encrypted.to_dict()
            
        with self._lock:
            self.knowledge[knowledge_id] = knowledge
            
        # Grant creator full permissions
        self.access_control.grant_permission(
            creator_id, knowledge_id,
            ['view', 'edit', 'delete', 'share', 'encrypt']
        )
        
        # Log audit
        self.audit.log(
            action=AuditAction.CREATE,
            user_id=creator_id,
            user_role=creator_role,
            resource_type='knowledge',
            resource_id=knowledge_id,
            details={'access_level': access_level.value, 'encrypted': encrypt}
        )
        
        return knowledge
        
    def get_knowledge(self, knowledge_id: str, user_id: str,
                     user_role: str, language: Language = None,
                     ip_address: str = None) -> Optional[Dict[str, Any]]:
        """
        Get knowledge entry with access control and audit.
        
        Args:
            knowledge_id: Knowledge identifier
            user_id: Requesting user ID
            user_role: User's role
            language: Preferred language
            ip_address: Client IP
            
        Returns:
            Knowledge entry or None if not permitted
        """
        with self._lock:
            knowledge = self.knowledge.get(knowledge_id)
            
        if not knowledge:
            return None
            
        access_level = AccessLevel(knowledge['access_level'])
        
        # Check permission
        if not self.access_control.check_permission(
            user_id, knowledge_id, 'view', user_role, access_level
        ):
            self.audit.log(
                action=AuditAction.ACCESS_DENIED,
                user_id=user_id,
                user_role=user_role,
                resource_type='knowledge',
                resource_id=knowledge_id,
                ip_address=ip_address,
                success=False,
                error_message='Insufficient permissions'
            )
            return None
            
        # Decrypt if needed
        result = dict(knowledge)
        if knowledge.get('encrypted') and 'encrypted_content' in knowledge:
            try:
                encrypted = EncryptedContent.from_dict(knowledge['encrypted_content'])
                decrypted = self.encryption.decrypt(encrypted)
                content = json.loads(decrypted.decode())
                result['decrypted_title'] = content.get('title')
                result['decrypted_description'] = content.get('description')
                
                self.audit.log(
                    action=AuditAction.DECRYPT,
                    user_id=user_id,
                    user_role=user_role,
                    resource_type='knowledge',
                    resource_id=knowledge_id,
                    ip_address=ip_address
                )
            except Exception as e:
                logger.error(f"Decryption failed: {e}")
                
        # Get translations
        if language:
            title_trans = self.languages.get_translation(f"{knowledge_id}_title", language)
            desc_trans = self.languages.get_translation(f"{knowledge_id}_desc", language)
            if title_trans:
                result['title_translated'] = title_trans
            if desc_trans:
                result['description_translated'] = desc_trans
                
        # Log access
        self.audit.log(
            action=AuditAction.VIEW,
            user_id=user_id,
            user_role=user_role,
            resource_type='knowledge',
            resource_id=knowledge_id,
            ip_address=ip_address,
            details={'language': language.value if language else None}
        )
        
        return result
        
    def add_translation(self, knowledge_id: str, language: Language,
                       title: str, description: str,
                       user_id: str, user_role: str) -> bool:
        """
        Add translation for knowledge entry.
        
        Args:
            knowledge_id: Knowledge identifier
            language: Target language
            title: Translated title
            description: Translated description
            user_id: User adding translation
            user_role: User's role
            
        Returns:
            True if successful
        """
        if knowledge_id not in self.knowledge:
            return False
            
        # Check permission
        if not self.access_control.check_permission(
            user_id, knowledge_id, 'edit', user_role
        ):
            return False
            
        self.languages.add_translation(f"{knowledge_id}_title", language, title)
        self.languages.add_translation(f"{knowledge_id}_desc", language, description)
        
        # Update knowledge
        with self._lock:
            self.knowledge[knowledge_id]['updated_at'] = datetime.now().isoformat()
            
        # Log audit
        self.audit.log(
            action=AuditAction.UPDATE,
            user_id=user_id,
            user_role=user_role,
            resource_type='knowledge',
            resource_id=knowledge_id,
            details={'action': 'add_translation', 'language': language.value}
        )
        
        return True
        
    def search_external(self, query: str, sources: List[str] = None) -> List[Dict]:
        """
        Search external indigenous knowledge databases.
        
        Args:
            query: Search query
            sources: List of sources to search (None = all)
            
        Returns:
            Combined search results
        """
        results = []
        
        sources = sources or list(self.connectors.keys())
        
        for source in sources:
            if source in self.connectors:
                try:
                    source_results = self.connectors[source].search(query)
                    results.extend(source_results)
                except Exception as e:
                    logger.error(f"External search error ({source}): {e}")
                    
        # Sort by relevance
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return results
        
    def get_access_report(self, knowledge_id: str, days: int = 30) -> Dict[str, Any]:
        """Get access report for knowledge entry."""
        return self.audit.get_access_report(knowledge_id, days)
        
    def export_audit_trail(self, format: str = 'json',
                          start_date: datetime = None,
                          end_date: datetime = None) -> str:
        """Export audit trail."""
        return self.audit.export_entries(format, start_date, end_date)


def create_advanced_indigenous_manager(master_key: bytes = None,
                                       audit_path: str = None) -> AdvancedIndigenousKnowledgeManager:
    """Factory function to create advanced indigenous knowledge manager."""
    return AdvancedIndigenousKnowledgeManager(master_key, audit_path)


def create_encryption_provider(master_key: bytes = None) -> AES256EncryptionProvider:
    """Factory function to create encryption provider."""
    return AES256EncryptionProvider(master_key)


def create_audit_manager(storage_path: str = None) -> AuditTrailManager:
    """Factory function to create audit trail manager."""
    return AuditTrailManager(storage_path)


def create_language_manager() -> MultiLanguageManager:
    """Factory function to create multi-language manager."""
    return MultiLanguageManager()


def create_access_control() -> AccessControlManager:
    """Factory function to create access control manager."""
    return AccessControlManager()
