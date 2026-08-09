"""
Indigenous Knowledge Integration Module for MineralVision.

This module provides comprehensive indigenous knowledge capabilities including:
- AES-256 encryption for confidential knowledge
- Comprehensive audit trail for access tracking
- Multi-language support for traditional names/descriptions
- Integration with external indigenous knowledge databases (TKDL, UNESCO ICH)
"""

from .core import (
    KnowledgeHolder,
    TraditionalKnowledge,
    CulturalHeritageSite,
    ResourceArea,
    ConsultationRecord,
    BenefitSharingAgreement,
    IndigenousKnowledgeManager
)
from .advanced_indigenous import (
    AccessLevel,
    AuditAction,
    Language,
    AuditEntry,
    MultiLanguageText,
    EncryptedContent,
    EncryptionProvider,
    AES256EncryptionProvider,
    AuditTrailManager,
    MultiLanguageManager,
    ExternalDatabaseConnector,
    TKDLConnector,
    ICHConnector,
    AccessControlManager,
    AdvancedIndigenousKnowledgeManager,
    create_advanced_indigenous_manager,
    create_encryption_provider,
    create_audit_manager,
    create_language_manager,
    create_access_control
)

__all__ = [
    'KnowledgeHolder',
    'TraditionalKnowledge',
    'CulturalHeritageSite',
    'ResourceArea',
    'ConsultationRecord',
    'BenefitSharingAgreement',
    'IndigenousKnowledgeManager',
    'AccessLevel',
    'AuditAction',
    'Language',
    'AuditEntry',
    'MultiLanguageText',
    'EncryptedContent',
    'EncryptionProvider',
    'AES256EncryptionProvider',
    'AuditTrailManager',
    'MultiLanguageManager',
    'ExternalDatabaseConnector',
    'TKDLConnector',
    'ICHConnector',
    'AccessControlManager',
    'AdvancedIndigenousKnowledgeManager',
    'create_advanced_indigenous_manager',
    'create_encryption_provider',
    'create_audit_manager',
    'create_language_manager',
    'create_access_control'
]
