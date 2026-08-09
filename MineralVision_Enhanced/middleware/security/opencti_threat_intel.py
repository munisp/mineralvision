"""
OpenCTI Threat Intelligence Integration
========================================

Production-grade cyber threat intelligence platform integration:
- STIX/TAXII threat data ingestion
- Indicator of Compromise (IoC) management
- Threat actor tracking
- Attack pattern analysis
- Vulnerability correlation
- Intelligence sharing

OpenCTI provides a unified platform for managing
cyber threat intelligence knowledge.
"""

import asyncio
import json
import logging
import uuid
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import re

logger = logging.getLogger(__name__)

from .._mock_fallback import probe_url, real_client_unavailable


class IndicatorType(Enum):
    """Types of indicators."""
    IP_ADDRESS = "ipv4-addr"
    DOMAIN = "domain-name"
    URL = "url"
    FILE_HASH_MD5 = "file:hashes.MD5"
    FILE_HASH_SHA1 = "file:hashes.SHA-1"
    FILE_HASH_SHA256 = "file:hashes.SHA-256"
    EMAIL = "email-addr"
    REGISTRY_KEY = "windows-registry-key"
    USER_AGENT = "user-agent"
    CVE = "vulnerability"


class ThreatActorType(Enum):
    """Types of threat actors."""
    NATION_STATE = "nation-state"
    CRIME_SYNDICATE = "crime-syndicate"
    HACTIVIST = "hactivist"
    INSIDER = "insider-threat"
    UNKNOWN = "unknown"


class MalwareType(Enum):
    """Types of malware."""
    RANSOMWARE = "ransomware"
    TROJAN = "trojan"
    WORM = "worm"
    SPYWARE = "spyware"
    ROOTKIT = "rootkit"
    BACKDOOR = "backdoor"
    BOTNET = "botnet"
    CRYPTOMINER = "cryptominer"


class ConfidenceLevel(Enum):
    """Confidence levels for intelligence."""
    LOW = 25
    MEDIUM = 50
    HIGH = 75
    CONFIRMED = 100


@dataclass
class Indicator:
    """Threat indicator (IoC)."""
    id: str
    type: IndicatorType
    value: str
    name: str
    description: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    valid_from: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    kill_chain_phases: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    external_references: List[Dict[str, str]] = field(default_factory=list)
    created_by: Optional[str] = None
    
    def to_stix(self) -> Dict[str, Any]:
        """Convert to STIX format."""
        return {
            "type": "indicator",
            "spec_version": "2.1",
            "id": f"indicator--{self.id}",
            "created": self.valid_from.isoformat(),
            "name": self.name,
            "description": self.description,
            "indicator_types": [self.type.value],
            "pattern": f"[{self.type.value} = '{self.value}']",
            "pattern_type": "stix",
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "confidence": self.confidence.value,
            "labels": self.labels,
            "kill_chain_phases": [
                {"kill_chain_name": "lockheed-martin-cyber-kill-chain", "phase_name": phase}
                for phase in self.kill_chain_phases
            ],
            "external_references": self.external_references
        }


@dataclass
class ThreatActor:
    """Threat actor entity."""
    id: str
    name: str
    actor_type: ThreatActorType
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    sophistication: str = "intermediate"
    resource_level: str = "organization"
    primary_motivation: str = "unknown"
    secondary_motivations: List[str] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    labels: List[str] = field(default_factory=list)
    
    def to_stix(self) -> Dict[str, Any]:
        """Convert to STIX format."""
        return {
            "type": "threat-actor",
            "spec_version": "2.1",
            "id": f"threat-actor--{self.id}",
            "name": self.name,
            "description": self.description,
            "threat_actor_types": [self.actor_type.value],
            "aliases": self.aliases,
            "goals": self.goals,
            "sophistication": self.sophistication,
            "resource_level": self.resource_level,
            "primary_motivation": self.primary_motivation,
            "secondary_motivations": self.secondary_motivations,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "labels": self.labels
        }


@dataclass
class AttackPattern:
    """Attack pattern (TTP)."""
    id: str
    name: str
    description: str = ""
    mitre_id: Optional[str] = None
    kill_chain_phases: List[str] = field(default_factory=list)
    external_references: List[Dict[str, str]] = field(default_factory=list)
    
    def to_stix(self) -> Dict[str, Any]:
        """Convert to STIX format."""
        refs = self.external_references.copy()
        if self.mitre_id:
            refs.append({
                "source_name": "mitre-attack",
                "external_id": self.mitre_id,
                "url": f"https://attack.mitre.org/techniques/{self.mitre_id}"
            })
        
        return {
            "type": "attack-pattern",
            "spec_version": "2.1",
            "id": f"attack-pattern--{self.id}",
            "name": self.name,
            "description": self.description,
            "kill_chain_phases": [
                {"kill_chain_name": "mitre-attack", "phase_name": phase}
                for phase in self.kill_chain_phases
            ],
            "external_references": refs
        }


@dataclass
class Malware:
    """Malware entity."""
    id: str
    name: str
    malware_type: MalwareType
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    is_family: bool = False
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    capabilities: List[str] = field(default_factory=list)
    
    def to_stix(self) -> Dict[str, Any]:
        """Convert to STIX format."""
        return {
            "type": "malware",
            "spec_version": "2.1",
            "id": f"malware--{self.id}",
            "name": self.name,
            "description": self.description,
            "malware_types": [self.malware_type.value],
            "aliases": self.aliases,
            "is_family": self.is_family,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "capabilities": self.capabilities
        }


@dataclass
class OpenCTIConfig:
    """OpenCTI configuration."""
    url: str = "http://localhost:8080"
    token: str = ""
    ssl_verify: bool = True
    proxy: Optional[str] = None


class MockOpenCTIClient:
    """Mock OpenCTI GraphQL client."""
    
    def __init__(self, config: OpenCTIConfig):
        self.config = config
        self._indicators: Dict[str, Indicator] = {}
        self._threat_actors: Dict[str, ThreatActor] = {}
        self._attack_patterns: Dict[str, AttackPattern] = {}
        self._malware: Dict[str, Malware] = {}
        self._relationships: List[Dict[str, Any]] = []
        
        # Initialize with sample data
        self._init_sample_data()
    
    def _init_sample_data(self):
        """Initialize sample threat intelligence."""
        # Sample indicators
        indicators = [
            Indicator(
                id=str(uuid.uuid4()),
                type=IndicatorType.IP_ADDRESS,
                value="192.168.1.100",
                name="Suspicious IP",
                description="Known C2 server",
                confidence=ConfidenceLevel.HIGH,
                labels=["malicious", "c2"]
            ),
            Indicator(
                id=str(uuid.uuid4()),
                type=IndicatorType.DOMAIN,
                value="malicious-domain.com",
                name="Malicious Domain",
                description="Phishing domain",
                confidence=ConfidenceLevel.CONFIRMED,
                labels=["phishing"]
            ),
            Indicator(
                id=str(uuid.uuid4()),
                type=IndicatorType.FILE_HASH_SHA256,
                value="a" * 64,
                name="Malware Hash",
                description="Known ransomware sample",
                confidence=ConfidenceLevel.CONFIRMED,
                labels=["ransomware"]
            )
        ]
        
        for ind in indicators:
            self._indicators[ind.id] = ind
        
        # Sample threat actors
        actors = [
            ThreatActor(
                id=str(uuid.uuid4()),
                name="APT29",
                actor_type=ThreatActorType.NATION_STATE,
                description="Russian state-sponsored threat actor",
                aliases=["Cozy Bear", "The Dukes"],
                sophistication="expert",
                resource_level="government"
            ),
            ThreatActor(
                id=str(uuid.uuid4()),
                name="FIN7",
                actor_type=ThreatActorType.CRIME_SYNDICATE,
                description="Financially motivated threat group",
                aliases=["Carbanak"],
                sophistication="expert",
                primary_motivation="financial-gain"
            )
        ]
        
        for actor in actors:
            self._threat_actors[actor.id] = actor
        
        # Sample attack patterns
        patterns = [
            AttackPattern(
                id=str(uuid.uuid4()),
                name="Spearphishing Attachment",
                description="Adversaries send spearphishing emails with malicious attachments",
                mitre_id="T1566.001",
                kill_chain_phases=["initial-access"]
            ),
            AttackPattern(
                id=str(uuid.uuid4()),
                name="PowerShell",
                description="Adversaries use PowerShell for execution",
                mitre_id="T1059.001",
                kill_chain_phases=["execution"]
            )
        ]
        
        for pattern in patterns:
            self._attack_patterns[pattern.id] = pattern
    
    async def create_indicator(self, indicator: Indicator) -> Indicator:
        """Create an indicator."""
        self._indicators[indicator.id] = indicator
        return indicator
    
    async def get_indicator(self, indicator_id: str) -> Optional[Indicator]:
        """Get indicator by ID."""
        return self._indicators.get(indicator_id)
    
    async def search_indicators(self, value: str = None,
                               indicator_type: IndicatorType = None,
                               labels: List[str] = None) -> List[Indicator]:
        """Search indicators."""
        results = list(self._indicators.values())
        
        if value:
            results = [i for i in results if value.lower() in i.value.lower()]
        if indicator_type:
            results = [i for i in results if i.type == indicator_type]
        if labels:
            results = [i for i in results if any(l in i.labels for l in labels)]
        
        return results
    
    async def delete_indicator(self, indicator_id: str) -> bool:
        """Delete an indicator."""
        if indicator_id in self._indicators:
            del self._indicators[indicator_id]
            return True
        return False
    
    async def create_threat_actor(self, actor: ThreatActor) -> ThreatActor:
        """Create a threat actor."""
        self._threat_actors[actor.id] = actor
        return actor
    
    async def get_threat_actor(self, actor_id: str) -> Optional[ThreatActor]:
        """Get threat actor by ID."""
        return self._threat_actors.get(actor_id)
    
    async def search_threat_actors(self, name: str = None,
                                   actor_type: ThreatActorType = None) -> List[ThreatActor]:
        """Search threat actors."""
        results = list(self._threat_actors.values())
        
        if name:
            results = [a for a in results if name.lower() in a.name.lower() or 
                      any(name.lower() in alias.lower() for alias in a.aliases)]
        if actor_type:
            results = [a for a in results if a.actor_type == actor_type]
        
        return results
    
    async def create_attack_pattern(self, pattern: AttackPattern) -> AttackPattern:
        """Create an attack pattern."""
        self._attack_patterns[pattern.id] = pattern
        return pattern
    
    async def get_attack_pattern(self, pattern_id: str) -> Optional[AttackPattern]:
        """Get attack pattern by ID."""
        return self._attack_patterns.get(pattern_id)
    
    async def search_attack_patterns(self, name: str = None,
                                     mitre_id: str = None) -> List[AttackPattern]:
        """Search attack patterns."""
        results = list(self._attack_patterns.values())
        
        if name:
            results = [p for p in results if name.lower() in p.name.lower()]
        if mitre_id:
            results = [p for p in results if p.mitre_id == mitre_id]
        
        return results
    
    async def create_relationship(self, source_id: str, target_id: str,
                                  relationship_type: str) -> Dict[str, Any]:
        """Create a relationship between entities."""
        rel = {
            'id': str(uuid.uuid4()),
            'source_ref': source_id,
            'target_ref': target_id,
            'relationship_type': relationship_type,
            'created': datetime.now().isoformat()
        }
        self._relationships.append(rel)
        return rel
    
    async def get_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get relationships for an entity."""
        return [
            r for r in self._relationships
            if r['source_ref'] == entity_id or r['target_ref'] == entity_id
        ]
    
    async def export_stix_bundle(self) -> Dict[str, Any]:
        """Export all data as STIX bundle."""
        objects = []
        
        for ind in self._indicators.values():
            objects.append(ind.to_stix())
        for actor in self._threat_actors.values():
            objects.append(actor.to_stix())
        for pattern in self._attack_patterns.values():
            objects.append(pattern.to_stix())
        for malware in self._malware.values():
            objects.append(malware.to_stix())
        
        return {
            "type": "bundle",
            "id": f"bundle--{uuid.uuid4()}",
            "objects": objects
        }


class ThreatMatcher:
    """Match observables against threat intelligence."""
    
    def __init__(self, client: MockOpenCTIClient):
        self.client = client
    
    async def match_ip(self, ip: str) -> List[Indicator]:
        """Match an IP address against indicators."""
        return await self.client.search_indicators(
            value=ip,
            indicator_type=IndicatorType.IP_ADDRESS
        )
    
    async def match_domain(self, domain: str) -> List[Indicator]:
        """Match a domain against indicators."""
        return await self.client.search_indicators(
            value=domain,
            indicator_type=IndicatorType.DOMAIN
        )
    
    async def match_hash(self, file_hash: str) -> List[Indicator]:
        """Match a file hash against indicators."""
        # Determine hash type by length
        if len(file_hash) == 32:
            hash_type = IndicatorType.FILE_HASH_MD5
        elif len(file_hash) == 40:
            hash_type = IndicatorType.FILE_HASH_SHA1
        elif len(file_hash) == 64:
            hash_type = IndicatorType.FILE_HASH_SHA256
        else:
            return []
        
        return await self.client.search_indicators(
            value=file_hash,
            indicator_type=hash_type
        )
    
    async def match_url(self, url: str) -> List[Indicator]:
        """Match a URL against indicators."""
        return await self.client.search_indicators(
            value=url,
            indicator_type=IndicatorType.URL
        )
    
    async def enrich_observable(self, observable_type: str,
                               observable_value: str) -> Dict[str, Any]:
        """Enrich an observable with threat intelligence."""
        result = {
            'observable_type': observable_type,
            'observable_value': observable_value,
            'indicators': [],
            'threat_actors': [],
            'attack_patterns': [],
            'risk_score': 0
        }
        
        # Match against indicators
        if observable_type == 'ip':
            indicators = await self.match_ip(observable_value)
        elif observable_type == 'domain':
            indicators = await self.match_domain(observable_value)
        elif observable_type == 'hash':
            indicators = await self.match_hash(observable_value)
        elif observable_type == 'url':
            indicators = await self.match_url(observable_value)
        else:
            indicators = []
        
        result['indicators'] = [i.to_stix() for i in indicators]
        
        # Calculate risk score
        if indicators:
            max_confidence = max(i.confidence.value for i in indicators)
            result['risk_score'] = max_confidence
        
        return result


class OpenCTIThreatIntel:
    """
    OpenCTI threat intelligence integration for MineralVision.
    
    Provides cyber threat intelligence capabilities:
    - Indicator management
    - Threat actor tracking
    - Attack pattern analysis
    - Observable enrichment
    - STIX data exchange
    
    Example:
        cti = OpenCTIThreatIntel()
        await cti.connect()
        
        # Search for indicators
        indicators = await cti.search_indicators(labels=["ransomware"])
        
        # Enrich an IP
        enrichment = await cti.enrich_observable("ip", "192.168.1.100")
        
        # Export STIX bundle
        bundle = await cti.export_stix()
    """
    
    def __init__(self, config: OpenCTIConfig = None):
        self.config = config or OpenCTIConfig()
        self.client: Optional[MockOpenCTIClient] = None
        self.matcher: Optional[ThreatMatcher] = None
        self._connected = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """True when running on the explicit in-memory mock fallback."""
        return self._degraded
    
    async def connect(self) -> 'OpenCTIThreatIntel':
        """
        Connect to OpenCTI (real connection first).

        A real GraphQL client implementation is not available yet, so this
        falls back to the in-memory mock ONLY when
        MV_ALLOW_MOCK_FALLBACK=true; otherwise raises RuntimeError.
        """
        reachable = probe_url(self.config.url, timeout=2.0)
        reason = (
            f"server reachable at {self.config.url} but real GraphQL client not implemented"
            if reachable else f"no OpenCTI server reachable at {self.config.url}"
        )
        if real_client_unavailable("OpenCTI", reason):
            self._degraded = True
            self.client = MockOpenCTIClient(self.config)
        self.matcher = ThreatMatcher(self.client)
        self._connected = True
        logger.info(f"Connected to OpenCTI at {self.config.url}")
        return self
    
    async def create_indicator(self, indicator_type: IndicatorType,
                              value: str, name: str,
                              description: str = "",
                              confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
                              labels: List[str] = None) -> Indicator:
        """Create a new indicator."""
        if not self.client:
            raise RuntimeError("Not connected")
        
        indicator = Indicator(
            id=str(uuid.uuid4()),
            type=indicator_type,
            value=value,
            name=name,
            description=description,
            confidence=confidence,
            labels=labels or []
        )
        
        return await self.client.create_indicator(indicator)
    
    async def search_indicators(self, value: str = None,
                               indicator_type: IndicatorType = None,
                               labels: List[str] = None) -> List[Indicator]:
        """Search indicators."""
        if not self.client:
            raise RuntimeError("Not connected")
        return await self.client.search_indicators(value, indicator_type, labels)
    
    async def search_threat_actors(self, name: str = None,
                                   actor_type: ThreatActorType = None) -> List[ThreatActor]:
        """Search threat actors."""
        if not self.client:
            raise RuntimeError("Not connected")
        return await self.client.search_threat_actors(name, actor_type)
    
    async def search_attack_patterns(self, name: str = None,
                                     mitre_id: str = None) -> List[AttackPattern]:
        """Search attack patterns."""
        if not self.client:
            raise RuntimeError("Not connected")
        return await self.client.search_attack_patterns(name, mitre_id)
    
    async def enrich_observable(self, observable_type: str,
                               observable_value: str) -> Dict[str, Any]:
        """Enrich an observable with threat intelligence."""
        if not self.matcher:
            raise RuntimeError("Not connected")
        return await self.matcher.enrich_observable(observable_type, observable_value)
    
    async def export_stix(self) -> Dict[str, Any]:
        """Export all intelligence as STIX bundle."""
        if not self.client:
            raise RuntimeError("Not connected")
        return await self.client.export_stix_bundle()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get threat intelligence statistics."""
        if not self.client:
            raise RuntimeError("Not connected")
        
        return {
            'indicators': len(self.client._indicators),
            'threat_actors': len(self.client._threat_actors),
            'attack_patterns': len(self.client._attack_patterns),
            'malware': len(self.client._malware),
            'relationships': len(self.client._relationships)
        }
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Factory functions

def create_opencti(config: OpenCTIConfig = None) -> OpenCTIThreatIntel:
    """Create an OpenCTI instance."""
    return OpenCTIThreatIntel(config)


async def create_and_connect_opencti(config: OpenCTIConfig = None) -> OpenCTIThreatIntel:
    """Create and connect OpenCTI."""
    cti = OpenCTIThreatIntel(config)
    await cti.connect()
    return cti
