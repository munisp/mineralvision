"""
Wazuh SIEM Integration
=======================

Production-grade Security Information and Event Management for MineralVision:
- Log collection and analysis
- Intrusion detection
- File integrity monitoring
- Vulnerability detection
- Compliance monitoring (PCI-DSS, GDPR, HIPAA)
- Incident response automation

Wazuh provides unified XDR and SIEM protection for
endpoints and cloud workloads.
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
import os

logger = logging.getLogger(__name__)

from .._mock_fallback import probe_url, real_client_unavailable


class AlertLevel(Enum):
    """Wazuh alert severity levels."""
    LOW = 3
    MEDIUM = 7
    HIGH = 10
    CRITICAL = 12


class RuleGroup(Enum):
    """Wazuh rule groups."""
    AUTHENTICATION = "authentication"
    SYSLOG = "syslog"
    WEB = "web"
    FIREWALL = "firewall"
    IDS = "ids"
    ROOTCHECK = "rootcheck"
    SYSCHECK = "syscheck"
    VULNERABILITY = "vulnerability"
    COMPLIANCE = "compliance"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class ComplianceStandard(Enum):
    """Compliance standards."""
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    NIST_800_53 = "nist_800_53"
    TSC = "tsc"
    GPG13 = "gpg13"


@dataclass
class WazuhAlert:
    """Represents a Wazuh alert."""
    alert_id: str
    timestamp: datetime
    rule_id: int
    rule_description: str
    rule_level: int
    rule_groups: List[str]
    agent_id: str
    agent_name: str
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    user: Optional[str] = None
    full_log: Optional[str] = None
    decoder_name: Optional[str] = None
    compliance: Dict[str, List[str]] = field(default_factory=dict)
    mitre: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'rule': {
                'id': self.rule_id,
                'description': self.rule_description,
                'level': self.rule_level,
                'groups': self.rule_groups
            },
            'agent': {
                'id': self.agent_id,
                'name': self.agent_name
            },
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'user': self.user,
            'full_log': self.full_log,
            'compliance': self.compliance,
            'mitre': self.mitre,
            'data': self.data
        }
    
    @property
    def severity(self) -> str:
        if self.rule_level >= 12:
            return "critical"
        elif self.rule_level >= 10:
            return "high"
        elif self.rule_level >= 7:
            return "medium"
        else:
            return "low"


@dataclass
class FileIntegrityEvent:
    """File integrity monitoring event."""
    event_id: str
    timestamp: datetime
    path: str
    event_type: str
    agent_id: str
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    old_size: Optional[int] = None
    new_size: Optional[int] = None
    old_permissions: Optional[str] = None
    new_permissions: Optional[str] = None
    user: Optional[str] = None
    process: Optional[str] = None


@dataclass
class VulnerabilityEvent:
    """Vulnerability detection event."""
    event_id: str
    timestamp: datetime
    agent_id: str
    cve_id: str
    severity: str
    package_name: str
    package_version: str
    fixed_version: Optional[str] = None
    title: str = ""
    description: str = ""
    cvss_score: float = 0.0
    references: List[str] = field(default_factory=list)


@dataclass
class WazuhConfig:
    """Wazuh configuration."""
    manager_url: str = "https://localhost:55000"
    api_user: str = "wazuh"
    api_password: str = "wazuh"
    verify_ssl: bool = False
    agents_auto_register: bool = True
    syscheck_frequency: int = 43200
    rootcheck_frequency: int = 43200
    vulnerability_scan_frequency: int = 86400


class MockWazuhAPI:
    """Mock Wazuh API client."""
    
    def __init__(self, config: WazuhConfig):
        self.config = config
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._alerts: List[WazuhAlert] = []
        self._fim_events: List[FileIntegrityEvent] = []
        self._vulnerabilities: List[VulnerabilityEvent] = []
        self._rules: Dict[int, Dict[str, Any]] = {}
        self._token: Optional[str] = None
        
        # Add default agent
        self._agents['000'] = {
            'id': '000',
            'name': 'manager',
            'ip': '127.0.0.1',
            'status': 'active',
            'os': {'name': 'Ubuntu', 'version': '22.04'},
            'version': 'Wazuh v4.7.0'
        }
        
        # Add sample rules
        self._init_rules()
    
    def _init_rules(self):
        """Initialize sample rules."""
        rules = [
            (5501, "Login session opened", 3, ["authentication", "pam"]),
            (5502, "Login session closed", 3, ["authentication", "pam"]),
            (5503, "User login failed", 5, ["authentication", "pam"]),
            (5710, "SSH authentication attempt", 5, ["authentication", "sshd"]),
            (5712, "SSH brute force attack", 10, ["authentication", "sshd"]),
            (31100, "Web attack detected", 10, ["web", "attack"]),
            (31101, "SQL injection attempt", 12, ["web", "attack", "sql_injection"]),
            (31102, "XSS attempt", 10, ["web", "attack", "xss"]),
            (550, "File integrity changed", 7, ["syscheck"]),
            (553, "File deleted", 7, ["syscheck"]),
            (554, "File added", 5, ["syscheck"]),
            (23501, "Vulnerability detected", 7, ["vulnerability-detector"]),
            (87101, "Docker container started", 3, ["docker"]),
            (87102, "Docker container stopped", 3, ["docker"]),
            (87901, "Kubernetes pod created", 3, ["kubernetes"]),
        ]
        
        for rule_id, desc, level, groups in rules:
            self._rules[rule_id] = {
                'id': rule_id,
                'description': desc,
                'level': level,
                'groups': groups
            }
    
    async def authenticate(self) -> str:
        """Authenticate and get token."""
        self._token = str(uuid.uuid4())
        return self._token
    
    async def get_agents(self, status: str = None) -> List[Dict[str, Any]]:
        """Get list of agents."""
        agents = list(self._agents.values())
        if status:
            agents = [a for a in agents if a.get('status') == status]
        return agents
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID."""
        return self._agents.get(agent_id)
    
    async def register_agent(self, name: str, ip: str) -> Dict[str, Any]:
        """Register a new agent."""
        agent_id = str(len(self._agents)).zfill(3)
        agent = {
            'id': agent_id,
            'name': name,
            'ip': ip,
            'status': 'pending',
            'registered_at': datetime.now().isoformat()
        }
        self._agents[agent_id] = agent
        return agent
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        if agent_id in self._agents and agent_id != '000':
            del self._agents[agent_id]
            return True
        return False
    
    async def get_alerts(self, limit: int = 100, 
                        level_min: int = None,
                        rule_groups: List[str] = None) -> List[WazuhAlert]:
        """Get alerts."""
        alerts = self._alerts[-limit:]
        
        if level_min:
            alerts = [a for a in alerts if a.rule_level >= level_min]
        if rule_groups:
            alerts = [a for a in alerts if any(g in a.rule_groups for g in rule_groups)]
        
        return alerts
    
    async def get_fim_events(self, agent_id: str = None,
                            limit: int = 100) -> List[FileIntegrityEvent]:
        """Get file integrity monitoring events."""
        events = self._fim_events[-limit:]
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        return events
    
    async def get_vulnerabilities(self, agent_id: str = None,
                                  severity: str = None,
                                  limit: int = 100) -> List[VulnerabilityEvent]:
        """Get vulnerability events."""
        vulns = self._vulnerabilities[-limit:]
        if agent_id:
            vulns = [v for v in vulns if v.agent_id == agent_id]
        if severity:
            vulns = [v for v in vulns if v.severity == severity]
        return vulns
    
    async def get_rules(self, rule_id: int = None) -> List[Dict[str, Any]]:
        """Get rules."""
        if rule_id:
            rule = self._rules.get(rule_id)
            return [rule] if rule else []
        return list(self._rules.values())
    
    async def run_syscheck(self, agent_id: str) -> Dict[str, Any]:
        """Run syscheck scan on agent."""
        return {'status': 'started', 'agent_id': agent_id}
    
    async def run_rootcheck(self, agent_id: str) -> Dict[str, Any]:
        """Run rootcheck scan on agent."""
        return {'status': 'started', 'agent_id': agent_id}
    
    async def get_summary(self) -> Dict[str, Any]:
        """Get security summary."""
        return {
            'agents': {
                'total': len(self._agents),
                'active': len([a for a in self._agents.values() if a.get('status') == 'active']),
                'disconnected': len([a for a in self._agents.values() if a.get('status') == 'disconnected'])
            },
            'alerts': {
                'total': len(self._alerts),
                'critical': len([a for a in self._alerts if a.rule_level >= 12]),
                'high': len([a for a in self._alerts if 10 <= a.rule_level < 12]),
                'medium': len([a for a in self._alerts if 7 <= a.rule_level < 10]),
                'low': len([a for a in self._alerts if a.rule_level < 7])
            },
            'vulnerabilities': {
                'total': len(self._vulnerabilities),
                'critical': len([v for v in self._vulnerabilities if v.severity == 'critical']),
                'high': len([v for v in self._vulnerabilities if v.severity == 'high'])
            }
        }
    
    def add_alert(self, alert: WazuhAlert) -> None:
        """Add an alert (for testing)."""
        self._alerts.append(alert)
    
    def add_fim_event(self, event: FileIntegrityEvent) -> None:
        """Add FIM event (for testing)."""
        self._fim_events.append(event)
    
    def add_vulnerability(self, vuln: VulnerabilityEvent) -> None:
        """Add vulnerability (for testing)."""
        self._vulnerabilities.append(vuln)


class AlertAnalyzer:
    """Analyze and correlate alerts."""
    
    def __init__(self):
        self._correlation_rules: List[Dict[str, Any]] = []
    
    def add_correlation_rule(self, name: str, conditions: Dict[str, Any],
                            action: Callable) -> None:
        """Add a correlation rule."""
        self._correlation_rules.append({
            'name': name,
            'conditions': conditions,
            'action': action
        })
    
    def analyze(self, alerts: List[WazuhAlert]) -> List[Dict[str, Any]]:
        """Analyze alerts for patterns."""
        findings = []
        
        # Check for brute force
        auth_failures = [a for a in alerts if 'authentication' in a.rule_groups and a.rule_level >= 5]
        if len(auth_failures) >= 5:
            by_source = {}
            for alert in auth_failures:
                if alert.source_ip:
                    by_source[alert.source_ip] = by_source.get(alert.source_ip, 0) + 1
            
            for ip, count in by_source.items():
                if count >= 5:
                    findings.append({
                        'type': 'brute_force_detected',
                        'source_ip': ip,
                        'attempt_count': count,
                        'severity': 'high'
                    })
        
        # Check for web attacks
        web_attacks = [a for a in alerts if 'web' in a.rule_groups and 'attack' in a.rule_groups]
        if web_attacks:
            findings.append({
                'type': 'web_attacks_detected',
                'count': len(web_attacks),
                'severity': 'high' if any(a.rule_level >= 10 for a in web_attacks) else 'medium'
            })
        
        # Check for file integrity changes
        fim_alerts = [a for a in alerts if 'syscheck' in a.rule_groups]
        if len(fim_alerts) >= 10:
            findings.append({
                'type': 'mass_file_changes',
                'count': len(fim_alerts),
                'severity': 'high'
            })
        
        return findings


class ComplianceChecker:
    """Check compliance against standards."""
    
    def __init__(self):
        self._requirements: Dict[ComplianceStandard, List[Dict[str, Any]]] = {
            ComplianceStandard.PCI_DSS: [
                {'id': '1.1', 'description': 'Install and maintain firewall', 'rule_groups': ['firewall']},
                {'id': '2.1', 'description': 'Change default passwords', 'rule_groups': ['authentication']},
                {'id': '6.5', 'description': 'Address common vulnerabilities', 'rule_groups': ['web', 'vulnerability']},
                {'id': '10.1', 'description': 'Implement audit trails', 'rule_groups': ['syslog']},
                {'id': '11.5', 'description': 'Deploy file integrity monitoring', 'rule_groups': ['syscheck']},
            ],
            ComplianceStandard.GDPR: [
                {'id': '32', 'description': 'Security of processing', 'rule_groups': ['authentication', 'syscheck']},
                {'id': '33', 'description': 'Notification of breach', 'rule_groups': ['ids', 'web']},
            ],
            ComplianceStandard.HIPAA: [
                {'id': '164.312(a)', 'description': 'Access control', 'rule_groups': ['authentication']},
                {'id': '164.312(b)', 'description': 'Audit controls', 'rule_groups': ['syslog']},
                {'id': '164.312(c)', 'description': 'Integrity', 'rule_groups': ['syscheck']},
            ]
        }
    
    def check_compliance(self, standard: ComplianceStandard,
                        alerts: List[WazuhAlert]) -> Dict[str, Any]:
        """Check compliance status."""
        requirements = self._requirements.get(standard, [])
        results = []
        
        for req in requirements:
            # Check if we have alerts for this requirement
            relevant_alerts = [
                a for a in alerts
                if any(g in a.rule_groups for g in req['rule_groups'])
            ]
            
            status = 'compliant' if not relevant_alerts else 'needs_review'
            if any(a.rule_level >= 10 for a in relevant_alerts):
                status = 'non_compliant'
            
            results.append({
                'requirement_id': req['id'],
                'description': req['description'],
                'status': status,
                'alert_count': len(relevant_alerts)
            })
        
        compliant_count = len([r for r in results if r['status'] == 'compliant'])
        
        return {
            'standard': standard.value,
            'total_requirements': len(requirements),
            'compliant': compliant_count,
            'compliance_percentage': (compliant_count / len(requirements) * 100) if requirements else 100,
            'requirements': results
        }


class WazuhSIEM:
    """
    Wazuh SIEM integration for MineralVision.
    
    Provides comprehensive security monitoring:
    - Alert management and analysis
    - File integrity monitoring
    - Vulnerability detection
    - Compliance checking
    - Agent management
    
    Example:
        siem = WazuhSIEM()
        await siem.connect()
        
        # Get security summary
        summary = await siem.get_summary()
        
        # Get recent alerts
        alerts = await siem.get_alerts(level_min=7)
        
        # Check compliance
        compliance = siem.check_compliance(ComplianceStandard.PCI_DSS)
    """
    
    def __init__(self, config: WazuhConfig = None):
        self.config = config or WazuhConfig()
        self.api: Optional[MockWazuhAPI] = None
        self.analyzer = AlertAnalyzer()
        self.compliance_checker = ComplianceChecker()
        self._connected = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """True when running on the explicit in-memory mock fallback."""
        return self._degraded
    
    async def connect(self) -> 'WazuhSIEM':
        """
        Connect to Wazuh manager (real connection first).

        A real REST client implementation is not available yet, so this
        falls back to the in-memory mock ONLY when
        MV_ALLOW_MOCK_FALLBACK=true; otherwise raises RuntimeError.
        """
        reachable = probe_url(self.config.manager_url, timeout=2.0)
        reason = (
            f"server reachable at {self.config.manager_url} but real REST client not implemented"
            if reachable else f"no Wazuh manager reachable at {self.config.manager_url}"
        )
        if real_client_unavailable("Wazuh SIEM", reason):
            self._degraded = True
            self.api = MockWazuhAPI(self.config)
        await self.api.authenticate()
        self._connected = True
        logger.info(f"Connected to Wazuh at {self.config.manager_url}")
        return self
    
    async def get_summary(self) -> Dict[str, Any]:
        """Get security summary."""
        if not self.api:
            raise RuntimeError("Not connected")
        return await self.api.get_summary()
    
    async def get_agents(self, status: str = None) -> List[Dict[str, Any]]:
        """Get agents."""
        if not self.api:
            raise RuntimeError("Not connected")
        return await self.api.get_agents(status)
    
    async def register_agent(self, name: str, ip: str) -> Dict[str, Any]:
        """Register a new agent."""
        if not self.api:
            raise RuntimeError("Not connected")
        return await self.api.register_agent(name, ip)
    
    async def get_alerts(self, limit: int = 100,
                        level_min: int = None,
                        rule_groups: List[str] = None) -> List[WazuhAlert]:
        """Get alerts."""
        if not self.api:
            raise RuntimeError("Not connected")
        return await self.api.get_alerts(limit, level_min, rule_groups)
    
    async def get_fim_events(self, agent_id: str = None,
                            limit: int = 100) -> List[FileIntegrityEvent]:
        """Get file integrity events."""
        if not self.api:
            raise RuntimeError("Not connected")
        return await self.api.get_fim_events(agent_id, limit)
    
    async def get_vulnerabilities(self, agent_id: str = None,
                                  severity: str = None,
                                  limit: int = 100) -> List[VulnerabilityEvent]:
        """Get vulnerabilities."""
        if not self.api:
            raise RuntimeError("Not connected")
        return await self.api.get_vulnerabilities(agent_id, severity, limit)
    
    async def analyze_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Analyze recent alerts for patterns."""
        alerts = await self.get_alerts(limit=1000)
        return self.analyzer.analyze(alerts)
    
    def check_compliance(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """Check compliance against a standard."""
        # Use cached alerts for compliance check
        alerts = self.api._alerts if self.api else []
        return self.compliance_checker.check_compliance(standard, alerts)
    
    async def run_scan(self, agent_id: str, scan_type: str = "syscheck") -> Dict[str, Any]:
        """Run a security scan on an agent."""
        if not self.api:
            raise RuntimeError("Not connected")
        
        if scan_type == "syscheck":
            return await self.api.run_syscheck(agent_id)
        elif scan_type == "rootcheck":
            return await self.api.run_rootcheck(agent_id)
        else:
            raise ValueError(f"Unknown scan type: {scan_type}")
    
    def create_alert(self, rule_id: int, agent_id: str,
                    source_ip: str = None, user: str = None,
                    full_log: str = None, data: Dict[str, Any] = None) -> WazuhAlert:
        """Create and record an alert."""
        if not self.api:
            raise RuntimeError("Not connected")
        
        rule = self.api._rules.get(rule_id, {
            'id': rule_id,
            'description': 'Unknown rule',
            'level': 5,
            'groups': []
        })
        
        agent = self.api._agents.get(agent_id, {
            'id': agent_id,
            'name': 'unknown'
        })
        
        alert = WazuhAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            rule_id=rule['id'],
            rule_description=rule['description'],
            rule_level=rule['level'],
            rule_groups=rule['groups'],
            agent_id=agent['id'],
            agent_name=agent.get('name', 'unknown'),
            source_ip=source_ip,
            user=user,
            full_log=full_log,
            data=data or {}
        )
        
        self.api.add_alert(alert)
        return alert
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Factory functions

def create_wazuh_siem(config: WazuhConfig = None) -> WazuhSIEM:
    """Create a Wazuh SIEM instance."""
    return WazuhSIEM(config)


async def create_and_connect_siem(config: WazuhConfig = None) -> WazuhSIEM:
    """Create and connect Wazuh SIEM."""
    siem = WazuhSIEM(config)
    await siem.connect()
    return siem
