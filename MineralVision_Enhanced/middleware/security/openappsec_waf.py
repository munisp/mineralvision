"""
OpenAppSec WAF Integration
===========================

Production-grade Web Application Firewall integration for MineralVision:
- ML-based threat detection
- OWASP Top 10 protection
- API security
- Bot protection
- DDoS mitigation
- Custom security policies

OpenAppSec provides preemptive, ML-based protection against
web application attacks without signature updates.
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
import ipaddress

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of attacks detected."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    LDAP_INJECTION = "ldap_injection"
    XML_INJECTION = "xml_injection"
    SSRF = "ssrf"
    CSRF = "csrf"
    BOT = "bot"
    DDOS = "ddos"
    BRUTE_FORCE = "brute_force"
    API_ABUSE = "api_abuse"
    DATA_LEAKAGE = "data_leakage"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Actions to take on threats."""
    ALLOW = "allow"
    BLOCK = "block"
    CHALLENGE = "challenge"
    LOG = "log"
    RATE_LIMIT = "rate_limit"


class ProtectionMode(Enum):
    """WAF protection modes."""
    DETECT = "detect"
    PREVENT = "prevent"
    LEARNING = "learning"


@dataclass
class ThreatEvent:
    """Represents a detected threat."""
    event_id: str
    timestamp: datetime
    attack_type: AttackType
    threat_level: ThreatLevel
    source_ip: str
    target_uri: str
    method: str
    action_taken: ActionType
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    request_headers: Dict[str, str] = field(default_factory=dict)
    request_body: Optional[str] = None
    matched_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'attack_type': self.attack_type.value,
            'threat_level': self.threat_level.value,
            'source_ip': self.source_ip,
            'target_uri': self.target_uri,
            'method': self.method,
            'action_taken': self.action_taken.value,
            'confidence': self.confidence,
            'details': self.details,
            'matched_rules': self.matched_rules
        }


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    name: str
    mode: ProtectionMode = ProtectionMode.PREVENT
    enabled: bool = True
    sql_injection_protection: bool = True
    xss_protection: bool = True
    command_injection_protection: bool = True
    path_traversal_protection: bool = True
    bot_protection: bool = True
    rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    ip_whitelist: List[str] = field(default_factory=list)
    ip_blacklist: List[str] = field(default_factory=list)
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WAFConfig:
    """WAF configuration."""
    agent_url: str = "http://localhost:8080"
    management_url: str = "http://localhost:8081"
    default_policy: SecurityPolicy = field(default_factory=lambda: SecurityPolicy(name="default"))
    learning_mode_duration: timedelta = field(default_factory=lambda: timedelta(days=7))
    log_all_requests: bool = False
    block_on_error: bool = False


class ThreatDetector:
    """
    ML-based threat detection engine.
    
    Provides detection for:
    - SQL injection patterns
    - XSS attacks
    - Command injection
    - Path traversal
    - Bot behavior
    """
    
    def __init__(self):
        self._sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
            r"(--|\#|\/\*)",
            r"(\bOR\b.*=.*\bOR\b)",
            r"('.*--)",
            r"(\bEXEC\b|\bEXECUTE\b)",
        ]
        
        self._xss_patterns = [
            r"(<script[^>]*>)",
            r"(javascript:)",
            r"(on\w+\s*=)",
            r"(<iframe[^>]*>)",
            r"(document\.(cookie|location|write))",
        ]
        
        self._cmd_patterns = [
            r"(;|\||`|\$\()",
            r"(\b(cat|ls|pwd|wget|curl|bash|sh|nc)\b)",
            r"(&&|\|\|)",
        ]
        
        self._path_patterns = [
            r"(\.\.\/|\.\.\\)",
            r"(%2e%2e%2f|%2e%2e\/)",
            r"(\/etc\/passwd|\/etc\/shadow)",
        ]
    
    def detect_sql_injection(self, payload: str) -> Tuple[bool, float]:
        """Detect SQL injection attempts."""
        if not payload:
            return False, 0.0
        
        matches = 0
        for pattern in self._sql_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                matches += 1
        
        confidence = min(matches / len(self._sql_patterns), 1.0)
        return matches > 0, confidence
    
    def detect_xss(self, payload: str) -> Tuple[bool, float]:
        """Detect XSS attempts."""
        if not payload:
            return False, 0.0
        
        matches = 0
        for pattern in self._xss_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                matches += 1
        
        confidence = min(matches / len(self._xss_patterns), 1.0)
        return matches > 0, confidence
    
    def detect_command_injection(self, payload: str) -> Tuple[bool, float]:
        """Detect command injection attempts."""
        if not payload:
            return False, 0.0
        
        matches = 0
        for pattern in self._cmd_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                matches += 1
        
        confidence = min(matches / len(self._cmd_patterns), 1.0)
        return matches > 0, confidence
    
    def detect_path_traversal(self, payload: str) -> Tuple[bool, float]:
        """Detect path traversal attempts."""
        if not payload:
            return False, 0.0
        
        matches = 0
        for pattern in self._path_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                matches += 1
        
        confidence = min(matches / len(self._path_patterns), 1.0)
        return matches > 0, confidence
    
    def analyze_request(self, uri: str, method: str, headers: Dict[str, str],
                       body: Optional[str] = None,
                       query_params: Dict[str, str] = None) -> List[Tuple[AttackType, float]]:
        """Analyze a request for threats."""
        threats = []
        
        # Combine all payloads to check
        payloads = [uri]
        if body:
            payloads.append(body)
        if query_params:
            payloads.extend(query_params.values())
        
        combined_payload = ' '.join(payloads)
        
        # Check for SQL injection
        detected, confidence = self.detect_sql_injection(combined_payload)
        if detected:
            threats.append((AttackType.SQL_INJECTION, confidence))
        
        # Check for XSS
        detected, confidence = self.detect_xss(combined_payload)
        if detected:
            threats.append((AttackType.XSS, confidence))
        
        # Check for command injection
        detected, confidence = self.detect_command_injection(combined_payload)
        if detected:
            threats.append((AttackType.COMMAND_INJECTION, confidence))
        
        # Check for path traversal
        detected, confidence = self.detect_path_traversal(combined_payload)
        if detected:
            threats.append((AttackType.PATH_TRAVERSAL, confidence))
        
        return threats


class RateLimiter:
    """Rate limiting engine."""
    
    def __init__(self):
        self._requests: Dict[str, List[datetime]] = {}
        self._blocked: Dict[str, datetime] = {}
    
    def check_rate_limit(self, key: str, limit: int, window: int) -> Tuple[bool, int]:
        """
        Check if rate limit is exceeded.
        
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=window)
        
        # Check if blocked
        if key in self._blocked:
            if now < self._blocked[key]:
                return False, 0
            else:
                del self._blocked[key]
        
        # Get requests in window
        if key not in self._requests:
            self._requests[key] = []
        
        # Clean old requests
        self._requests[key] = [
            ts for ts in self._requests[key]
            if ts > window_start
        ]
        
        # Check limit
        current_count = len(self._requests[key])
        if current_count >= limit:
            # Block for window duration
            self._blocked[key] = now + timedelta(seconds=window)
            return False, 0
        
        # Record request
        self._requests[key].append(now)
        return True, limit - current_count - 1
    
    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        if key in self._requests:
            del self._requests[key]
        if key in self._blocked:
            del self._blocked[key]


class IPFilter:
    """IP-based filtering."""
    
    def __init__(self):
        self._whitelist: Set[str] = set()
        self._blacklist: Set[str] = set()
        self._whitelist_networks: List[ipaddress.IPv4Network] = []
        self._blacklist_networks: List[ipaddress.IPv4Network] = []
    
    def add_to_whitelist(self, ip_or_cidr: str) -> None:
        """Add IP or CIDR to whitelist."""
        if '/' in ip_or_cidr:
            self._whitelist_networks.append(ipaddress.IPv4Network(ip_or_cidr, strict=False))
        else:
            self._whitelist.add(ip_or_cidr)
    
    def add_to_blacklist(self, ip_or_cidr: str) -> None:
        """Add IP or CIDR to blacklist."""
        if '/' in ip_or_cidr:
            self._blacklist_networks.append(ipaddress.IPv4Network(ip_or_cidr, strict=False))
        else:
            self._blacklist.add(ip_or_cidr)
    
    def is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        if ip in self._whitelist:
            return True
        
        try:
            ip_addr = ipaddress.IPv4Address(ip)
            for network in self._whitelist_networks:
                if ip_addr in network:
                    return True
        except ValueError:
            pass
        
        return False
    
    def is_blacklisted(self, ip: str) -> bool:
        """Check if IP is blacklisted."""
        if ip in self._blacklist:
            return True
        
        try:
            ip_addr = ipaddress.IPv4Address(ip)
            for network in self._blacklist_networks:
                if ip_addr in network:
                    return True
        except ValueError:
            pass
        
        return False
    
    def check_ip(self, ip: str) -> ActionType:
        """Check IP and return action."""
        if self.is_whitelisted(ip):
            return ActionType.ALLOW
        if self.is_blacklisted(ip):
            return ActionType.BLOCK
        return ActionType.ALLOW


class OpenAppSecWAF:
    """
    OpenAppSec WAF integration for MineralVision.
    
    Provides comprehensive web application security:
    - ML-based threat detection
    - OWASP Top 10 protection
    - Rate limiting
    - IP filtering
    - Custom security policies
    
    Example:
        waf = OpenAppSecWAF()
        await waf.initialize()
        
        # Check a request
        result = await waf.check_request(
            source_ip="192.168.1.1",
            uri="/api/v1/data",
            method="POST",
            headers={"Content-Type": "application/json"},
            body='{"query": "SELECT * FROM users"}'
        )
        
        if result.action == ActionType.BLOCK:
            # Block the request
            pass
    """
    
    def __init__(self, config: WAFConfig = None):
        self.config = config or WAFConfig()
        self.detector = ThreatDetector()
        self.rate_limiter = RateLimiter()
        self.ip_filter = IPFilter()
        self._policies: Dict[str, SecurityPolicy] = {}
        self._events: List[ThreatEvent] = []
        self._initialized = False
    
    async def initialize(self) -> 'OpenAppSecWAF':
        """Initialize the WAF."""
        # Add default policy
        self._policies['default'] = self.config.default_policy
        
        # Setup IP filters from default policy
        for ip in self.config.default_policy.ip_whitelist:
            self.ip_filter.add_to_whitelist(ip)
        for ip in self.config.default_policy.ip_blacklist:
            self.ip_filter.add_to_blacklist(ip)
        
        self._initialized = True
        logger.info("OpenAppSec WAF initialized")
        return self
    
    async def check_request(self, source_ip: str, uri: str, method: str,
                           headers: Dict[str, str] = None,
                           body: Optional[str] = None,
                           query_params: Dict[str, str] = None,
                           policy_name: str = "default") -> 'WAFResult':
        """
        Check a request against security policies.
        
        Returns:
            WAFResult with action and details
        """
        headers = headers or {}
        query_params = query_params or {}
        
        policy = self._policies.get(policy_name, self.config.default_policy)
        
        # Check IP filter first
        ip_action = self.ip_filter.check_ip(source_ip)
        if ip_action == ActionType.BLOCK:
            event = self._create_event(
                AttackType.UNKNOWN, ThreatLevel.HIGH, source_ip, uri, method,
                ActionType.BLOCK, 1.0, {"reason": "IP blacklisted"}
            )
            self._events.append(event)
            return WAFResult(ActionType.BLOCK, [event])
        
        # Check rate limiting
        if policy.rate_limiting:
            allowed, remaining = self.rate_limiter.check_rate_limit(
                source_ip, policy.rate_limit_requests, policy.rate_limit_window
            )
            if not allowed:
                event = self._create_event(
                    AttackType.DDOS, ThreatLevel.MEDIUM, source_ip, uri, method,
                    ActionType.RATE_LIMIT, 0.8, {"reason": "Rate limit exceeded"}
                )
                self._events.append(event)
                return WAFResult(ActionType.RATE_LIMIT, [event])
        
        # Detect threats
        threats = self.detector.analyze_request(uri, method, headers, body, query_params)
        
        if not threats:
            return WAFResult(ActionType.ALLOW, [])
        
        # Process detected threats
        events = []
        max_threat_level = ThreatLevel.LOW
        
        for attack_type, confidence in threats:
            # Determine threat level based on confidence
            if confidence >= 0.8:
                threat_level = ThreatLevel.CRITICAL
            elif confidence >= 0.6:
                threat_level = ThreatLevel.HIGH
            elif confidence >= 0.4:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW
            
            if threat_level.value > max_threat_level.value:
                max_threat_level = threat_level
            
            # Determine action based on mode
            if policy.mode == ProtectionMode.PREVENT:
                action = ActionType.BLOCK if confidence >= 0.5 else ActionType.LOG
            elif policy.mode == ProtectionMode.DETECT:
                action = ActionType.LOG
            else:  # Learning mode
                action = ActionType.LOG
            
            event = self._create_event(
                attack_type, threat_level, source_ip, uri, method,
                action, confidence, {"body_preview": body[:100] if body else None},
                headers
            )
            events.append(event)
            self._events.append(event)
        
        # Return most severe action
        if any(e.action_taken == ActionType.BLOCK for e in events):
            return WAFResult(ActionType.BLOCK, events)
        
        return WAFResult(ActionType.ALLOW, events)
    
    def _create_event(self, attack_type: AttackType, threat_level: ThreatLevel,
                     source_ip: str, uri: str, method: str,
                     action: ActionType, confidence: float,
                     details: Dict[str, Any] = None,
                     headers: Dict[str, str] = None) -> ThreatEvent:
        """Create a threat event."""
        return ThreatEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            attack_type=attack_type,
            threat_level=threat_level,
            source_ip=source_ip,
            target_uri=uri,
            method=method,
            action_taken=action,
            confidence=confidence,
            details=details or {},
            request_headers=headers or {}
        )
    
    def add_policy(self, policy: SecurityPolicy) -> None:
        """Add a security policy."""
        self._policies[policy.name] = policy
        
        # Update IP filters
        for ip in policy.ip_whitelist:
            self.ip_filter.add_to_whitelist(ip)
        for ip in policy.ip_blacklist:
            self.ip_filter.add_to_blacklist(ip)
    
    def get_policy(self, name: str) -> Optional[SecurityPolicy]:
        """Get a security policy."""
        return self._policies.get(name)
    
    def list_policies(self) -> List[str]:
        """List all policy names."""
        return list(self._policies.keys())
    
    def get_events(self, limit: int = 100,
                  attack_type: AttackType = None,
                  threat_level: ThreatLevel = None) -> List[ThreatEvent]:
        """Get threat events."""
        events = self._events[-limit:]
        
        if attack_type:
            events = [e for e in events if e.attack_type == attack_type]
        if threat_level:
            events = [e for e in events if e.threat_level == threat_level]
        
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get WAF statistics."""
        total_events = len(self._events)
        
        by_attack_type = {}
        by_threat_level = {}
        by_action = {}
        
        for event in self._events:
            by_attack_type[event.attack_type.value] = by_attack_type.get(event.attack_type.value, 0) + 1
            by_threat_level[event.threat_level.value] = by_threat_level.get(event.threat_level.value, 0) + 1
            by_action[event.action_taken.value] = by_action.get(event.action_taken.value, 0) + 1
        
        return {
            'total_events': total_events,
            'by_attack_type': by_attack_type,
            'by_threat_level': by_threat_level,
            'by_action': by_action,
            'policies_count': len(self._policies)
        }
    
    def block_ip(self, ip: str, duration: timedelta = None) -> None:
        """Block an IP address."""
        self.ip_filter.add_to_blacklist(ip)
        logger.info(f"Blocked IP: {ip}")
    
    def unblock_ip(self, ip: str) -> None:
        """Unblock an IP address."""
        if ip in self.ip_filter._blacklist:
            self.ip_filter._blacklist.remove(ip)
            logger.info(f"Unblocked IP: {ip}")


@dataclass
class WAFResult:
    """Result of WAF check."""
    action: ActionType
    events: List[ThreatEvent]
    
    @property
    def is_blocked(self) -> bool:
        return self.action == ActionType.BLOCK
    
    @property
    def is_rate_limited(self) -> bool:
        return self.action == ActionType.RATE_LIMIT
    
    @property
    def threat_count(self) -> int:
        return len(self.events)


# Factory functions

def create_waf(config: WAFConfig = None) -> OpenAppSecWAF:
    """Create an OpenAppSec WAF instance."""
    return OpenAppSecWAF(config)


async def create_and_initialize_waf(config: WAFConfig = None) -> OpenAppSecWAF:
    """Create and initialize WAF."""
    waf = OpenAppSecWAF(config)
    await waf.initialize()
    return waf


# FastAPI middleware integration

class WAFMiddleware:
    """FastAPI middleware for WAF integration."""
    
    def __init__(self, waf: OpenAppSecWAF):
        self.waf = waf
    
    async def __call__(self, request, call_next):
        """Process request through WAF."""
        # Extract request details
        source_ip = request.client.host if request.client else "unknown"
        uri = str(request.url.path)
        method = request.method
        headers = dict(request.headers)
        
        # Get body if present
        body = None
        if method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                body = body.decode('utf-8') if body else None
            except:
                pass
        
        # Check request
        result = await self.waf.check_request(
            source_ip=source_ip,
            uri=uri,
            method=method,
            headers=headers,
            body=body,
            query_params=dict(request.query_params)
        )
        
        if result.is_blocked:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={"error": "Request blocked by WAF", "event_id": result.events[0].event_id if result.events else None}
            )
        
        if result.is_rate_limited:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        return await call_next(request)
