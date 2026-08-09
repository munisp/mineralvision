"""
Alert System Module for MineralVision Crop Monitoring.

Comprehensive alert and notification system:
- Vegetation stress detection alerts
- Weather risk alerts
- Pest/disease risk notifications
- Growth stage reminders
- Irrigation scheduling alerts
- Harvest timing alerts
- Email/push notification integration
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, date, timedelta
import uuid
import logging
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert categories."""
    VEGETATION = "vegetation"
    WEATHER = "weather"
    PEST_DISEASE = "pest_disease"
    IRRIGATION = "irrigation"
    FERTILIZATION = "fertilization"
    HARVEST = "harvest"
    GROWTH_STAGE = "growth_stage"
    EQUIPMENT = "equipment"
    COMPLIANCE = "compliance"
    SYSTEM = "system"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    SNOOZED = "snoozed"


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEBHOOK = "webhook"
    SLACK = "slack"
    WHATSAPP = "whatsapp"


class TriggerType(Enum):
    """Alert trigger types."""
    THRESHOLD = "threshold"
    CHANGE_RATE = "change_rate"
    ANOMALY = "anomaly"
    SCHEDULE = "schedule"
    FORECAST = "forecast"
    MANUAL = "manual"


@dataclass
class AlertRule:
    """Rule for generating alerts."""
    rule_id: str
    name: str
    category: AlertCategory
    
    # Trigger conditions
    trigger_type: TriggerType = TriggerType.THRESHOLD
    metric: str = ""  # e.g., "ndvi", "temperature", "soil_moisture"
    operator: str = "<"  # <, >, <=, >=, ==, !=, change
    threshold: float = 0.0
    threshold_unit: str = ""
    
    # Change rate specific
    change_period_days: int = 7
    change_threshold: float = 0.0
    
    # Severity mapping
    severity: AlertSeverity = AlertSeverity.MEDIUM
    
    # Scope
    field_ids: List[str] = field(default_factory=list)  # Empty = all fields
    crop_types: List[str] = field(default_factory=list)  # Empty = all crops
    
    # Notification settings
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    
    # Timing
    cooldown_hours: int = 24  # Minimum time between alerts
    active_hours: Tuple[int, int] = (6, 22)  # Hours when alerts can be sent
    
    # Status
    is_enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'category': self.category.value,
            'trigger_type': self.trigger_type.value,
            'metric': self.metric,
            'operator': self.operator,
            'threshold': self.threshold,
            'severity': self.severity.value,
            'is_enabled': self.is_enabled,
            'trigger_count': self.trigger_count,
            'notification_channels': [c.value for c in self.notification_channels]
        }


@dataclass
class Alert:
    """Individual alert instance."""
    alert_id: str
    rule_id: str
    
    # Alert details
    category: AlertCategory
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    
    # Content
    title: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    field_id: str = ""
    field_name: str = ""
    crop_type: str = ""
    
    # Trigger info
    trigger_value: float = 0.0
    threshold_value: float = 0.0
    trigger_metric: str = ""
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    action_items: List[Dict[str, str]] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # User interaction
    acknowledged_by: str = ""
    resolved_by: str = ""
    notes: str = ""
    
    # Notification tracking
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'rule_id': self.rule_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'status': self.status.value,
            'title': self.title,
            'message': self.message,
            'field_id': self.field_id,
            'field_name': self.field_name,
            'crop_type': self.crop_type,
            'trigger_value': self.trigger_value,
            'threshold_value': self.threshold_value,
            'recommendations': self.recommendations,
            'created_at': self.created_at.isoformat(),
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class NotificationTemplate:
    """Template for alert notifications."""
    template_id: str
    name: str
    category: AlertCategory
    
    # Templates by channel
    email_subject: str = ""
    email_body: str = ""
    sms_message: str = ""
    push_title: str = ""
    push_body: str = ""
    
    # Placeholders: {field_name}, {crop_type}, {value}, {threshold}, {date}, etc.
    
    def render(self, channel: NotificationChannel, context: Dict[str, Any]) -> Dict[str, str]:
        """Render template with context."""
        if channel == NotificationChannel.EMAIL:
            return {
                'subject': self._render_string(self.email_subject, context),
                'body': self._render_string(self.email_body, context)
            }
        elif channel == NotificationChannel.SMS:
            return {
                'message': self._render_string(self.sms_message, context)
            }
        elif channel == NotificationChannel.PUSH:
            return {
                'title': self._render_string(self.push_title, context),
                'body': self._render_string(self.push_body, context)
            }
        return {}
    
    def _render_string(self, template: str, context: Dict[str, Any]) -> str:
        """Render template string with context."""
        result = template
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


@dataclass
class AlertSubscription:
    """User subscription to alerts."""
    subscription_id: str
    user_id: str
    user_email: str = ""
    user_phone: str = ""
    
    # Subscription settings
    categories: List[AlertCategory] = field(default_factory=list)  # Empty = all
    severities: List[AlertSeverity] = field(default_factory=list)  # Empty = all
    field_ids: List[str] = field(default_factory=list)  # Empty = all
    
    # Channels
    channels: List[NotificationChannel] = field(default_factory=list)
    
    # Preferences
    digest_mode: bool = False  # Send daily digest instead of immediate
    digest_time: str = "08:00"  # Time for daily digest
    quiet_hours: Tuple[int, int] = (22, 7)  # No notifications during these hours
    
    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'subscription_id': self.subscription_id,
            'user_id': self.user_id,
            'user_email': self.user_email,
            'categories': [c.value for c in self.categories],
            'severities': [s.value for s in self.severities],
            'channels': [c.value for c in self.channels],
            'digest_mode': self.digest_mode,
            'is_active': self.is_active
        }


class AlertRuleEngine:
    """Engine for evaluating alert rules."""
    
    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._default_rules = self._create_default_rules()
    
    def _create_default_rules(self) -> List[AlertRule]:
        """Create default alert rules."""
        return [
            # Vegetation stress alerts
            AlertRule(
                rule_id="veg_stress_critical",
                name="Critical Vegetation Stress",
                category=AlertCategory.VEGETATION,
                trigger_type=TriggerType.THRESHOLD,
                metric="ndvi",
                operator="<",
                threshold=0.25,
                severity=AlertSeverity.CRITICAL,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.PUSH],
                description="Alert when NDVI drops below critical threshold"
            ),
            AlertRule(
                rule_id="veg_stress_warning",
                name="Vegetation Stress Warning",
                category=AlertCategory.VEGETATION,
                trigger_type=TriggerType.THRESHOLD,
                metric="ndvi",
                operator="<",
                threshold=0.40,
                severity=AlertSeverity.MEDIUM,
                notification_channels=[NotificationChannel.IN_APP],
                description="Alert when NDVI indicates moderate stress"
            ),
            AlertRule(
                rule_id="ndvi_rapid_decline",
                name="Rapid NDVI Decline",
                category=AlertCategory.VEGETATION,
                trigger_type=TriggerType.CHANGE_RATE,
                metric="ndvi",
                operator="change",
                change_period_days=7,
                change_threshold=-0.15,
                severity=AlertSeverity.HIGH,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.PUSH],
                description="Alert when NDVI drops rapidly over 7 days"
            ),
            
            # Weather alerts
            AlertRule(
                rule_id="drought_risk",
                name="Drought Risk Alert",
                category=AlertCategory.WEATHER,
                trigger_type=TriggerType.FORECAST,
                metric="precipitation_days",
                operator="<",
                threshold=1,
                severity=AlertSeverity.HIGH,
                notification_channels=[NotificationChannel.EMAIL],
                description="Alert when no rain forecast for extended period"
            ),
            AlertRule(
                rule_id="flood_risk",
                name="Flood Risk Alert",
                category=AlertCategory.WEATHER,
                trigger_type=TriggerType.FORECAST,
                metric="precipitation_mm",
                operator=">",
                threshold=100,
                severity=AlertSeverity.HIGH,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
                description="Alert when heavy rainfall forecast"
            ),
            AlertRule(
                rule_id="heat_stress",
                name="Heat Stress Alert",
                category=AlertCategory.WEATHER,
                trigger_type=TriggerType.THRESHOLD,
                metric="temperature_max",
                operator=">",
                threshold=38,
                threshold_unit="C",
                severity=AlertSeverity.MEDIUM,
                notification_channels=[NotificationChannel.PUSH],
                description="Alert when high temperatures forecast"
            ),
            
            # Irrigation alerts
            AlertRule(
                rule_id="irrigation_needed",
                name="Irrigation Required",
                category=AlertCategory.IRRIGATION,
                trigger_type=TriggerType.THRESHOLD,
                metric="soil_moisture",
                operator="<",
                threshold=30,
                threshold_unit="%",
                severity=AlertSeverity.MEDIUM,
                notification_channels=[NotificationChannel.IN_APP],
                description="Alert when soil moisture drops below threshold"
            ),
            
            # Harvest alerts
            AlertRule(
                rule_id="harvest_ready",
                name="Harvest Ready",
                category=AlertCategory.HARVEST,
                trigger_type=TriggerType.SCHEDULE,
                metric="growth_stage",
                operator="==",
                threshold=1,  # Maturity stage
                severity=AlertSeverity.INFO,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP],
                description="Alert when crop reaches harvest maturity"
            )
        ]
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self._rules[rule.rule_id] = rule
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get rule by ID."""
        return self._rules.get(rule_id)
    
    def list_rules(self, category: AlertCategory = None, enabled_only: bool = True) -> List[AlertRule]:
        """List alert rules."""
        rules = list(self._rules.values())
        
        if category:
            rules = [r for r in rules if r.category == category]
        
        if enabled_only:
            rules = [r for r in rules if r.is_enabled]
        
        return rules
    
    def evaluate_rule(
        self,
        rule: AlertRule,
        current_value: float,
        previous_value: float = None,
        context: Dict[str, Any] = None
    ) -> Optional[Alert]:
        """Evaluate a rule and generate alert if triggered."""
        if not rule.is_enabled:
            return None
        
        # Check cooldown
        if rule.last_triggered:
            cooldown_end = rule.last_triggered + timedelta(hours=rule.cooldown_hours)
            if datetime.utcnow() < cooldown_end:
                return None
        
        triggered = False
        
        if rule.trigger_type == TriggerType.THRESHOLD:
            triggered = self._evaluate_threshold(rule, current_value)
        elif rule.trigger_type == TriggerType.CHANGE_RATE:
            if previous_value is not None:
                change = current_value - previous_value
                triggered = self._evaluate_threshold(
                    AlertRule(
                        rule_id=rule.rule_id,
                        name=rule.name,
                        category=rule.category,
                        operator=rule.operator if rule.operator != "change" else "<",
                        threshold=rule.change_threshold
                    ),
                    change
                )
        
        if triggered:
            rule.last_triggered = datetime.utcnow()
            rule.trigger_count += 1
            
            return self._create_alert(rule, current_value, context or {})
        
        return None
    
    def _evaluate_threshold(self, rule: AlertRule, value: float) -> bool:
        """Evaluate threshold condition."""
        ops = {
            '<': lambda v, t: v < t,
            '>': lambda v, t: v > t,
            '<=': lambda v, t: v <= t,
            '>=': lambda v, t: v >= t,
            '==': lambda v, t: v == t,
            '!=': lambda v, t: v != t
        }
        
        op_func = ops.get(rule.operator, lambda v, t: False)
        return op_func(value, rule.threshold)
    
    def _create_alert(
        self,
        rule: AlertRule,
        trigger_value: float,
        context: Dict[str, Any]
    ) -> Alert:
        """Create alert from triggered rule."""
        # Generate alert message
        title, message = self._generate_alert_content(rule, trigger_value, context)
        recommendations = self._generate_recommendations(rule, context)
        
        return Alert(
            alert_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            category=rule.category,
            severity=rule.severity,
            title=title,
            message=message,
            field_id=context.get('field_id', ''),
            field_name=context.get('field_name', ''),
            crop_type=context.get('crop_type', ''),
            trigger_value=trigger_value,
            threshold_value=rule.threshold,
            trigger_metric=rule.metric,
            recommendations=recommendations,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
    
    def _generate_alert_content(
        self,
        rule: AlertRule,
        value: float,
        context: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Generate alert title and message."""
        field_name = context.get('field_name', 'Unknown field')
        
        title = f"{rule.severity.value.upper()}: {rule.name}"
        
        message = f"{rule.name} triggered for {field_name}. "
        message += f"Current {rule.metric}: {value:.2f}"
        
        if rule.threshold_unit:
            message += f" {rule.threshold_unit}"
        
        message += f" (threshold: {rule.threshold}"
        if rule.threshold_unit:
            message += f" {rule.threshold_unit}"
        message += ")"
        
        return title, message
    
    def _generate_recommendations(
        self,
        rule: AlertRule,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on alert type."""
        recommendations = {
            AlertCategory.VEGETATION: [
                "Conduct field inspection to identify cause of stress",
                "Check for pest or disease symptoms",
                "Review recent irrigation and fertilization records",
                "Consider soil sampling if issue persists"
            ],
            AlertCategory.WEATHER: [
                "Review weather forecast for the coming week",
                "Adjust irrigation schedule accordingly",
                "Prepare protective measures if severe weather expected",
                "Delay field operations if conditions unfavorable"
            ],
            AlertCategory.IRRIGATION: [
                "Schedule irrigation within 24-48 hours",
                "Check irrigation system for proper function",
                "Monitor soil moisture levels closely",
                "Consider mulching to reduce evaporation"
            ],
            AlertCategory.HARVEST: [
                "Schedule harvest operations",
                "Ensure harvesting equipment is ready",
                "Arrange transportation and storage",
                "Check market conditions for optimal timing"
            ]
        }
        
        return recommendations.get(rule.category, ["Review field conditions"])


class NotificationService:
    """Service for sending notifications."""
    
    def __init__(self):
        self._templates: Dict[str, NotificationTemplate] = {}
        self._subscriptions: Dict[str, AlertSubscription] = {}
        self._notification_log: List[Dict[str, Any]] = []
        
        # Initialize default templates
        self._init_default_templates()
    
    def _init_default_templates(self):
        """Initialize default notification templates."""
        self._templates['vegetation_alert'] = NotificationTemplate(
            template_id="vegetation_alert",
            name="Vegetation Alert",
            category=AlertCategory.VEGETATION,
            email_subject="[MineralVision] {severity}: Vegetation Alert for {field_name}",
            email_body="""
Dear Farm Manager,

A vegetation alert has been triggered for your field.

Field: {field_name}
Alert: {title}
Severity: {severity}

Details:
{message}

Recommendations:
{recommendations}

Please log in to MineralVision for more details and to acknowledge this alert.

Best regards,
MineralVision Crop Monitoring
            """,
            sms_message="MineralVision Alert: {title} for {field_name}. {message}",
            push_title="{severity}: {title}",
            push_body="{message}"
        )
        
        self._templates['weather_alert'] = NotificationTemplate(
            template_id="weather_alert",
            name="Weather Alert",
            category=AlertCategory.WEATHER,
            email_subject="[MineralVision] Weather Alert: {title}",
            email_body="""
Weather Alert for your fields.

Alert: {title}
{message}

Affected Fields: {field_name}

Recommendations:
{recommendations}

Stay safe and take necessary precautions.

MineralVision Crop Monitoring
            """,
            sms_message="Weather Alert: {title}. {message}",
            push_title="Weather: {title}",
            push_body="{message}"
        )
    
    def add_subscription(self, subscription: AlertSubscription) -> None:
        """Add user subscription."""
        self._subscriptions[subscription.subscription_id] = subscription
    
    def get_subscriptions_for_alert(self, alert: Alert) -> List[AlertSubscription]:
        """Get subscriptions that match an alert."""
        matching = []
        
        for sub in self._subscriptions.values():
            if not sub.is_active:
                continue
            
            # Check category filter
            if sub.categories and alert.category not in sub.categories:
                continue
            
            # Check severity filter
            if sub.severities and alert.severity not in sub.severities:
                continue
            
            # Check field filter
            if sub.field_ids and alert.field_id not in sub.field_ids:
                continue
            
            matching.append(sub)
        
        return matching
    
    def send_notification(
        self,
        alert: Alert,
        subscription: AlertSubscription,
        channel: NotificationChannel
    ) -> bool:
        """Send notification for an alert."""
        # Get template
        template_key = f"{alert.category.value}_alert"
        template = self._templates.get(template_key)
        
        if not template:
            template = self._templates.get('vegetation_alert')  # Default
        
        # Build context
        context = {
            'field_name': alert.field_name,
            'crop_type': alert.crop_type,
            'title': alert.title,
            'message': alert.message,
            'severity': alert.severity.value.upper(),
            'recommendations': '\n'.join(f"- {r}" for r in alert.recommendations),
            'date': datetime.utcnow().strftime('%Y-%m-%d %H:%M')
        }
        
        # Render template
        content = template.render(channel, context)
        
        # Log notification (in production, this would actually send)
        notification_record = {
            'alert_id': alert.alert_id,
            'subscription_id': subscription.subscription_id,
            'channel': channel.value,
            'recipient': subscription.user_email if channel == NotificationChannel.EMAIL else subscription.user_phone,
            'content': content,
            'sent_at': datetime.utcnow().isoformat(),
            'status': 'sent'
        }
        
        self._notification_log.append(notification_record)
        alert.notifications_sent.append(notification_record)
        
        logger.info(f"Notification sent: {channel.value} to {subscription.user_id} for alert {alert.alert_id}")
        
        return True
    
    def send_alert_notifications(self, alert: Alert) -> int:
        """Send notifications for an alert to all matching subscriptions."""
        subscriptions = self.get_subscriptions_for_alert(alert)
        sent_count = 0
        
        for sub in subscriptions:
            for channel in sub.channels:
                if self.send_notification(alert, sub, channel):
                    sent_count += 1
        
        return sent_count


class AlertManager:
    """Main alert management service."""
    
    def __init__(self):
        self.rule_engine = AlertRuleEngine()
        self.notification_service = NotificationService()
        
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        
        # Initialize default rules
        for rule in self.rule_engine._default_rules:
            self.rule_engine.add_rule(rule)
    
    def create_rule(self, **kwargs) -> AlertRule:
        """Create new alert rule."""
        rule_id = kwargs.get('rule_id', str(uuid.uuid4()))
        rule = AlertRule(rule_id=rule_id, **kwargs)
        self.rule_engine.add_rule(rule)
        return rule
    
    def evaluate_vegetation_data(
        self,
        field_id: str,
        field_name: str,
        crop_type: str,
        ndvi_current: float,
        ndvi_previous: float = None
    ) -> List[Alert]:
        """Evaluate vegetation data against rules."""
        context = {
            'field_id': field_id,
            'field_name': field_name,
            'crop_type': crop_type
        }
        
        alerts = []
        
        for rule in self.rule_engine.list_rules(category=AlertCategory.VEGETATION):
            if rule.metric == 'ndvi':
                alert = self.rule_engine.evaluate_rule(
                    rule, ndvi_current, ndvi_previous, context
                )
                if alert:
                    alerts.append(alert)
                    self._process_alert(alert)
        
        return alerts
    
    def evaluate_weather_data(
        self,
        field_id: str,
        field_name: str,
        weather_data: Dict[str, Any]
    ) -> List[Alert]:
        """Evaluate weather data against rules."""
        context = {
            'field_id': field_id,
            'field_name': field_name
        }
        
        alerts = []
        
        for rule in self.rule_engine.list_rules(category=AlertCategory.WEATHER):
            value = weather_data.get(rule.metric, 0)
            alert = self.rule_engine.evaluate_rule(rule, value, context=context)
            if alert:
                alerts.append(alert)
                self._process_alert(alert)
        
        return alerts
    
    def _process_alert(self, alert: Alert) -> None:
        """Process a new alert."""
        self._alerts[alert.alert_id] = alert
        self._alert_history.append(alert)
        
        # Send notifications
        self.notification_service.send_alert_notifications(alert)
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self._alerts.get(alert_id)
    
    def list_alerts(
        self,
        status: AlertStatus = None,
        category: AlertCategory = None,
        severity: AlertSeverity = None,
        field_id: str = "",
        limit: int = 100
    ) -> List[Alert]:
        """List alerts with filters."""
        alerts = list(self._alerts.values())
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if field_id:
            alerts = [a for a in alerts if a.field_id == field_id]
        
        # Sort by created_at descending
        alerts.sort(key=lambda a: a.created_at, reverse=True)
        
        return alerts[:limit]
    
    def acknowledge_alert(self, alert_id: str, user_id: str, notes: str = "") -> bool:
        """Acknowledge an alert."""
        alert = self._alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = user_id
        alert.notes = notes
        
        return True
    
    def resolve_alert(self, alert_id: str, user_id: str, notes: str = "") -> bool:
        """Resolve an alert."""
        alert = self._alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.resolved_by = user_id
        if notes:
            alert.notes = notes
        
        return True
    
    def snooze_alert(self, alert_id: str, hours: int = 24) -> bool:
        """Snooze an alert."""
        alert = self._alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.SNOOZED
        alert.expires_at = datetime.utcnow() + timedelta(hours=hours)
        
        return True
    
    def get_alert_summary(self, field_id: str = "") -> Dict[str, Any]:
        """Get summary of alerts."""
        alerts = self.list_alerts(field_id=field_id)
        
        summary = {
            'total': len(alerts),
            'active': len([a for a in alerts if a.status == AlertStatus.ACTIVE]),
            'acknowledged': len([a for a in alerts if a.status == AlertStatus.ACKNOWLEDGED]),
            'resolved': len([a for a in alerts if a.status == AlertStatus.RESOLVED]),
            'by_severity': {
                'critical': len([a for a in alerts if a.severity == AlertSeverity.CRITICAL]),
                'high': len([a for a in alerts if a.severity == AlertSeverity.HIGH]),
                'medium': len([a for a in alerts if a.severity == AlertSeverity.MEDIUM]),
                'low': len([a for a in alerts if a.severity == AlertSeverity.LOW]),
                'info': len([a for a in alerts if a.severity == AlertSeverity.INFO])
            },
            'by_category': {}
        }
        
        for category in AlertCategory:
            count = len([a for a in alerts if a.category == category])
            if count > 0:
                summary['by_category'][category.value] = count
        
        return summary
    
    def subscribe_user(
        self,
        user_id: str,
        user_email: str,
        channels: List[NotificationChannel] = None,
        categories: List[AlertCategory] = None,
        severities: List[AlertSeverity] = None
    ) -> AlertSubscription:
        """Subscribe user to alerts."""
        subscription = AlertSubscription(
            subscription_id=str(uuid.uuid4()),
            user_id=user_id,
            user_email=user_email,
            channels=channels or [NotificationChannel.EMAIL, NotificationChannel.IN_APP],
            categories=categories or [],
            severities=severities or [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        )
        
        self.notification_service.add_subscription(subscription)
        return subscription


class AlertService:
    """High-level alert service for crop monitoring."""
    
    def __init__(self):
        self.manager = AlertManager()
    
    def check_field_health(
        self,
        field_id: str,
        field_name: str,
        crop_type: str,
        ndvi_values: Dict[str, float]
    ) -> List[Alert]:
        """Check field health and generate alerts."""
        current_ndvi = ndvi_values.get('current', 0.5)
        previous_ndvi = ndvi_values.get('previous')
        
        return self.manager.evaluate_vegetation_data(
            field_id, field_name, crop_type,
            current_ndvi, previous_ndvi
        )
    
    def check_weather_risks(
        self,
        field_id: str,
        field_name: str,
        forecast_data: Dict[str, Any]
    ) -> List[Alert]:
        """Check weather forecast and generate alerts."""
        return self.manager.evaluate_weather_data(
            field_id, field_name, forecast_data
        )
    
    def get_active_alerts(self, field_id: str = "") -> List[Dict[str, Any]]:
        """Get active alerts as dictionaries."""
        alerts = self.manager.list_alerts(
            status=AlertStatus.ACTIVE,
            field_id=field_id
        )
        return [a.to_dict() for a in alerts]
    
    def get_alert_dashboard(self) -> Dict[str, Any]:
        """Get alert dashboard data."""
        summary = self.manager.get_alert_summary()
        recent_alerts = self.manager.list_alerts(limit=10)
        
        return {
            'summary': summary,
            'recent_alerts': [a.to_dict() for a in recent_alerts],
            'rules_count': len(self.manager.rule_engine.list_rules())
        }


def create_alert_service() -> AlertService:
    """Factory function to create alert service."""
    return AlertService()


def create_sample_alerts(service: AlertService) -> List[Alert]:
    """Create sample alerts for demonstration."""
    # Simulate vegetation stress alert
    alerts = service.check_field_health(
        field_id="field_001",
        field_name="Block A - Oil Palm",
        crop_type="oil_palm",
        ndvi_values={'current': 0.22, 'previous': 0.45}
    )
    
    # Simulate weather alert
    weather_alerts = service.check_weather_risks(
        field_id="field_001",
        field_name="Block A - Oil Palm",
        forecast_data={
            'precipitation_mm': 150,
            'temperature_max': 39
        }
    )
    
    return alerts + weather_alerts
