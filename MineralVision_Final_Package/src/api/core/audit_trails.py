"""
QA/QC and Audit Trails for MineralVision.

This module provides:
- Lineage graphs for derived products
- QC metrics and validation
- Chain-of-custody tracking
- NI 43-101 / JORC compliance support
- Tamper-evident logging

Essential for regulatory compliance and data governance.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import json
import hashlib
import uuid

logger = logging.getLogger(__name__)


class QCStatus(Enum):
    """QC status levels."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    PENDING = "pending"


class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"           # Meets all standards
    ACCEPTABLE = "acceptable"  # Minor issues
    LOW = "low"             # Significant issues
    REJECTED = "rejected"   # Does not meet standards


class AuditAction(Enum):
    """Audit action types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    PROCESS = "process"
    EXPORT = "export"
    APPROVE = "approve"
    REJECT = "reject"


@dataclass
class LineageNode:
    """Node in lineage graph."""
    node_id: str
    node_type: str  # 'dataset', 'process', 'model', 'output'
    name: str
    version: str
    created_at: datetime
    created_by: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'name': self.name,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'metadata': self.metadata,
            'checksum': self.checksum
        }


@dataclass
class LineageEdge:
    """Edge in lineage graph."""
    source_id: str
    target_id: str
    relationship: str  # 'derived_from', 'processed_by', 'input_to'
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship': self.relationship,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class QCMetric:
    """Quality control metric."""
    metric_name: str
    value: float
    threshold_min: Optional[float]
    threshold_max: Optional[float]
    unit: str
    status: QCStatus
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'threshold_min': self.threshold_min,
            'threshold_max': self.threshold_max,
            'unit': self.unit,
            'status': self.status.value,
            'message': self.message
        }


@dataclass
class QCReport:
    """Complete QC report for a dataset."""
    report_id: str
    dataset_id: str
    dataset_version: str
    metrics: List[QCMetric]
    overall_status: QCStatus
    overall_quality: DataQuality
    created_at: datetime
    created_by: str
    notes: str = ""
    
    @property
    def passed_metrics(self) -> int:
        return len([m for m in self.metrics if m.status == QCStatus.PASSED])
    
    @property
    def failed_metrics(self) -> int:
        return len([m for m in self.metrics if m.status == QCStatus.FAILED])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'dataset_id': self.dataset_id,
            'dataset_version': self.dataset_version,
            'metrics': [m.to_dict() for m in self.metrics],
            'overall_status': self.overall_status.value,
            'overall_quality': self.overall_quality.value,
            'passed_metrics': self.passed_metrics,
            'failed_metrics': self.failed_metrics,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'notes': self.notes
        }


@dataclass
class AuditEntry:
    """Audit log entry."""
    entry_id: str
    timestamp: datetime
    action: AuditAction
    resource_type: str
    resource_id: str
    user_id: str
    user_name: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action.value,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'user_id': self.user_id,
            'user_name': self.user_name,
            'details': self.details,
            'ip_address': self.ip_address,
            'session_id': self.session_id,
            'checksum': self.checksum
        }
    
    def compute_checksum(self, previous_checksum: str = "") -> str:
        """Compute tamper-evident checksum."""
        data = json.dumps({
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action.value,
            'resource_id': self.resource_id,
            'user_id': self.user_id,
            'previous': previous_checksum
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


class LineageGraph:
    """
    Data lineage tracking graph.
    
    Tracks relationships between datasets, processes, and outputs.
    """
    
    def __init__(self):
        self._nodes: Dict[str, LineageNode] = {}
        self._edges: List[LineageEdge] = []
        
    def add_node(self, node_type: str, name: str, version: str,
                created_by: str, metadata: Dict[str, Any] = None,
                checksum: str = None) -> LineageNode:
        """
        Add a node to the lineage graph.
        
        Args:
            node_type: Type of node
            name: Node name
            version: Version string
            created_by: Creator identifier
            metadata: Additional metadata
            checksum: Data checksum
            
        Returns:
            LineageNode
        """
        node_id = f"{node_type}_{uuid.uuid4().hex[:8]}"
        
        node = LineageNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            version=version,
            created_at=datetime.now(),
            created_by=created_by,
            metadata=metadata or {},
            checksum=checksum
        )
        
        self._nodes[node_id] = node
        return node
    
    def add_edge(self, source_id: str, target_id: str,
                relationship: str, metadata: Dict[str, Any] = None) -> LineageEdge:
        """
        Add an edge to the lineage graph.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Relationship type
            metadata: Additional metadata
            
        Returns:
            LineageEdge
        """
        edge = LineageEdge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self._edges.append(edge)
        return edge
    
    def get_node(self, node_id: str) -> Optional[LineageNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)
    
    def get_ancestors(self, node_id: str) -> List[LineageNode]:
        """Get all ancestor nodes."""
        ancestors = []
        visited = set()
        
        def traverse(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            
            for edge in self._edges:
                if edge.target_id == nid:
                    source = self._nodes.get(edge.source_id)
                    if source:
                        ancestors.append(source)
                        traverse(edge.source_id)
                        
        traverse(node_id)
        return ancestors
    
    def get_descendants(self, node_id: str) -> List[LineageNode]:
        """Get all descendant nodes."""
        descendants = []
        visited = set()
        
        def traverse(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            
            for edge in self._edges:
                if edge.source_id == nid:
                    target = self._nodes.get(edge.target_id)
                    if target:
                        descendants.append(target)
                        traverse(edge.target_id)
                        
        traverse(node_id)
        return descendants
    
    def get_lineage_path(self, node_id: str) -> Dict[str, Any]:
        """Get complete lineage path for a node."""
        node = self._nodes.get(node_id)
        if not node:
            return {'error': 'Node not found'}
            
        ancestors = self.get_ancestors(node_id)
        descendants = self.get_descendants(node_id)
        
        # Get relevant edges
        relevant_edges = [e for e in self._edges 
                        if e.source_id == node_id or e.target_id == node_id
                        or any(a.node_id in [e.source_id, e.target_id] for a in ancestors)
                        or any(d.node_id in [e.source_id, e.target_id] for d in descendants)]
        
        return {
            'node': node.to_dict(),
            'ancestors': [a.to_dict() for a in ancestors],
            'descendants': [d.to_dict() for d in descendants],
            'edges': [e.to_dict() for e in relevant_edges]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary."""
        return {
            'nodes': [n.to_dict() for n in self._nodes.values()],
            'edges': [e.to_dict() for e in self._edges]
        }


class QCValidator:
    """
    Quality control validator.
    
    Validates data against configurable thresholds.
    """
    
    def __init__(self):
        self.thresholds: Dict[str, Dict[str, Any]] = {}
        self._register_default_thresholds()
        
    def _register_default_thresholds(self):
        """Register default QC thresholds."""
        # Geophysical data thresholds
        self.thresholds['magnetics'] = {
            'coverage_percent': {'min': 80, 'max': None},
            'line_spacing_m': {'min': None, 'max': 200},
            'noise_nt': {'min': None, 'max': 5},
            'spike_percent': {'min': None, 'max': 1}
        }
        
        self.thresholds['radiometrics'] = {
            'coverage_percent': {'min': 80, 'max': None},
            'dead_time_percent': {'min': None, 'max': 5},
            'altitude_std_m': {'min': None, 'max': 10}
        }
        
        self.thresholds['geochemistry'] = {
            'duplicate_rsd_percent': {'min': None, 'max': 20},
            'blank_contamination': {'min': None, 'max': 0.1},
            'standard_accuracy_percent': {'min': 90, 'max': None}
        }
        
        self.thresholds['soil'] = {
            'sample_density_per_ha': {'min': 1, 'max': None},
            'ph_range': {'min': 3.5, 'max': 9.5},
            'organic_matter_percent': {'min': 0, 'max': 100}
        }
        
    def validate(self, data_type: str, metrics: Dict[str, float]) -> QCReport:
        """
        Validate data against thresholds.
        
        Args:
            data_type: Type of data
            metrics: Dict of metric name to value
            
        Returns:
            QCReport
        """
        thresholds = self.thresholds.get(data_type, {})
        qc_metrics = []
        
        for metric_name, value in metrics.items():
            threshold = thresholds.get(metric_name, {})
            min_val = threshold.get('min')
            max_val = threshold.get('max')
            
            # Determine status
            status = QCStatus.PASSED
            message = ""
            
            if min_val is not None and value < min_val:
                status = QCStatus.FAILED
                message = f"Below minimum threshold ({min_val})"
            elif max_val is not None and value > max_val:
                status = QCStatus.FAILED
                message = f"Above maximum threshold ({max_val})"
            elif min_val is not None and value < min_val * 1.1:
                status = QCStatus.WARNING
                message = f"Close to minimum threshold ({min_val})"
            elif max_val is not None and value > max_val * 0.9:
                status = QCStatus.WARNING
                message = f"Close to maximum threshold ({max_val})"
                
            qc_metrics.append(QCMetric(
                metric_name=metric_name,
                value=value,
                threshold_min=min_val,
                threshold_max=max_val,
                unit="",
                status=status,
                message=message
            ))
            
        # Determine overall status
        if any(m.status == QCStatus.FAILED for m in qc_metrics):
            overall_status = QCStatus.FAILED
            overall_quality = DataQuality.REJECTED
        elif any(m.status == QCStatus.WARNING for m in qc_metrics):
            overall_status = QCStatus.WARNING
            overall_quality = DataQuality.ACCEPTABLE
        else:
            overall_status = QCStatus.PASSED
            overall_quality = DataQuality.HIGH
            
        return QCReport(
            report_id=f"qc_{uuid.uuid4().hex[:8]}",
            dataset_id="",
            dataset_version="",
            metrics=qc_metrics,
            overall_status=overall_status,
            overall_quality=overall_quality,
            created_at=datetime.now(),
            created_by="system"
        )
    
    def validate_geophysics(self, data: np.ndarray,
                           data_type: str = 'magnetics') -> QCReport:
        """
        Validate geophysical data.
        
        Args:
            data: Geophysical data array
            data_type: Type of geophysical data
            
        Returns:
            QCReport
        """
        metrics = {}
        
        # Calculate coverage
        valid_count = np.sum(~np.isnan(data))
        total_count = data.size
        metrics['coverage_percent'] = (valid_count / total_count) * 100
        
        # Calculate noise (std of differences)
        if data.ndim == 1:
            diffs = np.diff(data[~np.isnan(data)])
            metrics['noise_nt'] = np.std(diffs) if len(diffs) > 0 else 0
            
        # Calculate spike percentage
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            spikes = np.sum(np.abs(valid_data - mean) > 3 * std)
            metrics['spike_percent'] = (spikes / len(valid_data)) * 100
            
        return self.validate(data_type, metrics)
    
    def validate_geochemistry(self, samples: np.ndarray,
                             duplicates: np.ndarray = None,
                             standards: np.ndarray = None,
                             standard_expected: float = None) -> QCReport:
        """
        Validate geochemistry data.
        
        Args:
            samples: Sample data
            duplicates: Duplicate sample pairs
            standards: Standard reference values
            standard_expected: Expected standard value
            
        Returns:
            QCReport
        """
        metrics = {}
        
        # Duplicate precision
        if duplicates is not None and duplicates.shape[0] > 0:
            rsd = np.std(duplicates, axis=1) / np.mean(duplicates, axis=1) * 100
            metrics['duplicate_rsd_percent'] = np.mean(rsd)
            
        # Standard accuracy
        if standards is not None and standard_expected is not None:
            accuracy = np.abs(standards - standard_expected) / standard_expected * 100
            metrics['standard_accuracy_percent'] = 100 - np.mean(accuracy)
            
        return self.validate('geochemistry', metrics)


class AuditLogger:
    """
    Tamper-evident audit logging.
    
    Provides chain-of-custody tracking with cryptographic verification.
    """
    
    def __init__(self):
        self._entries: List[AuditEntry] = []
        self._last_checksum = ""
        
    def log(self, action: AuditAction,
           resource_type: str,
           resource_id: str,
           user_id: str,
           user_name: str,
           details: Dict[str, Any],
           ip_address: str = None,
           session_id: str = None) -> AuditEntry:
        """
        Log an audit entry.
        
        Args:
            action: Action type
            resource_type: Type of resource
            resource_id: Resource identifier
            user_id: User identifier
            user_name: User name
            details: Action details
            ip_address: Client IP address
            session_id: Session identifier
            
        Returns:
            AuditEntry
        """
        entry = AuditEntry(
            entry_id=f"audit_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(),
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            user_name=user_name,
            details=details,
            ip_address=ip_address,
            session_id=session_id
        )
        
        # Compute tamper-evident checksum
        entry.checksum = entry.compute_checksum(self._last_checksum)
        self._last_checksum = entry.checksum
        
        self._entries.append(entry)
        logger.info(f"Audit: {action.value} on {resource_type}/{resource_id} by {user_name}")
        
        return entry
    
    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify audit chain integrity.
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        previous_checksum = ""
        
        for i, entry in enumerate(self._entries):
            expected_checksum = entry.compute_checksum(previous_checksum)
            
            if entry.checksum != expected_checksum:
                issues.append(f"Entry {i} ({entry.entry_id}): checksum mismatch")
                
            previous_checksum = entry.checksum
            
        return len(issues) == 0, issues
    
    def get_entries(self, resource_type: str = None,
                   resource_id: str = None,
                   user_id: str = None,
                   action: AuditAction = None,
                   since: datetime = None) -> List[AuditEntry]:
        """
        Query audit entries.
        
        Args:
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            user_id: Filter by user
            action: Filter by action
            since: Filter by timestamp
            
        Returns:
            List of matching entries
        """
        entries = self._entries
        
        if resource_type:
            entries = [e for e in entries if e.resource_type == resource_type]
        if resource_id:
            entries = [e for e in entries if e.resource_id == resource_id]
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        if action:
            entries = [e for e in entries if e.action == action]
        if since:
            entries = [e for e in entries if e.timestamp >= since]
            
        return entries
    
    def export(self, format: str = 'json') -> str:
        """Export audit log."""
        if format == 'json':
            return json.dumps([e.to_dict() for e in self._entries], indent=2)
        else:
            # CSV format
            lines = ['entry_id,timestamp,action,resource_type,resource_id,user_id,user_name']
            for e in self._entries:
                lines.append(f"{e.entry_id},{e.timestamp.isoformat()},{e.action.value},"
                           f"{e.resource_type},{e.resource_id},{e.user_id},{e.user_name}")
            return '\n'.join(lines)


class ComplianceReporter:
    """
    Generate compliance reports for NI 43-101 / JORC.
    
    Provides standardized reporting formats.
    """
    
    def __init__(self, lineage_graph: LineageGraph,
                qc_validator: QCValidator,
                audit_logger: AuditLogger):
        self.lineage = lineage_graph
        self.qc = qc_validator
        self.audit = audit_logger
        
    def generate_ni43101_report(self, project_name: str,
                               dataset_ids: List[str],
                               author: str) -> Dict[str, Any]:
        """
        Generate NI 43-101 compliant report.
        
        Args:
            project_name: Project name
            dataset_ids: Dataset identifiers
            author: Report author
            
        Returns:
            Report dictionary
        """
        report = {
            'report_type': 'NI 43-101',
            'project_name': project_name,
            'author': author,
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }
        
        # Section 1: Data Sources and Lineage
        lineage_info = []
        for ds_id in dataset_ids:
            path = self.lineage.get_lineage_path(ds_id)
            if 'error' not in path:
                lineage_info.append(path)
        report['sections']['data_lineage'] = lineage_info
        
        # Section 2: QC Summary
        report['sections']['qc_summary'] = {
            'methodology': 'Standard QC protocols applied',
            'thresholds': self.qc.thresholds
        }
        
        # Section 3: Audit Trail
        is_valid, issues = self.audit.verify_chain()
        report['sections']['audit_trail'] = {
            'chain_verified': is_valid,
            'issues': issues,
            'n_entries': len(self.audit._entries)
        }
        
        # Section 4: Qualified Person Statement
        report['sections']['qp_statement'] = {
            'statement': f"This report was prepared by {author} in accordance with NI 43-101 requirements.",
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return report
    
    def generate_jorc_report(self, project_name: str,
                            dataset_ids: List[str],
                            competent_person: str) -> Dict[str, Any]:
        """
        Generate JORC compliant report.
        
        Args:
            project_name: Project name
            dataset_ids: Dataset identifiers
            competent_person: Competent Person name
            
        Returns:
            Report dictionary
        """
        report = {
            'report_type': 'JORC 2012',
            'project_name': project_name,
            'competent_person': competent_person,
            'generated_at': datetime.now().isoformat(),
            'table_1': {}
        }
        
        # Table 1 - Sampling Techniques and Data
        report['table_1']['sampling_techniques'] = {
            'description': 'See data lineage for sampling methodology',
            'datasets': dataset_ids
        }
        
        report['table_1']['drilling_techniques'] = {
            'description': 'Not applicable for this report'
        }
        
        report['table_1']['sample_preparation'] = {
            'description': 'Standard sample preparation protocols'
        }
        
        report['table_1']['qaqc'] = {
            'description': 'QC protocols applied',
            'thresholds': self.qc.thresholds
        }
        
        report['table_1']['data_verification'] = {
            'audit_chain_verified': self.audit.verify_chain()[0]
        }
        
        # Competent Person Statement
        report['competent_person_statement'] = {
            'name': competent_person,
            'statement': f"The information in this report that relates to Exploration Results "
                        f"is based on information compiled by {competent_person}.",
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return report


class AuditTrailManager:
    """
    Unified audit trail manager for MineralVision.
    
    Integrates lineage, QC, and audit logging.
    """
    
    def __init__(self):
        self.lineage = LineageGraph()
        self.qc = QCValidator()
        self.audit = AuditLogger()
        self.compliance = ComplianceReporter(self.lineage, self.qc, self.audit)
        
    def track_data_creation(self, name: str, version: str,
                           created_by: str, data: np.ndarray,
                           data_type: str) -> Tuple[LineageNode, QCReport]:
        """
        Track data creation with QC.
        
        Args:
            name: Dataset name
            version: Version string
            created_by: Creator
            data: Data array
            data_type: Type of data
            
        Returns:
            Tuple of (LineageNode, QCReport)
        """
        # Calculate checksum
        checksum = hashlib.sha256(data.tobytes()).hexdigest()[:16]
        
        # Create lineage node
        node = self.lineage.add_node(
            node_type='dataset',
            name=name,
            version=version,
            created_by=created_by,
            metadata={'shape': data.shape, 'dtype': str(data.dtype)},
            checksum=checksum
        )
        
        # Run QC
        qc_report = self.qc.validate_geophysics(data, data_type)
        qc_report.dataset_id = node.node_id
        qc_report.dataset_version = version
        
        # Log audit
        self.audit.log(
            action=AuditAction.CREATE,
            resource_type='dataset',
            resource_id=node.node_id,
            user_id=created_by,
            user_name=created_by,
            details={
                'name': name,
                'version': version,
                'checksum': checksum,
                'qc_status': qc_report.overall_status.value
            }
        )
        
        return node, qc_report
    
    def track_processing(self, input_node_id: str,
                        output_name: str,
                        process_name: str,
                        parameters: Dict[str, Any],
                        processed_by: str,
                        output_data: np.ndarray) -> LineageNode:
        """
        Track data processing.
        
        Args:
            input_node_id: Input node ID
            output_name: Output name
            process_name: Processing step name
            parameters: Processing parameters
            processed_by: Processor
            output_data: Output data
            
        Returns:
            Output LineageNode
        """
        # Create process node
        process_node = self.lineage.add_node(
            node_type='process',
            name=process_name,
            version='1.0',
            created_by=processed_by,
            metadata={'parameters': parameters}
        )
        
        # Link input to process
        self.lineage.add_edge(
            source_id=input_node_id,
            target_id=process_node.node_id,
            relationship='input_to'
        )
        
        # Create output node
        checksum = hashlib.sha256(output_data.tobytes()).hexdigest()[:16]
        output_node = self.lineage.add_node(
            node_type='dataset',
            name=output_name,
            version='1.0',
            created_by=processed_by,
            metadata={'shape': output_data.shape},
            checksum=checksum
        )
        
        # Link process to output
        self.lineage.add_edge(
            source_id=process_node.node_id,
            target_id=output_node.node_id,
            relationship='derived_from'
        )
        
        # Log audit
        self.audit.log(
            action=AuditAction.PROCESS,
            resource_type='dataset',
            resource_id=output_node.node_id,
            user_id=processed_by,
            user_name=processed_by,
            details={
                'process': process_name,
                'input': input_node_id,
                'parameters': parameters
            }
        )
        
        return output_node
    
    def get_full_lineage(self, node_id: str) -> Dict[str, Any]:
        """Get full lineage for a node."""
        return self.lineage.get_lineage_path(node_id)
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify system integrity."""
        audit_valid, audit_issues = self.audit.verify_chain()
        
        return {
            'audit_chain_valid': audit_valid,
            'audit_issues': audit_issues,
            'n_lineage_nodes': len(self.lineage._nodes),
            'n_lineage_edges': len(self.lineage._edges),
            'n_audit_entries': len(self.audit._entries)
        }


# Factory functions
def create_audit_trail_manager() -> AuditTrailManager:
    """Create audit trail manager."""
    return AuditTrailManager()


def create_lineage_graph() -> LineageGraph:
    """Create lineage graph."""
    return LineageGraph()


def create_qc_validator() -> QCValidator:
    """Create QC validator."""
    return QCValidator()
