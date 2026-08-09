"""
QA/QC Analysis Module for MineralVision Platform.

Comprehensive quality assurance and quality control including:
1. Standards (CRM) monitoring and control charts
2. Blanks analysis for contamination detection
3. Duplicates analysis (field, coarse, pulp)
4. Umpire/check assay comparison
5. Control charts (Shewhart, CUSUM, EWMA)
6. Bias detection and correction
7. Precision and accuracy metrics
8. Laboratory performance tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import numpy as np
from collections import defaultdict


class QAQCType(Enum):
    """Types of QA/QC samples."""
    STANDARD = "standard"
    BLANK = "blank"
    FIELD_DUPLICATE = "field_duplicate"
    COARSE_DUPLICATE = "coarse_duplicate"
    PULP_DUPLICATE = "pulp_duplicate"
    UMPIRE = "umpire"
    CHECK_ASSAY = "check_assay"
    REPLICATE = "replicate"


class ControlChartType(Enum):
    """Types of control charts."""
    SHEWHART = "shewhart"
    CUSUM = "cusum"
    EWMA = "ewma"
    MOVING_RANGE = "moving_range"


class AlertLevel(Enum):
    """Alert severity levels."""
    OK = "ok"
    WARNING = "warning"
    ACTION = "action"
    CRITICAL = "critical"


class FailureType(Enum):
    """Types of QA/QC failures."""
    BIAS = "bias"
    PRECISION = "precision"
    CONTAMINATION = "contamination"
    DRIFT = "drift"
    OUTLIER = "outlier"
    TREND = "trend"


@dataclass
class StandardReference:
    """Certified Reference Material (CRM) definition."""
    standard_id: str
    name: str
    certified_values: Dict[str, float]
    uncertainties: Dict[str, float]
    units: Dict[str, str] = field(default_factory=dict)
    matrix: str = ""
    supplier: str = ""
    lot_number: str = ""
    expiry_date: Optional[datetime] = None
    warning_limits_sigma: float = 2.0
    action_limits_sigma: float = 3.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_limits(self, element: str) -> Tuple[float, float, float, float]:
        """Get warning and action limits for an element."""
        if element not in self.certified_values:
            return (0, 0, 0, 0)
        
        cert = self.certified_values[element]
        unc = self.uncertainties.get(element, cert * 0.05)
        
        warn_low = cert - self.warning_limits_sigma * unc
        warn_high = cert + self.warning_limits_sigma * unc
        action_low = cert - self.action_limits_sigma * unc
        action_high = cert + self.action_limits_sigma * unc
        
        return (action_low, warn_low, warn_high, action_high)


@dataclass
class BlankReference:
    """Blank material definition."""
    blank_id: str
    name: str
    expected_values: Dict[str, float]
    detection_limits: Dict[str, float]
    contamination_thresholds: Dict[str, float]
    matrix: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAQCSample:
    """QA/QC sample result."""
    sample_id: str
    qaqc_type: QAQCType
    reference_id: str
    batch_id: str
    lab_id: str
    submission_date: datetime
    analysis_date: Optional[datetime] = None
    values: Dict[str, float] = field(default_factory=dict)
    original_sample_id: str = ""
    pair_sample_id: str = ""
    sequence_number: int = 0
    comments: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlPoint:
    """Single point on a control chart."""
    sample_id: str
    date: datetime
    value: float
    certified_value: float
    deviation: float
    z_score: float
    alert_level: AlertLevel
    cumsum: float = 0.0
    ewma: float = 0.0
    moving_range: float = 0.0


@dataclass
class DuplicatePair:
    """Duplicate sample pair for precision analysis."""
    original_id: str
    duplicate_id: str
    original_value: float
    duplicate_value: float
    element: str
    duplicate_type: QAQCType
    absolute_difference: float
    relative_difference: float
    half_absolute_relative_difference: float
    mean_value: float
    batch_id: str = ""
    date: Optional[datetime] = None


@dataclass
class QAQCAlert:
    """QA/QC alert notification."""
    alert_id: str
    alert_type: FailureType
    severity: AlertLevel
    element: str
    reference_id: str
    batch_id: str
    lab_id: str
    message: str
    value: float
    expected: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: str = ""


@dataclass
class LabPerformance:
    """Laboratory performance metrics."""
    lab_id: str
    period_start: datetime
    period_end: datetime
    total_samples: int
    standards_analyzed: int
    standards_passed: int
    blanks_analyzed: int
    blanks_passed: int
    duplicates_analyzed: int
    duplicates_passed: int
    mean_bias: Dict[str, float] = field(default_factory=dict)
    precision: Dict[str, float] = field(default_factory=dict)
    accuracy: Dict[str, float] = field(default_factory=dict)
    turnaround_days: float = 0.0
    failure_rate: float = 0.0


class StandardsAnalyzer:
    """Analyze certified reference material (CRM) results."""
    
    def __init__(self):
        self.standards: Dict[str, StandardReference] = {}
        self.results: Dict[str, List[QAQCSample]] = {}
        self.control_points: Dict[str, Dict[str, List[ControlPoint]]] = {}
    
    def add_standard(self, standard: StandardReference):
        """Register a CRM."""
        self.standards[standard.standard_id] = standard
        self.results[standard.standard_id] = []
        self.control_points[standard.standard_id] = {}
    
    def add_result(self, sample: QAQCSample) -> List[QAQCAlert]:
        """Add a standard result and check for alerts."""
        if sample.reference_id not in self.standards:
            return []
        
        self.results[sample.reference_id].append(sample)
        
        alerts = []
        standard = self.standards[sample.reference_id]
        
        for element, value in sample.values.items():
            if element not in standard.certified_values:
                continue
            
            cert = standard.certified_values[element]
            unc = standard.uncertainties.get(element, cert * 0.05)
            
            deviation = value - cert
            z_score = deviation / unc if unc > 0 else 0
            
            action_low, warn_low, warn_high, action_high = standard.get_limits(element)
            
            if value < action_low or value > action_high:
                alert_level = AlertLevel.ACTION
            elif value < warn_low or value > warn_high:
                alert_level = AlertLevel.WARNING
            else:
                alert_level = AlertLevel.OK
            
            if element not in self.control_points[sample.reference_id]:
                self.control_points[sample.reference_id][element] = []
            
            prev_points = self.control_points[sample.reference_id][element]
            cumsum = (prev_points[-1].cumsum if prev_points else 0) + z_score
            
            lambda_ewma = 0.2
            ewma = (lambda_ewma * z_score + 
                   (1 - lambda_ewma) * (prev_points[-1].ewma if prev_points else 0))
            
            moving_range = abs(z_score - prev_points[-1].z_score) if prev_points else 0
            
            point = ControlPoint(
                sample_id=sample.sample_id,
                date=sample.submission_date,
                value=value,
                certified_value=cert,
                deviation=deviation,
                z_score=z_score,
                alert_level=alert_level,
                cumsum=cumsum,
                ewma=ewma,
                moving_range=moving_range
            )
            
            self.control_points[sample.reference_id][element].append(point)
            
            if alert_level != AlertLevel.OK:
                alerts.append(QAQCAlert(
                    alert_id=f"STD-{sample.sample_id}-{element}",
                    alert_type=FailureType.BIAS,
                    severity=alert_level,
                    element=element,
                    reference_id=sample.reference_id,
                    batch_id=sample.batch_id,
                    lab_id=sample.lab_id,
                    message=f"Standard {sample.reference_id} {element}: {value:.4f} vs certified {cert:.4f}",
                    value=value,
                    expected=cert,
                    threshold=unc * standard.action_limits_sigma,
                    timestamp=sample.submission_date
                ))
            
            trend_alert = self._check_trends(sample.reference_id, element)
            if trend_alert:
                alerts.append(trend_alert)
        
        return alerts
    
    def _check_trends(self, standard_id: str, element: str) -> Optional[QAQCAlert]:
        """Check for trending patterns (Western Electric rules)."""
        points = self.control_points.get(standard_id, {}).get(element, [])
        
        if len(points) < 7:
            return None
        
        recent = points[-7:]
        z_scores = [p.z_score for p in recent]
        
        if all(z > 0 for z in z_scores) or all(z < 0 for z in z_scores):
            return QAQCAlert(
                alert_id=f"TREND-{standard_id}-{element}-{datetime.now().isoformat()}",
                alert_type=FailureType.TREND,
                severity=AlertLevel.WARNING,
                element=element,
                reference_id=standard_id,
                batch_id=points[-1].sample_id,
                lab_id="",
                message=f"7 consecutive points on same side of mean for {element}",
                value=z_scores[-1],
                expected=0,
                threshold=0,
                timestamp=datetime.now()
            )
        
        if len(points) >= 8:
            recent_8 = [p.z_score for p in points[-8:]]
            increasing = all(recent_8[i] < recent_8[i+1] for i in range(7))
            decreasing = all(recent_8[i] > recent_8[i+1] for i in range(7))
            
            if increasing or decreasing:
                return QAQCAlert(
                    alert_id=f"DRIFT-{standard_id}-{element}-{datetime.now().isoformat()}",
                    alert_type=FailureType.DRIFT,
                    severity=AlertLevel.WARNING,
                    element=element,
                    reference_id=standard_id,
                    batch_id=points[-1].sample_id,
                    lab_id="",
                    message=f"8 consecutive {'increasing' if increasing else 'decreasing'} points for {element}",
                    value=z_scores[-1],
                    expected=0,
                    threshold=0,
                    timestamp=datetime.now()
                )
        
        return None
    
    def get_control_chart_data(self, standard_id: str, element: str,
                               chart_type: ControlChartType = ControlChartType.SHEWHART
                              ) -> Dict[str, Any]:
        """Get data for control chart visualization."""
        if standard_id not in self.standards:
            return {}
        
        standard = self.standards[standard_id]
        points = self.control_points.get(standard_id, {}).get(element, [])
        
        if not points:
            return {}
        
        cert = standard.certified_values.get(element, 0)
        unc = standard.uncertainties.get(element, cert * 0.05)
        
        action_low, warn_low, warn_high, action_high = standard.get_limits(element)
        
        if chart_type == ControlChartType.SHEWHART:
            y_values = [p.value for p in points]
            center_line = cert
        elif chart_type == ControlChartType.CUSUM:
            y_values = [p.cumsum for p in points]
            center_line = 0
            h_value = 5
            action_high = h_value
            action_low = -h_value
            warn_high = h_value * 0.67
            warn_low = -h_value * 0.67
        elif chart_type == ControlChartType.EWMA:
            y_values = [p.ewma for p in points]
            center_line = 0
            lambda_ewma = 0.2
            ewma_sigma = math.sqrt(lambda_ewma / (2 - lambda_ewma))
            action_high = 3 * ewma_sigma
            action_low = -3 * ewma_sigma
            warn_high = 2 * ewma_sigma
            warn_low = -2 * ewma_sigma
        else:
            y_values = [p.moving_range for p in points]
            center_line = np.mean(y_values) if y_values else 0
            action_high = center_line * 3.267
            action_low = 0
            warn_high = center_line * 2.5
            warn_low = 0
        
        return {
            "standard_id": standard_id,
            "element": element,
            "chart_type": chart_type.value,
            "dates": [p.date.isoformat() for p in points],
            "sample_ids": [p.sample_id for p in points],
            "values": y_values,
            "center_line": center_line,
            "certified_value": cert,
            "uncertainty": unc,
            "warning_low": warn_low,
            "warning_high": warn_high,
            "action_low": action_low,
            "action_high": action_high,
            "alert_levels": [p.alert_level.value for p in points],
            "z_scores": [p.z_score for p in points],
            "total_points": len(points),
            "failures": len([p for p in points if p.alert_level != AlertLevel.OK])
        }
    
    def calculate_bias(self, standard_id: str, element: str,
                      n_recent: Optional[int] = None) -> Dict[str, float]:
        """Calculate bias statistics for a standard."""
        points = self.control_points.get(standard_id, {}).get(element, [])
        
        if not points:
            return {}
        
        if n_recent:
            points = points[-n_recent:]
        
        standard = self.standards[standard_id]
        cert = standard.certified_values.get(element, 0)
        
        values = [p.value for p in points]
        deviations = [p.deviation for p in points]
        z_scores = [p.z_score for p in points]
        
        mean_value = np.mean(values)
        std_value = np.std(values, ddof=1) if len(values) > 1 else 0
        mean_bias = np.mean(deviations)
        relative_bias = (mean_bias / cert * 100) if cert != 0 else 0
        
        t_stat = (mean_bias / (std_value / math.sqrt(len(values)))) if std_value > 0 else 0
        
        return {
            "n_samples": len(points),
            "mean_value": mean_value,
            "std_value": std_value,
            "certified_value": cert,
            "mean_bias": mean_bias,
            "relative_bias_percent": relative_bias,
            "mean_z_score": np.mean(z_scores),
            "t_statistic": t_stat,
            "cv_percent": (std_value / mean_value * 100) if mean_value != 0 else 0,
            "pass_rate": len([p for p in points if p.alert_level == AlertLevel.OK]) / len(points) * 100
        }


class BlanksAnalyzer:
    """Analyze blank samples for contamination."""
    
    def __init__(self):
        self.blanks: Dict[str, BlankReference] = {}
        self.results: List[QAQCSample] = []
        self.alerts: List[QAQCAlert] = []
    
    def add_blank(self, blank: BlankReference):
        """Register a blank material."""
        self.blanks[blank.blank_id] = blank
    
    def add_result(self, sample: QAQCSample) -> List[QAQCAlert]:
        """Add a blank result and check for contamination."""
        self.results.append(sample)
        alerts = []
        
        blank = self.blanks.get(sample.reference_id)
        
        for element, value in sample.values.items():
            threshold = 0.0
            
            if blank:
                threshold = blank.contamination_thresholds.get(
                    element, 
                    blank.detection_limits.get(element, 0) * 3
                )
            
            if value > threshold:
                severity = AlertLevel.ACTION if value > threshold * 2 else AlertLevel.WARNING
                
                alert = QAQCAlert(
                    alert_id=f"BLANK-{sample.sample_id}-{element}",
                    alert_type=FailureType.CONTAMINATION,
                    severity=severity,
                    element=element,
                    reference_id=sample.reference_id,
                    batch_id=sample.batch_id,
                    lab_id=sample.lab_id,
                    message=f"Blank contamination: {element} = {value:.4f} (threshold: {threshold:.4f})",
                    value=value,
                    expected=0,
                    threshold=threshold,
                    timestamp=sample.submission_date
                )
                alerts.append(alert)
                self.alerts.append(alert)
        
        return alerts
    
    def get_contamination_summary(self, element: str) -> Dict[str, Any]:
        """Get contamination statistics for an element."""
        values = []
        contaminated = 0
        
        for sample in self.results:
            if element in sample.values:
                value = sample.values[element]
                values.append(value)
                
                blank = self.blanks.get(sample.reference_id)
                if blank:
                    threshold = blank.contamination_thresholds.get(element, 0)
                    if value > threshold:
                        contaminated += 1
        
        if not values:
            return {}
        
        return {
            "element": element,
            "n_samples": len(values),
            "n_contaminated": contaminated,
            "contamination_rate": contaminated / len(values) * 100,
            "mean_value": np.mean(values),
            "max_value": max(values),
            "median_value": np.median(values),
            "std_value": np.std(values, ddof=1) if len(values) > 1 else 0
        }


class DuplicatesAnalyzer:
    """Analyze duplicate samples for precision."""
    
    def __init__(self):
        self.pairs: Dict[str, List[DuplicatePair]] = defaultdict(list)
        self.precision_limits: Dict[str, Dict[str, float]] = {}
    
    def set_precision_limits(self, element: str, limits: Dict[str, float]):
        """Set acceptable precision limits by grade range."""
        self.precision_limits[element] = limits
    
    def add_pair(self, original: QAQCSample, duplicate: QAQCSample,
                duplicate_type: QAQCType) -> List[QAQCAlert]:
        """Add a duplicate pair and analyze precision."""
        alerts = []
        
        for element in original.values:
            if element not in duplicate.values:
                continue
            
            orig_val = original.values[element]
            dup_val = duplicate.values[element]
            
            abs_diff = abs(orig_val - dup_val)
            mean_val = (orig_val + dup_val) / 2
            
            if mean_val > 0:
                rel_diff = abs_diff / mean_val * 100
                hard = abs_diff / mean_val * 100 / 2
            else:
                rel_diff = 0
                hard = 0
            
            pair = DuplicatePair(
                original_id=original.sample_id,
                duplicate_id=duplicate.sample_id,
                original_value=orig_val,
                duplicate_value=dup_val,
                element=element,
                duplicate_type=duplicate_type,
                absolute_difference=abs_diff,
                relative_difference=rel_diff,
                half_absolute_relative_difference=hard,
                mean_value=mean_val,
                batch_id=original.batch_id,
                date=original.submission_date
            )
            
            self.pairs[element].append(pair)
            
            limit = self._get_precision_limit(element, mean_val, duplicate_type)
            
            if hard > limit:
                severity = AlertLevel.ACTION if hard > limit * 1.5 else AlertLevel.WARNING
                
                alerts.append(QAQCAlert(
                    alert_id=f"DUP-{original.sample_id}-{element}",
                    alert_type=FailureType.PRECISION,
                    severity=severity,
                    element=element,
                    reference_id=duplicate_type.value,
                    batch_id=original.batch_id,
                    lab_id=original.lab_id,
                    message=f"Duplicate precision failure: {element} HARD={hard:.1f}% (limit: {limit:.1f}%)",
                    value=hard,
                    expected=0,
                    threshold=limit,
                    timestamp=original.submission_date
                ))
        
        return alerts
    
    def _get_precision_limit(self, element: str, grade: float,
                            duplicate_type: QAQCType) -> float:
        """Get precision limit based on grade and duplicate type."""
        base_limits = {
            QAQCType.FIELD_DUPLICATE: 20.0,
            QAQCType.COARSE_DUPLICATE: 15.0,
            QAQCType.PULP_DUPLICATE: 10.0,
            QAQCType.REPLICATE: 5.0
        }
        
        base = base_limits.get(duplicate_type, 15.0)
        
        if element in self.precision_limits:
            limits = self.precision_limits[element]
            for grade_range, limit in sorted(limits.items(), reverse=True):
                try:
                    if grade >= float(grade_range):
                        return limit
                except ValueError:
                    pass
        
        if grade < 0.1:
            return base * 2
        elif grade < 1.0:
            return base * 1.5
        else:
            return base
    
    def calculate_precision(self, element: str, 
                           duplicate_type: Optional[QAQCType] = None,
                           min_grade: float = 0) -> Dict[str, Any]:
        """Calculate precision statistics using Thompson-Howarth method."""
        pairs = self.pairs.get(element, [])
        
        if duplicate_type:
            pairs = [p for p in pairs if p.duplicate_type == duplicate_type]
        
        pairs = [p for p in pairs if p.mean_value >= min_grade]
        
        if len(pairs) < 2:
            return {}
        
        hard_values = [p.half_absolute_relative_difference for p in pairs]
        abs_diffs = [p.absolute_difference for p in pairs]
        mean_values = [p.mean_value for p in pairs]
        
        mean_hard = np.mean(hard_values)
        
        variance = sum(d**2 for d in abs_diffs) / (2 * len(pairs))
        precision_std = math.sqrt(variance)
        
        mean_grade = np.mean(mean_values)
        cv = (precision_std / mean_grade * 100) if mean_grade > 0 else 0
        
        sorted_hard = sorted(hard_values)
        p90 = sorted_hard[int(len(sorted_hard) * 0.9)] if sorted_hard else 0
        p95 = sorted_hard[int(len(sorted_hard) * 0.95)] if sorted_hard else 0
        
        limit = self._get_precision_limit(element, mean_grade, 
                                         duplicate_type or QAQCType.PULP_DUPLICATE)
        failures = len([h for h in hard_values if h > limit])
        
        return {
            "element": element,
            "duplicate_type": duplicate_type.value if duplicate_type else "all",
            "n_pairs": len(pairs),
            "mean_hard": mean_hard,
            "precision_std": precision_std,
            "cv_percent": cv,
            "mean_grade": mean_grade,
            "p90_hard": p90,
            "p95_hard": p95,
            "precision_limit": limit,
            "n_failures": failures,
            "pass_rate": (len(pairs) - failures) / len(pairs) * 100 if pairs else 0
        }
    
    def get_scatter_plot_data(self, element: str,
                             duplicate_type: Optional[QAQCType] = None) -> Dict[str, Any]:
        """Get data for duplicate scatter plot."""
        pairs = self.pairs.get(element, [])
        
        if duplicate_type:
            pairs = [p for p in pairs if p.duplicate_type == duplicate_type]
        
        if not pairs:
            return {}
        
        original = [p.original_value for p in pairs]
        duplicate = [p.duplicate_value for p in pairs]
        
        max_val = max(max(original), max(duplicate))
        
        slope, intercept = 1.0, 0.0
        if len(pairs) > 1:
            n = len(pairs)
            sum_x = sum(original)
            sum_y = sum(duplicate)
            sum_xy = sum(o * d for o, d in zip(original, duplicate))
            sum_x2 = sum(o**2 for o in original)
            
            denom = n * sum_x2 - sum_x**2
            if denom != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denom
                intercept = (sum_y - slope * sum_x) / n
        
        ss_res = sum((d - (slope * o + intercept))**2 for o, d in zip(original, duplicate))
        mean_dup = np.mean(duplicate)
        ss_tot = sum((d - mean_dup)**2 for d in duplicate)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            "element": element,
            "duplicate_type": duplicate_type.value if duplicate_type else "all",
            "original_values": original,
            "duplicate_values": duplicate,
            "sample_ids": [(p.original_id, p.duplicate_id) for p in pairs],
            "hard_values": [p.half_absolute_relative_difference for p in pairs],
            "max_value": max_val,
            "regression_slope": slope,
            "regression_intercept": intercept,
            "r_squared": r_squared,
            "n_pairs": len(pairs)
        }


class UmpireAnalyzer:
    """Analyze umpire/check assay results."""
    
    def __init__(self):
        self.comparisons: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.bias_threshold: float = 10.0
    
    def add_comparison(self, primary_lab: str, primary_value: float,
                      umpire_lab: str, umpire_value: float,
                      element: str, sample_id: str,
                      batch_id: str = "") -> Optional[QAQCAlert]:
        """Add an umpire comparison."""
        mean_val = (primary_value + umpire_value) / 2
        
        if mean_val > 0:
            rel_diff = (primary_value - umpire_value) / mean_val * 100
        else:
            rel_diff = 0
        
        comparison = {
            "sample_id": sample_id,
            "primary_lab": primary_lab,
            "primary_value": primary_value,
            "umpire_lab": umpire_lab,
            "umpire_value": umpire_value,
            "mean_value": mean_val,
            "relative_difference": rel_diff,
            "absolute_difference": abs(primary_value - umpire_value),
            "batch_id": batch_id,
            "timestamp": datetime.now()
        }
        
        self.comparisons[element].append(comparison)
        
        if abs(rel_diff) > self.bias_threshold:
            return QAQCAlert(
                alert_id=f"UMPIRE-{sample_id}-{element}",
                alert_type=FailureType.BIAS,
                severity=AlertLevel.WARNING,
                element=element,
                reference_id=f"{primary_lab}_vs_{umpire_lab}",
                batch_id=batch_id,
                lab_id=primary_lab,
                message=f"Umpire bias: {element} {rel_diff:+.1f}% ({primary_lab}: {primary_value:.4f} vs {umpire_lab}: {umpire_value:.4f})",
                value=rel_diff,
                expected=0,
                threshold=self.bias_threshold,
                timestamp=datetime.now()
            )
        
        return None
    
    def calculate_lab_bias(self, element: str, primary_lab: str,
                          umpire_lab: str) -> Dict[str, Any]:
        """Calculate systematic bias between labs."""
        comparisons = [c for c in self.comparisons.get(element, [])
                      if c["primary_lab"] == primary_lab and c["umpire_lab"] == umpire_lab]
        
        if len(comparisons) < 3:
            return {}
        
        rel_diffs = [c["relative_difference"] for c in comparisons]
        primary_vals = [c["primary_value"] for c in comparisons]
        umpire_vals = [c["umpire_value"] for c in comparisons]
        
        mean_bias = np.mean(rel_diffs)
        std_bias = np.std(rel_diffs, ddof=1)
        
        t_stat = mean_bias / (std_bias / math.sqrt(len(rel_diffs))) if std_bias > 0 else 0
        
        n = len(comparisons)
        sum_x = sum(primary_vals)
        sum_y = sum(umpire_vals)
        sum_xy = sum(p * u for p, u in zip(primary_vals, umpire_vals))
        sum_x2 = sum(p**2 for p in primary_vals)
        
        denom = n * sum_x2 - sum_x**2
        if denom != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
        else:
            slope, intercept = 1.0, 0.0
        
        return {
            "element": element,
            "primary_lab": primary_lab,
            "umpire_lab": umpire_lab,
            "n_comparisons": len(comparisons),
            "mean_bias_percent": mean_bias,
            "std_bias_percent": std_bias,
            "t_statistic": t_stat,
            "significant_bias": abs(t_stat) > 2.0,
            "regression_slope": slope,
            "regression_intercept": intercept,
            "bias_direction": "primary_high" if mean_bias > 0 else "primary_low"
        }


class QAQCDashboard:
    """
    Comprehensive QA/QC dashboard integrating all analyzers.
    """
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.standards_analyzer = StandardsAnalyzer()
        self.blanks_analyzer = BlanksAnalyzer()
        self.duplicates_analyzer = DuplicatesAnalyzer()
        self.umpire_analyzer = UmpireAnalyzer()
        
        self.all_alerts: List[QAQCAlert] = []
        self.lab_submissions: Dict[str, List[QAQCSample]] = defaultdict(list)
    
    def register_standard(self, standard: StandardReference):
        """Register a CRM."""
        self.standards_analyzer.add_standard(standard)
    
    def register_blank(self, blank: BlankReference):
        """Register a blank material."""
        self.blanks_analyzer.add_blank(blank)
    
    def process_sample(self, sample: QAQCSample) -> List[QAQCAlert]:
        """Process a QA/QC sample and return any alerts."""
        alerts = []
        
        self.lab_submissions[sample.lab_id].append(sample)
        
        if sample.qaqc_type == QAQCType.STANDARD:
            alerts.extend(self.standards_analyzer.add_result(sample))
        
        elif sample.qaqc_type == QAQCType.BLANK:
            alerts.extend(self.blanks_analyzer.add_result(sample))
        
        elif sample.qaqc_type in [QAQCType.FIELD_DUPLICATE, QAQCType.COARSE_DUPLICATE,
                                  QAQCType.PULP_DUPLICATE, QAQCType.REPLICATE]:
            pass
        
        self.all_alerts.extend(alerts)
        return alerts
    
    def process_duplicate_pair(self, original: QAQCSample, duplicate: QAQCSample,
                              duplicate_type: QAQCType) -> List[QAQCAlert]:
        """Process a duplicate pair."""
        alerts = self.duplicates_analyzer.add_pair(original, duplicate, duplicate_type)
        self.all_alerts.extend(alerts)
        return alerts
    
    def get_summary(self, lab_id: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get comprehensive QA/QC summary."""
        
        standards_summary = {}
        for std_id, std in self.standards_analyzer.standards.items():
            for element in std.certified_values:
                key = f"{std_id}_{element}"
                standards_summary[key] = self.standards_analyzer.calculate_bias(std_id, element)
        
        blanks_summary = {}
        all_elements = set()
        for sample in self.blanks_analyzer.results:
            all_elements.update(sample.values.keys())
        for element in all_elements:
            blanks_summary[element] = self.blanks_analyzer.get_contamination_summary(element)
        
        duplicates_summary = {}
        for element in self.duplicates_analyzer.pairs:
            duplicates_summary[element] = {
                "field": self.duplicates_analyzer.calculate_precision(
                    element, QAQCType.FIELD_DUPLICATE),
                "coarse": self.duplicates_analyzer.calculate_precision(
                    element, QAQCType.COARSE_DUPLICATE),
                "pulp": self.duplicates_analyzer.calculate_precision(
                    element, QAQCType.PULP_DUPLICATE)
            }
        
        active_alerts = [a for a in self.all_alerts if not a.resolved]
        
        return {
            "project": self.project_name,
            "summary_date": datetime.now().isoformat(),
            "total_alerts": len(self.all_alerts),
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == AlertLevel.CRITICAL]),
            "action_alerts": len([a for a in active_alerts if a.severity == AlertLevel.ACTION]),
            "warning_alerts": len([a for a in active_alerts if a.severity == AlertLevel.WARNING]),
            "standards": standards_summary,
            "blanks": blanks_summary,
            "duplicates": duplicates_summary,
            "labs_tracked": list(self.lab_submissions.keys()),
            "total_qaqc_samples": sum(len(s) for s in self.lab_submissions.values())
        }
    
    def get_lab_performance(self, lab_id: str,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> LabPerformance:
        """Calculate laboratory performance metrics."""
        samples = self.lab_submissions.get(lab_id, [])
        
        if start_date:
            samples = [s for s in samples if s.submission_date >= start_date]
        if end_date:
            samples = [s for s in samples if s.submission_date <= end_date]
        
        standards = [s for s in samples if s.qaqc_type == QAQCType.STANDARD]
        blanks = [s for s in samples if s.qaqc_type == QAQCType.BLANK]
        
        standards_passed = 0
        for std in standards:
            std_alerts = [a for a in self.all_alerts 
                        if a.batch_id == std.batch_id and a.severity == AlertLevel.ACTION]
            if not std_alerts:
                standards_passed += 1
        
        blanks_passed = 0
        for blank in blanks:
            blank_alerts = [a for a in self.all_alerts
                          if a.batch_id == blank.batch_id and 
                          a.alert_type == FailureType.CONTAMINATION]
            if not blank_alerts:
                blanks_passed += 1
        
        return LabPerformance(
            lab_id=lab_id,
            period_start=start_date or datetime.min,
            period_end=end_date or datetime.now(),
            total_samples=len(samples),
            standards_analyzed=len(standards),
            standards_passed=standards_passed,
            blanks_analyzed=len(blanks),
            blanks_passed=blanks_passed,
            duplicates_analyzed=0,
            duplicates_passed=0,
            failure_rate=(len(samples) - standards_passed - blanks_passed) / len(samples) * 100 if samples else 0
        )
    
    def export_alerts(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Export alerts for reporting."""
        alerts = self.all_alerts if include_resolved else [a for a in self.all_alerts if not a.resolved]
        
        return [{
            "alert_id": a.alert_id,
            "type": a.alert_type.value,
            "severity": a.severity.value,
            "element": a.element,
            "reference": a.reference_id,
            "batch": a.batch_id,
            "lab": a.lab_id,
            "message": a.message,
            "value": a.value,
            "expected": a.expected,
            "threshold": a.threshold,
            "timestamp": a.timestamp.isoformat(),
            "resolved": a.resolved
        } for a in alerts]


def create_qaqc_dashboard(project_name: str = "default") -> QAQCDashboard:
    """
    Factory function to create a QA/QC dashboard.
    
    Args:
        project_name: Project identifier
    
    Returns:
        Configured QAQCDashboard instance
    """
    return QAQCDashboard(project_name)


def create_standard_reference(standard_id: str, name: str,
                             certified_values: Dict[str, float],
                             uncertainties: Optional[Dict[str, float]] = None,
                             supplier: str = "") -> StandardReference:
    """
    Factory function to create a CRM reference.
    
    Args:
        standard_id: Unique identifier
        name: Display name
        certified_values: Dict of element -> certified value
        uncertainties: Dict of element -> 1-sigma uncertainty
        supplier: CRM supplier name
    
    Returns:
        Configured StandardReference
    """
    if uncertainties is None:
        uncertainties = {k: v * 0.05 for k, v in certified_values.items()}
    
    return StandardReference(
        standard_id=standard_id,
        name=name,
        certified_values=certified_values,
        uncertainties=uncertainties,
        supplier=supplier
    )


class QAQCAnalyzer:
    """
    Facade over QAQCDashboard providing the per-project analysis interface
    consumed by the API layer.

    Wraps a QAQCDashboard and exposes a small, stable surface:
    analyze(), generate_control_chart() and get_standards(). All metrics are
    computed from the registered QA/QC data via the underlying analyzers;
    when no data has been registered the methods return empty/zero summaries.
    """

    def __init__(self, project_name: str = "default"):
        self._dashboards: Dict[str, QAQCDashboard] = {}
        self._default_project = project_name

    def _dashboard(self, project_id: str) -> QAQCDashboard:
        """Get (or lazily create) the dashboard for a project."""
        if project_id not in self._dashboards:
            self._dashboards[project_id] = QAQCDashboard(project_id)
        return self._dashboards[project_id]

    def register_sample(self, project_id: str, sample: QAQCSample) -> List[QAQCAlert]:
        """Register a QA/QC sample with the project's dashboard."""
        return self._dashboard(project_id).process_sample(sample)

    def analyze(self, project_id: str, qaqc_type: QAQCType,
                element: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a QA/QC analysis for a project.

        Computes pass/fail statistics from alerts and control points tracked
        by the underlying analyzers for the requested QA/QC type.
        """
        dashboard = self._dashboard(project_id)
        summary = dashboard.get_summary()

        if qaqc_type == QAQCType.STANDARD:
            section = summary.get("standards", {})
            stats = list(section.values())
            if element:
                stats = [s for k, s in section.items()
                         if k.endswith(f"_{element}") and s]
            stats = [s for s in stats if s]
            total = sum(s.get("n_samples", 0) for s in stats)
            if total == 0:
                pass_rate = 0.0
            else:
                pass_rate = sum(s.get("pass_rate", 0.0) * s.get("n_samples", 0)
                                for s in stats) / total
            mean_bias = (sum(s.get("mean_bias", 0.0) for s in stats) / len(stats)) if stats else 0.0
            return {
                "total_samples": total,
                "pass_count": int(round(total * pass_rate / 100.0)),
                "fail_count": total - int(round(total * pass_rate / 100.0)),
                "pass_rate": pass_rate / 100.0,
                "mean_deviation": mean_bias,
                "std_deviation": (sum(s.get("std_value", 0.0) for s in stats) / len(stats)) if stats else 0.0,
                "outliers": [a["alert_id"] for a in dashboard.export_alerts()
                             if not element or a.get("element") == element]
            }

        if qaqc_type == QAQCType.BLANK:
            section = summary.get("blanks", {})
            stats = [s for k, s in section.items()
                     if s and (not element or k == element)]
            total = sum(s.get("n_samples", s.get("total", 0)) for s in stats)
            contamination = [a for a in dashboard.export_alerts()
                             if a.get("type") == FailureType.CONTAMINATION.value
                             and (not element or a.get("element") == element)]
            fail = len(contamination)
            return {
                "total_samples": total,
                "pass_count": max(total - fail, 0),
                "fail_count": fail,
                "pass_rate": ((total - fail) / total) if total else 0.0,
                "mean_deviation": 0.0,
                "std_deviation": 0.0,
                "outliers": [a["alert_id"] for a in contamination]
            }

        # Duplicates / umpire: precision statistics
        section = summary.get("duplicates", {})
        stats = [s for k, s in section.items() if s and (not element or k == element)]
        precisions = []
        for s in stats:
            for key in ("field", "coarse", "pulp"):
                entry = s.get(key) or {}
                if entry.get("precision_percent") is not None:
                    precisions.append(entry["precision_percent"])
        total = summary.get("total_qaqc_samples", 0)
        return {
            "total_samples": total,
            "pass_count": total,
            "fail_count": 0,
            "pass_rate": 1.0 if total else 0.0,
            "mean_deviation": (sum(precisions) / len(precisions)) if precisions else 0.0,
            "std_deviation": 0.0,
            "outliers": []
        }

    def generate_control_chart(self, project_id: str, standard_id: str,
                               element: str,
                               chart_type: ControlChartType = ControlChartType.SHEWHART
                               ) -> Dict[str, Any]:
        """Generate control chart data for a standard/element pair."""
        dashboard = self._dashboard(project_id)
        data = dashboard.standards_analyzer.get_control_chart_data(
            standard_id, element, chart_type
        )
        if not data:
            return {}
        return {
            "centerLine": data.get("center_line", 0.0),
            "upperControlLimit": data.get("action_high", 0.0),
            "lowerControlLimit": data.get("action_low", 0.0),
            "upperWarningLimit": data.get("warning_high", 0.0),
            "lowerWarningLimit": data.get("warning_low", 0.0),
            "dataPoints": [
                {
                    "date": d,
                    "sampleId": sid,
                    "value": v,
                    "zScore": z,
                    "alertLevel": lvl
                }
                for d, sid, v, z, lvl in zip(
                    data.get("dates", []),
                    data.get("sample_ids", []),
                    data.get("values", []),
                    data.get("z_scores", []),
                    data.get("alert_levels", [])
                )
            ],
            "outOfControl": [
                sid for sid, lvl in zip(data.get("sample_ids", []),
                                        data.get("alert_levels", []))
                if lvl != AlertLevel.OK.value
            ]
        }

    def get_standards(self, project_id: str) -> List[Dict[str, Any]]:
        """List all registered standards (CRMs) for a project."""
        dashboard = self._dashboard(project_id)
        return [
            {
                "standardId": std.standard_id,
                "name": std.name,
                "certifiedValues": std.certified_values,
                "uncertainties": std.uncertainties,
                "supplier": std.supplier
            }
            for std in dashboard.standards_analyzer.standards.values()
        ]


def create_qaqc_analyzer(project_name: str = "default") -> QAQCAnalyzer:
    """Factory function to create a QAQCAnalyzer facade."""
    return QAQCAnalyzer(project_name)
