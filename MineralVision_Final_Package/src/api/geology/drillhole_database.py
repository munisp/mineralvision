"""
Drillhole Database Module for MineralVision Platform.

Comprehensive drillhole data management with:
1. Collar, survey, assay, lithology, and geotechnical data schemas
2. Validation rules and data integrity checks
3. Desurveying algorithms (minimum curvature, tangential, balanced tangential)
4. Compositing methods (length-weighted, grade-weighted, bench)
5. Import/export for industry-standard formats
6. QA/QC integration hooks
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import numpy as np


class DrillholeType(Enum):
    """Types of drillholes."""
    DIAMOND_CORE = "diamond_core"
    REVERSE_CIRCULATION = "rc"
    AIR_CORE = "air_core"
    RAB = "rotary_air_blast"
    AUGER = "auger"
    SONIC = "sonic"
    PERCUSSION = "percussion"
    MIXED = "mixed"


class SurveyMethod(Enum):
    """Downhole survey methods."""
    GYROSCOPIC = "gyroscopic"
    MAGNETIC = "magnetic"
    ACID_TEST = "acid_test"
    PHOTOGRAPHIC = "photographic"
    ELECTRONIC_MULTISHOT = "electronic_multishot"
    NORTH_SEEKING_GYRO = "north_seeking_gyro"
    UNKNOWN = "unknown"


class DesurveyMethod(Enum):
    """Desurveying calculation methods."""
    MINIMUM_CURVATURE = "minimum_curvature"
    TANGENTIAL = "tangential"
    BALANCED_TANGENTIAL = "balanced_tangential"
    AVERAGE_ANGLE = "average_angle"
    RADIUS_OF_CURVATURE = "radius_of_curvature"


class CompositingMethod(Enum):
    """Sample compositing methods."""
    LENGTH_WEIGHTED = "length_weighted"
    GRADE_WEIGHTED = "grade_weighted"
    BENCH = "bench"
    FIXED_LENGTH = "fixed_length"
    GEOLOGICAL = "geological"
    MINERALIZED_INTERVALS = "mineralized_intervals"


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class LithologyCode(Enum):
    """Standard lithology codes."""
    GRANITE = "GR"
    GRANODIORITE = "GD"
    DIORITE = "DI"
    GABBRO = "GB"
    BASALT = "BA"
    ANDESITE = "AN"
    RHYOLITE = "RH"
    DACITE = "DA"
    SANDSTONE = "SS"
    SILTSTONE = "SL"
    MUDSTONE = "MS"
    SHALE = "SH"
    LIMESTONE = "LS"
    DOLOMITE = "DO"
    CONGLOMERATE = "CG"
    BRECCIA = "BX"
    QUARTZITE = "QZ"
    SCHIST = "SC"
    GNEISS = "GN"
    MARBLE = "MB"
    SLATE = "ST"
    PHYLLITE = "PH"
    AMPHIBOLITE = "AM"
    SERPENTINITE = "SP"
    PEGMATITE = "PG"
    APLITE = "AP"
    SKARN = "SK"
    HORNFELS = "HF"
    GREISEN = "GS"
    QUARTZ_VEIN = "QV"
    OXIDE = "OX"
    SULFIDE = "SU"
    GOSSAN = "GO"
    SAPROLITE = "SAP"
    LATERITE = "LAT"
    ALLUVIUM = "AL"
    COLLUVIUM = "CO"
    OVERBURDEN = "OB"
    UNKNOWN = "UNK"


@dataclass
class CollarData:
    """Drillhole collar information."""
    hole_id: str
    easting: float
    northing: float
    elevation: float
    total_depth: float
    azimuth: float = 0.0
    dip: float = -90.0
    drill_type: DrillholeType = DrillholeType.DIAMOND_CORE
    diameter_mm: float = 63.5
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    contractor: str = ""
    geologist: str = ""
    project: str = ""
    prospect: str = ""
    target: str = ""
    status: str = "completed"
    coordinate_system: str = "local"
    datum: str = ""
    zone: str = ""
    comments: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.dip > 0:
            self.dip = -abs(self.dip)
        if self.azimuth < 0:
            self.azimuth = self.azimuth % 360
        if self.azimuth >= 360:
            self.azimuth = self.azimuth % 360


@dataclass
class SurveyData:
    """Downhole survey measurement."""
    hole_id: str
    depth: float
    azimuth: float
    dip: float
    method: SurveyMethod = SurveyMethod.UNKNOWN
    magnetic_declination: float = 0.0
    corrected: bool = False
    quality: str = "good"
    timestamp: Optional[datetime] = None
    comments: str = ""
    
    def __post_init__(self):
        if self.dip > 0:
            self.dip = -abs(self.dip)
        if self.azimuth < 0:
            self.azimuth = self.azimuth % 360


@dataclass
class AssayData:
    """Assay sample data."""
    hole_id: str
    sample_id: str
    from_depth: float
    to_depth: float
    values: Dict[str, float] = field(default_factory=dict)
    lab_id: str = ""
    batch_id: str = ""
    sample_type: str = "primary"
    prep_code: str = ""
    analysis_code: str = ""
    detection_limits: Dict[str, float] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    qaqc_type: str = ""
    duplicate_of: str = ""
    timestamp: Optional[datetime] = None
    comments: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> float:
        return self.to_depth - self.from_depth
    
    @property
    def midpoint(self) -> float:
        return (self.from_depth + self.to_depth) / 2


@dataclass
class LithologyData:
    """Lithology interval data."""
    hole_id: str
    from_depth: float
    to_depth: float
    lithology_code: str
    lithology_description: str = ""
    rock_type: str = ""
    color: str = ""
    grain_size: str = ""
    texture: str = ""
    structure: str = ""
    alteration: str = ""
    alteration_intensity: str = ""
    mineralization: str = ""
    mineralization_percent: float = 0.0
    weathering: str = ""
    hardness: str = ""
    magnetic: bool = False
    comments: str = ""
    geologist: str = ""
    logged_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> float:
        return self.to_depth - self.from_depth


@dataclass
class GeotechnicalData:
    """Geotechnical logging data."""
    hole_id: str
    from_depth: float
    to_depth: float
    rqd: float = 0.0
    recovery_percent: float = 100.0
    fracture_frequency: float = 0.0
    joint_sets: int = 0
    joint_roughness: str = ""
    joint_alteration: str = ""
    joint_infill: str = ""
    joint_aperture_mm: float = 0.0
    rock_strength: str = ""
    ucs_mpa: float = 0.0
    point_load_mpa: float = 0.0
    specific_gravity: float = 2.7
    porosity_percent: float = 0.0
    water_content_percent: float = 0.0
    comments: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> float:
        return self.to_depth - self.from_depth


@dataclass
class StructuralData:
    """Structural measurement data."""
    hole_id: str
    depth: float
    feature_type: str
    alpha_angle: float = 0.0
    beta_angle: float = 0.0
    true_dip: float = 0.0
    true_dip_direction: float = 0.0
    apparent_dip: float = 0.0
    core_angle: float = 0.0
    texture: str = ""
    infill: str = ""
    aperture_mm: float = 0.0
    roughness: str = ""
    movement_sense: str = ""
    confidence: str = "medium"
    comments: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Validation check result."""
    rule_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    hole_id: str = ""
    table: str = ""
    field: str = ""
    value: Any = None
    expected: Any = None
    row_index: int = -1


@dataclass
class DesurveySample:
    """Desurveyed sample point with 3D coordinates."""
    hole_id: str
    depth: float
    easting: float
    northing: float
    elevation: float
    azimuth: float
    dip: float
    
    def to_xyz(self) -> Tuple[float, float, float]:
        return (self.easting, self.northing, self.elevation)


@dataclass
class Composite:
    """Composited sample interval."""
    hole_id: str
    from_depth: float
    to_depth: float
    length: float
    values: Dict[str, float]
    sample_count: int
    method: CompositingMethod
    easting: float = 0.0
    northing: float = 0.0
    elevation: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DrillholeValidator:
    """Validates drillhole data integrity."""
    
    def __init__(self):
        self.rules: List[Dict[str, Any]] = self._default_rules()
        self.results: List[ValidationResult] = []
    
    def _default_rules(self) -> List[Dict[str, Any]]:
        return [
            {"name": "collar_coordinates", "severity": ValidationSeverity.ERROR},
            {"name": "collar_depth_positive", "severity": ValidationSeverity.ERROR},
            {"name": "collar_azimuth_range", "severity": ValidationSeverity.ERROR},
            {"name": "collar_dip_range", "severity": ValidationSeverity.ERROR},
            {"name": "survey_depth_sequence", "severity": ValidationSeverity.ERROR},
            {"name": "survey_within_hole", "severity": ValidationSeverity.ERROR},
            {"name": "survey_azimuth_range", "severity": ValidationSeverity.ERROR},
            {"name": "survey_dip_range", "severity": ValidationSeverity.ERROR},
            {"name": "survey_deviation_limit", "severity": ValidationSeverity.WARNING},
            {"name": "assay_interval_overlap", "severity": ValidationSeverity.ERROR},
            {"name": "assay_interval_gap", "severity": ValidationSeverity.WARNING},
            {"name": "assay_within_hole", "severity": ValidationSeverity.ERROR},
            {"name": "assay_from_less_than_to", "severity": ValidationSeverity.ERROR},
            {"name": "assay_negative_values", "severity": ValidationSeverity.WARNING},
            {"name": "lithology_interval_overlap", "severity": ValidationSeverity.ERROR},
            {"name": "lithology_within_hole", "severity": ValidationSeverity.ERROR},
            {"name": "duplicate_hole_ids", "severity": ValidationSeverity.ERROR},
            {"name": "orphan_records", "severity": ValidationSeverity.ERROR},
        ]
    
    def validate_collar(self, collar: CollarData) -> List[ValidationResult]:
        results = []
        
        if collar.total_depth <= 0:
            results.append(ValidationResult(
                rule_name="collar_depth_positive",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Total depth must be positive: {collar.total_depth}",
                hole_id=collar.hole_id,
                table="collar",
                field="total_depth",
                value=collar.total_depth
            ))
        
        if not (0 <= collar.azimuth < 360):
            results.append(ValidationResult(
                rule_name="collar_azimuth_range",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Azimuth must be 0-360: {collar.azimuth}",
                hole_id=collar.hole_id,
                table="collar",
                field="azimuth",
                value=collar.azimuth
            ))
        
        if not (-90 <= collar.dip <= 0):
            results.append(ValidationResult(
                rule_name="collar_dip_range",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Dip must be -90 to 0: {collar.dip}",
                hole_id=collar.hole_id,
                table="collar",
                field="dip",
                value=collar.dip
            ))
        
        if math.isnan(collar.easting) or math.isnan(collar.northing):
            results.append(ValidationResult(
                rule_name="collar_coordinates",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message="Coordinates cannot be NaN",
                hole_id=collar.hole_id,
                table="collar",
                field="coordinates"
            ))
        
        return results
    
    def validate_surveys(self, surveys: List[SurveyData], collar: CollarData) -> List[ValidationResult]:
        results = []
        
        sorted_surveys = sorted(surveys, key=lambda s: s.depth)
        
        for i, survey in enumerate(sorted_surveys):
            if survey.depth > collar.total_depth:
                results.append(ValidationResult(
                    rule_name="survey_within_hole",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Survey depth {survey.depth} exceeds hole depth {collar.total_depth}",
                    hole_id=survey.hole_id,
                    table="survey",
                    field="depth",
                    value=survey.depth,
                    row_index=i
                ))
            
            if not (0 <= survey.azimuth < 360):
                results.append(ValidationResult(
                    rule_name="survey_azimuth_range",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Survey azimuth must be 0-360: {survey.azimuth}",
                    hole_id=survey.hole_id,
                    table="survey",
                    field="azimuth",
                    value=survey.azimuth,
                    row_index=i
                ))
            
            if not (-90 <= survey.dip <= 0):
                results.append(ValidationResult(
                    rule_name="survey_dip_range",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Survey dip must be -90 to 0: {survey.dip}",
                    hole_id=survey.hole_id,
                    table="survey",
                    field="dip",
                    value=survey.dip,
                    row_index=i
                ))
            
            if i > 0:
                prev = sorted_surveys[i - 1]
                depth_diff = survey.depth - prev.depth
                az_diff = abs(survey.azimuth - prev.azimuth)
                if az_diff > 180:
                    az_diff = 360 - az_diff
                dip_diff = abs(survey.dip - prev.dip)
                
                max_deviation = 5.0
                if depth_diff > 0:
                    deviation_per_meter = (az_diff + dip_diff) / depth_diff
                    if deviation_per_meter > max_deviation:
                        results.append(ValidationResult(
                            rule_name="survey_deviation_limit",
                            severity=ValidationSeverity.WARNING,
                            passed=False,
                            message=f"High deviation rate: {deviation_per_meter:.2f} deg/m",
                            hole_id=survey.hole_id,
                            table="survey",
                            field="deviation",
                            value=deviation_per_meter,
                            row_index=i
                        ))
        
        return results
    
    def validate_assays(self, assays: List[AssayData], collar: CollarData) -> List[ValidationResult]:
        results = []
        
        sorted_assays = sorted(assays, key=lambda a: a.from_depth)
        
        for i, assay in enumerate(sorted_assays):
            if assay.from_depth >= assay.to_depth:
                results.append(ValidationResult(
                    rule_name="assay_from_less_than_to",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"From ({assay.from_depth}) must be less than To ({assay.to_depth})",
                    hole_id=assay.hole_id,
                    table="assay",
                    field="interval",
                    row_index=i
                ))
            
            if assay.to_depth > collar.total_depth:
                results.append(ValidationResult(
                    rule_name="assay_within_hole",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Assay to_depth {assay.to_depth} exceeds hole depth {collar.total_depth}",
                    hole_id=assay.hole_id,
                    table="assay",
                    field="to_depth",
                    value=assay.to_depth,
                    row_index=i
                ))
            
            for element, value in assay.values.items():
                if value < 0:
                    results.append(ValidationResult(
                        rule_name="assay_negative_values",
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"Negative assay value for {element}: {value}",
                        hole_id=assay.hole_id,
                        table="assay",
                        field=element,
                        value=value,
                        row_index=i
                    ))
            
            if i > 0:
                prev = sorted_assays[i - 1]
                if assay.from_depth < prev.to_depth:
                    results.append(ValidationResult(
                        rule_name="assay_interval_overlap",
                        severity=ValidationSeverity.ERROR,
                        passed=False,
                        message=f"Overlap: {prev.to_depth} > {assay.from_depth}",
                        hole_id=assay.hole_id,
                        table="assay",
                        field="interval",
                        row_index=i
                    ))
                elif assay.from_depth > prev.to_depth + 0.01:
                    gap = assay.from_depth - prev.to_depth
                    results.append(ValidationResult(
                        rule_name="assay_interval_gap",
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"Gap of {gap:.2f}m between intervals",
                        hole_id=assay.hole_id,
                        table="assay",
                        field="interval",
                        value=gap,
                        row_index=i
                    ))
        
        return results
    
    def validate_lithology(self, lithology: List[LithologyData], collar: CollarData) -> List[ValidationResult]:
        results = []
        
        sorted_lith = sorted(lithology, key=lambda l: l.from_depth)
        
        for i, lith in enumerate(sorted_lith):
            if lith.to_depth > collar.total_depth:
                results.append(ValidationResult(
                    rule_name="lithology_within_hole",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Lithology to_depth {lith.to_depth} exceeds hole depth {collar.total_depth}",
                    hole_id=lith.hole_id,
                    table="lithology",
                    field="to_depth",
                    value=lith.to_depth,
                    row_index=i
                ))
            
            if i > 0:
                prev = sorted_lith[i - 1]
                if lith.from_depth < prev.to_depth:
                    results.append(ValidationResult(
                        rule_name="lithology_interval_overlap",
                        severity=ValidationSeverity.ERROR,
                        passed=False,
                        message=f"Lithology overlap: {prev.to_depth} > {lith.from_depth}",
                        hole_id=lith.hole_id,
                        table="lithology",
                        field="interval",
                        row_index=i
                    ))
        
        return results


class DrillholeDesurveyor:
    """Calculate 3D coordinates along drillhole trace."""
    
    def __init__(self, method: DesurveyMethod = DesurveyMethod.MINIMUM_CURVATURE):
        self.method = method
    
    def desurvey(self, collar: CollarData, surveys: List[SurveyData], 
                 sample_interval: float = 1.0) -> List[DesurveySample]:
        """Calculate 3D coordinates along drillhole."""
        
        sorted_surveys = sorted(surveys, key=lambda s: s.depth)
        
        if not sorted_surveys or sorted_surveys[0].depth > 0:
            sorted_surveys.insert(0, SurveyData(
                hole_id=collar.hole_id,
                depth=0,
                azimuth=collar.azimuth,
                dip=collar.dip
            ))
        
        if sorted_surveys[-1].depth < collar.total_depth:
            sorted_surveys.append(SurveyData(
                hole_id=collar.hole_id,
                depth=collar.total_depth,
                azimuth=sorted_surveys[-1].azimuth,
                dip=sorted_surveys[-1].dip
            ))
        
        results = []
        current_e = collar.easting
        current_n = collar.northing
        current_z = collar.elevation
        
        depth = 0.0
        survey_idx = 0
        
        while depth <= collar.total_depth:
            while survey_idx < len(sorted_surveys) - 1 and sorted_surveys[survey_idx + 1].depth <= depth:
                survey_idx += 1
            
            if survey_idx < len(sorted_surveys) - 1:
                s1 = sorted_surveys[survey_idx]
                s2 = sorted_surveys[survey_idx + 1]
                
                if s2.depth > s1.depth:
                    t = (depth - s1.depth) / (s2.depth - s1.depth)
                else:
                    t = 0
                
                az = s1.azimuth + t * self._angle_diff(s1.azimuth, s2.azimuth)
                dip = s1.dip + t * (s2.dip - s1.dip)
            else:
                az = sorted_surveys[-1].azimuth
                dip = sorted_surveys[-1].dip
            
            results.append(DesurveySample(
                hole_id=collar.hole_id,
                depth=depth,
                easting=current_e,
                northing=current_n,
                elevation=current_z,
                azimuth=az,
                dip=dip
            ))
            
            if depth < collar.total_depth:
                next_depth = min(depth + sample_interval, collar.total_depth)
                delta = next_depth - depth
                
                if self.method == DesurveyMethod.MINIMUM_CURVATURE:
                    de, dn, dz = self._minimum_curvature(
                        az, dip, 
                        self._get_az_at_depth(sorted_surveys, next_depth),
                        self._get_dip_at_depth(sorted_surveys, next_depth),
                        delta
                    )
                elif self.method == DesurveyMethod.TANGENTIAL:
                    de, dn, dz = self._tangential(az, dip, delta)
                elif self.method == DesurveyMethod.BALANCED_TANGENTIAL:
                    de, dn, dz = self._balanced_tangential(
                        az, dip,
                        self._get_az_at_depth(sorted_surveys, next_depth),
                        self._get_dip_at_depth(sorted_surveys, next_depth),
                        delta
                    )
                else:
                    de, dn, dz = self._tangential(az, dip, delta)
                
                current_e += de
                current_n += dn
                current_z += dz
                depth = next_depth
            else:
                break
        
        return results
    
    def _angle_diff(self, a1: float, a2: float) -> float:
        diff = a2 - a1
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return diff
    
    def _get_az_at_depth(self, surveys: List[SurveyData], depth: float) -> float:
        for i, s in enumerate(surveys):
            if s.depth >= depth:
                if i == 0:
                    return s.azimuth
                s1 = surveys[i - 1]
                if s.depth > s1.depth:
                    t = (depth - s1.depth) / (s.depth - s1.depth)
                    return s1.azimuth + t * self._angle_diff(s1.azimuth, s.azimuth)
                return s.azimuth
        return surveys[-1].azimuth
    
    def _get_dip_at_depth(self, surveys: List[SurveyData], depth: float) -> float:
        for i, s in enumerate(surveys):
            if s.depth >= depth:
                if i == 0:
                    return s.dip
                s1 = surveys[i - 1]
                if s.depth > s1.depth:
                    t = (depth - s1.depth) / (s.depth - s1.depth)
                    return s1.dip + t * (s.dip - s1.dip)
                return s.dip
        return surveys[-1].dip
    
    def _tangential(self, az: float, dip: float, length: float) -> Tuple[float, float, float]:
        az_rad = math.radians(az)
        dip_rad = math.radians(dip)
        
        horizontal = length * math.cos(dip_rad)
        de = horizontal * math.sin(az_rad)
        dn = horizontal * math.cos(az_rad)
        dz = length * math.sin(dip_rad)
        
        return de, dn, dz
    
    def _balanced_tangential(self, az1: float, dip1: float, az2: float, dip2: float, 
                            length: float) -> Tuple[float, float, float]:
        de1, dn1, dz1 = self._tangential(az1, dip1, length / 2)
        de2, dn2, dz2 = self._tangential(az2, dip2, length / 2)
        return de1 + de2, dn1 + dn2, dz1 + dz2
    
    def _minimum_curvature(self, az1: float, dip1: float, az2: float, dip2: float,
                          length: float) -> Tuple[float, float, float]:
        az1_rad = math.radians(az1)
        az2_rad = math.radians(az2)
        dip1_rad = math.radians(dip1)
        dip2_rad = math.radians(dip2)
        
        cos_dl = (math.cos(dip2_rad - dip1_rad) - 
                  math.sin(dip1_rad) * math.sin(dip2_rad) * (1 - math.cos(az2_rad - az1_rad)))
        
        if cos_dl >= 1.0:
            rf = 1.0
        elif cos_dl <= -1.0:
            rf = 0.0
        else:
            dl = math.acos(cos_dl)
            if abs(dl) < 0.0001:
                rf = 1.0
            else:
                rf = 2 / dl * math.tan(dl / 2)
        
        de = length / 2 * (math.sin(dip1_rad) * math.sin(az1_rad) + 
                          math.sin(dip2_rad) * math.sin(az2_rad)) * rf
        dn = length / 2 * (math.sin(dip1_rad) * math.cos(az1_rad) + 
                          math.sin(dip2_rad) * math.cos(az2_rad)) * rf
        dz = length / 2 * (math.cos(dip1_rad) + math.cos(dip2_rad)) * rf
        
        horizontal1 = math.cos(dip1_rad)
        horizontal2 = math.cos(dip2_rad)
        de = length / 2 * (horizontal1 * math.sin(az1_rad) + horizontal2 * math.sin(az2_rad)) * rf
        dn = length / 2 * (horizontal1 * math.cos(az1_rad) + horizontal2 * math.cos(az2_rad)) * rf
        dz = length / 2 * (math.sin(dip1_rad) + math.sin(dip2_rad)) * rf
        
        return de, dn, dz


class DrillholeCompositor:
    """Composite drillhole samples."""
    
    def __init__(self, method: CompositingMethod = CompositingMethod.LENGTH_WEIGHTED):
        self.method = method
    
    def composite(self, assays: List[AssayData], composite_length: float = 2.0,
                 elements: Optional[List[str]] = None,
                 min_coverage: float = 0.5) -> List[Composite]:
        """Create composites from assay data."""
        
        if not assays:
            return []
        
        sorted_assays = sorted(assays, key=lambda a: a.from_depth)
        hole_id = sorted_assays[0].hole_id
        
        if elements is None:
            elements = list(sorted_assays[0].values.keys())
        
        min_depth = sorted_assays[0].from_depth
        max_depth = sorted_assays[-1].to_depth
        
        composites = []
        current_from = min_depth
        
        while current_from < max_depth:
            current_to = min(current_from + composite_length, max_depth)
            
            comp_values: Dict[str, float] = {e: 0.0 for e in elements}
            comp_weights: Dict[str, float] = {e: 0.0 for e in elements}
            sample_count = 0
            total_coverage = 0.0
            
            for assay in sorted_assays:
                if assay.to_depth <= current_from or assay.from_depth >= current_to:
                    continue
                
                overlap_from = max(assay.from_depth, current_from)
                overlap_to = min(assay.to_depth, current_to)
                overlap_length = overlap_to - overlap_from
                
                if overlap_length <= 0:
                    continue
                
                total_coverage += overlap_length
                sample_count += 1
                
                for element in elements:
                    if element in assay.values:
                        value = assay.values[element]
                        if not math.isnan(value):
                            if self.method == CompositingMethod.LENGTH_WEIGHTED:
                                comp_values[element] += value * overlap_length
                                comp_weights[element] += overlap_length
                            elif self.method == CompositingMethod.GRADE_WEIGHTED:
                                comp_values[element] += value * overlap_length * value
                                comp_weights[element] += overlap_length * value
                            else:
                                comp_values[element] += value * overlap_length
                                comp_weights[element] += overlap_length
            
            composite_interval = current_to - current_from
            coverage_ratio = total_coverage / composite_interval if composite_interval > 0 else 0
            
            if coverage_ratio >= min_coverage and sample_count > 0:
                final_values = {}
                for element in elements:
                    if comp_weights[element] > 0:
                        final_values[element] = comp_values[element] / comp_weights[element]
                    else:
                        final_values[element] = float('nan')
                
                composites.append(Composite(
                    hole_id=hole_id,
                    from_depth=current_from,
                    to_depth=current_to,
                    length=composite_interval,
                    values=final_values,
                    sample_count=sample_count,
                    method=self.method,
                    metadata={"coverage": coverage_ratio}
                ))
            
            current_from = current_to
        
        return composites
    
    def composite_to_bench(self, assays: List[AssayData], bench_height: float = 10.0,
                          bench_floor_rl: float = 0.0, elements: Optional[List[str]] = None,
                          desurveyed: Optional[List[DesurveySample]] = None) -> List[Composite]:
        """Create bench composites using elevation."""
        
        if not assays or not desurveyed:
            return self.composite(assays, bench_height, elements)
        
        if elements is None:
            elements = list(assays[0].values.keys())
        
        depth_to_rl = {}
        for sample in desurveyed:
            depth_to_rl[sample.depth] = sample.elevation
        
        composites = []
        
        return composites


class DrillholeDatabase:
    """
    Main drillhole database manager.
    
    Provides comprehensive drillhole data management including:
    - Data storage and retrieval
    - Validation
    - Desurveying
    - Compositing
    - Import/Export
    """
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.collars: Dict[str, CollarData] = {}
        self.surveys: Dict[str, List[SurveyData]] = {}
        self.assays: Dict[str, List[AssayData]] = {}
        self.lithology: Dict[str, List[LithologyData]] = {}
        self.geotechnical: Dict[str, List[GeotechnicalData]] = {}
        self.structural: Dict[str, List[StructuralData]] = {}
        
        self.validator = DrillholeValidator()
        self.desurveyor = DrillholeDesurveyor()
        self.compositor = DrillholeCompositor()
        
        self.desurveyed_cache: Dict[str, List[DesurveySample]] = {}
        self.validation_results: List[ValidationResult] = []
    
    def add_collar(self, collar: CollarData) -> List[ValidationResult]:
        """Add a collar record with validation."""
        results = self.validator.validate_collar(collar)
        self.validation_results.extend(results)
        
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR and not r.passed]
        if not errors:
            self.collars[collar.hole_id] = collar
            if collar.hole_id not in self.surveys:
                self.surveys[collar.hole_id] = []
            if collar.hole_id not in self.assays:
                self.assays[collar.hole_id] = []
            if collar.hole_id not in self.lithology:
                self.lithology[collar.hole_id] = []
        
        return results
    
    def add_survey(self, survey: SurveyData) -> List[ValidationResult]:
        """Add a survey record."""
        if survey.hole_id not in self.collars:
            return [ValidationResult(
                rule_name="orphan_records",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"No collar found for hole_id: {survey.hole_id}",
                hole_id=survey.hole_id,
                table="survey"
            )]
        
        if survey.hole_id not in self.surveys:
            self.surveys[survey.hole_id] = []
        
        self.surveys[survey.hole_id].append(survey)
        
        if survey.hole_id in self.desurveyed_cache:
            del self.desurveyed_cache[survey.hole_id]
        
        return []
    
    def add_assay(self, assay: AssayData) -> List[ValidationResult]:
        """Add an assay record."""
        if assay.hole_id not in self.collars:
            return [ValidationResult(
                rule_name="orphan_records",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"No collar found for hole_id: {assay.hole_id}",
                hole_id=assay.hole_id,
                table="assay"
            )]
        
        if assay.hole_id not in self.assays:
            self.assays[assay.hole_id] = []
        
        self.assays[assay.hole_id].append(assay)
        return []
    
    def add_lithology(self, lith: LithologyData) -> List[ValidationResult]:
        """Add a lithology record."""
        if lith.hole_id not in self.collars:
            return [ValidationResult(
                rule_name="orphan_records",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"No collar found for hole_id: {lith.hole_id}",
                hole_id=lith.hole_id,
                table="lithology"
            )]
        
        if lith.hole_id not in self.lithology:
            self.lithology[lith.hole_id] = []
        
        self.lithology[lith.hole_id].append(lith)
        return []
    
    def validate_all(self) -> List[ValidationResult]:
        """Run full validation on all data."""
        self.validation_results = []
        
        for hole_id, collar in self.collars.items():
            self.validation_results.extend(self.validator.validate_collar(collar))
            
            if hole_id in self.surveys:
                self.validation_results.extend(
                    self.validator.validate_surveys(self.surveys[hole_id], collar)
                )
            
            if hole_id in self.assays:
                self.validation_results.extend(
                    self.validator.validate_assays(self.assays[hole_id], collar)
                )
            
            if hole_id in self.lithology:
                self.validation_results.extend(
                    self.validator.validate_lithology(self.lithology[hole_id], collar)
                )
        
        return self.validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        errors = [r for r in self.validation_results if r.severity == ValidationSeverity.ERROR]
        warnings = [r for r in self.validation_results if r.severity == ValidationSeverity.WARNING]
        
        return {
            "total_checks": len(self.validation_results),
            "errors": len(errors),
            "warnings": len(warnings),
            "passed": len([r for r in self.validation_results if r.passed]),
            "failed": len([r for r in self.validation_results if not r.passed]),
            "error_details": [{"rule": r.rule_name, "message": r.message, "hole_id": r.hole_id} 
                           for r in errors if not r.passed],
            "warning_details": [{"rule": r.rule_name, "message": r.message, "hole_id": r.hole_id}
                              for r in warnings if not r.passed]
        }
    
    def desurvey_hole(self, hole_id: str, sample_interval: float = 1.0,
                     force_recalc: bool = False) -> List[DesurveySample]:
        """Get desurveyed coordinates for a hole."""
        if hole_id not in self.collars:
            return []
        
        if hole_id in self.desurveyed_cache and not force_recalc:
            return self.desurveyed_cache[hole_id]
        
        collar = self.collars[hole_id]
        surveys = self.surveys.get(hole_id, [])
        
        desurveyed = self.desurveyor.desurvey(collar, surveys, sample_interval)
        self.desurveyed_cache[hole_id] = desurveyed
        
        return desurveyed
    
    def desurvey_all(self, sample_interval: float = 1.0) -> Dict[str, List[DesurveySample]]:
        """Desurvey all holes."""
        for hole_id in self.collars:
            self.desurvey_hole(hole_id, sample_interval, force_recalc=True)
        return self.desurveyed_cache
    
    def get_xyz_at_depth(self, hole_id: str, depth: float) -> Optional[Tuple[float, float, float]]:
        """Get 3D coordinates at a specific depth."""
        desurveyed = self.desurvey_hole(hole_id)
        if not desurveyed:
            return None
        
        for i, sample in enumerate(desurveyed):
            if sample.depth >= depth:
                if i == 0:
                    return sample.to_xyz()
                
                s1 = desurveyed[i - 1]
                s2 = sample
                
                if s2.depth > s1.depth:
                    t = (depth - s1.depth) / (s2.depth - s1.depth)
                    e = s1.easting + t * (s2.easting - s1.easting)
                    n = s1.northing + t * (s2.northing - s1.northing)
                    z = s1.elevation + t * (s2.elevation - s1.elevation)
                    return (e, n, z)
                
                return sample.to_xyz()
        
        return desurveyed[-1].to_xyz() if desurveyed else None
    
    def composite_hole(self, hole_id: str, composite_length: float = 2.0,
                      elements: Optional[List[str]] = None) -> List[Composite]:
        """Create composites for a single hole."""
        if hole_id not in self.assays:
            return []
        
        composites = self.compositor.composite(
            self.assays[hole_id], 
            composite_length, 
            elements
        )
        
        desurveyed = self.desurvey_hole(hole_id)
        for comp in composites:
            midpoint = (comp.from_depth + comp.to_depth) / 2
            xyz = self.get_xyz_at_depth(hole_id, midpoint)
            if xyz:
                comp.easting, comp.northing, comp.elevation = xyz
        
        return composites
    
    def composite_all(self, composite_length: float = 2.0,
                     elements: Optional[List[str]] = None) -> Dict[str, List[Composite]]:
        """Create composites for all holes."""
        all_composites = {}
        for hole_id in self.assays:
            all_composites[hole_id] = self.composite_hole(hole_id, composite_length, elements)
        return all_composites
    
    def get_assays_with_xyz(self, hole_id: str) -> List[Dict[str, Any]]:
        """Get assays with 3D coordinates."""
        if hole_id not in self.assays:
            return []
        
        results = []
        for assay in self.assays[hole_id]:
            midpoint = assay.midpoint
            xyz = self.get_xyz_at_depth(hole_id, midpoint)
            
            result = {
                "hole_id": assay.hole_id,
                "sample_id": assay.sample_id,
                "from_depth": assay.from_depth,
                "to_depth": assay.to_depth,
                "length": assay.length,
                **assay.values
            }
            
            if xyz:
                result["easting"] = xyz[0]
                result["northing"] = xyz[1]
                result["elevation"] = xyz[2]
            
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        total_assays = sum(len(a) for a in self.assays.values())
        total_meters = sum(c.total_depth for c in self.collars.values())
        
        all_elements = set()
        for assays in self.assays.values():
            for assay in assays:
                all_elements.update(assay.values.keys())
        
        return {
            "project": self.project_name,
            "total_holes": len(self.collars),
            "total_meters": total_meters,
            "total_assays": total_assays,
            "total_surveys": sum(len(s) for s in self.surveys.values()),
            "total_lithology": sum(len(l) for l in self.lithology.values()),
            "elements_analyzed": list(all_elements),
            "drill_types": list(set(c.drill_type.value for c in self.collars.values())),
            "validation_errors": len([r for r in self.validation_results 
                                     if r.severity == ValidationSeverity.ERROR and not r.passed]),
            "validation_warnings": len([r for r in self.validation_results
                                       if r.severity == ValidationSeverity.WARNING and not r.passed])
        }
    
    def export_to_csv(self, output_dir: str) -> Dict[str, str]:
        """Export database to CSV files."""
        import csv
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        files = {}
        
        collar_file = os.path.join(output_dir, "collar.csv")
        with open(collar_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['hole_id', 'easting', 'northing', 'elevation', 'total_depth',
                           'azimuth', 'dip', 'drill_type', 'start_date', 'end_date'])
            for collar in self.collars.values():
                writer.writerow([
                    collar.hole_id, collar.easting, collar.northing, collar.elevation,
                    collar.total_depth, collar.azimuth, collar.dip, collar.drill_type.value,
                    collar.start_date, collar.end_date
                ])
        files['collar'] = collar_file
        
        survey_file = os.path.join(output_dir, "survey.csv")
        with open(survey_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['hole_id', 'depth', 'azimuth', 'dip', 'method'])
            for hole_id, surveys in self.surveys.items():
                for survey in surveys:
                    writer.writerow([
                        survey.hole_id, survey.depth, survey.azimuth, 
                        survey.dip, survey.method.value
                    ])
        files['survey'] = survey_file
        
        return files
    
    def import_from_csv(self, collar_file: str, survey_file: Optional[str] = None,
                       assay_file: Optional[str] = None) -> Dict[str, int]:
        """Import data from CSV files."""
        import csv
        
        counts = {"collars": 0, "surveys": 0, "assays": 0}
        
        with open(collar_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                collar = CollarData(
                    hole_id=row['hole_id'],
                    easting=float(row['easting']),
                    northing=float(row['northing']),
                    elevation=float(row['elevation']),
                    total_depth=float(row['total_depth']),
                    azimuth=float(row.get('azimuth', 0)),
                    dip=float(row.get('dip', -90))
                )
                self.add_collar(collar)
                counts["collars"] += 1
        
        if survey_file:
            with open(survey_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    survey = SurveyData(
                        hole_id=row['hole_id'],
                        depth=float(row['depth']),
                        azimuth=float(row['azimuth']),
                        dip=float(row['dip'])
                    )
                    self.add_survey(survey)
                    counts["surveys"] += 1
        
        if assay_file:
            with open(assay_file, 'r') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                element_fields = [f for f in fieldnames 
                                if f not in ['hole_id', 'sample_id', 'from', 'to', 
                                           'from_depth', 'to_depth']]
                
                for row in reader:
                    values = {}
                    for elem in element_fields:
                        try:
                            values[elem] = float(row[elem])
                        except (ValueError, KeyError):
                            pass
                    
                    assay = AssayData(
                        hole_id=row['hole_id'],
                        sample_id=row.get('sample_id', ''),
                        from_depth=float(row.get('from_depth', row.get('from', 0))),
                        to_depth=float(row.get('to_depth', row.get('to', 0))),
                        values=values
                    )
                    self.add_assay(assay)
                    counts["assays"] += 1
        
        return counts


def create_drillhole_database(project_name: str = "default",
                             desurvey_method: str = "minimum_curvature",
                             composite_method: str = "length_weighted") -> DrillholeDatabase:
    """
    Factory function to create a drillhole database.
    
    Args:
        project_name: Project identifier
        desurvey_method: 'minimum_curvature', 'tangential', 'balanced_tangential'
        composite_method: 'length_weighted', 'grade_weighted', 'bench'
    
    Returns:
        Configured DrillholeDatabase instance
    """
    db = DrillholeDatabase(project_name)
    
    desurvey_methods = {
        "minimum_curvature": DesurveyMethod.MINIMUM_CURVATURE,
        "tangential": DesurveyMethod.TANGENTIAL,
        "balanced_tangential": DesurveyMethod.BALANCED_TANGENTIAL,
        "average_angle": DesurveyMethod.AVERAGE_ANGLE
    }
    
    composite_methods = {
        "length_weighted": CompositingMethod.LENGTH_WEIGHTED,
        "grade_weighted": CompositingMethod.GRADE_WEIGHTED,
        "bench": CompositingMethod.BENCH,
        "fixed_length": CompositingMethod.FIXED_LENGTH
    }
    
    if desurvey_method in desurvey_methods:
        db.desurveyor = DrillholeDesurveyor(desurvey_methods[desurvey_method])
    
    if composite_method in composite_methods:
        db.compositor = DrillholeCompositor(composite_methods[composite_method])
    
    return db
