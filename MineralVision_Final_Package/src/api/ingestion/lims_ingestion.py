"""
Laboratory LIMS Data Ingestion Module for MineralVision Platform.

Supports ingestion of laboratory analytical data from various LIMS formats:
- Generic CSV/Excel exports
- ALS Global format
- SGS format
- Bureau Veritas format
- Intertek format
- ActLabs format

Provides method normalization, QC sample handling, and data validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from datetime import datetime
import csv
import re


class LIMSFormat(Enum):
    """Supported LIMS export formats."""
    GENERIC_CSV = "generic_csv"
    ALS_GLOBAL = "als_global"
    SGS = "sgs"
    BUREAU_VERITAS = "bureau_veritas"
    INTERTEK = "intertek"
    ACTLABS = "actlabs"
    MSALABS = "msalabs"
    CUSTOM = "custom"


class AnalyticalMethod(Enum):
    """Common analytical methods."""
    # Fire assay
    FA_AAS = "fa_aas"  # Fire assay with AAS finish
    FA_GRAV = "fa_grav"  # Fire assay gravimetric
    FA_ICP = "fa_icp"  # Fire assay with ICP finish
    
    # ICP methods
    ICP_OES = "icp_oes"  # ICP Optical Emission
    ICP_MS = "icp_ms"  # ICP Mass Spectrometry
    ICP_AES = "icp_aes"  # ICP Atomic Emission
    
    # Digestion methods
    AQUA_REGIA = "aqua_regia"
    FOUR_ACID = "four_acid"
    SODIUM_PEROXIDE = "sodium_peroxide"
    LITHIUM_BORATE = "lithium_borate"
    
    # XRF
    XRF_PRESSED = "xrf_pressed"
    XRF_FUSED = "xrf_fused"
    
    # Other
    LECO = "leco"  # Carbon/Sulfur
    SPECIFIC_ION = "specific_ion"
    TITRATION = "titration"
    COLORIMETRIC = "colorimetric"
    
    # Soil-specific
    MEHLICH_3 = "mehlich_3"
    BRAY_1 = "bray_1"
    OLSEN = "olsen"
    AMMONIUM_ACETATE = "ammonium_acetate"
    DTPA = "dtpa"
    
    UNKNOWN = "unknown"


class SampleType(Enum):
    """Sample types for QC tracking."""
    ROUTINE = "routine"
    DUPLICATE = "duplicate"
    STANDARD = "standard"
    BLANK = "blank"
    REPLICATE = "replicate"
    CHECK = "check"
    PREP_DUPLICATE = "prep_duplicate"
    PULP_DUPLICATE = "pulp_duplicate"


class ResultUnit(Enum):
    """Result units."""
    PPM = "ppm"
    PPB = "ppb"
    PERCENT = "%"
    G_T = "g/t"
    OZ_T = "oz/t"
    MG_KG = "mg/kg"
    UG_KG = "ug/kg"
    MEQL = "meq/L"
    CMOL_KG = "cmol/kg"
    PH = "pH"
    DS_M = "dS/m"  # Electrical conductivity
    RATIO = "ratio"


@dataclass
class LabResult:
    """Single analytical result."""
    element: str
    value: float
    unit: ResultUnit
    
    # Method information
    method: AnalyticalMethod = AnalyticalMethod.UNKNOWN
    method_code: str = ""
    
    # Detection limits
    detection_limit: float = 0.0
    upper_limit: float = float('inf')
    
    # Flags
    below_detection: bool = False
    above_upper_limit: bool = False
    
    # Quality
    precision: Optional[float] = None  # RSD %
    accuracy: Optional[float] = None  # Recovery %
    
    # Raw value
    raw_value: str = ""
    
    def to_ppm(self) -> float:
        """Convert value to ppm."""
        conversions = {
            ResultUnit.PPM: 1.0,
            ResultUnit.PPB: 0.001,
            ResultUnit.PERCENT: 10000.0,
            ResultUnit.G_T: 1.0,
            ResultUnit.OZ_T: 31103.5,  # Troy oz to g
            ResultUnit.MG_KG: 1.0,
            ResultUnit.UG_KG: 0.001
        }
        return self.value * conversions.get(self.unit, 1.0)


@dataclass
class LabSample:
    """Laboratory sample with all results."""
    sample_id: str
    lab_id: str
    
    # Sample information
    sample_type: SampleType = SampleType.ROUTINE
    batch_id: str = ""
    job_number: str = ""
    
    # Dates
    received_date: Optional[datetime] = None
    analyzed_date: Optional[datetime] = None
    reported_date: Optional[datetime] = None
    
    # Sample preparation
    prep_code: str = ""
    sample_weight: float = 0.0  # grams
    
    # Location (if provided)
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    
    # Results
    results: Dict[str, LabResult] = field(default_factory=dict)
    
    # QC information
    qc_batch: str = ""
    standard_name: str = ""  # For standard samples
    expected_values: Dict[str, float] = field(default_factory=dict)  # For standards
    
    # Raw data
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    is_valid: bool = True
    validation_notes: List[str] = field(default_factory=list)
    
    def get_result(self, element: str) -> Optional[LabResult]:
        """Get result for element."""
        return self.results.get(element)
    
    def get_value(self, element: str, default: float = 0.0) -> float:
        """Get value for element, handling below-detection."""
        result = self.results.get(element)
        if result is None:
            return default
        if result.below_detection:
            return result.detection_limit / 2
        return result.value
    
    def get_all_elements(self) -> List[str]:
        """Get list of all analyzed elements."""
        return list(self.results.keys())


@dataclass
class QCReport:
    """Quality control report for a batch."""
    batch_id: str
    
    # Standards
    standards_analyzed: int = 0
    standards_passed: int = 0
    standard_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Blanks
    blanks_analyzed: int = 0
    blanks_passed: int = 0
    blank_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Duplicates
    duplicates_analyzed: int = 0
    duplicates_passed: int = 0
    duplicate_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Overall
    overall_pass: bool = True
    issues: List[str] = field(default_factory=list)


class MethodNormalizer:
    """Normalize analytical methods across laboratories."""
    
    # Method code patterns
    METHOD_PATTERNS = {
        AnalyticalMethod.FA_AAS: [
            r'fa.*aas', r'fire.*assay.*aas', r'au-aa', r'pb.*collect'
        ],
        AnalyticalMethod.FA_GRAV: [
            r'fa.*grav', r'fire.*assay.*grav', r'au-gra'
        ],
        AnalyticalMethod.FA_ICP: [
            r'fa.*icp', r'fire.*assay.*icp'
        ],
        AnalyticalMethod.ICP_OES: [
            r'icp.*oes', r'icp.*aes', r'me-icp', r'icp41', r'icp61'
        ],
        AnalyticalMethod.ICP_MS: [
            r'icp.*ms', r'me-ms', r'icp.*mass'
        ],
        AnalyticalMethod.AQUA_REGIA: [
            r'aqua.*regia', r'ar.*digest', r'3.*acid'
        ],
        AnalyticalMethod.FOUR_ACID: [
            r'4.*acid', r'four.*acid', r'hf.*digest', r'total.*digest'
        ],
        AnalyticalMethod.LITHIUM_BORATE: [
            r'li.*borate', r'lithium.*borate', r'fusion', r'xrf.*fus'
        ],
        AnalyticalMethod.XRF_PRESSED: [
            r'xrf.*press', r'pressed.*pellet'
        ],
        AnalyticalMethod.XRF_FUSED: [
            r'xrf.*fus', r'fused.*bead', r'glass.*bead'
        ],
        AnalyticalMethod.LECO: [
            r'leco', r'c.*s.*analyzer', r'combustion'
        ],
        AnalyticalMethod.MEHLICH_3: [
            r'mehlich', r'm3', r'meh.*3'
        ],
        AnalyticalMethod.BRAY_1: [
            r'bray', r'bray.*1'
        ],
        AnalyticalMethod.OLSEN: [
            r'olsen', r'bicarbonate.*p'
        ],
        AnalyticalMethod.AMMONIUM_ACETATE: [
            r'ammonium.*acetate', r'nh4oac', r'exchangeable'
        ],
        AnalyticalMethod.DTPA: [
            r'dtpa', r'extractable.*micro'
        ]
    }
    
    @classmethod
    def normalize(cls, method_code: str) -> AnalyticalMethod:
        """Normalize method code to standard enum."""
        method_lower = method_code.lower()
        
        for method, patterns in cls.METHOD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, method_lower):
                    return method
        
        return AnalyticalMethod.UNKNOWN


class UnitNormalizer:
    """Normalize units across laboratories."""
    
    UNIT_PATTERNS = {
        ResultUnit.PPM: [r'^ppm$', r'^mg/kg$', r'^mg\.kg', r'^ug/g$', r'^g/t$'],
        ResultUnit.PPB: [r'^ppb$', r'^ug/kg$', r'^ng/g$'],
        ResultUnit.PERCENT: [r'^%$', r'^pct$', r'^percent$', r'^wt%$', r'^wt\.%$'],
        ResultUnit.G_T: [r'^g/t$', r'^g/ton', r'^gpt$'],
        ResultUnit.OZ_T: [r'^oz/t', r'^opt$', r'^troy'],
        ResultUnit.MEQL: [r'^meq/l', r'^meq\.l', r'^milliequiv'],
        ResultUnit.CMOL_KG: [r'^cmol', r'^centimol'],
        ResultUnit.PH: [r'^ph$'],
        ResultUnit.DS_M: [r'^ds/m', r'^ms/cm', r'^mmhos', r'^ec$']
    }
    
    @classmethod
    def normalize(cls, unit_str: str) -> ResultUnit:
        """Normalize unit string to standard enum."""
        unit_lower = unit_str.lower().strip()
        
        for unit, patterns in cls.UNIT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, unit_lower):
                    return unit
        
        return ResultUnit.PPM  # Default


class GenericCSVParser:
    """Parser for generic CSV LIMS exports."""
    
    # Common column name patterns
    COLUMN_PATTERNS = {
        'sample_id': ['sample', 'sample_id', 'sampleid', 'sample_no', 'sample_number'],
        'lab_id': ['lab_id', 'labid', 'lab_no', 'lab_number', 'certificate'],
        'batch_id': ['batch', 'batch_id', 'job', 'job_no', 'work_order'],
        'sample_type': ['type', 'sample_type', 'qc_type'],
        'received_date': ['received', 'date_received', 'recv_date'],
        'analyzed_date': ['analyzed', 'analysis_date', 'date_analyzed'],
        'x': ['x', 'easting', 'longitude', 'lon'],
        'y': ['y', 'northing', 'latitude', 'lat'],
        'z': ['z', 'elevation', 'depth']
    }
    
    def __init__(self, element_columns: Optional[Dict[str, str]] = None):
        """
        Initialize parser.
        
        Args:
            element_columns: Mapping of column names to element symbols
        """
        self.element_columns = element_columns or {}
    
    def parse_file(self, file_path: str) -> List[LabSample]:
        """Parse CSV file and return samples."""
        samples = []
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            # Detect delimiter
            sample = f.read(4096)
            f.seek(0)
            
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ','
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            # Map columns
            column_map = self._map_columns(reader.fieldnames or [])
            element_cols = self._identify_element_columns(reader.fieldnames or [])
            
            for row in reader:
                sample = self._parse_row(row, column_map, element_cols)
                if sample:
                    samples.append(sample)
        
        return samples
    
    def _map_columns(self, fieldnames: List[str]) -> Dict[str, str]:
        """Map CSV columns to standard names."""
        column_map = {}
        
        for field in fieldnames:
            field_lower = field.lower().strip()
            
            for standard_name, patterns in self.COLUMN_PATTERNS.items():
                for pattern in patterns:
                    if pattern in field_lower or field_lower == pattern:
                        column_map[standard_name] = field
                        break
        
        return column_map
    
    def _identify_element_columns(self, fieldnames: List[str]) -> Dict[str, Tuple[str, ResultUnit]]:
        """Identify element columns and their units."""
        elements = {}
        
        # Standard element symbols
        element_symbols = [
            'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
            'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ag', 'Cd', 'Sn',
            'Sb', 'Te', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
            'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
            'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U'
        ]
        
        # Soil parameters
        soil_params = [
            'pH', 'EC', 'CEC', 'OM', 'OC', 'TN', 'N', 'P', 'K', 'Ca', 'Mg', 'Na', 'S',
            'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo', 'Al', 'H', 'ESP', 'SAR', 'BS'
        ]
        
        for field in fieldnames:
            # Check custom mapping
            if field in self.element_columns:
                elem = self.element_columns[field]
                elements[field] = (elem, ResultUnit.PPM)
                continue
            
            # Extract element and unit from column name
            # Pattern: Element_Unit or Element (Unit) or Element-Unit
            match = re.match(r'([A-Za-z]{1,2})[\s_\-\(]*([^)]*)', field)
            if match:
                elem_str = match.group(1)
                unit_str = match.group(2).strip('() ')
                
                # Capitalize element properly
                elem = elem_str[0].upper() + elem_str[1:].lower() if len(elem_str) > 1 else elem_str.upper()
                
                if elem in element_symbols or elem in soil_params:
                    unit = UnitNormalizer.normalize(unit_str) if unit_str else ResultUnit.PPM
                    elements[field] = (elem, unit)
        
        return elements
    
    def _parse_row(
        self, row: Dict[str, str], column_map: Dict[str, str], element_cols: Dict[str, Tuple[str, ResultUnit]]
    ) -> Optional[LabSample]:
        """Parse single CSV row."""
        # Get sample ID
        sample_id_col = column_map.get('sample_id')
        if not sample_id_col or not row.get(sample_id_col):
            return None
        
        sample_id = row[sample_id_col].strip()
        
        # Get lab ID
        lab_id_col = column_map.get('lab_id')
        lab_id = row.get(lab_id_col, sample_id) if lab_id_col else sample_id
        
        # Get batch ID
        batch_col = column_map.get('batch_id')
        batch_id = row.get(batch_col, '') if batch_col else ''
        
        # Get sample type
        type_col = column_map.get('sample_type')
        sample_type = SampleType.ROUTINE
        if type_col and row.get(type_col):
            type_str = row[type_col].lower()
            if 'dup' in type_str:
                sample_type = SampleType.DUPLICATE
            elif 'std' in type_str or 'standard' in type_str:
                sample_type = SampleType.STANDARD
            elif 'blank' in type_str:
                sample_type = SampleType.BLANK
        
        # Get dates
        received_date = None
        recv_col = column_map.get('received_date')
        if recv_col and row.get(recv_col):
            received_date = self._parse_date(row[recv_col])
        
        analyzed_date = None
        anal_col = column_map.get('analyzed_date')
        if anal_col and row.get(anal_col):
            analyzed_date = self._parse_date(row[anal_col])
        
        # Get coordinates
        x = self._safe_float(row.get(column_map.get('x', '')))
        y = self._safe_float(row.get(column_map.get('y', '')))
        z = self._safe_float(row.get(column_map.get('z', '')))
        
        # Parse results
        results = {}
        for col, (element, unit) in element_cols.items():
            raw_value = row.get(col, '').strip()
            if not raw_value:
                continue
            
            result = self._parse_result(raw_value, element, unit)
            if result:
                results[element] = result
        
        return LabSample(
            sample_id=sample_id,
            lab_id=lab_id,
            sample_type=sample_type,
            batch_id=batch_id,
            received_date=received_date,
            analyzed_date=analyzed_date,
            x=x,
            y=y,
            z=z,
            results=results,
            raw_data=dict(row)
        )
    
    def _parse_result(self, raw_value: str, element: str, unit: ResultUnit) -> Optional[LabResult]:
        """Parse result value."""
        raw_value = raw_value.strip()
        
        # Check for below detection
        below_detection = False
        above_upper = False
        
        if raw_value.startswith('<') or raw_value.upper() in ['ND', 'BDL', 'N/D']:
            below_detection = True
            raw_value = raw_value.lstrip('<').strip()
        elif raw_value.startswith('>'):
            above_upper = True
            raw_value = raw_value.lstrip('>').strip()
        
        # Parse numeric value
        try:
            # Remove non-numeric characters except decimal point and minus
            cleaned = re.sub(r'[^\d.\-eE]', '', raw_value)
            if not cleaned:
                return None
            value = float(cleaned)
        except ValueError:
            return None
        
        return LabResult(
            element=element,
            value=value,
            unit=unit,
            below_detection=below_detection,
            above_upper_limit=above_upper,
            detection_limit=value if below_detection else 0.0,
            upper_limit=value if above_upper else float('inf'),
            raw_value=raw_value
        )
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string."""
        formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
            "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
            "%Y%m%d", "%d-%m-%Y", "%d.%m.%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely parse float."""
        if not value:
            return None
        try:
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return None


class ALSParser(GenericCSVParser):
    """Parser for ALS Global LIMS exports."""
    
    def __init__(self):
        super().__init__()
        # ALS-specific column patterns
        self.COLUMN_PATTERNS.update({
            'sample_id': ['sample', 'client_sample', 'sample_id'],
            'lab_id': ['lab_sample', 'als_sample', 'lab_id'],
            'batch_id': ['job_number', 'work_order', 'batch'],
            'method': ['method', 'method_code', 'analysis_code']
        })
    
    def _parse_row(self, row, column_map, element_cols):
        """Parse ALS-specific row format."""
        sample = super()._parse_row(row, column_map, element_cols)
        
        if sample:
            # Extract method information
            method_col = column_map.get('method')
            if method_col and row.get(method_col):
                method = MethodNormalizer.normalize(row[method_col])
                for result in sample.results.values():
                    result.method = method
                    result.method_code = row[method_col]
        
        return sample


class SGSParser(GenericCSVParser):
    """Parser for SGS LIMS exports."""
    
    def __init__(self):
        super().__init__()
        self.COLUMN_PATTERNS.update({
            'sample_id': ['sample_id', 'client_id', 'sample'],
            'lab_id': ['sgs_id', 'lab_id', 'certificate_no'],
            'batch_id': ['batch', 'job_no', 'work_order']
        })


class QCValidator:
    """Validate QC samples and generate reports."""
    
    def __init__(
        self,
        standard_tolerance: float = 10.0,  # % deviation allowed
        blank_threshold: float = 3.0,  # x detection limit
        duplicate_tolerance: float = 20.0  # % relative difference
    ):
        self.standard_tolerance = standard_tolerance
        self.blank_threshold = blank_threshold
        self.duplicate_tolerance = duplicate_tolerance
    
    def validate_samples(self, samples: List[LabSample]) -> QCReport:
        """Validate all QC samples and generate report."""
        # Group by batch
        batches = {}
        for sample in samples:
            batch = sample.batch_id or "default"
            if batch not in batches:
                batches[batch] = []
            batches[batch].append(sample)
        
        # Generate report for each batch
        reports = []
        for batch_id, batch_samples in batches.items():
            report = self._validate_batch(batch_id, batch_samples)
            reports.append(report)
        
        # Combine reports
        combined = QCReport(batch_id="combined")
        for report in reports:
            combined.standards_analyzed += report.standards_analyzed
            combined.standards_passed += report.standards_passed
            combined.standard_results.extend(report.standard_results)
            combined.blanks_analyzed += report.blanks_analyzed
            combined.blanks_passed += report.blanks_passed
            combined.blank_results.extend(report.blank_results)
            combined.duplicates_analyzed += report.duplicates_analyzed
            combined.duplicates_passed += report.duplicates_passed
            combined.duplicate_results.extend(report.duplicate_results)
            combined.issues.extend(report.issues)
        
        combined.overall_pass = len(combined.issues) == 0
        
        return combined
    
    def _validate_batch(self, batch_id: str, samples: List[LabSample]) -> QCReport:
        """Validate single batch."""
        report = QCReport(batch_id=batch_id)
        
        # Find QC samples
        standards = [s for s in samples if s.sample_type == SampleType.STANDARD]
        blanks = [s for s in samples if s.sample_type == SampleType.BLANK]
        duplicates = [s for s in samples if s.sample_type == SampleType.DUPLICATE]
        
        # Validate standards
        for std in standards:
            report.standards_analyzed += 1
            result = self._validate_standard(std)
            report.standard_results.append(result)
            if result['passed']:
                report.standards_passed += 1
            else:
                report.issues.append(f"Standard {std.sample_id} failed: {result['issues']}")
        
        # Validate blanks
        for blank in blanks:
            report.blanks_analyzed += 1
            result = self._validate_blank(blank)
            report.blank_results.append(result)
            if result['passed']:
                report.blanks_passed += 1
            else:
                report.issues.append(f"Blank {blank.sample_id} failed: {result['issues']}")
        
        # Validate duplicates
        routine_samples = {s.sample_id: s for s in samples if s.sample_type == SampleType.ROUTINE}
        for dup in duplicates:
            report.duplicates_analyzed += 1
            # Find original sample
            original_id = dup.sample_id.rstrip('_DUP').rstrip('-DUP').rstrip('D')
            original = routine_samples.get(original_id)
            
            result = self._validate_duplicate(dup, original)
            report.duplicate_results.append(result)
            if result['passed']:
                report.duplicates_passed += 1
            else:
                report.issues.append(f"Duplicate {dup.sample_id} failed: {result['issues']}")
        
        report.overall_pass = len(report.issues) == 0
        
        return report
    
    def _validate_standard(self, standard: LabSample) -> Dict[str, Any]:
        """Validate standard sample."""
        issues = []
        element_results = {}
        
        for element, result in standard.results.items():
            expected = standard.expected_values.get(element)
            if expected is None:
                continue
            
            if expected > 0:
                deviation = abs(result.value - expected) / expected * 100
                passed = deviation <= self.standard_tolerance
                
                element_results[element] = {
                    'expected': expected,
                    'measured': result.value,
                    'deviation_pct': deviation,
                    'passed': passed
                }
                
                if not passed:
                    issues.append(f"{element}: {deviation:.1f}% deviation")
        
        return {
            'sample_id': standard.sample_id,
            'standard_name': standard.standard_name,
            'passed': len(issues) == 0,
            'issues': issues,
            'element_results': element_results
        }
    
    def _validate_blank(self, blank: LabSample) -> Dict[str, Any]:
        """Validate blank sample."""
        issues = []
        element_results = {}
        
        for element, result in blank.results.items():
            threshold = result.detection_limit * self.blank_threshold
            passed = result.value <= threshold or result.below_detection
            
            element_results[element] = {
                'value': result.value,
                'threshold': threshold,
                'passed': passed
            }
            
            if not passed:
                issues.append(f"{element}: {result.value} > {threshold}")
        
        return {
            'sample_id': blank.sample_id,
            'passed': len(issues) == 0,
            'issues': issues,
            'element_results': element_results
        }
    
    def _validate_duplicate(
        self, duplicate: LabSample, original: Optional[LabSample]
    ) -> Dict[str, Any]:
        """Validate duplicate sample."""
        if original is None:
            return {
                'sample_id': duplicate.sample_id,
                'passed': False,
                'issues': ['Original sample not found'],
                'element_results': {}
            }
        
        issues = []
        element_results = {}
        
        for element, dup_result in duplicate.results.items():
            orig_result = original.results.get(element)
            if orig_result is None:
                continue
            
            # Calculate relative percent difference
            mean_val = (dup_result.value + orig_result.value) / 2
            if mean_val > 0:
                rpd = abs(dup_result.value - orig_result.value) / mean_val * 100
            else:
                rpd = 0
            
            passed = rpd <= self.duplicate_tolerance
            
            element_results[element] = {
                'original': orig_result.value,
                'duplicate': dup_result.value,
                'rpd': rpd,
                'passed': passed
            }
            
            if not passed:
                issues.append(f"{element}: RPD {rpd:.1f}%")
        
        return {
            'sample_id': duplicate.sample_id,
            'original_id': original.sample_id,
            'passed': len(issues) == 0,
            'issues': issues,
            'element_results': element_results
        }


class LIMSIngestionPipeline:
    """
    Complete LIMS data ingestion pipeline.
    
    Handles parsing, method normalization, QC validation, and export.
    """
    
    def __init__(
        self,
        format: LIMSFormat = LIMSFormat.GENERIC_CSV,
        validate_qc: bool = True
    ):
        self.format = format
        self.validate_qc = validate_qc
        
        self.qc_validator = QCValidator()
        self.samples: List[LabSample] = []
        self.qc_report: Optional[QCReport] = None
    
    def ingest(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest LIMS data file.
        
        Args:
            file_path: Path to LIMS export file
        
        Returns:
            Summary of ingested data
        """
        # Select parser
        parser = self._get_parser()
        
        # Parse file
        self.samples = parser.parse_file(file_path)
        
        if not self.samples:
            return {
                "file_path": file_path,
                "format": self.format.value,
                "sample_count": 0,
                "error": "No samples parsed"
            }
        
        # Validate QC
        if self.validate_qc:
            self.qc_report = self.qc_validator.validate_samples(self.samples)
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        return {
            "file_path": file_path,
            "format": self.format.value,
            "sample_count": len(self.samples),
            "routine_samples": sum(1 for s in self.samples if s.sample_type == SampleType.ROUTINE),
            "qc_samples": sum(1 for s in self.samples if s.sample_type != SampleType.ROUTINE),
            "elements_analyzed": stats["elements"],
            "qc_report": {
                "standards_pass_rate": self.qc_report.standards_passed / max(self.qc_report.standards_analyzed, 1) * 100 if self.qc_report else None,
                "blanks_pass_rate": self.qc_report.blanks_passed / max(self.qc_report.blanks_analyzed, 1) * 100 if self.qc_report else None,
                "duplicates_pass_rate": self.qc_report.duplicates_passed / max(self.qc_report.duplicates_analyzed, 1) * 100 if self.qc_report else None,
                "overall_pass": self.qc_report.overall_pass if self.qc_report else None
            } if self.qc_report else None,
            "statistics": stats
        }
    
    def _get_parser(self) -> GenericCSVParser:
        """Get appropriate parser for format."""
        if self.format == LIMSFormat.ALS_GLOBAL:
            return ALSParser()
        elif self.format == LIMSFormat.SGS:
            return SGSParser()
        else:
            return GenericCSVParser()
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not self.samples:
            return {"elements": [], "summary": {}}
        
        # Collect all elements
        all_elements = set()
        for sample in self.samples:
            all_elements.update(sample.results.keys())
        
        # Calculate stats per element
        summary = {}
        routine_samples = [s for s in self.samples if s.sample_type == SampleType.ROUTINE]
        
        for element in sorted(all_elements):
            values = []
            below_detection = 0
            
            for sample in routine_samples:
                result = sample.results.get(element)
                if result:
                    if result.below_detection:
                        below_detection += 1
                    else:
                        values.append(result.value)
            
            if values:
                summary[element] = {
                    "count": len(values),
                    "below_detection": below_detection,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values))
                }
        
        return {
            "elements": sorted(all_elements),
            "summary": summary
        }
    
    def to_geochem_samples(self) -> List[Dict[str, Any]]:
        """
        Convert to MineralVision GeochemSample format.
        """
        geochem_samples = []
        
        for sample in self.samples:
            if sample.sample_type != SampleType.ROUTINE:
                continue
            
            elements = {}
            for element, result in sample.results.items():
                elements[element] = result.to_ppm()
            
            geochem_sample = {
                "sample_id": sample.sample_id,
                "x": sample.x or 0.0,
                "y": sample.y or 0.0,
                "z": sample.z or 0.0,
                "sample_type": "soil",  # Default
                "sampling_date": sample.received_date,
                "elements": elements,
                "metadata": {
                    "source": "lims",
                    "lab_id": sample.lab_id,
                    "batch_id": sample.batch_id,
                    "analyzed_date": sample.analyzed_date.isoformat() if sample.analyzed_date else None
                }
            }
            geochem_samples.append(geochem_sample)
        
        return geochem_samples
    
    def to_soil_samples(self) -> List[Dict[str, Any]]:
        """
        Convert to MineralVision SoilSample format for agricultural analysis.
        """
        soil_samples = []
        
        for sample in self.samples:
            if sample.sample_type != SampleType.ROUTINE:
                continue
            
            # Map LIMS results to soil sample fields
            soil_sample = {
                "sample_id": sample.sample_id,
                "x": sample.x or 0.0,
                "y": sample.y or 0.0,
                "sampling_date": sample.received_date or datetime.now(),
                
                # Chemical properties
                "ph": sample.get_value("pH"),
                "ec_dsm": sample.get_value("EC"),
                "cec_cmol_kg": sample.get_value("CEC"),
                "organic_matter_pct": sample.get_value("OM") or sample.get_value("OC") * 1.724,
                
                # Macronutrients
                "nitrogen_pct": sample.get_value("N") / 10000 if sample.get_value("N") > 100 else sample.get_value("N"),
                "phosphorus_ppm": sample.get_value("P"),
                "potassium_ppm": sample.get_value("K"),
                
                # Secondary nutrients
                "calcium_ppm": sample.get_value("Ca"),
                "magnesium_ppm": sample.get_value("Mg"),
                "sulfur_ppm": sample.get_value("S"),
                
                # Micronutrients
                "iron_ppm": sample.get_value("Fe"),
                "manganese_ppm": sample.get_value("Mn"),
                "zinc_ppm": sample.get_value("Zn"),
                "copper_ppm": sample.get_value("Cu"),
                "boron_ppm": sample.get_value("B"),
                "molybdenum_ppm": sample.get_value("Mo"),
                
                # Problematic elements
                "aluminum_ppm": sample.get_value("Al"),
                "sodium_ppm": sample.get_value("Na"),
                
                "metadata": {
                    "source": "lims",
                    "lab_id": sample.lab_id,
                    "batch_id": sample.batch_id
                }
            }
            
            soil_samples.append(soil_sample)
        
        return soil_samples
    
    def export_csv(self, output_path: str) -> None:
        """Export samples to CSV."""
        if not self.samples:
            return
        
        # Collect all elements
        all_elements = set()
        for sample in self.samples:
            all_elements.update(sample.results.keys())
        elements = sorted(all_elements)
        
        # Build header
        header = ['sample_id', 'lab_id', 'sample_type', 'batch_id', 'x', 'y', 'z']
        for elem in elements:
            header.append(f"{elem}")
            header.append(f"{elem}_flag")
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for sample in self.samples:
                row = [
                    sample.sample_id,
                    sample.lab_id,
                    sample.sample_type.value,
                    sample.batch_id,
                    sample.x or '',
                    sample.y or '',
                    sample.z or ''
                ]
                
                for elem in elements:
                    result = sample.results.get(elem)
                    if result:
                        row.append(result.value)
                        flags = []
                        if result.below_detection:
                            flags.append('BDL')
                        if result.above_upper_limit:
                            flags.append('OVR')
                        row.append(','.join(flags) if flags else '')
                    else:
                        row.extend(['', ''])
                
                writer.writerow(row)
    
    def get_element_summary(self, element: str) -> Dict[str, Any]:
        """Get detailed summary for specific element."""
        values = []
        below_detection = 0
        methods = set()
        
        for sample in self.samples:
            if sample.sample_type != SampleType.ROUTINE:
                continue
            
            result = sample.results.get(element)
            if result:
                if result.below_detection:
                    below_detection += 1
                else:
                    values.append(result.value)
                
                if result.method != AnalyticalMethod.UNKNOWN:
                    methods.add(result.method.value)
        
        if not values:
            return {"element": element, "no_data": True}
        
        return {
            "element": element,
            "count": len(values),
            "below_detection": below_detection,
            "methods": list(methods),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "percentiles": {
                "p10": float(np.percentile(values, 10)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95))
            }
        }


def create_lims_pipeline(
    format: str = "generic",
    validate_qc: bool = True
) -> LIMSIngestionPipeline:
    """
    Factory function to create LIMS ingestion pipeline.
    
    Args:
        format: LIMS format ('generic', 'als', 'sgs', 'bureau_veritas', 'intertek', 'actlabs')
        validate_qc: Whether to validate QC samples
    
    Returns:
        Configured LIMSIngestionPipeline
    """
    format_map = {
        'generic': LIMSFormat.GENERIC_CSV,
        'als': LIMSFormat.ALS_GLOBAL,
        'als_global': LIMSFormat.ALS_GLOBAL,
        'sgs': LIMSFormat.SGS,
        'bureau_veritas': LIMSFormat.BUREAU_VERITAS,
        'bv': LIMSFormat.BUREAU_VERITAS,
        'intertek': LIMSFormat.INTERTEK,
        'actlabs': LIMSFormat.ACTLABS,
        'msalabs': LIMSFormat.MSALABS
    }
    
    lims_format = format_map.get(format.lower(), LIMSFormat.GENERIC_CSV)
    
    return LIMSIngestionPipeline(
        format=lims_format,
        validate_qc=validate_qc
    )
