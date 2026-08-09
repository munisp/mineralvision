"""
Portable XRF Data Ingestion Module for MineralVision Platform.

Supports ingestion of portable XRF data from various vendors:
- Olympus/Vanta
- Bruker (S1 TITAN, Tracer)
- Thermo Fisher (Niton)
- SciAps

Provides unit normalization, detection limit handling, and QC flagging.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from datetime import datetime
import csv
import re


class XRFVendor(Enum):
    """Supported XRF instrument vendors."""
    OLYMPUS_VANTA = "olympus_vanta"
    OLYMPUS_DELTA = "olympus_delta"
    BRUKER_S1_TITAN = "bruker_s1_titan"
    BRUKER_TRACER = "bruker_tracer"
    THERMO_NITON = "thermo_niton"
    SCIAPS_X = "sciaps_x"
    GENERIC = "generic"


class XRFMode(Enum):
    """XRF measurement modes."""
    SOIL = "soil"
    MINING = "mining"
    GEOCHEM = "geochem"
    ALLOY = "alloy"
    PRECIOUS_METALS = "precious_metals"
    LIGHT_ELEMENTS = "light_elements"
    CUSTOM = "custom"


class ElementUnit(Enum):
    """Element concentration units."""
    PPM = "ppm"
    PPB = "ppb"
    PERCENT = "percent"
    MG_KG = "mg/kg"
    UG_G = "ug/g"
    G_T = "g/t"  # grams per tonne (same as ppm)


@dataclass
class XRFReading:
    """Single XRF measurement reading."""
    reading_id: str
    sample_id: str
    timestamp: datetime
    
    # Location
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    
    # Instrument info
    vendor: XRFVendor = XRFVendor.GENERIC
    instrument_serial: str = ""
    mode: XRFMode = XRFMode.GEOCHEM
    
    # Measurement parameters
    measurement_time: float = 0.0  # seconds
    beam_current: float = 0.0  # uA
    voltage: float = 0.0  # kV
    filter_position: int = 0
    
    # Element concentrations (normalized to ppm)
    concentrations: Dict[str, float] = field(default_factory=dict)
    
    # Measurement errors (1-sigma, in same units as concentration)
    errors: Dict[str, float] = field(default_factory=dict)
    
    # Detection limits
    detection_limits: Dict[str, float] = field(default_factory=dict)
    
    # QC flags
    below_detection: Dict[str, bool] = field(default_factory=dict)
    quality_flags: Dict[str, str] = field(default_factory=dict)
    
    # Raw data
    raw_values: Dict[str, Any] = field(default_factory=dict)
    
    # Sample info
    sample_type: str = "soil"
    sample_prep: str = "none"
    moisture_content: Optional[float] = None
    
    # Overall QC
    is_valid: bool = True
    qc_notes: List[str] = field(default_factory=list)
    
    def get_concentration(self, element: str, default: float = 0.0) -> float:
        """Get concentration for element, handling below-detection."""
        if element in self.below_detection and self.below_detection[element]:
            # Return half detection limit for below-detection values
            if element in self.detection_limits:
                return self.detection_limits[element] / 2
        return self.concentrations.get(element, default)
    
    def get_error_percent(self, element: str) -> float:
        """Get relative error as percentage."""
        conc = self.concentrations.get(element, 0)
        error = self.errors.get(element, 0)
        if conc > 0:
            return (error / conc) * 100
        return 0.0


@dataclass
class XRFCalibration:
    """XRF calibration information."""
    calibration_id: str
    calibration_date: datetime
    
    # Calibration type
    calibration_type: str = "factory"  # factory, field, custom
    
    # Reference materials used
    reference_materials: List[str] = field(default_factory=list)
    
    # Element-specific calibration factors
    slope: Dict[str, float] = field(default_factory=dict)
    intercept: Dict[str, float] = field(default_factory=dict)
    r_squared: Dict[str, float] = field(default_factory=dict)
    
    # Valid range
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    
    # Notes
    notes: str = ""


class UnitConverter:
    """Convert between concentration units."""
    
    # Conversion factors to ppm
    TO_PPM = {
        ElementUnit.PPM: 1.0,
        ElementUnit.PPB: 0.001,
        ElementUnit.PERCENT: 10000.0,
        ElementUnit.MG_KG: 1.0,
        ElementUnit.UG_G: 1.0,
        ElementUnit.G_T: 1.0
    }
    
    @classmethod
    def to_ppm(cls, value: float, from_unit: ElementUnit) -> float:
        """Convert value to ppm."""
        return value * cls.TO_PPM[from_unit]
    
    @classmethod
    def from_ppm(cls, value: float, to_unit: ElementUnit) -> float:
        """Convert ppm to target unit."""
        return value / cls.TO_PPM[to_unit]
    
    @classmethod
    def detect_unit(cls, unit_str: str) -> ElementUnit:
        """Detect unit from string."""
        unit_str = unit_str.lower().strip()
        
        if unit_str in ['ppm', 'mg/kg', 'mg kg-1', 'ug/g', 'μg/g']:
            return ElementUnit.PPM
        elif unit_str in ['ppb', 'ug/kg', 'μg/kg', 'ng/g']:
            return ElementUnit.PPB
        elif unit_str in ['%', 'percent', 'wt%', 'wt.%']:
            return ElementUnit.PERCENT
        elif unit_str in ['g/t', 'g/ton', 'g/tonne']:
            return ElementUnit.G_T
        else:
            return ElementUnit.PPM  # Default


class VendorParser:
    """Base class for vendor-specific parsers."""
    
    # Standard element symbols
    ELEMENTS = [
        'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
        'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd',
        'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
        'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U'
    ]
    
    def parse(self, file_path: str) -> List[XRFReading]:
        """Parse vendor file and return readings."""
        raise NotImplementedError
    
    def _normalize_element(self, element: str) -> Optional[str]:
        """Normalize element symbol."""
        # Remove units and whitespace
        element = re.sub(r'\s*\(.*\)', '', element).strip()
        
        # Capitalize properly
        if len(element) == 1:
            element = element.upper()
        elif len(element) >= 2:
            element = element[0].upper() + element[1:].lower()
        
        # Check if valid element
        if element in self.ELEMENTS:
            return element
        
        # Try common variations
        variations = {
            'FE': 'Fe', 'CU': 'Cu', 'ZN': 'Zn', 'PB': 'Pb', 'AS': 'As',
            'AU': 'Au', 'AG': 'Ag', 'MN': 'Mn', 'NI': 'Ni', 'CR': 'Cr'
        }
        if element.upper() in variations:
            return variations[element.upper()]
        
        return None


class OlympusVantaParser(VendorParser):
    """Parser for Olympus Vanta XRF exports."""
    
    def parse(self, file_path: str) -> List[XRFReading]:
        """Parse Olympus Vanta CSV export."""
        readings = []
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            # Detect delimiter
            first_line = f.readline()
            f.seek(0)
            
            delimiter = ',' if ',' in first_line else '\t'
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                reading = self._parse_row(row)
                if reading:
                    readings.append(reading)
        
        return readings
    
    def _parse_row(self, row: Dict[str, str]) -> Optional[XRFReading]:
        """Parse single row from Olympus export."""
        # Find reading ID column
        reading_id = (
            row.get('Reading #') or 
            row.get('Reading') or 
            row.get('ID') or 
            str(hash(str(row)))
        )
        
        # Find sample ID
        sample_id = (
            row.get('Sample ID') or 
            row.get('Sample') or 
            row.get('Name') or 
            reading_id
        )
        
        # Parse timestamp
        timestamp = datetime.now()
        date_str = row.get('Date') or row.get('DateTime')
        time_str = row.get('Time')
        if date_str:
            try:
                if time_str:
                    timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                else:
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                pass
        
        # Parse coordinates
        x = self._parse_float(row.get('Longitude') or row.get('X'))
        y = self._parse_float(row.get('Latitude') or row.get('Y'))
        z = self._parse_float(row.get('Elevation') or row.get('Z'))
        
        # Parse measurement parameters
        meas_time = self._parse_float(row.get('Test Time') or row.get('Duration'))
        
        # Parse element concentrations
        concentrations = {}
        errors = {}
        detection_limits = {}
        below_detection = {}
        
        for key, value in row.items():
            # Check if this is an element column
            element = self._normalize_element(key.split('(')[0].strip())
            if not element:
                continue
            
            # Check for error column
            if 'Err' in key or 'Error' in key or '1s' in key:
                errors[element] = self._parse_float(value)
                continue
            
            # Check for LOD column
            if 'LOD' in key or 'DL' in key:
                detection_limits[element] = self._parse_float(value)
                continue
            
            # Parse concentration
            conc_value = value.strip()
            
            # Check for below detection indicators
            if conc_value.startswith('<') or conc_value.startswith('ND') or conc_value == 'BDL':
                below_detection[element] = True
                # Extract numeric value if present
                numeric = re.search(r'[\d.]+', conc_value)
                if numeric:
                    detection_limits[element] = float(numeric.group())
                    concentrations[element] = float(numeric.group()) / 2
                else:
                    concentrations[element] = 0.0
            else:
                conc = self._parse_float(conc_value)
                if conc is not None:
                    concentrations[element] = conc
                    below_detection[element] = False
        
        # Detect and convert units
        unit = ElementUnit.PPM  # Olympus typically exports in ppm
        for key in row.keys():
            if '%' in key:
                unit = ElementUnit.PERCENT
                break
        
        # Convert to ppm if needed
        if unit != ElementUnit.PPM:
            for element in concentrations:
                concentrations[element] = UnitConverter.to_ppm(concentrations[element], unit)
            for element in errors:
                errors[element] = UnitConverter.to_ppm(errors[element], unit)
            for element in detection_limits:
                detection_limits[element] = UnitConverter.to_ppm(detection_limits[element], unit)
        
        return XRFReading(
            reading_id=str(reading_id),
            sample_id=str(sample_id),
            timestamp=timestamp,
            x=x,
            y=y,
            z=z,
            vendor=XRFVendor.OLYMPUS_VANTA,
            measurement_time=meas_time or 0.0,
            concentrations=concentrations,
            errors=errors,
            detection_limits=detection_limits,
            below_detection=below_detection,
            raw_values=dict(row)
        )
    
    def _parse_float(self, value: Any) -> Optional[float]:
        """Parse float value, handling various formats."""
        if value is None:
            return None
        
        try:
            # Remove common non-numeric characters
            cleaned = str(value).replace(',', '').replace('<', '').replace('>', '')
            cleaned = re.sub(r'[^\d.\-eE]', '', cleaned)
            if cleaned:
                return float(cleaned)
        except (ValueError, TypeError):
            pass
        
        return None


class BrukerParser(VendorParser):
    """Parser for Bruker XRF exports."""
    
    def parse(self, file_path: str) -> List[XRFReading]:
        """Parse Bruker CSV/TXT export."""
        readings = []
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # Bruker exports can have different formats
        if '\t' in content:
            delimiter = '\t'
        else:
            delimiter = ','
        
        lines = content.strip().split('\n')
        
        # Find header row
        header_idx = 0
        for i, line in enumerate(lines):
            if any(elem in line for elem in ['Fe', 'Cu', 'Zn', 'Sample']):
                header_idx = i
                break
        
        headers = lines[header_idx].split(delimiter)
        
        for line in lines[header_idx + 1:]:
            if not line.strip():
                continue
            
            values = line.split(delimiter)
            if len(values) < len(headers):
                values.extend([''] * (len(headers) - len(values)))
            
            row = dict(zip(headers, values))
            reading = self._parse_bruker_row(row)
            if reading:
                readings.append(reading)
        
        return readings
    
    def _parse_bruker_row(self, row: Dict[str, str]) -> Optional[XRFReading]:
        """Parse single Bruker row."""
        reading_id = row.get('Spectrum', '') or row.get('Reading', '') or str(hash(str(row)))
        sample_id = row.get('Sample', '') or row.get('Name', '') or reading_id
        
        concentrations = {}
        errors = {}
        below_detection = {}
        
        for key, value in row.items():
            element = self._normalize_element(key)
            if not element:
                continue
            
            value = str(value).strip()
            
            if value.startswith('<') or value == 'ND':
                below_detection[element] = True
                numeric = re.search(r'[\d.]+', value)
                concentrations[element] = float(numeric.group()) / 2 if numeric else 0.0
            else:
                try:
                    concentrations[element] = float(value.replace(',', ''))
                    below_detection[element] = False
                except ValueError:
                    continue
        
        return XRFReading(
            reading_id=str(reading_id),
            sample_id=str(sample_id),
            timestamp=datetime.now(),
            vendor=XRFVendor.BRUKER_S1_TITAN,
            concentrations=concentrations,
            errors=errors,
            below_detection=below_detection,
            raw_values=dict(row)
        )


class GenericCSVParser(VendorParser):
    """Generic CSV parser for XRF data."""
    
    def __init__(self, element_columns: Optional[Dict[str, str]] = None):
        """
        Initialize parser.
        
        Args:
            element_columns: Mapping of column names to element symbols
        """
        self.element_columns = element_columns or {}
    
    def parse(self, file_path: str) -> List[XRFReading]:
        """Parse generic CSV file."""
        readings = []
        
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
            
            for i, row in enumerate(reader):
                reading = self._parse_generic_row(row, i)
                if reading:
                    readings.append(reading)
        
        return readings
    
    def _parse_generic_row(self, row: Dict[str, str], index: int) -> Optional[XRFReading]:
        """Parse generic CSV row."""
        # Try to find ID columns
        reading_id = None
        sample_id = None
        
        for key in ['id', 'reading_id', 'sample_id', 'name', 'sample']:
            for col in row.keys():
                if key in col.lower():
                    if reading_id is None:
                        reading_id = row[col]
                    if sample_id is None:
                        sample_id = row[col]
        
        if reading_id is None:
            reading_id = str(index)
        if sample_id is None:
            sample_id = reading_id
        
        # Parse coordinates
        x, y, z = None, None, None
        for key, value in row.items():
            key_lower = key.lower()
            if 'lon' in key_lower or key_lower == 'x':
                x = self._safe_float(value)
            elif 'lat' in key_lower or key_lower == 'y':
                y = self._safe_float(value)
            elif 'elev' in key_lower or key_lower == 'z':
                z = self._safe_float(value)
        
        # Parse elements
        concentrations = {}
        errors = {}
        below_detection = {}
        
        for key, value in row.items():
            # Check custom mapping first
            if key in self.element_columns:
                element = self.element_columns[key]
            else:
                element = self._normalize_element(key)
            
            if not element:
                continue
            
            value = str(value).strip()
            
            # Handle below detection
            if value.startswith('<') or value.upper() in ['ND', 'BDL', 'N/A']:
                below_detection[element] = True
                numeric = re.search(r'[\d.]+', value)
                concentrations[element] = float(numeric.group()) / 2 if numeric else 0.0
            else:
                conc = self._safe_float(value)
                if conc is not None:
                    concentrations[element] = conc
                    below_detection[element] = False
        
        if not concentrations:
            return None
        
        return XRFReading(
            reading_id=str(reading_id),
            sample_id=str(sample_id),
            timestamp=datetime.now(),
            x=x,
            y=y,
            z=z,
            vendor=XRFVendor.GENERIC,
            concentrations=concentrations,
            errors=errors,
            below_detection=below_detection,
            raw_values=dict(row)
        )
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely parse float."""
        try:
            cleaned = str(value).replace(',', '').strip()
            cleaned = re.sub(r'[^\d.\-eE]', '', cleaned)
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None


class QCChecker:
    """Quality control checker for XRF data."""
    
    # Typical detection limits (ppm) for portable XRF
    TYPICAL_DETECTION_LIMITS = {
        'Fe': 50, 'Cu': 10, 'Zn': 10, 'Pb': 5, 'As': 5,
        'Au': 5, 'Ag': 10, 'Mn': 50, 'Ni': 20, 'Cr': 20,
        'Ti': 50, 'V': 20, 'Co': 20, 'Mo': 5, 'Sn': 20,
        'Sb': 10, 'W': 10, 'Bi': 10, 'Rb': 5, 'Sr': 5,
        'Zr': 5, 'Nb': 5, 'Ba': 50, 'Th': 10, 'U': 10
    }
    
    # Reasonable concentration ranges (ppm) for soil/rock
    REASONABLE_RANGES = {
        'Fe': (1000, 500000),  # 0.1% to 50%
        'Cu': (1, 100000),
        'Zn': (1, 100000),
        'Pb': (1, 50000),
        'As': (0.1, 50000),
        'Au': (0.01, 10000),
        'Ag': (0.1, 10000),
        'Mn': (10, 100000),
        'Ni': (1, 50000),
        'Cr': (1, 50000)
    }
    
    def check_reading(self, reading: XRFReading) -> List[str]:
        """
        Check reading quality and return list of issues.
        """
        issues = []
        
        # Check for missing critical data
        if not reading.concentrations:
            issues.append("No element concentrations")
            return issues
        
        # Check measurement time
        if reading.measurement_time > 0 and reading.measurement_time < 10:
            issues.append(f"Short measurement time: {reading.measurement_time}s")
        
        # Check for unreasonable values
        for element, conc in reading.concentrations.items():
            if conc < 0:
                issues.append(f"{element}: Negative concentration")
            
            if element in self.REASONABLE_RANGES:
                min_val, max_val = self.REASONABLE_RANGES[element]
                if conc > max_val:
                    issues.append(f"{element}: Unusually high ({conc} ppm)")
        
        # Check error percentages
        for element, conc in reading.concentrations.items():
            if element in reading.errors and conc > 0:
                error_pct = (reading.errors[element] / conc) * 100
                if error_pct > 50:
                    issues.append(f"{element}: High error ({error_pct:.0f}%)")
        
        # Check for too many below-detection values
        n_below = sum(1 for v in reading.below_detection.values() if v)
        n_total = len(reading.concentrations)
        if n_total > 0 and n_below / n_total > 0.8:
            issues.append(f"Most elements below detection ({n_below}/{n_total})")
        
        return issues
    
    def flag_reading(self, reading: XRFReading) -> XRFReading:
        """Add QC flags to reading."""
        issues = self.check_reading(reading)
        reading.qc_notes = issues
        reading.is_valid = len(issues) == 0
        
        # Add element-specific flags
        for element, conc in reading.concentrations.items():
            flags = []
            
            if reading.below_detection.get(element, False):
                flags.append("BDL")
            
            if element in reading.errors and conc > 0:
                error_pct = (reading.errors[element] / conc) * 100
                if error_pct > 30:
                    flags.append(f"HIGH_ERR_{error_pct:.0f}")
            
            if conc < 0:
                flags.append("NEGATIVE")
            
            if flags:
                reading.quality_flags[element] = ",".join(flags)
        
        return reading


class XRFIngestionPipeline:
    """
    Complete XRF data ingestion pipeline.
    
    Handles parsing, unit conversion, QC, and export to MineralVision schemas.
    """
    
    def __init__(
        self,
        vendor: XRFVendor = XRFVendor.GENERIC,
        target_unit: ElementUnit = ElementUnit.PPM,
        apply_qc: bool = True
    ):
        self.vendor = vendor
        self.target_unit = target_unit
        self.apply_qc = apply_qc
        
        self.qc_checker = QCChecker()
        self.readings: List[XRFReading] = []
        self.calibration: Optional[XRFCalibration] = None
    
    def ingest(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest XRF data file.
        
        Args:
            file_path: Path to XRF export file
        
        Returns:
            Summary of ingested data
        """
        # Select parser based on vendor
        parser = self._get_parser()
        
        # Parse file
        self.readings = parser.parse(file_path)
        
        # Apply QC
        if self.apply_qc:
            self.readings = [self.qc_checker.flag_reading(r) for r in self.readings]
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        return {
            "file_path": file_path,
            "vendor": self.vendor.value,
            "reading_count": len(self.readings),
            "valid_readings": sum(1 for r in self.readings if r.is_valid),
            "elements_detected": stats["elements"],
            "statistics": stats
        }
    
    def _get_parser(self) -> VendorParser:
        """Get appropriate parser for vendor."""
        if self.vendor in [XRFVendor.OLYMPUS_VANTA, XRFVendor.OLYMPUS_DELTA]:
            return OlympusVantaParser()
        elif self.vendor in [XRFVendor.BRUKER_S1_TITAN, XRFVendor.BRUKER_TRACER]:
            return BrukerParser()
        else:
            return GenericCSVParser()
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for all readings."""
        if not self.readings:
            return {"elements": [], "summary": {}}
        
        # Collect all elements
        all_elements = set()
        for reading in self.readings:
            all_elements.update(reading.concentrations.keys())
        
        # Calculate stats per element
        summary = {}
        for element in sorted(all_elements):
            values = [
                r.concentrations[element]
                for r in self.readings
                if element in r.concentrations and not r.below_detection.get(element, False)
            ]
            
            if values:
                summary[element] = {
                    "count": len(values),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "below_detection_count": sum(
                        1 for r in self.readings
                        if element in r.below_detection and r.below_detection[element]
                    )
                }
        
        return {
            "elements": sorted(all_elements),
            "summary": summary
        }
    
    def apply_calibration(self, calibration: XRFCalibration) -> None:
        """Apply calibration correction to readings."""
        self.calibration = calibration
        
        for reading in self.readings:
            for element in reading.concentrations:
                if element in calibration.slope:
                    original = reading.concentrations[element]
                    slope = calibration.slope[element]
                    intercept = calibration.intercept.get(element, 0)
                    reading.concentrations[element] = original * slope + intercept
    
    def to_geochem_samples(self) -> List[Dict[str, Any]]:
        """
        Convert readings to MineralVision GeochemSample format.
        
        Returns list of dictionaries compatible with GeochemSample dataclass.
        """
        samples = []
        
        for reading in self.readings:
            sample = {
                "sample_id": reading.sample_id,
                "x": reading.x or 0.0,
                "y": reading.y or 0.0,
                "z": reading.z or 0.0,
                "sample_type": reading.sample_type,
                "sampling_date": reading.timestamp,
                "elements": reading.concentrations.copy(),
                "quality_flags": reading.quality_flags.copy(),
                "metadata": {
                    "source": "xrf",
                    "vendor": reading.vendor.value,
                    "measurement_time": reading.measurement_time,
                    "is_valid": reading.is_valid,
                    "qc_notes": reading.qc_notes
                }
            }
            samples.append(sample)
        
        return samples
    
    def export_csv(self, output_path: str, include_errors: bool = True) -> None:
        """Export readings to CSV file."""
        if not self.readings:
            return
        
        # Collect all elements
        all_elements = set()
        for reading in self.readings:
            all_elements.update(reading.concentrations.keys())
        elements = sorted(all_elements)
        
        # Build header
        header = ['reading_id', 'sample_id', 'timestamp', 'x', 'y', 'z', 'is_valid']
        for elem in elements:
            header.append(f"{elem}_ppm")
            if include_errors:
                header.append(f"{elem}_error")
                header.append(f"{elem}_flag")
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for reading in self.readings:
                row = [
                    reading.reading_id,
                    reading.sample_id,
                    reading.timestamp.isoformat(),
                    reading.x or '',
                    reading.y or '',
                    reading.z or '',
                    reading.is_valid
                ]
                
                for elem in elements:
                    row.append(reading.concentrations.get(elem, ''))
                    if include_errors:
                        row.append(reading.errors.get(elem, ''))
                        flag = reading.quality_flags.get(elem, '')
                        if reading.below_detection.get(elem, False):
                            flag = 'BDL' + (',' + flag if flag else '')
                        row.append(flag)
                
                writer.writerow(row)
    
    def get_element_summary(self, element: str) -> Dict[str, Any]:
        """Get detailed summary for specific element."""
        values = []
        errors = []
        below_detection_count = 0
        
        for reading in self.readings:
            if element in reading.concentrations:
                if reading.below_detection.get(element, False):
                    below_detection_count += 1
                else:
                    values.append(reading.concentrations[element])
                    if element in reading.errors:
                        errors.append(reading.errors[element])
        
        if not values:
            return {"element": element, "no_data": True}
        
        return {
            "element": element,
            "count": len(values),
            "below_detection": below_detection_count,
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
            },
            "mean_error": float(np.mean(errors)) if errors else None
        }


def create_xrf_pipeline(
    vendor: str = "generic",
    apply_qc: bool = True
) -> XRFIngestionPipeline:
    """
    Factory function to create XRF ingestion pipeline.
    
    Args:
        vendor: XRF vendor name ('olympus', 'bruker', 'thermo', 'sciaps', 'generic')
        apply_qc: Whether to apply QC checks
    
    Returns:
        Configured XRFIngestionPipeline
    """
    vendor_map = {
        'olympus': XRFVendor.OLYMPUS_VANTA,
        'olympus_vanta': XRFVendor.OLYMPUS_VANTA,
        'olympus_delta': XRFVendor.OLYMPUS_DELTA,
        'bruker': XRFVendor.BRUKER_S1_TITAN,
        'bruker_s1': XRFVendor.BRUKER_S1_TITAN,
        'bruker_tracer': XRFVendor.BRUKER_TRACER,
        'thermo': XRFVendor.THERMO_NITON,
        'niton': XRFVendor.THERMO_NITON,
        'sciaps': XRFVendor.SCIAPS_X,
        'generic': XRFVendor.GENERIC
    }
    
    xrf_vendor = vendor_map.get(vendor.lower(), XRFVendor.GENERIC)
    
    return XRFIngestionPipeline(
        vendor=xrf_vendor,
        apply_qc=apply_qc
    )
