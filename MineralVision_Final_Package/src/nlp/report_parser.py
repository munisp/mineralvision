"""
Geological Report Parser for MineralVision.

Parses structured and semi-structured geological reports,
drill logs, and technical documents into machine-readable format.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Types of geological reports."""
    DRILL_LOG = "drill_log"
    TECHNICAL_REPORT = "technical_report"
    NI_43_101 = "ni_43_101"
    JORC = "jorc"
    ASSAY_CERTIFICATE = "assay_certificate"
    GEOLOGICAL_MAP = "geological_map"
    PROSPECT_SUMMARY = "prospect_summary"
    ANNUAL_REPORT = "annual_report"


@dataclass
class ParsedInterval:
    """A parsed drill interval."""
    from_m: float
    to_m: float
    lithology: Optional[str] = None
    alteration: Optional[str] = None
    mineralization: Optional[str] = None
    structure: Optional[str] = None
    description: Optional[str] = None
    assays: Dict[str, float] = field(default_factory=dict)
    recovery: Optional[float] = None
    rqd: Optional[float] = None
    
    @property
    def length(self) -> float:
        return self.to_m - self.from_m
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'from_m': self.from_m,
            'to_m': self.to_m,
            'length_m': self.length,
            'lithology': self.lithology,
            'alteration': self.alteration,
            'mineralization': self.mineralization,
            'structure': self.structure,
            'description': self.description,
            'assays': self.assays,
            'recovery': self.recovery,
            'rqd': self.rqd
        }


@dataclass
class DrillHoleHeader:
    """Drill hole header information."""
    hole_id: str
    project: Optional[str] = None
    prospect: Optional[str] = None
    easting: Optional[float] = None
    northing: Optional[float] = None
    elevation: Optional[float] = None
    azimuth: Optional[float] = None
    dip: Optional[float] = None
    total_depth: Optional[float] = None
    drill_type: Optional[str] = None
    date_started: Optional[str] = None
    date_completed: Optional[str] = None
    geologist: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hole_id': self.hole_id,
            'project': self.project,
            'prospect': self.prospect,
            'easting': self.easting,
            'northing': self.northing,
            'elevation': self.elevation,
            'azimuth': self.azimuth,
            'dip': self.dip,
            'total_depth': self.total_depth,
            'drill_type': self.drill_type,
            'date_started': self.date_started,
            'date_completed': self.date_completed,
            'geologist': self.geologist
        }


@dataclass
class ParsedReport:
    """A fully parsed geological report."""
    report_type: ReportType
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    project: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    commodities: List[str] = field(default_factory=list)
    sections: Dict[str, str] = field(default_factory=dict)
    drill_holes: List[DrillHoleHeader] = field(default_factory=list)
    intervals: List[ParsedInterval] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_type': self.report_type.value,
            'title': self.title,
            'author': self.author,
            'date': self.date,
            'project': self.project,
            'company': self.company,
            'location': self.location,
            'commodities': self.commodities,
            'sections': self.sections,
            'drill_holes': [h.to_dict() for h in self.drill_holes],
            'intervals': [i.to_dict() for i in self.intervals],
            'resources': self.resources,
            'metadata': self.metadata
        }


class DrillLogParser:
    """
    Parser for drill log documents.
    
    Handles various drill log formats including:
    - CSV/Excel exports
    - PDF text extracts
    - Structured text logs
    """
    
    def __init__(self):
        self._build_patterns()
    
    def _build_patterns(self) -> None:
        """Build regex patterns for drill log parsing."""
        # Hole ID patterns
        self.hole_id_patterns = [
            re.compile(r'(?:hole|drill\s*hole|dh|ddh|rc|rac)\s*[:#]?\s*([A-Z]{1,4}[-_]?\d{2,6}[A-Z]?)', re.IGNORECASE),
            re.compile(r'([A-Z]{2,4}\d{3,6})', re.IGNORECASE),
        ]
        
        # Coordinate patterns
        self.coord_patterns = {
            'easting': re.compile(r'(?:easting|east|e)\s*[:#]?\s*(\d{5,7}\.?\d*)', re.IGNORECASE),
            'northing': re.compile(r'(?:northing|north|n)\s*[:#]?\s*(\d{5,8}\.?\d*)', re.IGNORECASE),
            'elevation': re.compile(r'(?:elevation|elev|rl|z)\s*[:#]?\s*(-?\d{1,4}\.?\d*)', re.IGNORECASE),
        }
        
        # Orientation patterns
        self.orientation_patterns = {
            'azimuth': re.compile(r'(?:azimuth|azi|az|bearing)\s*[:#]?\s*(\d{1,3}\.?\d*)', re.IGNORECASE),
            'dip': re.compile(r'(?:dip|inclination|incl)\s*[:#]?\s*(-?\d{1,2}\.?\d*)', re.IGNORECASE),
        }
        
        # Depth patterns
        self.depth_pattern = re.compile(
            r'(?:total\s*depth|td|eoh|end\s*of\s*hole)\s*[:#]?\s*(\d+\.?\d*)\s*(m|ft)?',
            re.IGNORECASE
        )
        
        # Interval pattern (from-to)
        self.interval_pattern = re.compile(
            r'(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)\s*(?:m|meters?)?',
            re.IGNORECASE
        )
        
        # Assay patterns
        self.assay_patterns = {
            'Au': re.compile(r'(?:Au|gold)\s*[:#=]?\s*(\d+\.?\d*)\s*(?:g/t|ppm|ppb)?', re.IGNORECASE),
            'Ag': re.compile(r'(?:Ag|silver)\s*[:#=]?\s*(\d+\.?\d*)\s*(?:g/t|ppm)?', re.IGNORECASE),
            'Cu': re.compile(r'(?:Cu|copper)\s*[:#=]?\s*(\d+\.?\d*)\s*%?', re.IGNORECASE),
            'Pb': re.compile(r'(?:Pb|lead)\s*[:#=]?\s*(\d+\.?\d*)\s*%?', re.IGNORECASE),
            'Zn': re.compile(r'(?:Zn|zinc)\s*[:#=]?\s*(\d+\.?\d*)\s*%?', re.IGNORECASE),
            'Li': re.compile(r'(?:Li|Li2O|lithium)\s*[:#=]?\s*(\d+\.?\d*)\s*(?:%|ppm)?', re.IGNORECASE),
        }
    
    def parse(self, text: str) -> Tuple[Optional[DrillHoleHeader], List[ParsedInterval]]:
        """Parse drill log text."""
        header = self._parse_header(text)
        intervals = self._parse_intervals(text)
        
        return header, intervals
    
    def _parse_header(self, text: str) -> Optional[DrillHoleHeader]:
        """Parse drill hole header information."""
        # Extract hole ID
        hole_id = None
        for pattern in self.hole_id_patterns:
            match = pattern.search(text)
            if match:
                hole_id = match.group(1)
                break
        
        if not hole_id:
            return None
        
        header = DrillHoleHeader(hole_id=hole_id)
        
        # Extract coordinates
        for coord_name, pattern in self.coord_patterns.items():
            match = pattern.search(text)
            if match:
                setattr(header, coord_name, float(match.group(1)))
        
        # Extract orientation
        for orient_name, pattern in self.orientation_patterns.items():
            match = pattern.search(text)
            if match:
                setattr(header, orient_name, float(match.group(1)))
        
        # Extract total depth
        match = self.depth_pattern.search(text)
        if match:
            header.total_depth = float(match.group(1))
        
        return header
    
    def _parse_intervals(self, text: str) -> List[ParsedInterval]:
        """Parse drill intervals from text."""
        intervals = []
        lines = text.split('\n')
        
        current_interval = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for interval definition
            interval_match = self.interval_pattern.search(line)
            if interval_match:
                # Save previous interval
                if current_interval:
                    intervals.append(current_interval)
                
                from_m = float(interval_match.group(1))
                to_m = float(interval_match.group(2))
                
                current_interval = ParsedInterval(
                    from_m=from_m,
                    to_m=to_m,
                    description=line
                )
                
                # Extract assays from same line
                for element, pattern in self.assay_patterns.items():
                    match = pattern.search(line)
                    if match:
                        current_interval.assays[element] = float(match.group(1))
                
                # Extract lithology keywords
                current_interval.lithology = self._extract_lithology(line)
                current_interval.alteration = self._extract_alteration(line)
                current_interval.mineralization = self._extract_mineralization(line)
        
        # Don't forget last interval
        if current_interval:
            intervals.append(current_interval)
        
        return intervals
    
    def _extract_lithology(self, text: str) -> Optional[str]:
        """Extract lithology from text."""
        lithologies = [
            'granite', 'granodiorite', 'diorite', 'gabbro', 'basalt',
            'andesite', 'rhyolite', 'dacite', 'pegmatite', 'schist',
            'gneiss', 'quartzite', 'sandstone', 'siltstone', 'mudstone',
            'shale', 'limestone', 'dolomite', 'breccia', 'conglomerate',
            'laterite', 'saprolite', 'regolith'
        ]
        
        text_lower = text.lower()
        for lith in lithologies:
            if lith in text_lower:
                return lith
        
        return None
    
    def _extract_alteration(self, text: str) -> Optional[str]:
        """Extract alteration from text."""
        alterations = [
            'silicified', 'sericitized', 'chloritized', 'carbonated',
            'propylitic', 'phyllic', 'argillic', 'potassic',
            'oxidized', 'weathered', 'leached'
        ]
        
        text_lower = text.lower()
        found = []
        for alt in alterations:
            if alt in text_lower:
                found.append(alt)
        
        return ', '.join(found) if found else None
    
    def _extract_mineralization(self, text: str) -> Optional[str]:
        """Extract mineralization from text."""
        minerals = [
            'pyrite', 'chalcopyrite', 'galena', 'sphalerite',
            'arsenopyrite', 'magnetite', 'hematite', 'gold',
            'quartz vein', 'sulfide', 'oxide'
        ]
        
        text_lower = text.lower()
        found = []
        for mineral in minerals:
            if mineral in text_lower:
                found.append(mineral)
        
        return ', '.join(found) if found else None


class GeologicalReportParser:
    """
    Parser for geological technical reports.
    
    Handles NI 43-101, JORC, and other technical report formats.
    """
    
    def __init__(self):
        self.drill_parser = DrillLogParser()
        self._build_patterns()
    
    def _build_patterns(self) -> None:
        """Build patterns for report parsing."""
        # Section headers
        self.section_patterns = {
            'summary': re.compile(r'^(?:\d+\.?\s*)?(?:executive\s+)?summary', re.IGNORECASE | re.MULTILINE),
            'introduction': re.compile(r'^(?:\d+\.?\s*)?introduction', re.IGNORECASE | re.MULTILINE),
            'location': re.compile(r'^(?:\d+\.?\s*)?(?:property\s+)?location', re.IGNORECASE | re.MULTILINE),
            'geology': re.compile(r'^(?:\d+\.?\s*)?(?:regional\s+|local\s+)?geology', re.IGNORECASE | re.MULTILINE),
            'mineralization': re.compile(r'^(?:\d+\.?\s*)?mineralization', re.IGNORECASE | re.MULTILINE),
            'exploration': re.compile(r'^(?:\d+\.?\s*)?exploration', re.IGNORECASE | re.MULTILINE),
            'drilling': re.compile(r'^(?:\d+\.?\s*)?drilling', re.IGNORECASE | re.MULTILINE),
            'sampling': re.compile(r'^(?:\d+\.?\s*)?sampling', re.IGNORECASE | re.MULTILINE),
            'resources': re.compile(r'^(?:\d+\.?\s*)?(?:mineral\s+)?resource', re.IGNORECASE | re.MULTILINE),
            'reserves': re.compile(r'^(?:\d+\.?\s*)?(?:mineral\s+)?reserve', re.IGNORECASE | re.MULTILINE),
            'conclusions': re.compile(r'^(?:\d+\.?\s*)?conclusions?', re.IGNORECASE | re.MULTILINE),
            'recommendations': re.compile(r'^(?:\d+\.?\s*)?recommendations?', re.IGNORECASE | re.MULTILINE),
        }
        
        # Resource patterns
        self.resource_patterns = {
            'measured': re.compile(r'measured\s*[:\s]+(\d+\.?\d*)\s*(?:Mt|million\s*t)', re.IGNORECASE),
            'indicated': re.compile(r'indicated\s*[:\s]+(\d+\.?\d*)\s*(?:Mt|million\s*t)', re.IGNORECASE),
            'inferred': re.compile(r'inferred\s*[:\s]+(\d+\.?\d*)\s*(?:Mt|million\s*t)', re.IGNORECASE),
        }
        
        # Commodity patterns
        self.commodity_pattern = re.compile(
            r'\b(gold|silver|copper|lead|zinc|nickel|cobalt|lithium|uranium|'
            r'platinum|palladium|iron|manganese|rare\s*earth|ree)\b',
            re.IGNORECASE
        )
    
    def parse(self, text: str, report_type: Optional[ReportType] = None) -> ParsedReport:
        """Parse a geological report."""
        # Detect report type if not provided
        if report_type is None:
            report_type = self._detect_report_type(text)
        
        report = ParsedReport(
            report_type=report_type,
            raw_text=text
        )
        
        # Extract metadata
        report.title = self._extract_title(text)
        report.author = self._extract_author(text)
        report.date = self._extract_date(text)
        report.project = self._extract_project(text)
        report.company = self._extract_company(text)
        
        # Extract commodities
        report.commodities = self._extract_commodities(text)
        
        # Extract sections
        report.sections = self._extract_sections(text)
        
        # Extract resources
        report.resources = self._extract_resources(text)
        
        # Parse drill data if present
        if 'drilling' in report.sections or 'drill' in text.lower():
            header, intervals = self.drill_parser.parse(text)
            if header:
                report.drill_holes.append(header)
            report.intervals.extend(intervals)
        
        return report
    
    def _detect_report_type(self, text: str) -> ReportType:
        """Detect report type from content."""
        text_lower = text.lower()
        
        if 'ni 43-101' in text_lower or 'national instrument 43-101' in text_lower:
            return ReportType.NI_43_101
        elif 'jorc' in text_lower:
            return ReportType.JORC
        elif 'drill log' in text_lower or 'drill hole' in text_lower:
            return ReportType.DRILL_LOG
        elif 'assay' in text_lower and 'certificate' in text_lower:
            return ReportType.ASSAY_CERTIFICATE
        else:
            return ReportType.TECHNICAL_REPORT
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract report title."""
        lines = text.split('\n')
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if len(line) > 20 and len(line) < 200:
                # Likely a title
                if any(word in line.lower() for word in ['report', 'project', 'property', 'assessment']):
                    return line
        return None
    
    def _extract_author(self, text: str) -> Optional[str]:
        """Extract report author."""
        patterns = [
            re.compile(r'(?:prepared\s+by|author)[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)', re.IGNORECASE),
            re.compile(r'(?:P\.?\s*Geo\.?|P\.?\s*Eng\.?)[,\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text[:5000])  # Check first part
            if match:
                return match.group(1)
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract report date."""
        patterns = [
            re.compile(r'(?:dated?|effective\s+date)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})', re.IGNORECASE),
            re.compile(r'(\w+\s+\d{4})', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text[:3000])
            if match:
                return match.group(1)
        
        return None
    
    def _extract_project(self, text: str) -> Optional[str]:
        """Extract project name."""
        patterns = [
            re.compile(r'(?:project|property)[:\s]+([A-Z][A-Za-z\s]+?)(?:\s+project|\s+property|,|\n)', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text[:5000])
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_company(self, text: str) -> Optional[str]:
        """Extract company name."""
        patterns = [
            re.compile(r'(?:prepared\s+for|client)[:\s]+([A-Z][A-Za-z\s]+?(?:Inc\.?|Ltd\.?|Corp\.?|LLC))', re.IGNORECASE),
            re.compile(r'([A-Z][A-Za-z\s]+?(?:Mining|Resources|Minerals|Exploration)\s+(?:Inc\.?|Ltd\.?|Corp\.?))', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text[:5000])
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_commodities(self, text: str) -> List[str]:
        """Extract target commodities."""
        matches = self.commodity_pattern.findall(text)
        return list(set(m.lower() for m in matches))
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract report sections."""
        sections = {}
        
        # Find all section positions
        section_positions = []
        for section_name, pattern in self.section_patterns.items():
            for match in pattern.finditer(text):
                section_positions.append((match.start(), section_name, match.group()))
        
        # Sort by position
        section_positions.sort(key=lambda x: x[0])
        
        # Extract section content
        for i, (pos, name, header) in enumerate(section_positions):
            start = pos + len(header)
            if i + 1 < len(section_positions):
                end = section_positions[i + 1][0]
            else:
                end = min(pos + 10000, len(text))  # Max 10k chars per section
            
            content = text[start:end].strip()
            if len(content) > 50:  # Only keep substantial sections
                sections[name] = content[:5000]  # Limit section size
        
        return sections
    
    def _extract_resources(self, text: str) -> Dict[str, Any]:
        """Extract resource estimates."""
        resources = {}
        
        for category, pattern in self.resource_patterns.items():
            match = pattern.search(text)
            if match:
                resources[category] = {
                    'tonnes_mt': float(match.group(1))
                }
        
        return resources


def parse_geological_report(
    text: str,
    report_type: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to parse a geological report."""
    parser = GeologicalReportParser()
    
    rtype = None
    if report_type:
        try:
            rtype = ReportType(report_type)
        except ValueError:
            pass
    
    report = parser.parse(text, rtype)
    return report.to_dict()
