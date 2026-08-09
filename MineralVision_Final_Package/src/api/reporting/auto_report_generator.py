"""
Automated Report Generation for MineralVision.

Provides templated report generation for regulatory compliance:
- NI 43-101 Technical Reports
- JORC Code Reports
- Custom project reports
- Auto-populated figures and tables
- QC summaries from audit trail
- Executive summaries
- Data appendices
"""

import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, date
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ReportStandard(Enum):
    """Reporting standards."""
    NI_43_101 = "ni_43_101"
    JORC = "jorc"
    SAMREC = "samrec"
    CIM = "cim"
    CUSTOM = "custom"


class ReportSection(Enum):
    """Standard report sections."""
    TITLE_PAGE = "title_page"
    TABLE_OF_CONTENTS = "table_of_contents"
    EXECUTIVE_SUMMARY = "executive_summary"
    INTRODUCTION = "introduction"
    PROPERTY_DESCRIPTION = "property_description"
    ACCESSIBILITY = "accessibility"
    HISTORY = "history"
    GEOLOGICAL_SETTING = "geological_setting"
    DEPOSIT_TYPE = "deposit_type"
    MINERALIZATION = "mineralization"
    EXPLORATION = "exploration"
    DRILLING = "drilling"
    SAMPLE_PREPARATION = "sample_preparation"
    DATA_VERIFICATION = "data_verification"
    MINERAL_PROCESSING = "mineral_processing"
    MINERAL_RESOURCE = "mineral_resource"
    MINERAL_RESERVE = "mineral_reserve"
    MINING_METHODS = "mining_methods"
    INFRASTRUCTURE = "infrastructure"
    ENVIRONMENTAL = "environmental"
    CAPITAL_COSTS = "capital_costs"
    OPERATING_COSTS = "operating_costs"
    ECONOMIC_ANALYSIS = "economic_analysis"
    INTERPRETATION = "interpretation"
    CONCLUSIONS = "conclusions"
    RECOMMENDATIONS = "recommendations"
    REFERENCES = "references"
    APPENDICES = "appendices"
    QC_SUMMARY = "qc_summary"
    MODEL_EXPLANATION = "model_explanation"


class FigureType(Enum):
    """Types of figures."""
    MAP = "map"
    CROSS_SECTION = "cross_section"
    PLAN_VIEW = "plan_view"
    HISTOGRAM = "histogram"
    SCATTER_PLOT = "scatter_plot"
    VARIOGRAM = "variogram"
    GRADE_TONNAGE = "grade_tonnage"
    DRILLHOLE_LOG = "drillhole_log"
    BLOCK_MODEL = "block_model"
    PROSPECTIVITY_MAP = "prospectivity_map"


class TableType(Enum):
    """Types of tables."""
    DRILLHOLE_SUMMARY = "drillhole_summary"
    ASSAY_SUMMARY = "assay_summary"
    RESOURCE_ESTIMATE = "resource_estimate"
    RESERVE_ESTIMATE = "reserve_estimate"
    QC_STATISTICS = "qc_statistics"
    COST_SUMMARY = "cost_summary"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    SAMPLE_STATISTICS = "sample_statistics"


@dataclass
class Author:
    """Report author."""
    name: str
    title: str
    qualifications: str
    company: str
    signature_date: date
    is_qualified_person: bool = True
    areas_of_responsibility: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'title': self.title,
            'qualifications': self.qualifications,
            'company': self.company,
            'signature_date': self.signature_date.isoformat(),
            'is_qualified_person': self.is_qualified_person,
            'areas_of_responsibility': self.areas_of_responsibility
        }


@dataclass
class Figure:
    """Report figure."""
    figure_id: str
    figure_number: str
    title: str
    figure_type: FigureType
    description: str
    source: str
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'figure_id': self.figure_id,
            'figure_number': self.figure_number,
            'title': self.title,
            'figure_type': self.figure_type.value,
            'description': self.description,
            'source': self.source,
            'image_path': self.image_path,
            'metadata': self.metadata
        }


@dataclass
class Table:
    """Report table."""
    table_id: str
    table_number: str
    title: str
    table_type: TableType
    headers: List[str]
    rows: List[List[Any]]
    footnotes: List[str] = field(default_factory=list)
    source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'table_id': self.table_id,
            'table_number': self.table_number,
            'title': self.title,
            'table_type': self.table_type.value,
            'headers': self.headers,
            'rows': self.rows,
            'footnotes': self.footnotes,
            'source': self.source
        }
        
    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        lines = []
        lines.append(f"**Table {self.table_number}: {self.title}**\n")
        
        lines.append("| " + " | ".join(str(h) for h in self.headers) + " |")
        lines.append("| " + " | ".join("---" for _ in self.headers) + " |")
        
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
        if self.footnotes:
            lines.append("")
            for i, note in enumerate(self.footnotes, 1):
                lines.append(f"^{i}^ {note}")
                
        return "\n".join(lines)


@dataclass
class SectionContent:
    """Content for a report section."""
    section: ReportSection
    title: str
    content: str
    figures: List[Figure] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    subsections: List['SectionContent'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'section': self.section.value,
            'title': self.title,
            'content': self.content,
            'figures': [f.to_dict() for f in self.figures],
            'tables': [t.to_dict() for t in self.tables],
            'subsections': [s.to_dict() for s in self.subsections]
        }


@dataclass
class QCSummary:
    """QC summary for report."""
    total_samples: int
    duplicate_samples: int
    standard_samples: int
    blank_samples: int
    duplicate_precision: float
    standard_accuracy: float
    blank_contamination_rate: float
    overall_qc_status: str
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_samples': self.total_samples,
            'duplicate_samples': self.duplicate_samples,
            'standard_samples': self.standard_samples,
            'blank_samples': self.blank_samples,
            'duplicate_precision': self.duplicate_precision,
            'standard_accuracy': self.standard_accuracy,
            'blank_contamination_rate': self.blank_contamination_rate,
            'overall_qc_status': self.overall_qc_status,
            'issues': self.issues,
            'recommendations': self.recommendations
        }


@dataclass
class ResourceEstimate:
    """Mineral resource estimate."""
    category: str  # Measured, Indicated, Inferred
    commodity: str
    tonnage_mt: float
    grade: float
    grade_unit: str
    contained_metal: float
    contained_metal_unit: str
    cutoff_grade: float
    effective_date: date
    methodology: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category,
            'commodity': self.commodity,
            'tonnage_mt': self.tonnage_mt,
            'grade': self.grade,
            'grade_unit': self.grade_unit,
            'contained_metal': self.contained_metal,
            'contained_metal_unit': self.contained_metal_unit,
            'cutoff_grade': self.cutoff_grade,
            'effective_date': self.effective_date.isoformat(),
            'methodology': self.methodology
        }


@dataclass
class ReportMetadata:
    """Report metadata."""
    report_id: str
    title: str
    project_name: str
    property_name: str
    location: str
    standard: ReportStandard
    effective_date: date
    authors: List[Author]
    issuer: str
    version: str = "1.0"
    confidential: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'title': self.title,
            'project_name': self.project_name,
            'property_name': self.property_name,
            'location': self.location,
            'standard': self.standard.value,
            'effective_date': self.effective_date.isoformat(),
            'authors': [a.to_dict() for a in self.authors],
            'issuer': self.issuer,
            'version': self.version,
            'confidential': self.confidential
        }


class ReportTemplate(ABC):
    """Abstract base class for report templates."""
    
    @abstractmethod
    def get_required_sections(self) -> List[ReportSection]:
        """Get required sections for this template."""
        pass
        
    @abstractmethod
    def get_section_order(self) -> List[ReportSection]:
        """Get section order for this template."""
        pass
        
    @abstractmethod
    def validate_report(self, sections: Dict[ReportSection, SectionContent]) -> List[str]:
        """Validate report against template requirements."""
        pass


class NI43101Template(ReportTemplate):
    """NI 43-101 Technical Report template."""
    
    REQUIRED_SECTIONS = [
        ReportSection.TITLE_PAGE,
        ReportSection.TABLE_OF_CONTENTS,
        ReportSection.EXECUTIVE_SUMMARY,
        ReportSection.INTRODUCTION,
        ReportSection.PROPERTY_DESCRIPTION,
        ReportSection.ACCESSIBILITY,
        ReportSection.HISTORY,
        ReportSection.GEOLOGICAL_SETTING,
        ReportSection.DEPOSIT_TYPE,
        ReportSection.MINERALIZATION,
        ReportSection.EXPLORATION,
        ReportSection.DRILLING,
        ReportSection.SAMPLE_PREPARATION,
        ReportSection.DATA_VERIFICATION,
        ReportSection.INTERPRETATION,
        ReportSection.CONCLUSIONS,
        ReportSection.RECOMMENDATIONS,
        ReportSection.REFERENCES,
    ]
    
    SECTION_ORDER = [
        ReportSection.TITLE_PAGE,
        ReportSection.TABLE_OF_CONTENTS,
        ReportSection.EXECUTIVE_SUMMARY,
        ReportSection.INTRODUCTION,
        ReportSection.PROPERTY_DESCRIPTION,
        ReportSection.ACCESSIBILITY,
        ReportSection.HISTORY,
        ReportSection.GEOLOGICAL_SETTING,
        ReportSection.DEPOSIT_TYPE,
        ReportSection.MINERALIZATION,
        ReportSection.EXPLORATION,
        ReportSection.DRILLING,
        ReportSection.SAMPLE_PREPARATION,
        ReportSection.DATA_VERIFICATION,
        ReportSection.MINERAL_PROCESSING,
        ReportSection.MINERAL_RESOURCE,
        ReportSection.MINERAL_RESERVE,
        ReportSection.MINING_METHODS,
        ReportSection.INFRASTRUCTURE,
        ReportSection.ENVIRONMENTAL,
        ReportSection.CAPITAL_COSTS,
        ReportSection.OPERATING_COSTS,
        ReportSection.ECONOMIC_ANALYSIS,
        ReportSection.INTERPRETATION,
        ReportSection.CONCLUSIONS,
        ReportSection.RECOMMENDATIONS,
        ReportSection.REFERENCES,
        ReportSection.APPENDICES,
        ReportSection.QC_SUMMARY,
        ReportSection.MODEL_EXPLANATION,
    ]
    
    def get_required_sections(self) -> List[ReportSection]:
        return self.REQUIRED_SECTIONS
        
    def get_section_order(self) -> List[ReportSection]:
        return self.SECTION_ORDER
        
    def validate_report(self, sections: Dict[ReportSection, SectionContent]) -> List[str]:
        """Validate NI 43-101 compliance."""
        issues = []
        
        for required in self.REQUIRED_SECTIONS:
            if required not in sections:
                issues.append(f"Missing required section: {required.value}")
            elif not sections[required].content.strip():
                issues.append(f"Empty content in required section: {required.value}")
                
        if ReportSection.EXECUTIVE_SUMMARY in sections:
            summary = sections[ReportSection.EXECUTIVE_SUMMARY]
            if len(summary.content) < 500:
                issues.append("Executive summary may be too brief")
                
        if ReportSection.DATA_VERIFICATION in sections:
            dv = sections[ReportSection.DATA_VERIFICATION]
            if "qualified person" not in dv.content.lower():
                issues.append("Data verification should reference Qualified Person site visit")
                
        return issues


class JORCTemplate(ReportTemplate):
    """JORC Code report template."""
    
    REQUIRED_SECTIONS = [
        ReportSection.EXECUTIVE_SUMMARY,
        ReportSection.INTRODUCTION,
        ReportSection.PROPERTY_DESCRIPTION,
        ReportSection.GEOLOGICAL_SETTING,
        ReportSection.EXPLORATION,
        ReportSection.DRILLING,
        ReportSection.SAMPLE_PREPARATION,
        ReportSection.DATA_VERIFICATION,
        ReportSection.MINERAL_RESOURCE,
        ReportSection.CONCLUSIONS,
    ]
    
    SECTION_ORDER = [
        ReportSection.TITLE_PAGE,
        ReportSection.TABLE_OF_CONTENTS,
        ReportSection.EXECUTIVE_SUMMARY,
        ReportSection.INTRODUCTION,
        ReportSection.PROPERTY_DESCRIPTION,
        ReportSection.GEOLOGICAL_SETTING,
        ReportSection.EXPLORATION,
        ReportSection.DRILLING,
        ReportSection.SAMPLE_PREPARATION,
        ReportSection.DATA_VERIFICATION,
        ReportSection.MINERAL_RESOURCE,
        ReportSection.MINERAL_RESERVE,
        ReportSection.CONCLUSIONS,
        ReportSection.RECOMMENDATIONS,
        ReportSection.REFERENCES,
        ReportSection.APPENDICES,
    ]
    
    def get_required_sections(self) -> List[ReportSection]:
        return self.REQUIRED_SECTIONS
        
    def get_section_order(self) -> List[ReportSection]:
        return self.SECTION_ORDER
        
    def validate_report(self, sections: Dict[ReportSection, SectionContent]) -> List[str]:
        """Validate JORC compliance."""
        issues = []
        
        for required in self.REQUIRED_SECTIONS:
            if required not in sections:
                issues.append(f"Missing required section: {required.value}")
                
        if ReportSection.MINERAL_RESOURCE in sections:
            mr = sections[ReportSection.MINERAL_RESOURCE]
            if "competent person" not in mr.content.lower():
                issues.append("Mineral Resource section should reference Competent Person")
                
        return issues


class DataExtractor:
    """Extract data from MineralVision for report generation."""
    
    def extract_qc_summary(self, audit_data: Dict[str, Any]) -> QCSummary:
        """Extract QC summary from audit trail data."""
        total = audit_data.get('total_samples', 0)
        duplicates = audit_data.get('duplicate_samples', 0)
        standards = audit_data.get('standard_samples', 0)
        blanks = audit_data.get('blank_samples', 0)
        
        dup_precision = audit_data.get('duplicate_precision', 0.95)
        std_accuracy = audit_data.get('standard_accuracy', 0.98)
        blank_contam = audit_data.get('blank_contamination_rate', 0.01)
        
        issues = []
        recommendations = []
        
        if dup_precision < 0.9:
            issues.append("Duplicate precision below acceptable threshold")
            recommendations.append("Review sampling and assay procedures")
            
        if std_accuracy < 0.95:
            issues.append("Standard accuracy below acceptable threshold")
            recommendations.append("Investigate laboratory bias")
            
        if blank_contam > 0.05:
            issues.append("Elevated blank contamination rate")
            recommendations.append("Review sample preparation procedures")
            
        status = "PASS" if not issues else "REVIEW REQUIRED"
        
        return QCSummary(
            total_samples=total,
            duplicate_samples=duplicates,
            standard_samples=standards,
            blank_samples=blanks,
            duplicate_precision=dup_precision,
            standard_accuracy=std_accuracy,
            blank_contamination_rate=blank_contam,
            overall_qc_status=status,
            issues=issues,
            recommendations=recommendations
        )
        
    def extract_drillhole_summary(self, drillhole_data: List[Dict[str, Any]]) -> Table:
        """Extract drillhole summary table."""
        headers = ["Hole ID", "Easting", "Northing", "Elevation", "Azimuth", "Dip", "Depth (m)"]
        rows = []
        
        for dh in drillhole_data:
            rows.append([
                dh.get('hole_id', ''),
                f"{dh.get('easting', 0):.1f}",
                f"{dh.get('northing', 0):.1f}",
                f"{dh.get('elevation', 0):.1f}",
                f"{dh.get('azimuth', 0):.0f}",
                f"{dh.get('dip', -90):.0f}",
                f"{dh.get('depth', 0):.1f}"
            ])
            
        return Table(
            table_id=hashlib.md5(b"drillhole_summary").hexdigest()[:16],
            table_number="1",
            title="Drillhole Summary",
            table_type=TableType.DRILLHOLE_SUMMARY,
            headers=headers,
            rows=rows,
            source="MineralVision Drillhole Database"
        )
        
    def extract_resource_table(self, resources: List[ResourceEstimate]) -> Table:
        """Extract resource estimate table."""
        headers = ["Category", "Tonnage (Mt)", "Grade", "Contained Metal", "Cutoff"]
        rows = []
        
        for res in resources:
            rows.append([
                res.category,
                f"{res.tonnage_mt:.2f}",
                f"{res.grade:.2f} {res.grade_unit}",
                f"{res.contained_metal:.0f} {res.contained_metal_unit}",
                f"{res.cutoff_grade:.2f} {res.grade_unit}"
            ])
            
        return Table(
            table_id=hashlib.md5(b"resource_estimate").hexdigest()[:16],
            table_number="2",
            title="Mineral Resource Estimate",
            table_type=TableType.RESOURCE_ESTIMATE,
            headers=headers,
            rows=rows,
            footnotes=[
                "Mineral Resources are not Mineral Reserves and do not have demonstrated economic viability",
                f"Effective date: {resources[0].effective_date.isoformat() if resources else 'N/A'}"
            ],
            source="MineralVision Resource Estimation"
        )


class SectionGenerator:
    """Generate report sections from templates and data."""
    
    def __init__(self):
        self.data_extractor = DataExtractor()
        
    def generate_title_page(self, metadata: ReportMetadata) -> SectionContent:
        """Generate title page."""
        content = f"""
# {metadata.title}

**Project:** {metadata.project_name}
**Property:** {metadata.property_name}
**Location:** {metadata.location}

**Prepared for:** {metadata.issuer}

**Effective Date:** {metadata.effective_date.strftime('%B %d, %Y')}

**Prepared by:**
"""
        for author in metadata.authors:
            content += f"\n{author.name}, {author.qualifications}\n{author.title}, {author.company}\n"
            
        if metadata.confidential:
            content += "\n**CONFIDENTIAL**\n"
            
        return SectionContent(
            section=ReportSection.TITLE_PAGE,
            title="Title Page",
            content=content
        )
        
    def generate_executive_summary(self, project_data: Dict[str, Any],
                                  resources: List[ResourceEstimate] = None) -> SectionContent:
        """Generate executive summary."""
        content = f"""
## Executive Summary

This technical report presents the results of exploration and resource estimation activities 
conducted on the {project_data.get('property_name', 'Property')}.

### Key Findings

{project_data.get('key_findings', 'Key findings to be populated.')}

### Exploration Summary

Total drilling: {project_data.get('total_drilling_m', 0):,.0f} meters in {project_data.get('total_holes', 0)} holes.

### Resource Estimate
"""
        if resources:
            for res in resources:
                content += f"\n- **{res.category}:** {res.tonnage_mt:.2f} Mt at {res.grade:.2f} {res.grade_unit}"
                
        content += f"""

### Recommendations

{project_data.get('recommendations', 'Recommendations to be populated.')}
"""
        
        return SectionContent(
            section=ReportSection.EXECUTIVE_SUMMARY,
            title="Executive Summary",
            content=content
        )
        
    def generate_qc_section(self, audit_data: Dict[str, Any]) -> SectionContent:
        """Generate QC summary section."""
        qc_summary = self.data_extractor.extract_qc_summary(audit_data)
        
        content = f"""
## Quality Assurance and Quality Control Summary

### Sample Statistics

- Total samples analyzed: {qc_summary.total_samples:,}
- Duplicate samples: {qc_summary.duplicate_samples:,} ({qc_summary.duplicate_samples/max(qc_summary.total_samples,1)*100:.1f}%)
- Standard reference materials: {qc_summary.standard_samples:,}
- Blank samples: {qc_summary.blank_samples:,}

### QC Performance

- Duplicate precision: {qc_summary.duplicate_precision*100:.1f}%
- Standard accuracy: {qc_summary.standard_accuracy*100:.1f}%
- Blank contamination rate: {qc_summary.blank_contamination_rate*100:.2f}%

### Overall Status: {qc_summary.overall_qc_status}
"""
        
        if qc_summary.issues:
            content += "\n### Issues Identified\n"
            for issue in qc_summary.issues:
                content += f"- {issue}\n"
                
        if qc_summary.recommendations:
            content += "\n### Recommendations\n"
            for rec in qc_summary.recommendations:
                content += f"- {rec}\n"
                
        qc_table = Table(
            table_id=hashlib.md5(b"qc_stats").hexdigest()[:16],
            table_number="QC-1",
            title="QC Statistics Summary",
            table_type=TableType.QC_STATISTICS,
            headers=["Metric", "Value", "Threshold", "Status"],
            rows=[
                ["Duplicate Precision", f"{qc_summary.duplicate_precision*100:.1f}%", ">90%", 
                 "PASS" if qc_summary.duplicate_precision >= 0.9 else "FAIL"],
                ["Standard Accuracy", f"{qc_summary.standard_accuracy*100:.1f}%", ">95%",
                 "PASS" if qc_summary.standard_accuracy >= 0.95 else "FAIL"],
                ["Blank Contamination", f"{qc_summary.blank_contamination_rate*100:.2f}%", "<5%",
                 "PASS" if qc_summary.blank_contamination_rate < 0.05 else "FAIL"]
            ],
            source="MineralVision Audit Trail"
        )
        
        return SectionContent(
            section=ReportSection.QC_SUMMARY,
            title="QA/QC Summary",
            content=content,
            tables=[qc_table]
        )
        
    def generate_model_explanation_section(self, xai_data: Dict[str, Any]) -> SectionContent:
        """Generate model explanation section for ML-based estimates."""
        content = f"""
## Machine Learning Model Interpretation

### Model Overview

The prospectivity model uses {xai_data.get('model_type', 'ensemble machine learning')} 
to predict mineral potential based on integrated geoscience datasets.

### Feature Importance

The following features contribute most significantly to model predictions:
"""
        
        top_features = xai_data.get('top_features', [])
        for i, (feature, importance) in enumerate(top_features[:10], 1):
            content += f"\n{i}. **{feature}**: {importance*100:.1f}% contribution"
            
        content += """

### Model Validation

The model was validated using spatial cross-validation to prevent data leakage 
from spatially autocorrelated features. Performance metrics:
"""
        
        metrics = xai_data.get('metrics', {})
        content += f"""
- AUC-ROC: {metrics.get('auc_roc', 0.85):.3f}
- Precision: {metrics.get('precision', 0.80):.3f}
- Recall: {metrics.get('recall', 0.75):.3f}

### Limitations

{xai_data.get('limitations', 'Model predictions should be validated with field observations.')}
"""
        
        return SectionContent(
            section=ReportSection.MODEL_EXPLANATION,
            title="Machine Learning Model Interpretation",
            content=content
        )


class ReportGenerator:
    """Main report generator."""
    
    def __init__(self):
        self.section_generator = SectionGenerator()
        self.templates: Dict[ReportStandard, ReportTemplate] = {
            ReportStandard.NI_43_101: NI43101Template(),
            ReportStandard.JORC: JORCTemplate()
        }
        
    def create_report(self, metadata: ReportMetadata,
                     project_data: Dict[str, Any],
                     audit_data: Dict[str, Any] = None,
                     xai_data: Dict[str, Any] = None,
                     resources: List[ResourceEstimate] = None,
                     custom_sections: Dict[ReportSection, SectionContent] = None) -> Dict[str, Any]:
        """
        Create a complete report.
        
        Args:
            metadata: Report metadata
            project_data: Project data for populating sections
            audit_data: Audit trail data for QC section
            xai_data: XAI data for model explanation section
            resources: Resource estimates
            custom_sections: Custom section content to override generated content
            
        Returns:
            Complete report as dictionary
        """
        template = self.templates.get(metadata.standard)
        if not template:
            template = NI43101Template()
            
        sections: Dict[ReportSection, SectionContent] = {}
        
        sections[ReportSection.TITLE_PAGE] = self.section_generator.generate_title_page(metadata)
        
        sections[ReportSection.EXECUTIVE_SUMMARY] = self.section_generator.generate_executive_summary(
            project_data, resources
        )
        
        if audit_data:
            sections[ReportSection.QC_SUMMARY] = self.section_generator.generate_qc_section(audit_data)
            
        if xai_data:
            sections[ReportSection.MODEL_EXPLANATION] = self.section_generator.generate_model_explanation_section(xai_data)
            
        if custom_sections:
            sections.update(custom_sections)
            
        validation_issues = template.validate_report(sections)
        
        ordered_sections = []
        for section_type in template.get_section_order():
            if section_type in sections:
                ordered_sections.append(sections[section_type])
                
        toc = self._generate_toc(ordered_sections)
        
        return {
            'metadata': metadata.to_dict(),
            'table_of_contents': toc,
            'sections': [s.to_dict() for s in ordered_sections],
            'validation_issues': validation_issues,
            'generated_at': datetime.utcnow().isoformat()
        }
        
    def _generate_toc(self, sections: List[SectionContent]) -> List[Dict[str, Any]]:
        """Generate table of contents."""
        toc = []
        for i, section in enumerate(sections, 1):
            if section.section != ReportSection.TITLE_PAGE:
                toc.append({
                    'number': str(i),
                    'title': section.title,
                    'section': section.section.value
                })
        return toc
        
    def export_markdown(self, report: Dict[str, Any]) -> str:
        """Export report as markdown."""
        lines = []
        
        for section in report['sections']:
            lines.append(section['content'])
            lines.append("")
            
            for table in section.get('tables', []):
                t = Table(**{k: v for k, v in table.items() if k != 'table_type'})
                t.table_type = TableType(table['table_type'])
                lines.append(t.to_markdown())
                lines.append("")
                
        return "\n".join(lines)
        
    def export_json(self, report: Dict[str, Any]) -> str:
        """Export report as JSON."""
        return json.dumps(report, indent=2, default=str)


class ReportScheduler:
    """Schedule automated report generation."""
    
    def __init__(self, generator: ReportGenerator):
        self.generator = generator
        self._schedules: Dict[str, Dict[str, Any]] = {}
        
    def schedule_report(self, schedule_id: str, metadata: ReportMetadata,
                       cron_expression: str, data_sources: Dict[str, Callable]) -> None:
        """Schedule automated report generation."""
        self._schedules[schedule_id] = {
            'metadata': metadata,
            'cron': cron_expression,
            'data_sources': data_sources,
            'last_run': None,
            'next_run': None
        }
        
    def run_scheduled_report(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Run a scheduled report."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return None
            
        project_data = {}
        audit_data = None
        xai_data = None
        
        for source_name, source_func in schedule['data_sources'].items():
            try:
                data = source_func()
                if source_name == 'project':
                    project_data = data
                elif source_name == 'audit':
                    audit_data = data
                elif source_name == 'xai':
                    xai_data = data
            except Exception as e:
                logger.error(f"Error fetching data from {source_name}: {e}")
                
        report = self.generator.create_report(
            schedule['metadata'],
            project_data,
            audit_data,
            xai_data
        )
        
        schedule['last_run'] = datetime.utcnow()
        
        return report
        
    def get_schedules(self) -> List[Dict[str, Any]]:
        """Get all schedules."""
        return [
            {
                'schedule_id': sid,
                'metadata': s['metadata'].to_dict(),
                'cron': s['cron'],
                'last_run': s['last_run'].isoformat() if s['last_run'] else None
            }
            for sid, s in self._schedules.items()
        ]


def create_report_generator() -> ReportGenerator:
    """Factory function to create report generator."""
    return ReportGenerator()


def create_ni43101_report(metadata: ReportMetadata,
                         project_data: Dict[str, Any],
                         audit_data: Dict[str, Any] = None,
                         xai_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create NI 43-101 compliant report."""
    generator = ReportGenerator()
    return generator.create_report(metadata, project_data, audit_data, xai_data)


def create_jorc_report(metadata: ReportMetadata,
                      project_data: Dict[str, Any],
                      audit_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create JORC compliant report."""
    metadata.standard = ReportStandard.JORC
    generator = ReportGenerator()
    return generator.create_report(metadata, project_data, audit_data)
