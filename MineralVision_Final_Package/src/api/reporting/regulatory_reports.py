"""
Regulatory Reporting Module for MineralVision Platform.

Comprehensive regulatory reporting including:
1. NI 43-101 (Canadian) report templates
2. JORC (Australian) report templates
3. SAMREC (South African) report templates
4. SEC S-K 1300 (US) report templates
5. Resource/Reserve statement generation
6. QP/CP certification support
7. Table generation and formatting
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import math


class ReportingStandard(Enum):
    """Regulatory reporting standards."""
    NI_43_101 = "ni_43_101"
    JORC_2012 = "jorc_2012"
    SAMREC = "samrec"
    SEC_SK_1300 = "sec_sk_1300"
    PERC = "perc"
    CIM = "cim"


class ResourceCategory(Enum):
    """Mineral resource categories."""
    MEASURED = "measured"
    INDICATED = "indicated"
    INFERRED = "inferred"


class ReserveCategory(Enum):
    """Mineral reserve categories."""
    PROVEN = "proven"
    PROBABLE = "probable"


class CommodityType(Enum):
    """Commodity types."""
    GOLD = "gold"
    SILVER = "silver"
    COPPER = "copper"
    ZINC = "zinc"
    LEAD = "lead"
    NICKEL = "nickel"
    IRON_ORE = "iron_ore"
    LITHIUM = "lithium"
    COBALT = "cobalt"
    URANIUM = "uranium"
    PLATINUM = "platinum"
    PALLADIUM = "palladium"
    COAL = "coal"
    DIAMONDS = "diamonds"
    OTHER = "other"


@dataclass
class QualifiedPerson:
    """Qualified Person / Competent Person details."""
    name: str
    title: str
    company: str
    professional_designation: str
    registration_number: str
    professional_body: str
    email: str
    phone: str = ""
    address: str = ""
    years_experience: int = 0
    areas_of_expertise: List[str] = field(default_factory=list)
    independence_statement: str = ""
    site_visit_date: Optional[date] = None
    site_visit_duration: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "company": self.company,
            "professional_designation": self.professional_designation,
            "registration_number": self.registration_number,
            "professional_body": self.professional_body,
            "email": self.email,
            "years_experience": self.years_experience,
            "areas_of_expertise": self.areas_of_expertise,
            "site_visit_date": self.site_visit_date.isoformat() if self.site_visit_date else None
        }


@dataclass
class ProjectInfo:
    """Project information."""
    name: str
    location: str
    country: str
    state_province: str = ""
    coordinates: Tuple[float, float] = (0.0, 0.0)
    coordinate_system: str = "WGS84"
    area_hectares: float = 0.0
    tenure_type: str = ""
    tenure_numbers: List[str] = field(default_factory=list)
    tenure_expiry: Optional[date] = None
    ownership: str = ""
    operator: str = ""
    stage: str = ""
    commodities: List[CommodityType] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "location": self.location,
            "country": self.country,
            "state_province": self.state_province,
            "coordinates": self.coordinates,
            "area_hectares": self.area_hectares,
            "tenure_type": self.tenure_type,
            "tenure_numbers": self.tenure_numbers,
            "ownership": self.ownership,
            "stage": self.stage,
            "commodities": [c.value for c in self.commodities]
        }


@dataclass
class ResourceEstimate:
    """Single resource estimate entry."""
    category: ResourceCategory
    domain: str
    tonnage_mt: float
    grade: float
    grade_unit: str
    contained_metal: float
    metal_unit: str
    cutoff_grade: float
    density: float = 2.7
    confidence_level: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "domain": self.domain,
            "tonnage_mt": self.tonnage_mt,
            "grade": self.grade,
            "grade_unit": self.grade_unit,
            "contained_metal": self.contained_metal,
            "metal_unit": self.metal_unit,
            "cutoff_grade": self.cutoff_grade,
            "density": self.density
        }


@dataclass
class ReserveEstimate:
    """Single reserve estimate entry."""
    category: ReserveCategory
    domain: str
    tonnage_mt: float
    grade: float
    grade_unit: str
    contained_metal: float
    metal_unit: str
    cutoff_grade: float
    dilution_percent: float = 0.0
    mining_recovery_percent: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "domain": self.domain,
            "tonnage_mt": self.tonnage_mt,
            "grade": self.grade,
            "grade_unit": self.grade_unit,
            "contained_metal": self.contained_metal,
            "metal_unit": self.metal_unit,
            "cutoff_grade": self.cutoff_grade,
            "dilution_percent": self.dilution_percent,
            "mining_recovery_percent": self.mining_recovery_percent
        }


@dataclass
class ResourceStatement:
    """Complete resource statement."""
    effective_date: date
    commodity: CommodityType
    estimates: List[ResourceEstimate]
    cutoff_basis: str = ""
    estimation_method: str = ""
    block_model_name: str = ""
    variogram_model: str = ""
    search_parameters: str = ""
    density_method: str = ""
    notes: List[str] = field(default_factory=list)
    
    def get_total_by_category(self, category: ResourceCategory) -> Dict[str, float]:
        """Get totals for a category."""
        estimates = [e for e in self.estimates if e.category == category]
        
        if not estimates:
            return {"tonnage_mt": 0, "grade": 0, "contained_metal": 0}
        
        total_tonnage = sum(e.tonnage_mt for e in estimates)
        total_metal = sum(e.contained_metal for e in estimates)
        avg_grade = sum(e.grade * e.tonnage_mt for e in estimates) / total_tonnage if total_tonnage > 0 else 0
        
        return {
            "tonnage_mt": total_tonnage,
            "grade": avg_grade,
            "contained_metal": total_metal
        }
    
    def get_measured_indicated_total(self) -> Dict[str, float]:
        """Get combined measured + indicated total."""
        measured = self.get_total_by_category(ResourceCategory.MEASURED)
        indicated = self.get_total_by_category(ResourceCategory.INDICATED)
        
        total_tonnage = measured["tonnage_mt"] + indicated["tonnage_mt"]
        total_metal = measured["contained_metal"] + indicated["contained_metal"]
        avg_grade = (measured["grade"] * measured["tonnage_mt"] + 
                    indicated["grade"] * indicated["tonnage_mt"]) / total_tonnage if total_tonnage > 0 else 0
        
        return {
            "tonnage_mt": total_tonnage,
            "grade": avg_grade,
            "contained_metal": total_metal
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "effective_date": self.effective_date.isoformat(),
            "commodity": self.commodity.value,
            "estimates": [e.to_dict() for e in self.estimates],
            "cutoff_basis": self.cutoff_basis,
            "estimation_method": self.estimation_method,
            "notes": self.notes,
            "totals": {
                "measured": self.get_total_by_category(ResourceCategory.MEASURED),
                "indicated": self.get_total_by_category(ResourceCategory.INDICATED),
                "inferred": self.get_total_by_category(ResourceCategory.INFERRED),
                "measured_indicated": self.get_measured_indicated_total()
            }
        }


@dataclass
class ReserveStatement:
    """Complete reserve statement."""
    effective_date: date
    commodity: CommodityType
    estimates: List[ReserveEstimate]
    cutoff_basis: str = ""
    mining_method: str = ""
    processing_method: str = ""
    metal_prices: Dict[str, float] = field(default_factory=dict)
    operating_costs: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    
    def get_total_by_category(self, category: ReserveCategory) -> Dict[str, float]:
        """Get totals for a category."""
        estimates = [e for e in self.estimates if e.category == category]
        
        if not estimates:
            return {"tonnage_mt": 0, "grade": 0, "contained_metal": 0}
        
        total_tonnage = sum(e.tonnage_mt for e in estimates)
        total_metal = sum(e.contained_metal for e in estimates)
        avg_grade = sum(e.grade * e.tonnage_mt for e in estimates) / total_tonnage if total_tonnage > 0 else 0
        
        return {
            "tonnage_mt": total_tonnage,
            "grade": avg_grade,
            "contained_metal": total_metal
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "effective_date": self.effective_date.isoformat(),
            "commodity": self.commodity.value,
            "estimates": [e.to_dict() for e in self.estimates],
            "cutoff_basis": self.cutoff_basis,
            "mining_method": self.mining_method,
            "metal_prices": self.metal_prices,
            "totals": {
                "proven": self.get_total_by_category(ReserveCategory.PROVEN),
                "probable": self.get_total_by_category(ReserveCategory.PROBABLE)
            }
        }


class NI43101ReportGenerator:
    """Generate NI 43-101 compliant reports."""
    
    REQUIRED_SECTIONS = [
        "1. Summary",
        "2. Introduction",
        "3. Reliance on Other Experts",
        "4. Property Description and Location",
        "5. Accessibility, Climate, Local Resources, Infrastructure and Physiography",
        "6. History",
        "7. Geological Setting and Mineralization",
        "8. Deposit Types",
        "9. Exploration",
        "10. Drilling",
        "11. Sample Preparation, Analyses and Security",
        "12. Data Verification",
        "13. Mineral Processing and Metallurgical Testing",
        "14. Mineral Resource Estimates",
        "15. Mineral Reserve Estimates",
        "16. Mining Methods",
        "17. Recovery Methods",
        "18. Project Infrastructure",
        "19. Market Studies and Contracts",
        "20. Environmental Studies, Permitting and Social or Community Impact",
        "21. Capital and Operating Costs",
        "22. Economic Analysis",
        "23. Adjacent Properties",
        "24. Other Relevant Data and Information",
        "25. Interpretation and Conclusions",
        "26. Recommendations"
    ]
    
    def __init__(self, project: ProjectInfo, qp: QualifiedPerson):
        self.project = project
        self.qp = qp
        self.resource_statement: Optional[ResourceStatement] = None
        self.reserve_statement: Optional[ReserveStatement] = None
        self.sections: Dict[str, str] = {}
        self.figures: List[Dict[str, str]] = []
        self.tables: List[Dict[str, Any]] = []
    
    def set_resource_statement(self, statement: ResourceStatement):
        """Set resource statement."""
        self.resource_statement = statement
    
    def set_reserve_statement(self, statement: ReserveStatement):
        """Set reserve statement."""
        self.reserve_statement = statement
    
    def add_section_content(self, section_number: int, content: str):
        """Add content to a section."""
        section_key = f"{section_number}"
        self.sections[section_key] = content
    
    def add_figure(self, number: int, title: str, filepath: str, caption: str = ""):
        """Add a figure reference."""
        self.figures.append({
            "number": number,
            "title": title,
            "filepath": filepath,
            "caption": caption
        })
    
    def add_table(self, number: int, title: str, data: List[List[Any]], 
                 headers: List[str], notes: List[str] = None):
        """Add a table."""
        self.tables.append({
            "number": number,
            "title": title,
            "headers": headers,
            "data": data,
            "notes": notes or []
        })
    
    def generate_title_page(self) -> str:
        """Generate title page content."""
        return f"""
TECHNICAL REPORT

{self.project.name.upper()}

{self.project.location}, {self.project.country}

Prepared for:
{self.project.ownership}

Prepared by:
{self.qp.name}, {self.qp.professional_designation}
{self.qp.company}

Effective Date: {self.resource_statement.effective_date if self.resource_statement else date.today()}
Report Date: {date.today()}

This Technical Report was prepared in accordance with the requirements of 
National Instrument 43-101 - Standards of Disclosure for Mineral Projects
"""
    
    def generate_certificate(self) -> str:
        """Generate QP certificate."""
        return f"""
CERTIFICATE OF QUALIFIED PERSON

I, {self.qp.name}, {self.qp.professional_designation}, do hereby certify that:

1. I am currently employed as {self.qp.title} by {self.qp.company}.

2. I am a member in good standing of {self.qp.professional_body} (Registration No. {self.qp.registration_number}).

3. I have {self.qp.years_experience} years of experience in mineral exploration and resource estimation.

4. I have read the definition of "Qualified Person" set out in National Instrument 43-101 and certify that by reason of my education, affiliation with a professional association, and past relevant work experience, I fulfill the requirements to be a "Qualified Person" for the purposes of NI 43-101.

5. I am responsible for the preparation of all sections of this Technical Report titled "{self.project.name}" dated {date.today()}.

6. I visited the {self.project.name} property on {self.qp.site_visit_date} for a duration of {self.qp.site_visit_duration}.

7. {self.qp.independence_statement}

8. I have read National Instrument 43-101 and Form 43-101F1 and this Technical Report has been prepared in compliance with that instrument and form.

9. As of the date of this certificate, to the best of my knowledge, information and belief, this Technical Report contains all scientific and technical information that is required to be disclosed to make the Technical Report not misleading.

Dated this {date.today().day} day of {date.today().strftime('%B')}, {date.today().year}.

_______________________________
{self.qp.name}, {self.qp.professional_designation}
"""
    
    def generate_resource_table(self) -> str:
        """Generate resource estimate table."""
        if not self.resource_statement:
            return "No resource statement available."
        
        lines = []
        lines.append(f"Table: Mineral Resource Estimate - {self.project.name}")
        lines.append(f"Effective Date: {self.resource_statement.effective_date}")
        lines.append(f"Cutoff Grade: {self.resource_statement.estimates[0].cutoff_grade if self.resource_statement.estimates else 'N/A'} {self.resource_statement.estimates[0].grade_unit if self.resource_statement.estimates else ''}")
        lines.append("")
        lines.append("| Category | Domain | Tonnage (Mt) | Grade | Contained Metal |")
        lines.append("|----------|--------|--------------|-------|-----------------|")
        
        for estimate in self.resource_statement.estimates:
            lines.append(f"| {estimate.category.value.capitalize()} | {estimate.domain} | {estimate.tonnage_mt:.2f} | {estimate.grade:.2f} {estimate.grade_unit} | {estimate.contained_metal:.0f} {estimate.metal_unit} |")
        
        totals = self.resource_statement.get_measured_indicated_total()
        lines.append(f"| **M+I Total** | - | {totals['tonnage_mt']:.2f} | {totals['grade']:.2f} | {totals['contained_metal']:.0f} |")
        
        inferred = self.resource_statement.get_total_by_category(ResourceCategory.INFERRED)
        if inferred['tonnage_mt'] > 0:
            lines.append(f"| **Inferred** | - | {inferred['tonnage_mt']:.2f} | {inferred['grade']:.2f} | {inferred['contained_metal']:.0f} |")
        
        lines.append("")
        lines.append("Notes:")
        for i, note in enumerate(self.resource_statement.notes, 1):
            lines.append(f"{i}. {note}")
        
        return "\n".join(lines)
    
    def generate_section_14(self) -> str:
        """Generate Section 14 - Mineral Resource Estimates."""
        content = []
        content.append("14. MINERAL RESOURCE ESTIMATES")
        content.append("")
        content.append("14.1 Introduction")
        content.append("")
        content.append(f"The mineral resource estimate for the {self.project.name} project was prepared in accordance with the CIM Definition Standards for Mineral Resources and Mineral Reserves (2014) and NI 43-101 guidelines.")
        content.append("")
        
        if self.resource_statement:
            content.append("14.2 Estimation Methodology")
            content.append("")
            content.append(f"Estimation Method: {self.resource_statement.estimation_method}")
            content.append(f"Block Model: {self.resource_statement.block_model_name}")
            content.append(f"Variogram Model: {self.resource_statement.variogram_model}")
            content.append(f"Search Parameters: {self.resource_statement.search_parameters}")
            content.append(f"Density Determination: {self.resource_statement.density_method}")
            content.append("")
            content.append("14.3 Resource Classification")
            content.append("")
            content.append("Resources were classified based on:")
            content.append("- Data density and distribution")
            content.append("- Geological continuity")
            content.append("- Estimation quality metrics (kriging variance, slope of regression)")
            content.append("- QP judgment of confidence in the estimate")
            content.append("")
            content.append("14.4 Mineral Resource Statement")
            content.append("")
            content.append(self.generate_resource_table())
        
        return "\n".join(content)
    
    def generate_full_report(self) -> Dict[str, str]:
        """Generate complete report structure."""
        report = {
            "title_page": self.generate_title_page(),
            "certificate": self.generate_certificate(),
            "table_of_contents": self._generate_toc(),
            "sections": {}
        }
        
        for section in self.REQUIRED_SECTIONS:
            section_num = int(section.split(".")[0])
            
            if section_num == 14 and self.resource_statement:
                report["sections"][section] = self.generate_section_14()
            elif str(section_num) in self.sections:
                report["sections"][section] = self.sections[str(section_num)]
            else:
                report["sections"][section] = f"[Content for {section} to be added]"
        
        return report
    
    def _generate_toc(self) -> str:
        """Generate table of contents."""
        lines = ["TABLE OF CONTENTS", ""]
        for section in self.REQUIRED_SECTIONS:
            lines.append(section)
        return "\n".join(lines)
    
    def export_to_markdown(self, filepath: str):
        """Export report to Markdown format."""
        report = self.generate_full_report()
        
        with open(filepath, 'w') as f:
            f.write("# " + self.project.name + " Technical Report\n\n")
            f.write(report["title_page"])
            f.write("\n\n---\n\n")
            f.write("## Certificate of Qualified Person\n\n")
            f.write(report["certificate"])
            f.write("\n\n---\n\n")
            f.write("## " + report["table_of_contents"].replace("\n", "\n## ", 1))
            f.write("\n\n---\n\n")
            
            for section, content in report["sections"].items():
                f.write(f"## {section}\n\n")
                f.write(content)
                f.write("\n\n---\n\n")


class JORCReportGenerator:
    """Generate JORC 2012 compliant reports."""
    
    TABLE_1_SECTIONS = [
        "Sampling Techniques and Data",
        "Reporting of Exploration Results",
        "Estimation and Reporting of Mineral Resources",
        "Estimation and Reporting of Ore Reserves"
    ]
    
    TABLE_1_CRITERIA = {
        "Sampling Techniques and Data": [
            "Sampling techniques",
            "Drilling techniques",
            "Drill sample recovery",
            "Logging",
            "Sub-sampling techniques and sample preparation",
            "Quality of assay data and laboratory tests",
            "Verification of sampling and assaying",
            "Location of data points",
            "Data spacing and distribution",
            "Orientation of data in relation to geological structure",
            "Sample security",
            "Audits or reviews"
        ],
        "Reporting of Exploration Results": [
            "Mineral tenement and land tenure status",
            "Exploration done by other parties",
            "Geology",
            "Drill hole Information",
            "Data aggregation methods",
            "Relationship between mineralisation widths and intercept lengths",
            "Diagrams",
            "Balanced reporting",
            "Other substantive exploration data",
            "Further work"
        ],
        "Estimation and Reporting of Mineral Resources": [
            "Database integrity",
            "Site visits",
            "Geological interpretation",
            "Dimensions",
            "Estimation and modelling techniques",
            "Moisture",
            "Cut-off parameters",
            "Mining factors or assumptions",
            "Metallurgical factors or assumptions",
            "Environmental factors or assumptions",
            "Bulk density",
            "Classification",
            "Audits or reviews",
            "Discussion of relative accuracy/confidence"
        ],
        "Estimation and Reporting of Ore Reserves": [
            "Mineral Resource estimate for conversion to Ore Reserves",
            "Site visits",
            "Study status",
            "Cut-off parameters",
            "Mining factors or assumptions",
            "Metallurgical factors or assumptions",
            "Infrastructure",
            "Costs",
            "Revenue factors",
            "Market assessment",
            "Economic",
            "Social",
            "Other",
            "Classification",
            "Audits or reviews",
            "Discussion of relative accuracy/confidence"
        ]
    }
    
    def __init__(self, project: ProjectInfo, cp: QualifiedPerson):
        self.project = project
        self.cp = cp
        self.resource_statement: Optional[ResourceStatement] = None
        self.reserve_statement: Optional[ReserveStatement] = None
        self.table_1_responses: Dict[str, Dict[str, str]] = {}
    
    def set_resource_statement(self, statement: ResourceStatement):
        """Set resource statement."""
        self.resource_statement = statement
    
    def set_reserve_statement(self, statement: ReserveStatement):
        """Set reserve statement."""
        self.reserve_statement = statement
    
    def set_table_1_response(self, section: str, criteria: str, response: str):
        """Set response for a Table 1 criteria."""
        if section not in self.table_1_responses:
            self.table_1_responses[section] = {}
        self.table_1_responses[section][criteria] = response
    
    def generate_table_1(self) -> str:
        """Generate JORC Table 1."""
        lines = []
        lines.append("JORC CODE, 2012 EDITION - TABLE 1")
        lines.append("")
        
        for section in self.TABLE_1_SECTIONS:
            lines.append(f"**Section: {section}**")
            lines.append("")
            lines.append("| Criteria | Commentary |")
            lines.append("|----------|------------|")
            
            criteria_list = self.TABLE_1_CRITERIA.get(section, [])
            responses = self.table_1_responses.get(section, {})
            
            for criteria in criteria_list:
                response = responses.get(criteria, "[To be completed]")
                response_escaped = response.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {criteria} | {response_escaped} |")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_resource_table(self) -> str:
        """Generate JORC-compliant resource table."""
        if not self.resource_statement:
            return "No resource statement available."
        
        lines = []
        lines.append(f"**Mineral Resource Estimate - {self.project.name}**")
        lines.append(f"Reported in accordance with the JORC Code (2012)")
        lines.append(f"Effective Date: {self.resource_statement.effective_date}")
        lines.append("")
        lines.append("| Classification | Tonnage (Mt) | Grade | Contained Metal |")
        lines.append("|----------------|--------------|-------|-----------------|")
        
        for category in [ResourceCategory.MEASURED, ResourceCategory.INDICATED, ResourceCategory.INFERRED]:
            totals = self.resource_statement.get_total_by_category(category)
            if totals['tonnage_mt'] > 0:
                unit = self.resource_statement.estimates[0].grade_unit if self.resource_statement.estimates else ""
                metal_unit = self.resource_statement.estimates[0].metal_unit if self.resource_statement.estimates else ""
                lines.append(f"| {category.value.capitalize()} | {totals['tonnage_mt']:.2f} | {totals['grade']:.2f} {unit} | {totals['contained_metal']:.0f} {metal_unit} |")
        
        mi_totals = self.resource_statement.get_measured_indicated_total()
        if mi_totals['tonnage_mt'] > 0:
            lines.append(f"| **Total M+I** | {mi_totals['tonnage_mt']:.2f} | {mi_totals['grade']:.2f} | {mi_totals['contained_metal']:.0f} |")
        
        lines.append("")
        lines.append("Notes:")
        lines.append("1. Mineral Resources are reported inclusive of Ore Reserves")
        lines.append("2. Mineral Resources are not Ore Reserves and do not have demonstrated economic viability")
        for i, note in enumerate(self.resource_statement.notes, 3):
            lines.append(f"{i}. {note}")
        
        return "\n".join(lines)
    
    def generate_competent_person_statement(self) -> str:
        """Generate Competent Person statement."""
        return f"""
COMPETENT PERSON STATEMENT

The information in this report that relates to Mineral Resources is based on information compiled by {self.cp.name}, who is a Member of {self.cp.professional_body}. {self.cp.name} is employed by {self.cp.company} and has sufficient experience which is relevant to the style of mineralisation and type of deposit under consideration and to the activity which they are undertaking to qualify as a Competent Person as defined in the 2012 Edition of the 'Australasian Code for Reporting of Exploration Results, Mineral Resources and Ore Reserves'. {self.cp.name} consents to the inclusion in the report of the matters based on their information in the form and context in which it appears.
"""
    
    def export_to_markdown(self, filepath: str):
        """Export JORC report to Markdown."""
        with open(filepath, 'w') as f:
            f.write(f"# {self.project.name} - JORC Report\n\n")
            f.write(f"**Effective Date:** {self.resource_statement.effective_date if self.resource_statement else date.today()}\n\n")
            f.write("---\n\n")
            f.write("## Mineral Resource Statement\n\n")
            f.write(self.generate_resource_table())
            f.write("\n\n---\n\n")
            f.write("## Competent Person Statement\n\n")
            f.write(self.generate_competent_person_statement())
            f.write("\n\n---\n\n")
            f.write("## JORC Table 1\n\n")
            f.write(self.generate_table_1())


class SAMRECReportGenerator:
    """Generate SAMREC compliant reports."""
    
    def __init__(self, project: ProjectInfo, cp: QualifiedPerson):
        self.project = project
        self.cp = cp
        self.resource_statement: Optional[ResourceStatement] = None
        self.reserve_statement: Optional[ReserveStatement] = None
    
    def set_resource_statement(self, statement: ResourceStatement):
        """Set resource statement."""
        self.resource_statement = statement
    
    def generate_resource_table(self) -> str:
        """Generate SAMREC-compliant resource table."""
        if not self.resource_statement:
            return "No resource statement available."
        
        lines = []
        lines.append(f"**Mineral Resource Statement - {self.project.name}**")
        lines.append(f"Reported in accordance with the SAMREC Code (2016)")
        lines.append(f"Effective Date: {self.resource_statement.effective_date}")
        lines.append("")
        lines.append("| Category | Tonnage (Mt) | Grade | Metal Content |")
        lines.append("|----------|--------------|-------|---------------|")
        
        for category in [ResourceCategory.MEASURED, ResourceCategory.INDICATED, ResourceCategory.INFERRED]:
            totals = self.resource_statement.get_total_by_category(category)
            if totals['tonnage_mt'] > 0:
                lines.append(f"| {category.value.capitalize()} | {totals['tonnage_mt']:.2f} | {totals['grade']:.2f} | {totals['contained_metal']:.0f} |")
        
        return "\n".join(lines)
    
    def generate_competent_person_statement(self) -> str:
        """Generate Competent Person statement for SAMREC."""
        return f"""
COMPETENT PERSON'S DECLARATION

I, {self.cp.name}, {self.cp.professional_designation}, hereby declare that:

1. I am registered with {self.cp.professional_body} (Registration No. {self.cp.registration_number}).

2. I have {self.cp.years_experience} years of relevant experience in the estimation, assessment and evaluation of Mineral Resources.

3. I have reviewed the Mineral Resource estimate for the {self.project.name} project and confirm that it has been prepared in accordance with the SAMREC Code (2016).

4. I consent to the publication of this Mineral Resource statement in the form and context in which it appears.

Signed: _______________________________
{self.cp.name}, {self.cp.professional_designation}
Date: {date.today()}
"""


class ReportingWorkflow:
    """
    Complete regulatory reporting workflow.
    """
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.project: Optional[ProjectInfo] = None
        self.qp: Optional[QualifiedPerson] = None
        self.resource_statement: Optional[ResourceStatement] = None
        self.reserve_statement: Optional[ReserveStatement] = None
        self.reporting_standard: ReportingStandard = ReportingStandard.NI_43_101
    
    def set_project(self, name: str, location: str, country: str,
                   coordinates: Tuple[float, float] = (0.0, 0.0),
                   area_hectares: float = 0.0,
                   commodities: List[str] = None,
                   **kwargs) -> ProjectInfo:
        """Set project information."""
        commodity_enums = []
        if commodities:
            for c in commodities:
                try:
                    commodity_enums.append(CommodityType(c.lower()))
                except ValueError:
                    commodity_enums.append(CommodityType.OTHER)
        
        self.project = ProjectInfo(
            name=name,
            location=location,
            country=country,
            coordinates=coordinates,
            area_hectares=area_hectares,
            commodities=commodity_enums,
            **kwargs
        )
        return self.project
    
    def set_qualified_person(self, name: str, title: str, company: str,
                            designation: str, registration: str,
                            professional_body: str, email: str,
                            years_experience: int = 0,
                            site_visit_date: Optional[str] = None,
                            **kwargs) -> QualifiedPerson:
        """Set qualified person information."""
        visit_date = None
        if site_visit_date:
            visit_date = date.fromisoformat(site_visit_date)
        
        self.qp = QualifiedPerson(
            name=name,
            title=title,
            company=company,
            professional_designation=designation,
            registration_number=registration,
            professional_body=professional_body,
            email=email,
            years_experience=years_experience,
            site_visit_date=visit_date,
            **kwargs
        )
        return self.qp
    
    def create_resource_statement(self, effective_date: str,
                                  commodity: str,
                                  estimates: List[Dict[str, Any]],
                                  **kwargs) -> ResourceStatement:
        """Create resource statement from estimates."""
        resource_estimates = []
        
        for est in estimates:
            category = ResourceCategory(est.get('category', 'inferred').lower())
            
            resource_estimates.append(ResourceEstimate(
                category=category,
                domain=est.get('domain', 'Main'),
                tonnage_mt=est.get('tonnage_mt', 0),
                grade=est.get('grade', 0),
                grade_unit=est.get('grade_unit', 'g/t'),
                contained_metal=est.get('contained_metal', 0),
                metal_unit=est.get('metal_unit', 'oz'),
                cutoff_grade=est.get('cutoff_grade', 0),
                density=est.get('density', 2.7)
            ))
        
        try:
            commodity_enum = CommodityType(commodity.lower())
        except ValueError:
            commodity_enum = CommodityType.OTHER
        
        self.resource_statement = ResourceStatement(
            effective_date=date.fromisoformat(effective_date),
            commodity=commodity_enum,
            estimates=resource_estimates,
            **kwargs
        )
        
        return self.resource_statement
    
    def set_reporting_standard(self, standard: str):
        """Set reporting standard."""
        try:
            self.reporting_standard = ReportingStandard(standard.lower())
        except ValueError:
            self.reporting_standard = ReportingStandard.NI_43_101
    
    def generate_report(self, output_path: str):
        """Generate report based on selected standard."""
        if not self.project or not self.qp:
            raise ValueError("Project and QP must be set before generating report")
        
        if self.reporting_standard == ReportingStandard.NI_43_101:
            generator = NI43101ReportGenerator(self.project, self.qp)
        elif self.reporting_standard == ReportingStandard.JORC_2012:
            generator = JORCReportGenerator(self.project, self.qp)
        elif self.reporting_standard == ReportingStandard.SAMREC:
            generator = SAMRECReportGenerator(self.project, self.qp)
        else:
            generator = NI43101ReportGenerator(self.project, self.qp)
        
        if self.resource_statement:
            generator.set_resource_statement(self.resource_statement)
        
        if self.reserve_statement:
            generator.set_reserve_statement(self.reserve_statement)
        
        generator.export_to_markdown(output_path)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow summary."""
        return {
            "project": self.project.to_dict() if self.project else None,
            "qualified_person": self.qp.to_dict() if self.qp else None,
            "reporting_standard": self.reporting_standard.value,
            "resource_statement": self.resource_statement.to_dict() if self.resource_statement else None,
            "reserve_statement": self.reserve_statement.to_dict() if self.reserve_statement else None
        }


def create_reporting_workflow(project_name: str = "default") -> ReportingWorkflow:
    """Factory function to create a reporting workflow."""
    return ReportingWorkflow(project_name)


def create_ni43101_report(project: ProjectInfo, qp: QualifiedPerson) -> NI43101ReportGenerator:
    """Factory function to create NI 43-101 report generator."""
    return NI43101ReportGenerator(project, qp)


def create_jorc_report(project: ProjectInfo, cp: QualifiedPerson) -> JORCReportGenerator:
    """Factory function to create JORC report generator."""
    return JORCReportGenerator(project, cp)
