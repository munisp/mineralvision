"""
Regulatory Reporting Module for MineralVision Platform.

Provides comprehensive regulatory reporting capabilities including:
- NI 43-101 (Canadian) report templates
- JORC (Australian) report templates
- SAMREC (South African) report templates
- SEC S-K 1300 (US) report templates
- Resource/Reserve statement generation
- QP/CP certification support
- Table generation and formatting
- Automated report generation
"""

from .regulatory_reports import (
    ReportingStandard,
    ResourceCategory,
    ReserveCategory,
    CommodityType,
    QualifiedPerson,
    ProjectInfo,
    ResourceEstimate,
    ReserveEstimate,
    ResourceStatement,
    ReserveStatement,
    NI43101ReportGenerator,
    JORCReportGenerator,
    SAMRECReportGenerator,
    ReportingWorkflow,
    create_reporting_workflow,
    create_ni43101_report,
    create_jorc_report,
)

from .auto_report_generator import (
    ReportStandard,
    ReportSection,
    FigureType,
    TableType,
    Author,
    Figure,
    Table,
    SectionContent,
    QCSummary,
    ResourceEstimate as AutoResourceEstimate,
    ReportMetadata,
    ReportTemplate,
    NI43101Template,
    JORCTemplate,
    DataExtractor,
    SectionGenerator,
    ReportGenerator,
    ReportScheduler,
    create_report_generator,
    create_ni43101_report as create_auto_ni43101_report,
    create_jorc_report as create_auto_jorc_report,
)

__all__ = [
    # Regulatory Reports
    "ReportingStandard",
    "ResourceCategory",
    "ReserveCategory",
    "CommodityType",
    "QualifiedPerson",
    "ProjectInfo",
    "ResourceEstimate",
    "ReserveEstimate",
    "ResourceStatement",
    "ReserveStatement",
    "NI43101ReportGenerator",
    "JORCReportGenerator",
    "SAMRECReportGenerator",
    "ReportingWorkflow",
    "create_reporting_workflow",
    "create_ni43101_report",
    "create_jorc_report",
    # Auto Report Generator
    "ReportStandard",
    "ReportSection",
    "FigureType",
    "TableType",
    "Author",
    "Figure",
    "Table",
    "SectionContent",
    "QCSummary",
    "AutoResourceEstimate",
    "ReportMetadata",
    "ReportTemplate",
    "NI43101Template",
    "JORCTemplate",
    "DataExtractor",
    "SectionGenerator",
    "ReportGenerator",
    "ReportScheduler",
    "create_report_generator",
    "create_auto_ni43101_report",
    "create_auto_jorc_report",
]
