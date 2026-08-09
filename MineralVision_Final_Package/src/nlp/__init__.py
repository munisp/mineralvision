"""
NLP Module for MineralVision

Natural Language Processing for geological reports, drill logs,
and unstructured geoscience text data.

Modules:
- geo_nlp: Core NLP functionality with GeoSciBERT-style embeddings
- entity_extraction: Named entity recognition for geological entities
- report_parser: Structured extraction from geological reports
- text_prospectivity: NLP features for prospectivity modeling
"""

from .geo_nlp import (
    GeoNLPProcessor,
    GeoEmbeddings,
    GeologyTokenizer,
    create_geo_nlp_processor,
)

from .entity_extraction import (
    GeoEntityExtractor,
    GeologyEntity,
    EntityType,
    extract_geological_entities,
)

from .report_parser import (
    GeologicalReportParser,
    DrillLogParser,
    ParsedReport,
    ParsedInterval,
    parse_geological_report,
)

from .text_prospectivity import (
    TextProspectivityFeatures,
    DocumentSimilarity,
    ProspectivityTextAnalyzer,
    extract_prospectivity_features,
)

__all__ = [
    # Core NLP
    "GeoNLPProcessor",
    "GeoEmbeddings",
    "GeologyTokenizer",
    "create_geo_nlp_processor",
    # Entity extraction
    "GeoEntityExtractor",
    "GeologyEntity",
    "EntityType",
    "extract_geological_entities",
    # Report parsing
    "GeologicalReportParser",
    "DrillLogParser",
    "ParsedReport",
    "ParsedInterval",
    "parse_geological_report",
    # Text prospectivity
    "TextProspectivityFeatures",
    "DocumentSimilarity",
    "ProspectivityTextAnalyzer",
    "extract_prospectivity_features",
]

__version__ = "1.0.0"
