"""
MineralVision Core Module.

This module provides core infrastructure components:
- Unified geospatial data model with CRS/units governance
- Storage tiering architecture for large surveys
- QA/QC and audit trails for compliance
"""

from .geospatial_model import (
    CoordinateSystem,
    VerticalDatum,
    DepthConvention,
    LengthUnit,
    AreaUnit,
    AngleUnit,
    TimeUnit,
    EPSGCode,
    CRSDefinition,
    UnitConverter,
    BoundingBox,
    GridDefinition,
    GeospatialMetadata,
    CoordinateTransformer,
    GeospatialValidator,
    GeospatialDataModel,
    create_geospatial_model,
    create_grid_definition,
)

from .storage_tiering import (
    StorageTier,
    DataFormat,
    AccessPattern,
    DataCharacteristics,
    TieringDecision,
    CacheEntry,
    LRUCache,
    TieringEngine,
    BenchmarkResult,
    PerformanceBenchmark,
    StorageManager,
    create_storage_manager,
    analyze_dataset_for_tiering,
)

from .audit_trails import (
    QCStatus,
    DataQuality,
    AuditAction,
    LineageNode,
    LineageEdge,
    QCMetric,
    QCReport,
    AuditEntry,
    LineageGraph,
    QCValidator,
    AuditLogger,
    ComplianceReporter,
    AuditTrailManager,
    create_audit_trail_manager,
    create_lineage_graph,
    create_qc_validator,
)

__all__ = [
    # Geospatial Model
    'CoordinateSystem',
    'VerticalDatum',
    'DepthConvention',
    'LengthUnit',
    'AreaUnit',
    'AngleUnit',
    'TimeUnit',
    'EPSGCode',
    'CRSDefinition',
    'UnitConverter',
    'BoundingBox',
    'GridDefinition',
    'GeospatialMetadata',
    'CoordinateTransformer',
    'GeospatialValidator',
    'GeospatialDataModel',
    'create_geospatial_model',
    'create_grid_definition',
    
    # Storage Tiering
    'StorageTier',
    'DataFormat',
    'AccessPattern',
    'DataCharacteristics',
    'TieringDecision',
    'CacheEntry',
    'LRUCache',
    'TieringEngine',
    'BenchmarkResult',
    'PerformanceBenchmark',
    'StorageManager',
    'create_storage_manager',
    'analyze_dataset_for_tiering',
    
    # Audit Trails
    'QCStatus',
    'DataQuality',
    'AuditAction',
    'LineageNode',
    'LineageEdge',
    'QCMetric',
    'QCReport',
    'AuditEntry',
    'LineageGraph',
    'QCValidator',
    'AuditLogger',
    'ComplianceReporter',
    'AuditTrailManager',
    'create_audit_trail_manager',
    'create_lineage_graph',
    'create_qc_validator',
]
