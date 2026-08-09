"""
MineralVision Data Ingestion Module.

Provides hardware data ingestion adapters for:
- LiDAR point clouds (LAS/LAZ)
- Portable XRF geochemistry
- GNSS/Survey data
- Laboratory LIMS data

All adapters convert raw hardware outputs to MineralVision schemas.
"""

from .lidar_ingestion import (
    LiDARFormat,
    LiDARClassification,
    LiDARPoint,
    LiDARMetadata,
    LiDARIngestionPipeline,
    create_lidar_pipeline
)

from .xrf_ingestion import (
    XRFVendor,
    XRFMode,
    XRFReading,
    XRFCalibration,
    XRFIngestionPipeline,
    create_xrf_pipeline
)

from .gnss_ingestion import (
    GNSSFormat,
    FixQuality,
    GNSSObservation,
    GNSSTrajectory,
    GNSSIngestionPipeline,
    create_gnss_pipeline
)

from .lims_ingestion import (
    LIMSFormat,
    AnalyticalMethod,
    LabSample,
    LabResult,
    LIMSIngestionPipeline,
    create_lims_pipeline
)

__all__ = [
    # LiDAR
    'LiDARFormat',
    'LiDARClassification',
    'LiDARPoint',
    'LiDARMetadata',
    'LiDARIngestionPipeline',
    'create_lidar_pipeline',
    
    # XRF
    'XRFVendor',
    'XRFMode',
    'XRFReading',
    'XRFCalibration',
    'XRFIngestionPipeline',
    'create_xrf_pipeline',
    
    # GNSS
    'GNSSFormat',
    'FixQuality',
    'GNSSObservation',
    'GNSSTrajectory',
    'GNSSIngestionPipeline',
    'create_gnss_pipeline',
    
    # LIMS
    'LIMSFormat',
    'AnalyticalMethod',
    'LabSample',
    'LabResult',
    'LIMSIngestionPipeline',
    'create_lims_pipeline'
]
