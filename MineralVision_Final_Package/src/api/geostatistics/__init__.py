"""
Geostatistics Module for MineralVision Platform.

Provides comprehensive geostatistical analysis capabilities including:
- Variography (experimental variogram calculation, model fitting)
- Kriging (ordinary, simple, universal, indicator, cokriging)
- Block modeling (creation, estimation, classification, reporting)
- Grade shell generation (isosurface extraction, smoothing, export)
"""

from .variography import (
    DirectionDefinition,
    LagBin,
    ExperimentalVariogram,
    VariogramStructure,
    FittedVariogramModel,
    ExperimentalVariogramCalculator,
    VariogramModelFitter,
    CrossVariogramCalculator,
    VariographyWorkflow,
    create_variography_workflow,
)

from .kriging import (
    KrigingType,
    DriftType,
    SearchEllipsoid,
    SearchParameters,
    VariogramModel,
    KrigingResult,
    CrossValidationResult,
    KrigingStatistics,
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
    IndicatorKriging,
    Cokriging,
    KrigingWorkflow,
    create_kriging_workflow,
)

from .block_model import (
    BlockModelType,
    EstimationMethod,
    ReserveCategory,
    Block,
    BlockModel,
    BlockModelEstimator,
    ResourceClassifier,
    BlockModelReporter,
    BlockModelIO,
    BlockModelWorkflow,
    create_block_model_workflow,
)

from .grade_shells import (
    ShellMethod,
    SurfaceType,
    Point3D,
    Triangle,
    Vertex,
    GradeShell,
    GridCell,
    MarchingCubes,
    GradeShellGenerator,
    ShellSmoother,
    ShellValidator,
    ShellExporter,
    GradeShellWorkflow,
    create_grade_shell_workflow,
)

__all__ = [
    "DirectionDefinition",
    "LagBin",
    "ExperimentalVariogram",
    "VariogramStructure",
    "FittedVariogramModel",
    "ExperimentalVariogramCalculator",
    "VariogramModelFitter",
    "CrossVariogramCalculator",
    "VariographyWorkflow",
    "create_variography_workflow",
    "KrigingType",
    "DriftType",
    "SearchEllipsoid",
    "SearchParameters",
    "VariogramModel",
    "KrigingResult",
    "CrossValidationResult",
    "KrigingStatistics",
    "OrdinaryKriging",
    "SimpleKriging",
    "UniversalKriging",
    "IndicatorKriging",
    "Cokriging",
    "KrigingWorkflow",
    "create_kriging_workflow",
    "BlockModelType",
    "EstimationMethod",
    "ReserveCategory",
    "Block",
    "BlockModel",
    "BlockModelEstimator",
    "ResourceClassifier",
    "BlockModelReporter",
    "BlockModelIO",
    "BlockModelWorkflow",
    "create_block_model_workflow",
    "ShellMethod",
    "SurfaceType",
    "Point3D",
    "Triangle",
    "Vertex",
    "GradeShell",
    "GridCell",
    "MarchingCubes",
    "GradeShellGenerator",
    "ShellSmoother",
    "ShellValidator",
    "ShellExporter",
    "GradeShellWorkflow",
    "create_grade_shell_workflow",
]
