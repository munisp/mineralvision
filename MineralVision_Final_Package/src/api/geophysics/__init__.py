"""
Geophysical Inversion Module for MineralVision Platform.

Provides comprehensive geophysical inversion capabilities including:
- Magnetic inversion (susceptibility)
- Gravity inversion (density)
- Electromagnetic inversion (conductivity)
- IP inversion (chargeability)
- DC resistivity inversion
- Forward modeling
- Regularization and constraints
- Depth weighting and reference models
- Sparse matrix solvers for scalability (100k+ cells)
- Octree/adaptive mesh support
- Joint inversion with cross-gradient coupling
- Topography handling for draped meshes
"""

from .inversion import (
    InversionType,
    RegularizationType,
    DepthWeightingType,
    Point3D,
    MeshCell,
    InversionMesh,
    ObservationPoint,
    SurveyData,
    InversionParameters,
    InversionResult,
    ForwardModeler,
    RegularizationOperator,
    DepthWeighting,
    GaussNewtonSolver,
    MagneticInversion,
    GravityInversion,
    EMInversion,
    InversionWorkflow,
    create_inversion_workflow,
    create_magnetic_inversion,
    create_gravity_inversion,
)

from .advanced_inversion import (
    MeshType,
    SolverType,
    JointInversionType,
    OctreeCell,
    TopographySurface,
    OctreeMesh,
    SparseMatrixBuilder,
    SparseForwardModeler,
    SparseRegularization,
    CrossGradientOperator,
    SparseSolver,
    JointInversionResult,
    JointInversion,
    AdvancedInversionWorkflow,
    create_advanced_inversion_workflow,
)

__all__ = [
    # Basic inversion
    "InversionType",
    "RegularizationType",
    "DepthWeightingType",
    "Point3D",
    "MeshCell",
    "InversionMesh",
    "ObservationPoint",
    "SurveyData",
    "InversionParameters",
    "InversionResult",
    "ForwardModeler",
    "RegularizationOperator",
    "DepthWeighting",
    "GaussNewtonSolver",
    "MagneticInversion",
    "GravityInversion",
    "EMInversion",
    "InversionWorkflow",
    "create_inversion_workflow",
    "create_magnetic_inversion",
    "create_gravity_inversion",
    # Advanced inversion
    "MeshType",
    "SolverType",
    "JointInversionType",
    "OctreeCell",
    "TopographySurface",
    "OctreeMesh",
    "SparseMatrixBuilder",
    "SparseForwardModeler",
    "SparseRegularization",
    "CrossGradientOperator",
    "SparseSolver",
    "JointInversionResult",
    "JointInversion",
    "AdvancedInversionWorkflow",
    "create_advanced_inversion_workflow",
]
