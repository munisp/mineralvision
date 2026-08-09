"""
3D Visualization Module for MineralVision Platform.

Provides comprehensive 3D visualization capabilities including:
- Drillhole visualization with traces and intervals
- Block model rendering with grade coloring
- Surface/wireframe display
- Cross-section planes
- Interactive camera controls
- Export to images and videos
- PyVista/VTK integration layer
"""

from .visualization_3d import (
    RenderMode,
    ColorMap,
    CameraView,
    Point3D,
    Color,
    BoundingBox,
    CameraSettings,
    LightSettings,
    ScalarBarSettings,
    VisualizationObject,
    DrillholeVisualization,
    BlockModelVisualization,
    SurfaceVisualization,
    PointCloudVisualization,
    SectionPlaneVisualization,
    ColorMapper,
    MeshBuilder,
    Scene3D,
    PyVistaRenderer,
    VisualizationWorkflow,
    create_visualization_workflow,
)

from .collaborative_3d import (
    UserRole,
    AnnotationType,
    ReviewStatus,
    ExportFormat,
    User as CollabUser,
    Annotation,
    Measurement,
    SceneView,
    ReviewComment,
    ReviewWorkflow,
    AnnotationManager,
    MeasurementTool,
    ReviewWorkflowManager,
    SceneExporter,
    CollaborativeScene,
    create_collaborative_scene,
    create_review_workflow,
)

__all__ = [
    # 3D Visualization
    "RenderMode",
    "ColorMap",
    "CameraView",
    "Point3D",
    "Color",
    "BoundingBox",
    "CameraSettings",
    "LightSettings",
    "ScalarBarSettings",
    "VisualizationObject",
    "DrillholeVisualization",
    "BlockModelVisualization",
    "SurfaceVisualization",
    "PointCloudVisualization",
    "SectionPlaneVisualization",
    "ColorMapper",
    "MeshBuilder",
    "Scene3D",
    "PyVistaRenderer",
    "VisualizationWorkflow",
    "create_visualization_workflow",
    
    # Collaborative 3D
    "UserRole",
    "AnnotationType",
    "ReviewStatus",
    "ExportFormat",
    "CollabUser",
    "Annotation",
    "Measurement",
    "SceneView",
    "ReviewComment",
    "ReviewWorkflow",
    "AnnotationManager",
    "MeasurementTool",
    "ReviewWorkflowManager",
    "SceneExporter",
    "CollaborativeScene",
    "create_collaborative_scene",
    "create_review_workflow",
]
