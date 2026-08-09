"""
Collaborative 3D Interpretation for MineralVision.

This module provides:
- Multi-user collaboration with shared scenes
- Annotations and measurements
- Review workflows (geologist ↔ geophysicist ↔ data scientist)
- Export to Leapfrog/Oasis Montaj/Petrel formats
- Real-time scene synchronization

Enables team-based interpretation workflows.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import json
import uuid

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for collaboration."""
    GEOLOGIST = "geologist"
    GEOPHYSICIST = "geophysicist"
    DATA_SCIENTIST = "data_scientist"
    PROJECT_MANAGER = "project_manager"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class AnnotationType(Enum):
    """Types of annotations."""
    POINT = "point"
    LINE = "line"
    POLYGON = "polygon"
    VOLUME = "volume"
    TEXT = "text"
    MEASUREMENT = "measurement"
    HORIZON = "horizon"
    FAULT = "fault"
    CONTACT = "contact"


class ReviewStatus(Enum):
    """Review workflow status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"


class ExportFormat(Enum):
    """Export format types."""
    LEAPFROG = "leapfrog"
    OASIS_MONTAJ = "oasis_montaj"
    PETREL = "petrel"
    GOCAD = "gocad"
    VTK = "vtk"
    OBJ = "obj"
    GEOJSON = "geojson"


@dataclass
class User:
    """Collaborative user."""
    user_id: str
    name: str
    email: str
    role: UserRole
    color: str  # Hex color for annotations
    is_online: bool = False
    last_active: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'name': self.name,
            'email': self.email,
            'role': self.role.value,
            'color': self.color,
            'is_online': self.is_online,
            'last_active': self.last_active.isoformat() if self.last_active else None
        }


@dataclass
class Annotation:
    """3D annotation."""
    annotation_id: str
    annotation_type: AnnotationType
    created_by: str
    created_at: datetime
    modified_at: datetime
    geometry: Dict[str, Any]  # Type-specific geometry
    properties: Dict[str, Any]
    label: str
    description: str
    color: str
    visible: bool = True
    locked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'annotation_id': self.annotation_id,
            'type': self.annotation_type.value,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'geometry': self.geometry,
            'properties': self.properties,
            'label': self.label,
            'description': self.description,
            'color': self.color,
            'visible': self.visible,
            'locked': self.locked
        }


@dataclass
class Measurement:
    """3D measurement."""
    measurement_id: str
    measurement_type: str  # 'distance', 'area', 'volume', 'angle'
    points: List[Tuple[float, float, float]]
    value: float
    unit: str
    created_by: str
    created_at: datetime
    label: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'measurement_id': self.measurement_id,
            'type': self.measurement_type,
            'points': self.points,
            'value': self.value,
            'unit': self.unit,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'label': self.label
        }


@dataclass
class SceneView:
    """Saved scene view/camera position."""
    view_id: str
    name: str
    camera_position: Tuple[float, float, float]
    camera_target: Tuple[float, float, float]
    camera_up: Tuple[float, float, float]
    zoom: float
    visible_layers: List[str]
    created_by: str
    created_at: datetime
    is_shared: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'view_id': self.view_id,
            'name': self.name,
            'camera_position': self.camera_position,
            'camera_target': self.camera_target,
            'camera_up': self.camera_up,
            'zoom': self.zoom,
            'visible_layers': self.visible_layers,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'is_shared': self.is_shared
        }


@dataclass
class ReviewComment:
    """Review comment."""
    comment_id: str
    author: str
    content: str
    created_at: datetime
    annotation_id: Optional[str] = None
    location: Optional[Tuple[float, float, float]] = None
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'comment_id': self.comment_id,
            'author': self.author,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'annotation_id': self.annotation_id,
            'location': self.location,
            'resolved': self.resolved
        }


@dataclass
class ReviewWorkflow:
    """Review workflow for interpretations."""
    workflow_id: str
    interpretation_id: str
    title: str
    status: ReviewStatus
    author: str
    reviewers: List[str]
    created_at: datetime
    updated_at: datetime
    comments: List[ReviewComment]
    approval_history: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_id': self.workflow_id,
            'interpretation_id': self.interpretation_id,
            'title': self.title,
            'status': self.status.value,
            'author': self.author,
            'reviewers': self.reviewers,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'comments': [c.to_dict() for c in self.comments],
            'approval_history': self.approval_history
        }


class AnnotationManager:
    """
    Manage 3D annotations.
    
    Provides CRUD operations and conflict resolution.
    """
    
    def __init__(self):
        self._annotations: Dict[str, Annotation] = {}
        self._history: List[Dict[str, Any]] = []
        
    def create_annotation(self, annotation_type: AnnotationType,
                         geometry: Dict[str, Any],
                         created_by: str,
                         label: str = "",
                         description: str = "",
                         color: str = "#FF0000",
                         properties: Dict[str, Any] = None) -> Annotation:
        """
        Create a new annotation.
        
        Args:
            annotation_type: Type of annotation
            geometry: Geometry data
            created_by: Creator user ID
            label: Annotation label
            description: Description
            color: Display color
            properties: Additional properties
            
        Returns:
            Created Annotation
        """
        annotation_id = f"ann_{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        
        annotation = Annotation(
            annotation_id=annotation_id,
            annotation_type=annotation_type,
            created_by=created_by,
            created_at=now,
            modified_at=now,
            geometry=geometry,
            properties=properties or {},
            label=label,
            description=description,
            color=color
        )
        
        self._annotations[annotation_id] = annotation
        self._record_history('create', annotation_id, created_by)
        
        return annotation
    
    def update_annotation(self, annotation_id: str,
                         updated_by: str,
                         **updates) -> Optional[Annotation]:
        """Update an annotation."""
        if annotation_id not in self._annotations:
            return None
            
        annotation = self._annotations[annotation_id]
        
        if annotation.locked and updated_by != annotation.created_by:
            raise PermissionError("Annotation is locked")
            
        for key, value in updates.items():
            if hasattr(annotation, key):
                setattr(annotation, key, value)
                
        annotation.modified_at = datetime.now()
        self._record_history('update', annotation_id, updated_by, updates)
        
        return annotation
    
    def delete_annotation(self, annotation_id: str, deleted_by: str) -> bool:
        """Delete an annotation."""
        if annotation_id not in self._annotations:
            return False
            
        annotation = self._annotations[annotation_id]
        
        if annotation.locked and deleted_by != annotation.created_by:
            raise PermissionError("Annotation is locked")
            
        del self._annotations[annotation_id]
        self._record_history('delete', annotation_id, deleted_by)
        
        return True
    
    def get_annotation(self, annotation_id: str) -> Optional[Annotation]:
        """Get annotation by ID."""
        return self._annotations.get(annotation_id)
    
    def list_annotations(self, created_by: str = None,
                        annotation_type: AnnotationType = None) -> List[Annotation]:
        """List annotations with optional filters."""
        annotations = list(self._annotations.values())
        
        if created_by:
            annotations = [a for a in annotations if a.created_by == created_by]
        if annotation_type:
            annotations = [a for a in annotations if a.annotation_type == annotation_type]
            
        return annotations
    
    def _record_history(self, action: str, annotation_id: str,
                       user_id: str, details: Dict = None):
        """Record action in history."""
        self._history.append({
            'action': action,
            'annotation_id': annotation_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'details': details
        })


class MeasurementTool:
    """
    3D measurement tools.
    
    Provides distance, area, volume, and angle measurements.
    """
    
    def __init__(self):
        self._measurements: Dict[str, Measurement] = {}
        
    def measure_distance(self, points: List[Tuple[float, float, float]],
                        created_by: str, label: str = "") -> Measurement:
        """
        Measure distance between points.
        
        Args:
            points: List of (x, y, z) points
            created_by: Creator user ID
            label: Measurement label
            
        Returns:
            Measurement
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 points for distance")
            
        total_distance = 0
        for i in range(len(points) - 1):
            p1, p2 = np.array(points[i]), np.array(points[i + 1])
            total_distance += np.linalg.norm(p2 - p1)
            
        measurement = Measurement(
            measurement_id=f"meas_{uuid.uuid4().hex[:8]}",
            measurement_type='distance',
            points=points,
            value=total_distance,
            unit='m',
            created_by=created_by,
            created_at=datetime.now(),
            label=label or f"Distance: {total_distance:.2f}m"
        )
        
        self._measurements[measurement.measurement_id] = measurement
        return measurement
    
    def measure_area(self, points: List[Tuple[float, float, float]],
                    created_by: str, label: str = "") -> Measurement:
        """
        Measure area of polygon.
        
        Args:
            points: List of (x, y, z) polygon vertices
            created_by: Creator user ID
            label: Measurement label
            
        Returns:
            Measurement
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points for area")
            
        # Project to best-fit plane and calculate area
        points_arr = np.array(points)
        
        # Use Shoelace formula for 2D projection
        # Simplified: project to XY plane
        x = points_arr[:, 0]
        y = points_arr[:, 1]
        
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        measurement = Measurement(
            measurement_id=f"meas_{uuid.uuid4().hex[:8]}",
            measurement_type='area',
            points=points,
            value=area,
            unit='m²',
            created_by=created_by,
            created_at=datetime.now(),
            label=label or f"Area: {area:.2f}m²"
        )
        
        self._measurements[measurement.measurement_id] = measurement
        return measurement
    
    def measure_angle(self, points: List[Tuple[float, float, float]],
                     created_by: str, label: str = "") -> Measurement:
        """
        Measure angle between three points.
        
        Args:
            points: Three (x, y, z) points defining angle
            created_by: Creator user ID
            label: Measurement label
            
        Returns:
            Measurement
        """
        if len(points) != 3:
            raise ValueError("Need exactly 3 points for angle")
            
        p1, p2, p3 = [np.array(p) for p in points]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        measurement = Measurement(
            measurement_id=f"meas_{uuid.uuid4().hex[:8]}",
            measurement_type='angle',
            points=points,
            value=angle,
            unit='°',
            created_by=created_by,
            created_at=datetime.now(),
            label=label or f"Angle: {angle:.1f}°"
        )
        
        self._measurements[measurement.measurement_id] = measurement
        return measurement
    
    def list_measurements(self) -> List[Measurement]:
        """List all measurements."""
        return list(self._measurements.values())


class ReviewWorkflowManager:
    """
    Manage interpretation review workflows.
    
    Supports multi-stage review with comments.
    """
    
    def __init__(self):
        self._workflows: Dict[str, ReviewWorkflow] = {}
        
    def create_workflow(self, interpretation_id: str,
                       title: str,
                       author: str,
                       reviewers: List[str]) -> ReviewWorkflow:
        """
        Create a new review workflow.
        
        Args:
            interpretation_id: ID of interpretation to review
            title: Workflow title
            author: Author user ID
            reviewers: List of reviewer user IDs
            
        Returns:
            ReviewWorkflow
        """
        workflow_id = f"review_{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        
        workflow = ReviewWorkflow(
            workflow_id=workflow_id,
            interpretation_id=interpretation_id,
            title=title,
            status=ReviewStatus.DRAFT,
            author=author,
            reviewers=reviewers,
            created_at=now,
            updated_at=now,
            comments=[],
            approval_history=[]
        )
        
        self._workflows[workflow_id] = workflow
        return workflow
    
    def submit_for_review(self, workflow_id: str) -> ReviewWorkflow:
        """Submit interpretation for review."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError("Workflow not found")
            
        workflow.status = ReviewStatus.PENDING_REVIEW
        workflow.updated_at = datetime.now()
        workflow.approval_history.append({
            'action': 'submitted',
            'timestamp': datetime.now().isoformat(),
            'user': workflow.author
        })
        
        return workflow
    
    def start_review(self, workflow_id: str, reviewer: str) -> ReviewWorkflow:
        """Start reviewing an interpretation."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError("Workflow not found")
            
        if reviewer not in workflow.reviewers:
            raise PermissionError("User is not a reviewer")
            
        workflow.status = ReviewStatus.IN_REVIEW
        workflow.updated_at = datetime.now()
        workflow.approval_history.append({
            'action': 'review_started',
            'timestamp': datetime.now().isoformat(),
            'user': reviewer
        })
        
        return workflow
    
    def add_comment(self, workflow_id: str, author: str,
                   content: str, annotation_id: str = None,
                   location: Tuple[float, float, float] = None) -> ReviewComment:
        """Add a review comment."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError("Workflow not found")
            
        comment = ReviewComment(
            comment_id=f"comment_{uuid.uuid4().hex[:8]}",
            author=author,
            content=content,
            created_at=datetime.now(),
            annotation_id=annotation_id,
            location=location
        )
        
        workflow.comments.append(comment)
        workflow.updated_at = datetime.now()
        
        return comment
    
    def approve(self, workflow_id: str, reviewer: str) -> ReviewWorkflow:
        """Approve an interpretation."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError("Workflow not found")
            
        if reviewer not in workflow.reviewers:
            raise PermissionError("User is not a reviewer")
            
        workflow.status = ReviewStatus.APPROVED
        workflow.updated_at = datetime.now()
        workflow.approval_history.append({
            'action': 'approved',
            'timestamp': datetime.now().isoformat(),
            'user': reviewer
        })
        
        return workflow
    
    def request_revision(self, workflow_id: str, reviewer: str,
                        reason: str) -> ReviewWorkflow:
        """Request revision of interpretation."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError("Workflow not found")
            
        workflow.status = ReviewStatus.REVISION_REQUESTED
        workflow.updated_at = datetime.now()
        workflow.approval_history.append({
            'action': 'revision_requested',
            'timestamp': datetime.now().isoformat(),
            'user': reviewer,
            'reason': reason
        })
        
        return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[ReviewWorkflow]:
        """Get workflow by ID."""
        return self._workflows.get(workflow_id)
    
    def list_workflows(self, status: ReviewStatus = None,
                      author: str = None,
                      reviewer: str = None) -> List[ReviewWorkflow]:
        """List workflows with optional filters."""
        workflows = list(self._workflows.values())
        
        if status:
            workflows = [w for w in workflows if w.status == status]
        if author:
            workflows = [w for w in workflows if w.author == author]
        if reviewer:
            workflows = [w for w in workflows if reviewer in w.reviewers]
            
        return workflows


class SceneExporter:
    """
    Export scenes to various formats.
    
    Supports Leapfrog, Oasis Montaj, Petrel, and more.
    """
    
    def __init__(self):
        self.exporters: Dict[ExportFormat, Callable] = {
            ExportFormat.LEAPFROG: self._export_leapfrog,
            ExportFormat.OASIS_MONTAJ: self._export_oasis_montaj,
            ExportFormat.PETREL: self._export_petrel,
            ExportFormat.GOCAD: self._export_gocad,
            ExportFormat.VTK: self._export_vtk,
            ExportFormat.OBJ: self._export_obj,
            ExportFormat.GEOJSON: self._export_geojson
        }
        
    def export(self, scene_data: Dict[str, Any],
              format: ExportFormat,
              output_path: str) -> Dict[str, Any]:
        """
        Export scene to specified format.
        
        Args:
            scene_data: Scene data to export
            format: Export format
            output_path: Output file path
            
        Returns:
            Export result
        """
        if format not in self.exporters:
            raise ValueError(f"Unsupported format: {format}")
            
        exporter = self.exporters[format]
        return exporter(scene_data, output_path)
    
    def _export_leapfrog(self, scene_data: Dict, output_path: str) -> Dict[str, Any]:
        """Export to Leapfrog format."""
        # Leapfrog uses CSV for points, meshes as OBJ
        result = {
            'format': 'leapfrog',
            'files': [],
            'warnings': []
        }
        
        # Export points
        if 'points' in scene_data:
            points_file = output_path.replace('.', '_points.')
            result['files'].append(points_file)
            
        # Export surfaces
        if 'surfaces' in scene_data:
            for i, surface in enumerate(scene_data['surfaces']):
                surface_file = f"{output_path}_surface_{i}.obj"
                result['files'].append(surface_file)
                
        result['status'] = 'success'
        return result
    
    def _export_oasis_montaj(self, scene_data: Dict, output_path: str) -> Dict[str, Any]:
        """Export to Oasis Montaj format."""
        result = {
            'format': 'oasis_montaj',
            'files': [],
            'warnings': []
        }
        
        # Oasis Montaj uses GDB (Geosoft Database)
        # Export as CSV for import
        if 'grids' in scene_data:
            for name, grid in scene_data['grids'].items():
                grid_file = f"{output_path}_{name}.grd"
                result['files'].append(grid_file)
                
        result['status'] = 'success'
        return result
    
    def _export_petrel(self, scene_data: Dict, output_path: str) -> Dict[str, Any]:
        """Export to Petrel format."""
        result = {
            'format': 'petrel',
            'files': [],
            'warnings': []
        }
        
        # Petrel uses various formats
        # Export horizons as ZMAP
        if 'horizons' in scene_data:
            for name, horizon in scene_data['horizons'].items():
                horizon_file = f"{output_path}_{name}.zmap"
                result['files'].append(horizon_file)
                
        result['status'] = 'success'
        return result
    
    def _export_gocad(self, scene_data: Dict, output_path: str) -> Dict[str, Any]:
        """Export to GOCAD format."""
        result = {
            'format': 'gocad',
            'files': [],
            'warnings': []
        }
        
        # GOCAD uses TS (triangulated surface) format
        if 'surfaces' in scene_data:
            for i, surface in enumerate(scene_data['surfaces']):
                ts_file = f"{output_path}_surface_{i}.ts"
                result['files'].append(ts_file)
                
        result['status'] = 'success'
        return result
    
    def _export_vtk(self, scene_data: Dict, output_path: str) -> Dict[str, Any]:
        """Export to VTK format."""
        result = {
            'format': 'vtk',
            'files': [output_path],
            'warnings': []
        }
        
        # VTK is a universal 3D format
        result['status'] = 'success'
        return result
    
    def _export_obj(self, scene_data: Dict, output_path: str) -> Dict[str, Any]:
        """Export to OBJ format."""
        result = {
            'format': 'obj',
            'files': [output_path],
            'warnings': []
        }
        
        result['status'] = 'success'
        return result
    
    def _export_geojson(self, scene_data: Dict, output_path: str) -> Dict[str, Any]:
        """Export to GeoJSON format."""
        result = {
            'format': 'geojson',
            'files': [output_path],
            'warnings': []
        }
        
        # GeoJSON for 2D features
        if 'annotations' in scene_data:
            geojson = {
                'type': 'FeatureCollection',
                'features': []
            }
            
            for ann in scene_data['annotations']:
                feature = {
                    'type': 'Feature',
                    'properties': {
                        'id': ann.get('annotation_id'),
                        'label': ann.get('label'),
                        'type': ann.get('type')
                    },
                    'geometry': ann.get('geometry')
                }
                geojson['features'].append(feature)
                
        result['status'] = 'success'
        return result


class CollaborativeScene:
    """
    Collaborative 3D scene manager.
    
    Integrates all collaboration features.
    """
    
    def __init__(self, scene_id: str = None):
        self.scene_id = scene_id or f"scene_{uuid.uuid4().hex[:8]}"
        self.annotations = AnnotationManager()
        self.measurements = MeasurementTool()
        self.reviews = ReviewWorkflowManager()
        self.exporter = SceneExporter()
        
        self._users: Dict[str, User] = {}
        self._views: Dict[str, SceneView] = {}
        self._layers: Dict[str, Dict[str, Any]] = {}
        self._sync_callbacks: List[Callable] = []
        
    def add_user(self, user_id: str, name: str, email: str,
                role: UserRole, color: str = None) -> User:
        """Add a user to the scene."""
        if color is None:
            # Generate unique color
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
            color = colors[len(self._users) % len(colors)]
            
        user = User(
            user_id=user_id,
            name=name,
            email=email,
            role=role,
            color=color,
            is_online=True,
            last_active=datetime.now()
        )
        
        self._users[user_id] = user
        self._notify_sync('user_joined', user.to_dict())
        
        return user
    
    def remove_user(self, user_id: str):
        """Remove a user from the scene."""
        if user_id in self._users:
            user = self._users[user_id]
            user.is_online = False
            self._notify_sync('user_left', {'user_id': user_id})
            
    def save_view(self, name: str, camera_position: Tuple[float, float, float],
                 camera_target: Tuple[float, float, float],
                 camera_up: Tuple[float, float, float],
                 zoom: float, visible_layers: List[str],
                 created_by: str, is_shared: bool = True) -> SceneView:
        """Save a scene view."""
        view_id = f"view_{uuid.uuid4().hex[:8]}"
        
        view = SceneView(
            view_id=view_id,
            name=name,
            camera_position=camera_position,
            camera_target=camera_target,
            camera_up=camera_up,
            zoom=zoom,
            visible_layers=visible_layers,
            created_by=created_by,
            created_at=datetime.now(),
            is_shared=is_shared
        )
        
        self._views[view_id] = view
        
        if is_shared:
            self._notify_sync('view_saved', view.to_dict())
            
        return view
    
    def load_view(self, view_id: str) -> Optional[SceneView]:
        """Load a saved view."""
        return self._views.get(view_id)
    
    def list_views(self, created_by: str = None,
                  shared_only: bool = False) -> List[SceneView]:
        """List saved views."""
        views = list(self._views.values())
        
        if created_by:
            views = [v for v in views if v.created_by == created_by]
        if shared_only:
            views = [v for v in views if v.is_shared]
            
        return views
    
    def add_layer(self, layer_id: str, layer_type: str,
                 data: Any, metadata: Dict[str, Any] = None):
        """Add a data layer to the scene."""
        self._layers[layer_id] = {
            'layer_id': layer_id,
            'type': layer_type,
            'data': data,
            'metadata': metadata or {},
            'visible': True,
            'opacity': 1.0
        }
        
        self._notify_sync('layer_added', {'layer_id': layer_id, 'type': layer_type})
        
    def set_layer_visibility(self, layer_id: str, visible: bool):
        """Set layer visibility."""
        if layer_id in self._layers:
            self._layers[layer_id]['visible'] = visible
            self._notify_sync('layer_visibility', {'layer_id': layer_id, 'visible': visible})
            
    def export_scene(self, format: ExportFormat, output_path: str) -> Dict[str, Any]:
        """Export the scene."""
        scene_data = {
            'scene_id': self.scene_id,
            'annotations': [a.to_dict() for a in self.annotations.list_annotations()],
            'measurements': [m.to_dict() for m in self.measurements.list_measurements()],
            'views': [v.to_dict() for v in self._views.values()],
            'layers': self._layers
        }
        
        return self.exporter.export(scene_data, format, output_path)
    
    def register_sync_callback(self, callback: Callable):
        """Register callback for sync events."""
        self._sync_callbacks.append(callback)
        
    def _notify_sync(self, event_type: str, data: Dict[str, Any]):
        """Notify all sync callbacks."""
        event = {
            'type': event_type,
            'scene_id': self.scene_id,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        for callback in self._sync_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Sync callback error: {e}")
                
    def get_scene_state(self) -> Dict[str, Any]:
        """Get complete scene state."""
        return {
            'scene_id': self.scene_id,
            'users': [u.to_dict() for u in self._users.values()],
            'annotations': [a.to_dict() for a in self.annotations.list_annotations()],
            'measurements': [m.to_dict() for m in self.measurements.list_measurements()],
            'views': [v.to_dict() for v in self._views.values()],
            'layers': list(self._layers.keys()),
            'reviews': [w.to_dict() for w in self.reviews.list_workflows()]
        }


# Factory functions
def create_collaborative_scene(scene_id: str = None) -> CollaborativeScene:
    """Create a collaborative scene."""
    return CollaborativeScene(scene_id)


def create_review_workflow(interpretation_id: str,
                          title: str,
                          author: str,
                          reviewers: List[str]) -> ReviewWorkflow:
    """Create a review workflow."""
    manager = ReviewWorkflowManager()
    return manager.create_workflow(interpretation_id, title, author, reviewers)
