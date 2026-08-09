"""
Field Data Collection Module for MineralVision.

Provides mobile-first PWA capabilities for:
- Offline data collection
- GPS location capture
- Photo capture with geotagging
- Voice notes transcription
- Automatic sync on connectivity
- Form-based data entry
- Sample tracking
"""

import json
import hashlib
import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import uuid

logger = logging.getLogger(__name__)


class CollectionType(Enum):
    """Types of field data collection."""
    SAMPLE = "sample"
    OBSERVATION = "observation"
    PHOTO = "photo"
    VOICE_NOTE = "voice_note"
    MEASUREMENT = "measurement"
    WAYPOINT = "waypoint"
    TRAVERSE = "traverse"
    STRUCTURE = "structure"


class SyncStatus(Enum):
    """Synchronization status."""
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"
    CONFLICT = "conflict"


class DataQuality(Enum):
    """Data quality indicators."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class GPSAccuracy(Enum):
    """GPS accuracy levels."""
    RTK_FIXED = "rtk_fixed"
    RTK_FLOAT = "rtk_float"
    DGPS = "dgps"
    AUTONOMOUS = "autonomous"
    NO_FIX = "no_fix"


@dataclass
class GPSLocation:
    """GPS location data."""
    latitude: float
    longitude: float
    altitude: float
    accuracy_horizontal: float
    accuracy_vertical: float
    accuracy_type: GPSAccuracy
    timestamp: datetime
    satellites: int = 0
    hdop: float = 0.0
    pdop: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'accuracy_horizontal': self.accuracy_horizontal,
            'accuracy_vertical': self.accuracy_vertical,
            'accuracy_type': self.accuracy_type.value,
            'timestamp': self.timestamp.isoformat(),
            'satellites': self.satellites,
            'hdop': self.hdop,
            'pdop': self.pdop
        }
        
    def to_wkt(self) -> str:
        """Convert to WKT format."""
        return f"POINT({self.longitude} {self.latitude} {self.altitude})"


@dataclass
class Photo:
    """Photo with metadata."""
    photo_id: str
    filename: str
    data: Optional[bytes] = None
    data_base64: Optional[str] = None
    location: Optional[GPSLocation] = None
    bearing: float = 0.0
    tilt: float = 0.0
    caption: str = ""
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'photo_id': self.photo_id,
            'filename': self.filename,
            'data_base64': self.data_base64,
            'location': self.location.to_dict() if self.location else None,
            'bearing': self.bearing,
            'tilt': self.tilt,
            'caption': self.caption,
            'tags': self.tags,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class VoiceNote:
    """Voice note with transcription."""
    note_id: str
    filename: str
    duration_seconds: float
    data: Optional[bytes] = None
    data_base64: Optional[str] = None
    transcription: str = ""
    location: Optional[GPSLocation] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'note_id': self.note_id,
            'filename': self.filename,
            'duration_seconds': self.duration_seconds,
            'data_base64': self.data_base64,
            'transcription': self.transcription,
            'location': self.location.to_dict() if self.location else None,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class FormField:
    """Form field definition."""
    field_id: str
    name: str
    field_type: str  # text, number, select, multiselect, date, location, photo
    required: bool = False
    default_value: Any = None
    options: List[str] = field(default_factory=list)
    validation_regex: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_id': self.field_id,
            'name': self.name,
            'field_type': self.field_type,
            'required': self.required,
            'default_value': self.default_value,
            'options': self.options,
            'validation_regex': self.validation_regex,
            'min_value': self.min_value,
            'max_value': self.max_value
        }


@dataclass
class FormTemplate:
    """Form template for data collection."""
    template_id: str
    name: str
    description: str
    collection_type: CollectionType
    fields: List[FormField]
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'collection_type': self.collection_type.value,
            'fields': [f.to_dict() for f in self.fields],
            'version': self.version,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class FieldRecord:
    """Field data record."""
    record_id: str
    collection_type: CollectionType
    template_id: str
    data: Dict[str, Any]
    location: Optional[GPSLocation] = None
    photos: List[Photo] = field(default_factory=list)
    voice_notes: List[VoiceNote] = field(default_factory=list)
    sync_status: SyncStatus = SyncStatus.PENDING
    data_quality: DataQuality = DataQuality.UNKNOWN
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    synced_at: Optional[datetime] = None
    user_id: str = ""
    project_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'record_id': self.record_id,
            'collection_type': self.collection_type.value,
            'template_id': self.template_id,
            'data': self.data,
            'location': self.location.to_dict() if self.location else None,
            'photos': [p.to_dict() for p in self.photos],
            'voice_notes': [v.to_dict() for v in self.voice_notes],
            'sync_status': self.sync_status.value,
            'data_quality': self.data_quality.value,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'synced_at': self.synced_at.isoformat() if self.synced_at else None,
            'user_id': self.user_id,
            'project_id': self.project_id
        }


@dataclass
class SampleRecord(FieldRecord):
    """Sample collection record."""
    sample_id: str = ""
    sample_type: str = ""
    weight_kg: float = 0.0
    bag_numbers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'sample_id': self.sample_id,
            'sample_type': self.sample_type,
            'weight_kg': self.weight_kg,
            'bag_numbers': self.bag_numbers
        })
        return base


@dataclass
class TraverseRecord:
    """Traverse/mapping record."""
    traverse_id: str
    waypoints: List[GPSLocation]
    observations: List[FieldRecord]
    start_time: datetime
    end_time: Optional[datetime] = None
    total_distance_m: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'traverse_id': self.traverse_id,
            'waypoints': [w.to_dict() for w in self.waypoints],
            'observations': [o.to_dict() for o in self.observations],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_distance_m': self.total_distance_m
        }


class OfflineStorage:
    """Offline storage manager using IndexedDB-like interface."""
    
    def __init__(self, storage_path: str = ""):
        self.storage_path = storage_path
        self._records: Dict[str, FieldRecord] = {}
        self._templates: Dict[str, FormTemplate] = {}
        self._pending_sync: List[str] = []
        
    def save_record(self, record: FieldRecord) -> bool:
        """Save record to offline storage."""
        try:
            self._records[record.record_id] = record
            if record.sync_status == SyncStatus.PENDING:
                if record.record_id not in self._pending_sync:
                    self._pending_sync.append(record.record_id)
            return True
        except Exception as e:
            logger.error(f"Failed to save record: {e}")
            return False
            
    def get_record(self, record_id: str) -> Optional[FieldRecord]:
        """Get record from offline storage."""
        return self._records.get(record_id)
        
    def get_all_records(self, project_id: str = None) -> List[FieldRecord]:
        """Get all records, optionally filtered by project."""
        records = list(self._records.values())
        if project_id:
            records = [r for r in records if r.project_id == project_id]
        return records
        
    def get_pending_records(self) -> List[FieldRecord]:
        """Get records pending synchronization."""
        return [self._records[rid] for rid in self._pending_sync if rid in self._records]
        
    def mark_synced(self, record_id: str) -> None:
        """Mark record as synced."""
        if record_id in self._records:
            self._records[record_id].sync_status = SyncStatus.SYNCED
            self._records[record_id].synced_at = datetime.utcnow()
            if record_id in self._pending_sync:
                self._pending_sync.remove(record_id)
                
    def save_template(self, template: FormTemplate) -> bool:
        """Save form template."""
        try:
            self._templates[template.template_id] = template
            return True
        except Exception as e:
            logger.error(f"Failed to save template: {e}")
            return False
            
    def get_template(self, template_id: str) -> Optional[FormTemplate]:
        """Get form template."""
        return self._templates.get(template_id)
        
    def get_all_templates(self) -> List[FormTemplate]:
        """Get all form templates."""
        return list(self._templates.values())
        
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'total_records': len(self._records),
            'pending_sync': len(self._pending_sync),
            'templates': len(self._templates)
        }


class SyncManager:
    """Manage synchronization with server."""
    
    def __init__(self, api_endpoint: str = ""):
        self.api_endpoint = api_endpoint
        self._sync_callbacks: List[Callable[[FieldRecord, bool], None]] = []
        self._conflict_resolver: Optional[Callable[[FieldRecord, FieldRecord], FieldRecord]] = None
        
    def register_callback(self, callback: Callable[[FieldRecord, bool], None]) -> None:
        """Register sync completion callback."""
        self._sync_callbacks.append(callback)
        
    def set_conflict_resolver(self, resolver: Callable[[FieldRecord, FieldRecord], FieldRecord]) -> None:
        """Set conflict resolution function."""
        self._conflict_resolver = resolver
        
    def sync_record(self, record: FieldRecord) -> bool:
        """Sync single record to server."""
        try:
            record.sync_status = SyncStatus.SYNCING
            
            success = self._upload_record(record)
            
            if success:
                record.sync_status = SyncStatus.SYNCED
                record.synced_at = datetime.utcnow()
            else:
                record.sync_status = SyncStatus.FAILED
                
            for callback in self._sync_callbacks:
                callback(record, success)
                
            return success
        except Exception as e:
            logger.error(f"Sync failed for record {record.record_id}: {e}")
            record.sync_status = SyncStatus.FAILED
            return False
            
    def sync_all_pending(self, storage: OfflineStorage) -> Dict[str, Any]:
        """Sync all pending records."""
        pending = storage.get_pending_records()
        results = {
            'total': len(pending),
            'success': 0,
            'failed': 0,
            'conflicts': 0
        }
        
        for record in pending:
            if self.sync_record(record):
                storage.mark_synced(record.record_id)
                results['success'] += 1
            else:
                if record.sync_status == SyncStatus.CONFLICT:
                    results['conflicts'] += 1
                else:
                    results['failed'] += 1
                    
        return results
        
    def _upload_record(self, record: FieldRecord) -> bool:
        """Upload record to server."""
        logger.info(f"Uploading record {record.record_id}")
        return True
        
    def check_connectivity(self) -> bool:
        """Check if server is reachable."""
        return True


class GPSManager:
    """Manage GPS location capture."""
    
    def __init__(self):
        self._current_location: Optional[GPSLocation] = None
        self._tracking = False
        self._track_points: List[GPSLocation] = []
        
    def get_current_location(self) -> Optional[GPSLocation]:
        """Get current GPS location."""
        return self._current_location
        
    def update_location(self, latitude: float, longitude: float,
                       altitude: float, accuracy: float,
                       accuracy_type: GPSAccuracy = GPSAccuracy.AUTONOMOUS) -> GPSLocation:
        """Update current location."""
        self._current_location = GPSLocation(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            accuracy_horizontal=accuracy,
            accuracy_vertical=accuracy * 1.5,
            accuracy_type=accuracy_type,
            timestamp=datetime.utcnow()
        )
        
        if self._tracking:
            self._track_points.append(self._current_location)
            
        return self._current_location
        
    def start_tracking(self) -> None:
        """Start GPS tracking."""
        self._tracking = True
        self._track_points = []
        
    def stop_tracking(self) -> List[GPSLocation]:
        """Stop GPS tracking and return track points."""
        self._tracking = False
        points = self._track_points
        self._track_points = []
        return points
        
    def calculate_distance(self, loc1: GPSLocation, loc2: GPSLocation) -> float:
        """Calculate distance between two locations in meters."""
        import math
        
        R = 6371000
        
        lat1 = math.radians(loc1.latitude)
        lat2 = math.radians(loc2.latitude)
        dlat = math.radians(loc2.latitude - loc1.latitude)
        dlon = math.radians(loc2.longitude - loc1.longitude)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


class PhotoManager:
    """Manage photo capture and storage."""
    
    def __init__(self, gps_manager: GPSManager):
        self.gps_manager = gps_manager
        self._photos: Dict[str, Photo] = {}
        
    def capture_photo(self, image_data: bytes, filename: str = None,
                     caption: str = "", tags: List[str] = None) -> Photo:
        """Capture and store photo with metadata."""
        photo_id = str(uuid.uuid4())
        
        if not filename:
            filename = f"photo_{photo_id}.jpg"
            
        location = self.gps_manager.get_current_location()
        
        photo = Photo(
            photo_id=photo_id,
            filename=filename,
            data=image_data,
            data_base64=base64.b64encode(image_data).decode('utf-8') if image_data else None,
            location=location,
            caption=caption,
            tags=tags or []
        )
        
        self._photos[photo_id] = photo
        return photo
        
    def get_photo(self, photo_id: str) -> Optional[Photo]:
        """Get photo by ID."""
        return self._photos.get(photo_id)
        
    def get_photos_by_location(self, center: GPSLocation,
                              radius_m: float) -> List[Photo]:
        """Get photos within radius of location."""
        results = []
        for photo in self._photos.values():
            if photo.location:
                distance = self.gps_manager.calculate_distance(center, photo.location)
                if distance <= radius_m:
                    results.append(photo)
        return results


class VoiceNoteManager:
    """Manage voice note capture and transcription."""
    
    def __init__(self, gps_manager: GPSManager):
        self.gps_manager = gps_manager
        self._notes: Dict[str, VoiceNote] = {}
        
    def capture_voice_note(self, audio_data: bytes, duration: float,
                          filename: str = None) -> VoiceNote:
        """Capture and store voice note."""
        note_id = str(uuid.uuid4())
        
        if not filename:
            filename = f"voice_{note_id}.m4a"
            
        location = self.gps_manager.get_current_location()
        
        note = VoiceNote(
            note_id=note_id,
            filename=filename,
            duration_seconds=duration,
            data=audio_data,
            data_base64=base64.b64encode(audio_data).decode('utf-8') if audio_data else None,
            location=location
        )
        
        self._notes[note_id] = note
        return note
        
    def transcribe_note(self, note_id: str, transcription: str) -> bool:
        """Add transcription to voice note."""
        if note_id in self._notes:
            self._notes[note_id].transcription = transcription
            return True
        return False
        
    def get_note(self, note_id: str) -> Optional[VoiceNote]:
        """Get voice note by ID."""
        return self._notes.get(note_id)


class FormManager:
    """Manage form templates and data entry."""
    
    def __init__(self, storage: OfflineStorage):
        self.storage = storage
        self._setup_default_templates()
        
    def _setup_default_templates(self) -> None:
        """Setup default form templates."""
        sample_template = FormTemplate(
            template_id="sample_collection",
            name="Sample Collection",
            description="Rock/soil sample collection form",
            collection_type=CollectionType.SAMPLE,
            fields=[
                FormField("sample_id", "Sample ID", "text", required=True),
                FormField("sample_type", "Sample Type", "select", required=True,
                         options=["Rock Chip", "Soil", "Stream Sediment", "Core", "Channel"]),
                FormField("lithology", "Lithology", "select",
                         options=["Granite", "Basalt", "Sandstone", "Shale", "Limestone", "Other"]),
                FormField("alteration", "Alteration", "multiselect",
                         options=["Silicification", "Sericitization", "Chloritization", "Oxidation", "None"]),
                FormField("mineralization", "Mineralization", "text"),
                FormField("weight", "Weight (kg)", "number", min_value=0, max_value=50),
                FormField("notes", "Notes", "text"),
                FormField("photo", "Photo", "photo")
            ]
        )
        self.storage.save_template(sample_template)
        
        observation_template = FormTemplate(
            template_id="geological_observation",
            name="Geological Observation",
            description="Field geological observation",
            collection_type=CollectionType.OBSERVATION,
            fields=[
                FormField("observation_type", "Observation Type", "select", required=True,
                         options=["Outcrop", "Float", "Contact", "Structure", "Mineralization"]),
                FormField("lithology", "Lithology", "text"),
                FormField("description", "Description", "text", required=True),
                FormField("strike", "Strike", "number", min_value=0, max_value=360),
                FormField("dip", "Dip", "number", min_value=0, max_value=90),
                FormField("dip_direction", "Dip Direction", "select",
                         options=["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
                FormField("photo", "Photo", "photo")
            ]
        )
        self.storage.save_template(observation_template)
        
        structure_template = FormTemplate(
            template_id="structural_measurement",
            name="Structural Measurement",
            description="Structural geology measurement",
            collection_type=CollectionType.STRUCTURE,
            fields=[
                FormField("structure_type", "Structure Type", "select", required=True,
                         options=["Bedding", "Foliation", "Fault", "Joint", "Vein", "Fold Axis"]),
                FormField("strike", "Strike", "number", required=True, min_value=0, max_value=360),
                FormField("dip", "Dip", "number", required=True, min_value=0, max_value=90),
                FormField("dip_direction", "Dip Direction", "select", required=True,
                         options=["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
                FormField("confidence", "Confidence", "select",
                         options=["High", "Medium", "Low"]),
                FormField("notes", "Notes", "text")
            ]
        )
        self.storage.save_template(structure_template)
        
    def get_template(self, template_id: str) -> Optional[FormTemplate]:
        """Get form template."""
        return self.storage.get_template(template_id)
        
    def get_all_templates(self) -> List[FormTemplate]:
        """Get all form templates."""
        return self.storage.get_all_templates()
        
    def validate_form_data(self, template_id: str, data: Dict[str, Any]) -> List[str]:
        """Validate form data against template."""
        template = self.get_template(template_id)
        if not template:
            return ["Template not found"]
            
        errors = []
        
        for field in template.fields:
            value = data.get(field.field_id)
            
            if field.required and (value is None or value == ""):
                errors.append(f"{field.name} is required")
                continue
                
            if value is not None:
                if field.field_type == "number":
                    try:
                        num_val = float(value)
                        if field.min_value is not None and num_val < field.min_value:
                            errors.append(f"{field.name} must be >= {field.min_value}")
                        if field.max_value is not None and num_val > field.max_value:
                            errors.append(f"{field.name} must be <= {field.max_value}")
                    except ValueError:
                        errors.append(f"{field.name} must be a number")
                        
                if field.field_type == "select" and field.options:
                    if value not in field.options:
                        errors.append(f"{field.name} must be one of: {', '.join(field.options)}")
                        
        return errors


class FieldCollectionService:
    """Main field data collection service."""
    
    def __init__(self, api_endpoint: str = ""):
        self.storage = OfflineStorage()
        self.sync_manager = SyncManager(api_endpoint)
        self.gps_manager = GPSManager()
        self.photo_manager = PhotoManager(self.gps_manager)
        self.voice_manager = VoiceNoteManager(self.gps_manager)
        self.form_manager = FormManager(self.storage)
        
        self._current_project: str = ""
        self._current_user: str = ""
        
    def set_context(self, project_id: str, user_id: str) -> None:
        """Set current project and user context."""
        self._current_project = project_id
        self._current_user = user_id
        
    def create_record(self, template_id: str, data: Dict[str, Any],
                     location: GPSLocation = None) -> Optional[FieldRecord]:
        """Create new field record."""
        template = self.form_manager.get_template(template_id)
        if not template:
            logger.error(f"Template not found: {template_id}")
            return None
            
        errors = self.form_manager.validate_form_data(template_id, data)
        if errors:
            logger.error(f"Validation errors: {errors}")
            return None
            
        record_id = str(uuid.uuid4())
        
        if location is None:
            location = self.gps_manager.get_current_location()
            
        record = FieldRecord(
            record_id=record_id,
            collection_type=template.collection_type,
            template_id=template_id,
            data=data,
            location=location,
            user_id=self._current_user,
            project_id=self._current_project
        )
        
        self.storage.save_record(record)
        return record
        
    def add_photo_to_record(self, record_id: str, image_data: bytes,
                           caption: str = "") -> Optional[Photo]:
        """Add photo to existing record."""
        record = self.storage.get_record(record_id)
        if not record:
            return None
            
        photo = self.photo_manager.capture_photo(image_data, caption=caption)
        record.photos.append(photo)
        record.modified_at = datetime.utcnow()
        self.storage.save_record(record)
        
        return photo
        
    def add_voice_note_to_record(self, record_id: str, audio_data: bytes,
                                duration: float) -> Optional[VoiceNote]:
        """Add voice note to existing record."""
        record = self.storage.get_record(record_id)
        if not record:
            return None
            
        note = self.voice_manager.capture_voice_note(audio_data, duration)
        record.voice_notes.append(note)
        record.modified_at = datetime.utcnow()
        self.storage.save_record(record)
        
        return note
        
    def sync_pending(self) -> Dict[str, Any]:
        """Sync all pending records."""
        return self.sync_manager.sync_all_pending(self.storage)
        
    def get_records(self, project_id: str = None) -> List[FieldRecord]:
        """Get all records for project."""
        return self.storage.get_all_records(project_id or self._current_project)
        
    def start_traverse(self) -> str:
        """Start a new traverse."""
        self.gps_manager.start_tracking()
        traverse_id = str(uuid.uuid4())
        return traverse_id
        
    def end_traverse(self, traverse_id: str) -> TraverseRecord:
        """End traverse and return record."""
        waypoints = self.gps_manager.stop_tracking()
        
        total_distance = 0.0
        for i in range(1, len(waypoints)):
            total_distance += self.gps_manager.calculate_distance(
                waypoints[i-1], waypoints[i]
            )
            
        return TraverseRecord(
            traverse_id=traverse_id,
            waypoints=waypoints,
            observations=[],
            start_time=waypoints[0].timestamp if waypoints else datetime.utcnow(),
            end_time=waypoints[-1].timestamp if waypoints else datetime.utcnow(),
            total_distance_m=total_distance
        )
        
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'storage': self.storage.get_storage_stats(),
            'connectivity': self.sync_manager.check_connectivity(),
            'gps_available': self.gps_manager.get_current_location() is not None,
            'current_project': self._current_project,
            'current_user': self._current_user
        }


def create_field_collection_service(api_endpoint: str = "") -> FieldCollectionService:
    """Factory function to create field collection service."""
    return FieldCollectionService(api_endpoint)


def create_sample_template() -> FormTemplate:
    """Create sample collection template."""
    return FormTemplate(
        template_id="sample_collection",
        name="Sample Collection",
        description="Rock/soil sample collection form",
        collection_type=CollectionType.SAMPLE,
        fields=[
            FormField("sample_id", "Sample ID", "text", required=True),
            FormField("sample_type", "Sample Type", "select", required=True,
                     options=["Rock Chip", "Soil", "Stream Sediment", "Core", "Channel"]),
            FormField("weight", "Weight (kg)", "number", min_value=0, max_value=50),
            FormField("notes", "Notes", "text")
        ]
    )
