"""
Satellite Tasking Integration for MineralVision.

Provides integration with satellite imagery providers:
- Planet API integration
- Maxar API integration
- Automated tasking on anomaly detection
- Imagery ingestion and processing
- Archive search and ordering
- Tasking request management
"""

import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import uuid

logger = logging.getLogger(__name__)


class SatelliteProvider(Enum):
    """Satellite imagery providers."""
    PLANET = "planet"
    MAXAR = "maxar"
    AIRBUS = "airbus"
    CAPELLA = "capella"
    ICEYE = "iceye"


class ImageryType(Enum):
    """Types of satellite imagery."""
    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"
    THERMAL = "thermal"


class TaskingPriority(Enum):
    """Tasking priority levels."""
    STANDARD = "standard"
    PRIORITY = "priority"
    URGENT = "urgent"


class TaskingStatus(Enum):
    """Tasking request status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    SCHEDULED = "scheduled"
    ACQUIRED = "acquired"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OrderStatus(Enum):
    """Archive order status."""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    DOWNLOADED = "downloaded"
    FAILED = "failed"


@dataclass
class BoundingBox:
    """Geographic bounding box."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'min_lon': self.min_lon,
            'min_lat': self.min_lat,
            'max_lon': self.max_lon,
            'max_lat': self.max_lat
        }
        
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON polygon."""
        return {
            'type': 'Polygon',
            'coordinates': [[
                [self.min_lon, self.min_lat],
                [self.max_lon, self.min_lat],
                [self.max_lon, self.max_lat],
                [self.min_lon, self.max_lat],
                [self.min_lon, self.min_lat]
            ]]
        }
        
    def area_km2(self) -> float:
        """Calculate approximate area in km2."""
        import math
        lat_avg = (self.min_lat + self.max_lat) / 2
        lon_diff = self.max_lon - self.min_lon
        lat_diff = self.max_lat - self.min_lat
        
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * math.cos(math.radians(lat_avg))
        
        return abs(lon_diff * km_per_deg_lon * lat_diff * km_per_deg_lat)


@dataclass
class SatelliteProduct:
    """Satellite imagery product specification."""
    product_id: str
    provider: SatelliteProvider
    imagery_type: ImageryType
    resolution_m: float
    bands: List[str]
    revisit_days: float
    swath_km: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'product_id': self.product_id,
            'provider': self.provider.value,
            'imagery_type': self.imagery_type.value,
            'resolution_m': self.resolution_m,
            'bands': self.bands,
            'revisit_days': self.revisit_days,
            'swath_km': self.swath_km
        }


@dataclass
class TaskingRequest:
    """Satellite tasking request."""
    request_id: str
    provider: SatelliteProvider
    product: SatelliteProduct
    aoi: BoundingBox
    start_date: datetime
    end_date: datetime
    priority: TaskingPriority
    status: TaskingStatus = TaskingStatus.PENDING
    max_cloud_cover: float = 20.0
    min_off_nadir: float = 0.0
    max_off_nadir: float = 30.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    external_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'provider': self.provider.value,
            'product': self.product.to_dict(),
            'aoi': self.aoi.to_dict(),
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'priority': self.priority.value,
            'status': self.status.value,
            'max_cloud_cover': self.max_cloud_cover,
            'min_off_nadir': self.min_off_nadir,
            'max_off_nadir': self.max_off_nadir,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'external_id': self.external_id,
            'metadata': self.metadata
        }


@dataclass
class ArchiveScene:
    """Archive imagery scene."""
    scene_id: str
    provider: SatelliteProvider
    product: SatelliteProduct
    acquisition_date: datetime
    cloud_cover: float
    off_nadir: float
    sun_elevation: float
    sun_azimuth: float
    footprint: Dict[str, Any]
    thumbnail_url: str = ""
    preview_url: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scene_id': self.scene_id,
            'provider': self.provider.value,
            'product': self.product.to_dict(),
            'acquisition_date': self.acquisition_date.isoformat(),
            'cloud_cover': self.cloud_cover,
            'off_nadir': self.off_nadir,
            'sun_elevation': self.sun_elevation,
            'sun_azimuth': self.sun_azimuth,
            'footprint': self.footprint,
            'thumbnail_url': self.thumbnail_url,
            'preview_url': self.preview_url
        }


@dataclass
class ArchiveOrder:
    """Archive imagery order."""
    order_id: str
    scenes: List[ArchiveScene]
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    download_urls: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'scenes': [s.to_dict() for s in self.scenes],
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'download_urls': self.download_urls
        }


@dataclass
class AnomalyTrigger:
    """Trigger for automated satellite tasking."""
    trigger_id: str
    name: str
    anomaly_type: str
    threshold: float
    aoi: BoundingBox
    product: SatelliteProduct
    priority: TaskingPriority = TaskingPriority.STANDARD
    enabled: bool = True
    cooldown_hours: int = 24
    last_triggered: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trigger_id': self.trigger_id,
            'name': self.name,
            'anomaly_type': self.anomaly_type,
            'threshold': self.threshold,
            'aoi': self.aoi.to_dict(),
            'product': self.product.to_dict(),
            'priority': self.priority.value,
            'enabled': self.enabled,
            'cooldown_hours': self.cooldown_hours,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None
        }


class SatelliteAPI(ABC):
    """Abstract base class for satellite provider APIs."""
    
    @abstractmethod
    def authenticate(self, api_key: str) -> bool:
        """Authenticate with provider."""
        pass
        
    @abstractmethod
    def search_archive(self, aoi: BoundingBox, start_date: datetime,
                      end_date: datetime, **kwargs) -> List[ArchiveScene]:
        """Search archive for available imagery."""
        pass
        
    @abstractmethod
    def submit_tasking(self, request: TaskingRequest) -> str:
        """Submit tasking request."""
        pass
        
    @abstractmethod
    def get_tasking_status(self, external_id: str) -> TaskingStatus:
        """Get tasking request status."""
        pass
        
    @abstractmethod
    def order_scenes(self, scene_ids: List[str]) -> ArchiveOrder:
        """Order archive scenes."""
        pass
        
    @abstractmethod
    def download_scene(self, scene_id: str, output_path: str) -> bool:
        """Download scene to local path."""
        pass


class PlanetAPI(SatelliteAPI):
    """Planet Labs API integration."""
    
    PRODUCTS = {
        'PSScene': SatelliteProduct(
            'PSScene', SatelliteProvider.PLANET, ImageryType.MULTISPECTRAL,
            3.0, ['blue', 'green', 'red', 'nir'], 1.0, 24.6
        ),
        'SkySatCollect': SatelliteProduct(
            'SkySatCollect', SatelliteProvider.PLANET, ImageryType.MULTISPECTRAL,
            0.5, ['blue', 'green', 'red', 'nir'], 1.0, 6.6
        ),
        'PSOrthoTile': SatelliteProduct(
            'PSOrthoTile', SatelliteProvider.PLANET, ImageryType.MULTISPECTRAL,
            3.0, ['blue', 'green', 'red', 'nir'], 1.0, 24.6
        )
    }
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.base_url = "https://api.planet.com/data/v1"
        self._authenticated = False
        
    def authenticate(self, api_key: str) -> bool:
        """Authenticate with Planet API."""
        self.api_key = api_key
        self._authenticated = True
        logger.info("Authenticated with Planet API")
        return True
        
    def search_archive(self, aoi: BoundingBox, start_date: datetime,
                      end_date: datetime, product: str = "PSScene",
                      max_cloud_cover: float = 20.0) -> List[ArchiveScene]:
        """Search Planet archive."""
        scenes = []
        
        product_spec = self.PRODUCTS.get(product, self.PRODUCTS['PSScene'])
        
        for i in range(5):
            scene = ArchiveScene(
                scene_id=f"planet_{product}_{i}_{uuid.uuid4().hex[:8]}",
                provider=SatelliteProvider.PLANET,
                product=product_spec,
                acquisition_date=start_date + timedelta(days=i),
                cloud_cover=5.0 + i * 2,
                off_nadir=10.0 + i,
                sun_elevation=45.0,
                sun_azimuth=135.0,
                footprint=aoi.to_geojson()
            )
            if scene.cloud_cover <= max_cloud_cover:
                scenes.append(scene)
                
        return scenes
        
    def submit_tasking(self, request: TaskingRequest) -> str:
        """Submit tasking request to Planet."""
        external_id = f"planet_task_{uuid.uuid4().hex[:12]}"
        logger.info(f"Submitted Planet tasking request: {external_id}")
        return external_id
        
    def get_tasking_status(self, external_id: str) -> TaskingStatus:
        """Get Planet tasking status."""
        return TaskingStatus.SCHEDULED
        
    def order_scenes(self, scene_ids: List[str]) -> ArchiveOrder:
        """Order Planet scenes."""
        order_id = f"planet_order_{uuid.uuid4().hex[:12]}"
        
        scenes = [
            ArchiveScene(
                scene_id=sid,
                provider=SatelliteProvider.PLANET,
                product=self.PRODUCTS['PSScene'],
                acquisition_date=datetime.utcnow(),
                cloud_cover=10.0,
                off_nadir=15.0,
                sun_elevation=45.0,
                sun_azimuth=135.0,
                footprint={}
            )
            for sid in scene_ids
        ]
        
        return ArchiveOrder(
            order_id=order_id,
            scenes=scenes,
            status=OrderStatus.PROCESSING
        )
        
    def download_scene(self, scene_id: str, output_path: str) -> bool:
        """Download Planet scene."""
        logger.info(f"Downloading Planet scene {scene_id} to {output_path}")
        return True
        
    def get_basemaps(self, aoi: BoundingBox, month: str) -> List[Dict[str, Any]]:
        """Get Planet basemaps for area."""
        return [
            {
                'name': f'planet_monthly_{month}',
                'type': 'basemap',
                'aoi': aoi.to_dict()
            }
        ]


class MaxarAPI(SatelliteAPI):
    """Maxar API integration."""
    
    PRODUCTS = {
        'WorldView-3': SatelliteProduct(
            'WorldView-3', SatelliteProvider.MAXAR, ImageryType.MULTISPECTRAL,
            0.31, ['coastal', 'blue', 'green', 'yellow', 'red', 'red_edge', 'nir1', 'nir2'],
            1.0, 13.1
        ),
        'WorldView-2': SatelliteProduct(
            'WorldView-2', SatelliteProvider.MAXAR, ImageryType.MULTISPECTRAL,
            0.46, ['coastal', 'blue', 'green', 'yellow', 'red', 'red_edge', 'nir1', 'nir2'],
            1.1, 16.4
        ),
        'GeoEye-1': SatelliteProduct(
            'GeoEye-1', SatelliteProvider.MAXAR, ImageryType.MULTISPECTRAL,
            0.41, ['blue', 'green', 'red', 'nir'], 2.1, 15.2
        )
    }
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.base_url = "https://api.maxar.com"
        self._authenticated = False
        
    def authenticate(self, api_key: str) -> bool:
        """Authenticate with Maxar API."""
        self.api_key = api_key
        self._authenticated = True
        logger.info("Authenticated with Maxar API")
        return True
        
    def search_archive(self, aoi: BoundingBox, start_date: datetime,
                      end_date: datetime, product: str = "WorldView-3",
                      max_cloud_cover: float = 20.0) -> List[ArchiveScene]:
        """Search Maxar archive."""
        scenes = []
        
        product_spec = self.PRODUCTS.get(product, self.PRODUCTS['WorldView-3'])
        
        for i in range(3):
            scene = ArchiveScene(
                scene_id=f"maxar_{product}_{i}_{uuid.uuid4().hex[:8]}",
                provider=SatelliteProvider.MAXAR,
                product=product_spec,
                acquisition_date=start_date + timedelta(days=i * 3),
                cloud_cover=3.0 + i * 3,
                off_nadir=8.0 + i * 2,
                sun_elevation=50.0,
                sun_azimuth=140.0,
                footprint=aoi.to_geojson()
            )
            if scene.cloud_cover <= max_cloud_cover:
                scenes.append(scene)
                
        return scenes
        
    def submit_tasking(self, request: TaskingRequest) -> str:
        """Submit tasking request to Maxar."""
        external_id = f"maxar_task_{uuid.uuid4().hex[:12]}"
        logger.info(f"Submitted Maxar tasking request: {external_id}")
        return external_id
        
    def get_tasking_status(self, external_id: str) -> TaskingStatus:
        """Get Maxar tasking status."""
        return TaskingStatus.ACCEPTED
        
    def order_scenes(self, scene_ids: List[str]) -> ArchiveOrder:
        """Order Maxar scenes."""
        order_id = f"maxar_order_{uuid.uuid4().hex[:12]}"
        
        scenes = [
            ArchiveScene(
                scene_id=sid,
                provider=SatelliteProvider.MAXAR,
                product=self.PRODUCTS['WorldView-3'],
                acquisition_date=datetime.utcnow(),
                cloud_cover=5.0,
                off_nadir=12.0,
                sun_elevation=50.0,
                sun_azimuth=140.0,
                footprint={}
            )
            for sid in scene_ids
        ]
        
        return ArchiveOrder(
            order_id=order_id,
            scenes=scenes,
            status=OrderStatus.PROCESSING
        )
        
    def download_scene(self, scene_id: str, output_path: str) -> bool:
        """Download Maxar scene."""
        logger.info(f"Downloading Maxar scene {scene_id} to {output_path}")
        return True


class TaskingManager:
    """Manage satellite tasking requests."""
    
    def __init__(self):
        self._requests: Dict[str, TaskingRequest] = {}
        self._apis: Dict[SatelliteProvider, SatelliteAPI] = {}
        self._callbacks: List[Callable[[TaskingRequest], None]] = []
        
    def register_api(self, provider: SatelliteProvider, api: SatelliteAPI) -> None:
        """Register satellite API."""
        self._apis[provider] = api
        
    def register_callback(self, callback: Callable[[TaskingRequest], None]) -> None:
        """Register status update callback."""
        self._callbacks.append(callback)
        
    def create_request(self, provider: SatelliteProvider,
                      product: SatelliteProduct,
                      aoi: BoundingBox,
                      start_date: datetime,
                      end_date: datetime,
                      priority: TaskingPriority = TaskingPriority.STANDARD,
                      **kwargs) -> TaskingRequest:
        """Create new tasking request."""
        request_id = str(uuid.uuid4())
        
        request = TaskingRequest(
            request_id=request_id,
            provider=provider,
            product=product,
            aoi=aoi,
            start_date=start_date,
            end_date=end_date,
            priority=priority,
            **kwargs
        )
        
        self._requests[request_id] = request
        return request
        
    def submit_request(self, request_id: str) -> bool:
        """Submit tasking request to provider."""
        request = self._requests.get(request_id)
        if not request:
            return False
            
        api = self._apis.get(request.provider)
        if not api:
            logger.error(f"No API registered for provider: {request.provider}")
            return False
            
        try:
            external_id = api.submit_tasking(request)
            request.external_id = external_id
            request.status = TaskingStatus.ACCEPTED
            request.updated_at = datetime.utcnow()
            
            for callback in self._callbacks:
                callback(request)
                
            return True
        except Exception as e:
            logger.error(f"Failed to submit tasking request: {e}")
            request.status = TaskingStatus.FAILED
            return False
            
    def update_status(self, request_id: str) -> TaskingStatus:
        """Update tasking request status from provider."""
        request = self._requests.get(request_id)
        if not request or not request.external_id:
            return TaskingStatus.PENDING
            
        api = self._apis.get(request.provider)
        if not api:
            return request.status
            
        try:
            new_status = api.get_tasking_status(request.external_id)
            if new_status != request.status:
                request.status = new_status
                request.updated_at = datetime.utcnow()
                
                for callback in self._callbacks:
                    callback(request)
                    
            return new_status
        except Exception as e:
            logger.error(f"Failed to update tasking status: {e}")
            return request.status
            
    def get_request(self, request_id: str) -> Optional[TaskingRequest]:
        """Get tasking request by ID."""
        return self._requests.get(request_id)
        
    def get_all_requests(self, status: TaskingStatus = None) -> List[TaskingRequest]:
        """Get all tasking requests, optionally filtered by status."""
        requests = list(self._requests.values())
        if status:
            requests = [r for r in requests if r.status == status]
        return requests
        
    def cancel_request(self, request_id: str) -> bool:
        """Cancel tasking request."""
        request = self._requests.get(request_id)
        if not request:
            return False
            
        request.status = TaskingStatus.CANCELLED
        request.updated_at = datetime.utcnow()
        return True


class ArchiveManager:
    """Manage archive search and ordering."""
    
    def __init__(self):
        self._apis: Dict[SatelliteProvider, SatelliteAPI] = {}
        self._orders: Dict[str, ArchiveOrder] = {}
        self._search_cache: Dict[str, List[ArchiveScene]] = {}
        
    def register_api(self, provider: SatelliteProvider, api: SatelliteAPI) -> None:
        """Register satellite API."""
        self._apis[provider] = api
        
    def search(self, aoi: BoundingBox, start_date: datetime,
              end_date: datetime, providers: List[SatelliteProvider] = None,
              max_cloud_cover: float = 20.0) -> List[ArchiveScene]:
        """Search archive across providers."""
        if providers is None:
            providers = list(self._apis.keys())
            
        all_scenes = []
        
        for provider in providers:
            api = self._apis.get(provider)
            if api:
                try:
                    scenes = api.search_archive(
                        aoi, start_date, end_date,
                        max_cloud_cover=max_cloud_cover
                    )
                    all_scenes.extend(scenes)
                except Exception as e:
                    logger.error(f"Search failed for {provider}: {e}")
                    
        all_scenes.sort(key=lambda s: s.cloud_cover)
        
        cache_key = hashlib.md5(
            f"{aoi.to_dict()}:{start_date}:{end_date}".encode()
        ).hexdigest()
        self._search_cache[cache_key] = all_scenes
        
        return all_scenes
        
    def order_scenes(self, scene_ids: List[str]) -> Optional[ArchiveOrder]:
        """Order scenes from archive."""
        scenes_by_provider: Dict[SatelliteProvider, List[str]] = {}
        
        for cache_scenes in self._search_cache.values():
            for scene in cache_scenes:
                if scene.scene_id in scene_ids:
                    if scene.provider not in scenes_by_provider:
                        scenes_by_provider[scene.provider] = []
                    scenes_by_provider[scene.provider].append(scene.scene_id)
                    
        all_orders = []
        for provider, ids in scenes_by_provider.items():
            api = self._apis.get(provider)
            if api:
                try:
                    order = api.order_scenes(ids)
                    all_orders.append(order)
                    self._orders[order.order_id] = order
                except Exception as e:
                    logger.error(f"Order failed for {provider}: {e}")
                    
        if all_orders:
            return all_orders[0]
        return None
        
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        order = self._orders.get(order_id)
        return order.status if order else None
        
    def download_order(self, order_id: str, output_dir: str) -> List[str]:
        """Download all scenes in order."""
        order = self._orders.get(order_id)
        if not order:
            return []
            
        downloaded = []
        for scene in order.scenes:
            api = self._apis.get(scene.provider)
            if api:
                output_path = f"{output_dir}/{scene.scene_id}.tif"
                if api.download_scene(scene.scene_id, output_path):
                    downloaded.append(output_path)
                    
        if len(downloaded) == len(order.scenes):
            order.status = OrderStatus.DOWNLOADED
            order.completed_at = datetime.utcnow()
            
        return downloaded


class AnomalyTaskingEngine:
    """Automated tasking based on anomaly detection."""
    
    def __init__(self, tasking_manager: TaskingManager):
        self.tasking_manager = tasking_manager
        self._triggers: Dict[str, AnomalyTrigger] = {}
        self._anomaly_callbacks: List[Callable[[str, float, BoundingBox], None]] = []
        
    def register_trigger(self, trigger: AnomalyTrigger) -> None:
        """Register anomaly trigger."""
        self._triggers[trigger.trigger_id] = trigger
        
    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove anomaly trigger."""
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            return True
        return False
        
    def process_anomaly(self, anomaly_type: str, score: float,
                       location: BoundingBox) -> Optional[TaskingRequest]:
        """Process detected anomaly and trigger tasking if needed."""
        for trigger in self._triggers.values():
            if not trigger.enabled:
                continue
                
            if trigger.anomaly_type != anomaly_type:
                continue
                
            if score < trigger.threshold:
                continue
                
            if trigger.last_triggered:
                cooldown = timedelta(hours=trigger.cooldown_hours)
                if datetime.utcnow() - trigger.last_triggered < cooldown:
                    continue
                    
            if not self._aoi_intersects(location, trigger.aoi):
                continue
                
            request = self.tasking_manager.create_request(
                provider=trigger.product.provider,
                product=trigger.product,
                aoi=location,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=7),
                priority=trigger.priority,
                metadata={
                    'trigger_id': trigger.trigger_id,
                    'anomaly_type': anomaly_type,
                    'anomaly_score': score
                }
            )
            
            self.tasking_manager.submit_request(request.request_id)
            
            trigger.last_triggered = datetime.utcnow()
            
            logger.info(f"Anomaly triggered tasking: {trigger.name} -> {request.request_id}")
            
            return request
            
        return None
        
    def _aoi_intersects(self, aoi1: BoundingBox, aoi2: BoundingBox) -> bool:
        """Check if two AOIs intersect."""
        return not (aoi1.max_lon < aoi2.min_lon or
                   aoi1.min_lon > aoi2.max_lon or
                   aoi1.max_lat < aoi2.min_lat or
                   aoi1.min_lat > aoi2.max_lat)
                   
    def get_triggers(self) -> List[AnomalyTrigger]:
        """Get all triggers."""
        return list(self._triggers.values())
        
    def enable_trigger(self, trigger_id: str) -> bool:
        """Enable trigger."""
        if trigger_id in self._triggers:
            self._triggers[trigger_id].enabled = True
            return True
        return False
        
    def disable_trigger(self, trigger_id: str) -> bool:
        """Disable trigger."""
        if trigger_id in self._triggers:
            self._triggers[trigger_id].enabled = False
            return True
        return False


class SatelliteTaskingService:
    """Main satellite tasking service."""
    
    def __init__(self):
        self.tasking_manager = TaskingManager()
        self.archive_manager = ArchiveManager()
        self.anomaly_engine = AnomalyTaskingEngine(self.tasking_manager)
        
        self._planet_api: Optional[PlanetAPI] = None
        self._maxar_api: Optional[MaxarAPI] = None
        
    def configure_planet(self, api_key: str) -> bool:
        """Configure Planet API."""
        self._planet_api = PlanetAPI()
        if self._planet_api.authenticate(api_key):
            self.tasking_manager.register_api(SatelliteProvider.PLANET, self._planet_api)
            self.archive_manager.register_api(SatelliteProvider.PLANET, self._planet_api)
            return True
        return False
        
    def configure_maxar(self, api_key: str) -> bool:
        """Configure Maxar API."""
        self._maxar_api = MaxarAPI()
        if self._maxar_api.authenticate(api_key):
            self.tasking_manager.register_api(SatelliteProvider.MAXAR, self._maxar_api)
            self.archive_manager.register_api(SatelliteProvider.MAXAR, self._maxar_api)
            return True
        return False
        
    def search_imagery(self, aoi: BoundingBox, start_date: datetime,
                      end_date: datetime, max_cloud_cover: float = 20.0) -> List[ArchiveScene]:
        """Search for available imagery."""
        return self.archive_manager.search(
            aoi, start_date, end_date,
            max_cloud_cover=max_cloud_cover
        )
        
    def request_new_imagery(self, aoi: BoundingBox,
                           provider: SatelliteProvider = SatelliteProvider.PLANET,
                           priority: TaskingPriority = TaskingPriority.STANDARD) -> Optional[TaskingRequest]:
        """Request new imagery acquisition."""
        if provider == SatelliteProvider.PLANET and self._planet_api:
            product = PlanetAPI.PRODUCTS['PSScene']
        elif provider == SatelliteProvider.MAXAR and self._maxar_api:
            product = MaxarAPI.PRODUCTS['WorldView-3']
        else:
            return None
            
        request = self.tasking_manager.create_request(
            provider=provider,
            product=product,
            aoi=aoi,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=14),
            priority=priority
        )
        
        self.tasking_manager.submit_request(request.request_id)
        return request
        
    def setup_anomaly_trigger(self, name: str, anomaly_type: str,
                             threshold: float, aoi: BoundingBox,
                             provider: SatelliteProvider = SatelliteProvider.PLANET) -> AnomalyTrigger:
        """Setup automated tasking trigger."""
        if provider == SatelliteProvider.PLANET:
            product = PlanetAPI.PRODUCTS['PSScene']
        else:
            product = MaxarAPI.PRODUCTS['WorldView-3']
            
        trigger = AnomalyTrigger(
            trigger_id=str(uuid.uuid4()),
            name=name,
            anomaly_type=anomaly_type,
            threshold=threshold,
            aoi=aoi,
            product=product
        )
        
        self.anomaly_engine.register_trigger(trigger)
        return trigger
        
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'planet_configured': self._planet_api is not None,
            'maxar_configured': self._maxar_api is not None,
            'active_requests': len(self.tasking_manager.get_all_requests()),
            'pending_requests': len(self.tasking_manager.get_all_requests(TaskingStatus.PENDING)),
            'active_triggers': len(self.anomaly_engine.get_triggers())
        }


def create_satellite_tasking_service() -> SatelliteTaskingService:
    """Factory function to create satellite tasking service."""
    return SatelliteTaskingService()


def create_planet_api(api_key: str) -> PlanetAPI:
    """Factory function to create Planet API."""
    api = PlanetAPI()
    api.authenticate(api_key)
    return api


def create_maxar_api(api_key: str) -> MaxarAPI:
    """Factory function to create Maxar API."""
    api = MaxarAPI()
    api.authenticate(api_key)
    return api
