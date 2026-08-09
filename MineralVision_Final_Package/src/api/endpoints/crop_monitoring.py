"""
Crop Monitoring API Endpoints for MineralVision.

REST API endpoints for:
- Vegetation index processing
- Field management
- Weather integration
- VRA map generation
- Alert management
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from pydantic import BaseModel, Field
import json
import numpy as np

from ..crop_monitoring import (
    # Vegetation Indices
    create_crop_monitoring_service,
    create_synthetic_spectral_bands,
    VegetationIndexType,
    
    # Field Management
    create_field_manager,
    create_sample_fields,
    BoundaryFormat,
    FieldStatus,
    
    # Weather
    create_weather_service,
    
    # VRA Maps
    create_vra_service,
    VRAMapType,
    ExportFormat,
    
    # Alerts
    create_alert_service,
    AlertStatus,
    AlertCategory,
    AlertSeverity
)

router = APIRouter(prefix="/crop-monitoring", tags=["Crop Monitoring"])

# Initialize services
crop_service = create_crop_monitoring_service("oil_palm")
field_manager = create_field_manager()
weather_service = create_weather_service()
vra_service = create_vra_service()
alert_service = create_alert_service()

# Create sample data
sample_farm = create_sample_fields(field_manager, "Demo Farm")


# ============== Pydantic Models ==============

class CoordinateModel(BaseModel):
    longitude: float
    latitude: float
    elevation: Optional[float] = None


class FieldBoundaryModel(BaseModel):
    coordinates: List[CoordinateModel]
    holes: Optional[List[List[CoordinateModel]]] = None


class CreateFieldRequest(BaseModel):
    name: str
    boundary: FieldBoundaryModel
    farm_id: Optional[str] = ""
    crop_type: Optional[str] = ""
    status: Optional[str] = "growing"


class UpdateFieldRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    crop_type: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


class VegetationAnalysisRequest(BaseModel):
    field_id: str
    scene_id: Optional[str] = "synthetic"
    indices: Optional[List[str]] = None  # List of index types to calculate


class WeatherRequest(BaseModel):
    latitude: float
    longitude: float
    days: Optional[int] = 14


class VRAMapRequest(BaseModel):
    field_id: str
    map_type: str  # sowing, nitrogen, pk
    crop_type: str
    target_value: float  # density for sowing, yield for fertilizer
    num_zones: Optional[int] = 3


class AlertRuleRequest(BaseModel):
    name: str
    category: str
    metric: str
    operator: str
    threshold: float
    severity: Optional[str] = "medium"


class AlertActionRequest(BaseModel):
    user_id: str
    notes: Optional[str] = ""


# ============== Field Management Endpoints ==============

@router.get("/farms")
async def list_farms():
    """List all farms."""
    farms = field_manager.list_farms()
    return {"farms": [f.to_dict() for f in farms]}


@router.get("/farms/{farm_id}")
async def get_farm(farm_id: str):
    """Get farm details."""
    farm = field_manager.get_farm(farm_id)
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")
    return farm.to_dict()


@router.get("/fields")
async def list_fields(
    farm_id: Optional[str] = None,
    crop_type: Optional[str] = None,
    status: Optional[str] = None
):
    """List all fields with optional filters."""
    status_enum = FieldStatus(status) if status else None
    fields = field_manager.list_fields(
        farm_id=farm_id or "",
        crop_type=crop_type or "",
        status=status_enum
    )
    return {"fields": [f.to_dict() for f in fields]}


@router.get("/fields/{field_id}")
async def get_field(field_id: str):
    """Get field details."""
    field = field_manager.get_field(field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return field.to_dict()


@router.post("/fields")
async def create_field(request: CreateFieldRequest):
    """Create a new field."""
    from ..crop_monitoring import Coordinate, FieldBoundary, GeometryCalculator
    import uuid
    
    # Convert coordinates
    coordinates = [
        Coordinate(c.longitude, c.latitude, c.elevation)
        for c in request.boundary.coordinates
    ]
    
    # Create boundary
    boundary = FieldBoundary(
        boundary_id=str(uuid.uuid4()),
        coordinates=coordinates
    )
    boundary.area_ha = GeometryCalculator.calculate_area_ha(coordinates)
    boundary.perimeter_m = GeometryCalculator.calculate_perimeter_m(coordinates)
    boundary.centroid = GeometryCalculator.calculate_centroid(coordinates)
    
    # Create field
    field = field_manager.create_field(
        name=request.name,
        boundary=boundary,
        farm_id=request.farm_id,
        crop_type=request.crop_type
    )
    
    return {"field": field.to_dict()}


@router.put("/fields/{field_id}")
async def update_field(field_id: str, request: UpdateFieldRequest):
    """Update field properties."""
    updates = {k: v for k, v in request.dict().items() if v is not None}
    
    if 'status' in updates:
        updates['status'] = FieldStatus(updates['status'])
    
    field = field_manager.update_field(field_id, **updates)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    
    return {"field": field.to_dict()}


@router.delete("/fields/{field_id}")
async def delete_field(field_id: str):
    """Delete a field."""
    success = field_manager.delete_field(field_id)
    if not success:
        raise HTTPException(status_code=404, detail="Field not found")
    return {"message": "Field deleted successfully"}


@router.post("/fields/import")
async def import_fields(
    file: UploadFile = File(...),
    farm_id: Optional[str] = "",
    crop_type: Optional[str] = ""
):
    """Import field boundaries from file (GeoJSON, KML, etc.)."""
    content = await file.read()
    
    # Determine format from filename
    filename = file.filename.lower()
    if filename.endswith('.geojson') or filename.endswith('.json'):
        format_type = BoundaryFormat.GEOJSON
        data = json.loads(content.decode('utf-8'))
    elif filename.endswith('.kml'):
        format_type = BoundaryFormat.KML
        data = content.decode('utf-8')
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    fields = field_manager.import_boundaries(
        data, format_type, farm_id, crop_type
    )
    
    return {
        "message": f"Imported {len(fields)} fields",
        "fields": [f.to_dict() for f in fields]
    }


@router.get("/fields/export")
async def export_fields(field_ids: Optional[str] = None):
    """Export fields as GeoJSON."""
    ids = field_ids.split(',') if field_ids else None
    geojson = field_manager.export_fields_geojson(ids)
    return geojson


# ============== Vegetation Index Endpoints ==============

@router.post("/vegetation/analyze")
async def analyze_vegetation(request: VegetationAnalysisRequest):
    """Analyze vegetation indices for a field."""
    field = field_manager.get_field(request.field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    
    # Create synthetic spectral bands for demo
    bands = create_synthetic_spectral_bands(
        width=100, height=100,
        scene_id=request.scene_id
    )
    
    # Process imagery
    results = crop_service.process_imagery(bands)
    
    # Filter indices if specified
    if request.indices:
        index_types = [VegetationIndexType(i) for i in request.indices]
        results = {k: v for k, v in results.items() if k in index_types}
    
    # Get health summary
    ndvi_result = results.get(VegetationIndexType.NDVI)
    health_summary = None
    if ndvi_result:
        health_summary = crop_service.get_field_health_summary(
            request.field_id, ndvi_result
        )
    
    return {
        "field_id": request.field_id,
        "scene_id": request.scene_id,
        "indices": {k.value: v.to_dict() for k, v in results.items()},
        "health_summary": health_summary
    }


@router.get("/vegetation/indices")
async def list_vegetation_indices():
    """List available vegetation indices."""
    indices = [
        {
            "type": "ndvi",
            "name": "Normalized Difference Vegetation Index",
            "description": "Most common vegetation index, measures plant health",
            "range": [-1, 1],
            "bands_required": ["red", "nir"]
        },
        {
            "type": "ndre",
            "name": "Normalized Difference Red Edge",
            "description": "Better for chlorophyll content in mature vegetation",
            "range": [-1, 1],
            "bands_required": ["red_edge", "nir"]
        },
        {
            "type": "savi",
            "name": "Soil Adjusted Vegetation Index",
            "description": "Better for areas with exposed soil",
            "range": [-1, 1],
            "bands_required": ["red", "nir"]
        },
        {
            "type": "evi",
            "name": "Enhanced Vegetation Index",
            "description": "Better for high biomass regions",
            "range": [-1, 1],
            "bands_required": ["blue", "red", "nir"]
        },
        {
            "type": "gndvi",
            "name": "Green NDVI",
            "description": "More sensitive to chlorophyll concentration",
            "range": [-1, 1],
            "bands_required": ["green", "nir"]
        },
        {
            "type": "ndwi",
            "name": "Normalized Difference Water Index",
            "description": "Detects water content in vegetation",
            "range": [-1, 1],
            "bands_required": ["green", "nir"]
        }
    ]
    return {"indices": indices}


@router.get("/vegetation/health/{field_id}")
async def get_field_health(field_id: str):
    """Get vegetation health status for a field."""
    field = field_manager.get_field(field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    
    # Generate synthetic analysis
    bands = create_synthetic_spectral_bands()
    results = crop_service.process_imagery(bands)
    
    ndvi_result = results.get(VegetationIndexType.NDVI)
    if ndvi_result:
        return crop_service.get_field_health_summary(field_id, ndvi_result)
    
    return {"error": "Unable to calculate health status"}


# ============== Weather Endpoints ==============

@router.post("/weather/forecast")
async def get_weather_forecast(request: WeatherRequest):
    """Get weather forecast for a location."""
    forecast = weather_service.get_forecast(
        request.latitude,
        request.longitude,
        request.days
    )
    return forecast.to_dict()


@router.post("/weather/historical")
async def get_historical_weather(
    latitude: float = Body(...),
    longitude: float = Body(...),
    start_date: str = Body(...),
    end_date: str = Body(...)
):
    """Get historical weather data."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    
    historical = weather_service.get_historical_weather(
        latitude, longitude, start, end
    )
    return historical.to_dict()


@router.post("/weather/risks")
async def get_weather_risks(
    latitude: float = Body(...),
    longitude: float = Body(...),
    crop_type: str = Body(default="general")
):
    """Get weather risk assessment."""
    risks = weather_service.get_weather_risks(latitude, longitude, crop_type)
    return risks


@router.post("/weather/water-balance")
async def calculate_water_balance(
    latitude: float = Body(...),
    longitude: float = Body(...),
    crop_type: str = Body(...),
    growth_stage: str = Body(default="mid")
):
    """Calculate water balance for irrigation planning."""
    balance = weather_service.calculate_water_balance(
        latitude, longitude, crop_type, growth_stage
    )
    return balance


# ============== VRA Map Endpoints ==============

@router.post("/vra/create")
async def create_vra_map(request: VRAMapRequest):
    """Create a VRA map."""
    field = field_manager.get_field(request.field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    
    # Get field boundary as list of tuples
    boundary = [(c.longitude, c.latitude) for c in field.boundary.coordinates]
    
    # Generate synthetic NDVI for demo
    np.random.seed(42)
    ndvi = 0.5 + 0.2 * np.random.randn(100, 100)
    ndvi = np.clip(ndvi, 0, 1)
    
    map_type = request.map_type.lower()
    
    if map_type == "sowing":
        vra_map = vra_service.create_sowing_map(
            request.field_id, boundary, ndvi,
            request.crop_type, request.target_value,
            request.num_zones
        )
    elif map_type == "nitrogen":
        vra_map = vra_service.create_nitrogen_map(
            request.field_id, boundary, ndvi,
            request.crop_type, request.target_value,
            num_zones=request.num_zones
        )
    elif map_type == "pk":
        vra_map = vra_service.create_pk_map(
            request.field_id, boundary,
            request.crop_type, request.target_value,
            soil_p_ppm=15, soil_k_ppm=150,
            num_zones=request.num_zones
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown map type: {map_type}")
    
    return {"vra_map": vra_map.to_dict()}


@router.get("/vra/maps")
async def list_vra_maps(field_id: Optional[str] = None):
    """List VRA maps."""
    maps = vra_service.list_maps(field_id or "")
    return {"maps": [m.to_dict() for m in maps]}


@router.get("/vra/maps/{map_id}")
async def get_vra_map(map_id: str):
    """Get VRA map details."""
    vra_map = vra_service.get_map(map_id)
    if not vra_map:
        raise HTTPException(status_code=404, detail="VRA map not found")
    return vra_map.to_dict()


@router.get("/vra/maps/{map_id}/export")
async def export_vra_map(
    map_id: str,
    format: str = Query(default="geojson", description="Export format")
):
    """Export VRA map to specified format."""
    try:
        format_type = ExportFormat(format.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown format: {format}")
    
    try:
        exported = vra_service.export_map(map_id, format_type)
        return exported
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/vra/maps/{map_id}/savings")
async def calculate_vra_savings(
    map_id: str,
    product_price: Optional[float] = None
):
    """Calculate savings from VRA application."""
    try:
        savings = vra_service.calculate_savings(map_id, product_price)
        return savings
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============== Alert Endpoints ==============

@router.get("/alerts")
async def list_alerts(
    status: Optional[str] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    field_id: Optional[str] = None,
    limit: int = Query(default=100, le=500)
):
    """List alerts with filters."""
    status_enum = AlertStatus(status) if status else None
    category_enum = AlertCategory(category) if category else None
    severity_enum = AlertSeverity(severity) if severity else None
    
    alerts = alert_service.manager.list_alerts(
        status=status_enum,
        category=category_enum,
        severity=severity_enum,
        field_id=field_id or "",
        limit=limit
    )
    
    return {"alerts": [a.to_dict() for a in alerts]}


@router.get("/alerts/dashboard")
async def get_alert_dashboard():
    """Get alert dashboard data."""
    return alert_service.get_alert_dashboard()


@router.get("/alerts/{alert_id}")
async def get_alert(alert_id: str):
    """Get alert details."""
    alert = alert_service.manager.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert.to_dict()


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, request: AlertActionRequest):
    """Acknowledge an alert."""
    success = alert_service.manager.acknowledge_alert(
        alert_id, request.user_id, request.notes
    )
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert acknowledged"}


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, request: AlertActionRequest):
    """Resolve an alert."""
    success = alert_service.manager.resolve_alert(
        alert_id, request.user_id, request.notes
    )
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert resolved"}


@router.post("/alerts/{alert_id}/snooze")
async def snooze_alert(alert_id: str, hours: int = Query(default=24)):
    """Snooze an alert."""
    success = alert_service.manager.snooze_alert(alert_id, hours)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": f"Alert snoozed for {hours} hours"}


@router.get("/alerts/rules")
async def list_alert_rules(
    category: Optional[str] = None,
    enabled_only: bool = True
):
    """List alert rules."""
    category_enum = AlertCategory(category) if category else None
    rules = alert_service.manager.rule_engine.list_rules(
        category=category_enum,
        enabled_only=enabled_only
    )
    return {"rules": [r.to_dict() for r in rules]}


@router.post("/alerts/rules")
async def create_alert_rule(request: AlertRuleRequest):
    """Create a new alert rule."""
    rule = alert_service.manager.create_rule(
        name=request.name,
        category=AlertCategory(request.category),
        metric=request.metric,
        operator=request.operator,
        threshold=request.threshold,
        severity=AlertSeverity(request.severity)
    )
    return {"rule": rule.to_dict()}


@router.post("/alerts/check-field")
async def check_field_alerts(
    field_id: str = Body(...),
    ndvi_current: float = Body(...),
    ndvi_previous: Optional[float] = Body(default=None)
):
    """Check field health and generate alerts."""
    field = field_manager.get_field(field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    
    alerts = alert_service.check_field_health(
        field_id,
        field.name,
        field.crop_info.crop_type if field.crop_info else "general",
        {'current': ndvi_current, 'previous': ndvi_previous}
    )
    
    return {"alerts": [a.to_dict() for a in alerts]}


# ============== Dashboard Endpoint ==============

@router.get("/dashboard")
async def get_crop_monitoring_dashboard():
    """Get comprehensive crop monitoring dashboard data."""
    # Get all fields
    fields = field_manager.list_fields()
    
    # Get alert summary
    alert_dashboard = alert_service.get_alert_dashboard()
    
    # Get VRA maps
    vra_maps = vra_service.list_maps()
    
    # Calculate field statistics
    total_area = sum(f.boundary.area_ha for f in fields)
    crops = {}
    for f in fields:
        if f.crop_info:
            crop = f.crop_info.crop_type
            if crop not in crops:
                crops[crop] = {'count': 0, 'area_ha': 0}
            crops[crop]['count'] += 1
            crops[crop]['area_ha'] += f.boundary.area_ha
    
    return {
        "summary": {
            "total_fields": len(fields),
            "total_area_ha": round(total_area, 2),
            "total_farms": len(field_manager.list_farms()),
            "crops": crops
        },
        "alerts": alert_dashboard,
        "vra_maps_count": len(vra_maps),
        "recent_vra_maps": [m.to_dict() for m in vra_maps[:5]]
    }
