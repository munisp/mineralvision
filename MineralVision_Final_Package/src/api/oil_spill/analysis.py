"""Oil-spill segmentation assessment and geospatial decision-support utilities.

The functions in this module turn a supplied or model-produced segmentation mask into
reviewable environmental-response evidence. They deliberately do not claim to infer oil
from unmodelled RGB imagery and do not trigger notifications or vehicle operations.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .schemas import GeographicBounds, Severity

EARTH_RADIUS_M = 6_371_008.8


class MaskValidationError(ValueError):
    """Raised when caller-provided segmentation evidence is invalid."""


@dataclass(frozen=True)
class MaskAssessment:
    """Internal normalized result of a segmentation-mask assessment."""

    binary_mask: np.ndarray
    probability_map: np.ndarray
    candidate_pixels: int
    retained_pixels: int
    component_count: int
    confidence: Optional[float]
    quality_flags: List[str]
    oil_area_m2: Optional[float]
    geometry_geojson: Optional[Dict[str, Any]]
    severity: Severity


def decode_probability_mask(mask_base64: str, expected_width: int, expected_height: int) -> np.ndarray:
    """Decode a base64 image mask into a normalized single-band probability raster."""
    try:
        image_bytes = base64.b64decode(mask_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MaskValidationError("mask_base64 must be valid base64 image data") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("L")
            if image.width != expected_width or image.height != expected_height:
                raise MaskValidationError(
                    "mask dimensions must match image_width_px and image_height_px "
                    f"(received {image.width}x{image.height})"
                )
            array = np.asarray(image, dtype=np.float32)
    except MaskValidationError:
        raise
    except Exception as exc:
        raise MaskValidationError("mask_base64 must decode to a PNG or JPEG image") from exc

    return array / 255.0


def retain_components(binary_mask: np.ndarray, min_component_area_px: int) -> Tuple[np.ndarray, int]:
    """Remove connected components too small to be actionable evidence."""
    labels, component_count = ndimage.label(binary_mask)
    if component_count == 0:
        return np.zeros_like(binary_mask, dtype=bool), 0

    sizes = np.bincount(labels.ravel())
    accepted_labels = np.flatnonzero(sizes >= min_component_area_px)
    accepted_labels = accepted_labels[accepted_labels != 0]
    retained = np.isin(labels, accepted_labels)
    return retained, int(len(accepted_labels))


def haversine_m(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    """Return the great-circle distance in metres between two WGS84 points."""
    lat_a, lon_a, lat_b, lon_b = map(math.radians, (latitude_a, longitude_a, latitude_b, longitude_b))
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    haversine = math.sin(delta_lat / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(haversine))


def pixel_area_from_bounds_m2(bounds: GeographicBounds, image_width_px: int, image_height_px: int) -> float:
    """Estimate pixel area from a rectangular WGS84 image footprint.

    This is an equirectangular approximation appropriate only for a nadir imagery
    footprint. Survey-grade measurements should provide a calibrated GSD instead.
    """
    midpoint_latitude = (bounds.north + bounds.south) / 2
    midpoint_longitude = (bounds.east + bounds.west) / 2
    width_m = haversine_m(midpoint_latitude, bounds.west, midpoint_latitude, bounds.east)
    height_m = haversine_m(bounds.south, midpoint_longitude, bounds.north, midpoint_longitude)
    return (width_m * height_m) / (image_width_px * image_height_px)


def _pixel_to_lon_lat(x: float, y: float, width: int, height: int, bounds: GeographicBounds) -> List[float]:
    """Project a pixel coordinate into an assumed rectilinear WGS84 footprint."""
    denominator_x = max(width - 1, 1)
    denominator_y = max(height - 1, 1)
    longitude = bounds.west + (x / denominator_x) * (bounds.east - bounds.west)
    latitude = bounds.north - (y / denominator_y) * (bounds.north - bounds.south)
    return [round(longitude, 8), round(latitude, 8)]


def mask_to_geojson(binary_mask: np.ndarray, bounds: Optional[GeographicBounds]) -> Optional[Dict[str, Any]]:
    """Polygonize an accepted mask into WGS84 GeoJSON when image bounds are available."""
    if bounds is None or not binary_mask.any():
        return None

    # OpenCV is already a core MineralVision computer-vision dependency. Importing
    # locally makes the rest of the assessment module usable in non-CV environments.
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - requirements includes OpenCV
        raise RuntimeError("OpenCV is required to polygonize an oil-spill mask") from exc

    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary_mask.shape
    polygons: List[List[List[List[float]]]] = []

    for contour in contours:
        if len(contour) < 3:
            continue
        ring = [_pixel_to_lon_lat(float(point[0][0]), float(point[0][1]), width, height, bounds) for point in contour]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        polygons.append([ring])

    if not polygons:
        return None
    return {"type": "MultiPolygon", "coordinates": polygons}


def classify_severity(oil_area_m2: Optional[float]) -> Severity:
    """Apply conservative, configurable-at-policy-layer surface-area screening bands."""
    if oil_area_m2 is None:
        return Severity.UNKNOWN
    if oil_area_m2 < 1_000:
        return Severity.LOW
    if oil_area_m2 < 10_000:
        return Severity.MODERATE
    if oil_area_m2 < 100_000:
        return Severity.HIGH
    return Severity.CRITICAL


def assess_mask(
    probability_map: np.ndarray,
    *,
    threshold: float,
    min_component_area_px: int,
    ground_sampling_distance_m: Optional[float],
    geographic_bounds: Optional[GeographicBounds],
) -> MaskAssessment:
    """Assess a normalized probability map and derive reviewable evidence."""
    if probability_map.ndim != 2:
        raise MaskValidationError("probability_map must be a 2-D single-band array")
    if not np.isfinite(probability_map).all():
        raise MaskValidationError("probability_map must contain only finite values")
    if probability_map.min() < 0 or probability_map.max() > 1:
        raise MaskValidationError("probability_map values must be normalized to the range [0, 1]")

    candidate = probability_map >= threshold
    candidate_pixels = int(candidate.sum())
    accepted_mask, component_count = retain_components(candidate, min_component_area_px)
    retained_pixels = int(accepted_mask.sum())

    flags: List[str] = []
    if candidate_pixels == 0:
        flags.append("no_oil_pixels_above_threshold")
    if candidate_pixels > 0 and retained_pixels == 0:
        flags.append("all_candidate_components_below_minimum_area")
    if component_count > 10:
        flags.append("fragmented_detection")
    if ground_sampling_distance_m is None and geographic_bounds is None:
        flags.append("not_georeferenced")

    confidence: Optional[float]
    if retained_pixels == 0:
        confidence = None
    else:
        retention_ratio = retained_pixels / max(candidate_pixels, 1)
        confidence = round(float(probability_map[accepted_mask].mean()) * retention_ratio, 4)
        if confidence < 0.6:
            flags.append("low_model_or_annotation_confidence")

    pixel_area_m2: Optional[float] = None
    if ground_sampling_distance_m is not None:
        pixel_area_m2 = ground_sampling_distance_m ** 2
    elif geographic_bounds is not None:
        pixel_area_m2 = pixel_area_from_bounds_m2(
            geographic_bounds,
            image_width_px=probability_map.shape[1],
            image_height_px=probability_map.shape[0],
        )
        flags.append("area_estimated_from_geographic_bounds")

    oil_area_m2 = None if pixel_area_m2 is None else round(retained_pixels * pixel_area_m2, 3)
    geometry_geojson = mask_to_geojson(accepted_mask, geographic_bounds)
    return MaskAssessment(
        binary_mask=accepted_mask,
        probability_map=probability_map,
        candidate_pixels=candidate_pixels,
        retained_pixels=retained_pixels,
        component_count=component_count,
        confidence=confidence,
        quality_flags=flags,
        oil_area_m2=oil_area_m2,
        geometry_geojson=geometry_geojson,
        severity=classify_severity(oil_area_m2),
    )


def geometry_bounds(geometry_geojson: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    """Return west, south, east, north for a GeoJSON MultiPolygon."""
    if not geometry_geojson or geometry_geojson.get("type") != "MultiPolygon":
        return None
    points = [point for polygon in geometry_geojson.get("coordinates", []) for ring in polygon for point in ring]
    if not points:
        return None
    longitudes = [float(point[0]) for point in points]
    latitudes = [float(point[1]) for point in points]
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def build_coverage_priority_cells(
    geometry_geojson: Optional[Dict[str, Any]],
    *,
    cell_size_m: float,
    drone_count: int,
    buffer_m: float,
) -> Tuple[Optional[float], List[Dict[str, Any]], List[str]]:
    """Build a bounded, advisory survey-priority grid around an incident geometry.

    The returned grid is a recommendation for an approved mission planner. It contains
    no command, connection, or side effect that could fly an aircraft.
    """
    bounds = geometry_bounds(geometry_geojson)
    if bounds is None:
        return None, [], ["No geographic footprint is available; create a georeferenced assessment before planning coverage."]

    west, south, east, north = bounds
    centre_latitude = (south + north) / 2
    centre_longitude = (west + east) / 2
    width_m = haversine_m(centre_latitude, west, centre_latitude, east)
    height_m = haversine_m(south, centre_longitude, north, centre_longitude)
    buffered_width_m = width_m + 2 * buffer_m
    buffered_height_m = height_m + 2 * buffer_m
    recommended_area_m2 = round(buffered_width_m * buffered_height_m, 2)

    # Keep payloads and mission recommendations bounded while retaining an evidence-led
    # centre-outward priority order.
    columns = max(1, min(25, math.ceil(buffered_width_m / cell_size_m)))
    rows = max(1, min(25, math.ceil(buffered_height_m / cell_size_m)))
    longitude_padding = (east - west) * (buffer_m / width_m) if width_m else 0
    latitude_padding = (north - south) * (buffer_m / height_m) if height_m else 0
    expanded_west, expanded_east = west - longitude_padding, east + longitude_padding
    expanded_south, expanded_north = south - latitude_padding, north + latitude_padding

    cells: List[Dict[str, Any]] = []
    for row in range(rows):
        for column in range(columns):
            cell_west = expanded_west + (expanded_east - expanded_west) * column / columns
            cell_east = expanded_west + (expanded_east - expanded_west) * (column + 1) / columns
            cell_north = expanded_north - (expanded_north - expanded_south) * row / rows
            cell_south = expanded_north - (expanded_north - expanded_south) * (row + 1) / rows
            cell_lon = (cell_west + cell_east) / 2
            cell_lat = (cell_north + cell_south) / 2
            distance = haversine_m(centre_latitude, centre_longitude, cell_lat, cell_lon)
            cells.append(
                {
                    "cell_id": f"r{row + 1}-c{column + 1}",
                    "priority_score": round(1 / (1 + distance / max(cell_size_m, 1)), 4),
                    "centre": {"longitude": round(cell_lon, 8), "latitude": round(cell_lat, 8)},
                    "bounds": {"west": round(cell_west, 8), "south": round(cell_south, 8), "east": round(cell_east, 8), "north": round(cell_north, 8)},
                }
            )
    cells.sort(key=lambda cell: cell["priority_score"], reverse=True)
    recommended_cells = cells[: min(len(cells), max(drone_count * 5, 10))]
    notes = [
        "Advisory only: this output must be reviewed by an authorized operator before any flight action.",
        "Priority falls with distance from the assessed slick footprint; live weather, airspace, battery, and regulatory constraints are not evaluated here.",
    ]
    return recommended_area_m2, recommended_cells, notes
