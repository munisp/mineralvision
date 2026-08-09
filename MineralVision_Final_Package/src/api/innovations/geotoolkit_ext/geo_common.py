"""Shared geospatial helpers for geotoolkit_ext."""

import math
from typing import Any, Dict, List, Tuple

from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry
from pyproj import Transformer


def fc_to_geometries(fc: Dict[str, Any]) -> List[Tuple[BaseGeometry, Dict[str, Any]]]:
    """GeoJSON FeatureCollection -> [(geometry, properties)]."""
    if fc.get("type") == "FeatureCollection":
        feats = fc.get("features", [])
    elif fc.get("type") == "Feature":
        feats = [fc]
    else:  # bare geometry
        feats = [{"type": "Feature", "geometry": fc, "properties": {}}]
    out = []
    for f in feats:
        geom = shape(f["geometry"])
        out.append((geom, dict(f.get("properties") or {})))
    return out


def geom_to_feature(geom: BaseGeometry, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "Feature", "geometry": mapping(geom), "properties": properties}


def looks_geographic(geoms: List[BaseGeometry]) -> bool:
    """Heuristic: all coords within lon/lat bounds -> geographic CRS."""
    minx, miny, maxx, maxy = float("inf"), float("inf"), float("-inf"), float("-inf")
    for g in geoms:
        b = g.bounds
        minx, miny = min(minx, b[0]), min(miny, b[1])
        maxx, maxy = max(maxx, b[2]), max(maxy, b[3])
    return -180 <= minx and maxx <= 180 and -90 <= miny and maxy <= 90


def utm_zone(lon: float, lat: float) -> Tuple[int, bool]:
    """UTM zone number and northern-hemisphere flag for lon/lat."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return zone, lat >= 0.0


def utm_epsg_for(lon: float, lat: float) -> int:
    zone, north = utm_zone(lon, lat)
    return (32600 + zone) if north else (32700 + zone)


def area_m2(geom: BaseGeometry, epsg: int) -> float:
    """Project a lon/lat geometry to the given projected CRS and return area in m^2."""
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    from shapely.ops import transform as shp_transform
    return shp_transform(transformer.transform, geom).area
