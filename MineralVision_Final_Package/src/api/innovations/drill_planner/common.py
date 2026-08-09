"""Shared geodesy helpers for drill_planner (local UTM via pyproj)."""

import math
from typing import Tuple

import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 + zone) if lat >= 0 else (32700 + zone)


class LocalProjector:
    """lon/lat <-> local UTM metres around a reference point."""

    def __init__(self, ref_lon: float, ref_lat: float):
        self.epsg = utm_epsg(ref_lon, ref_lat)
        self.fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
        self.inv = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)

    def to_m(self, lon: float, lat: float) -> Tuple[float, float]:
        return self.fwd.transform(lon, lat)

    def to_lonlat(self, x: float, y: float) -> Tuple[float, float]:
        return self.inv.transform(x, y)

    def to_m_array(self, lons, lats):
        return self.fwd.transform(np.asarray(lons, float), np.asarray(lats, float))

    def to_lonlat_array(self, xs, ys):
        return self.inv.transform(np.asarray(xs, float), np.asarray(ys, float))


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
