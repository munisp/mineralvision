"""
Shared geospatial core for the geotoolkit innovations module.

- Web-Mercator (slippy map) tile math
- Bilinear raster resampling with numpy
- Real colormap lookup tables (viridis / terrain / iron-oxide), pure numpy/PIL
  (no matplotlib dependency in the request path)
- In-memory raster registry shared by raster-tiles and targeting-tiles
"""

from __future__ import annotations

import io
import math
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Web-Mercator tile math (standard slippy-map formulas)
# ---------------------------------------------------------------------------

WEBMERC_MAX = 20037508.342789244  # pi * R, metres
WEBMERC_MIN = -WEBMERC_MAX
MAX_LAT = 85.05112878


def num_tiles(z: int) -> int:
    return 1 << z


def tile_size_m(z: int) -> float:
    """Ground size (metres) of one XYZ tile at zoom z in Web Mercator."""
    return (2.0 * WEBMERC_MAX) / num_tiles(z)


def tile_bounds_merc(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) of tile in Web-Mercator metres."""
    size = tile_size_m(z)
    minx = WEBMERC_MIN + x * size
    maxx = minx + size
    maxy = WEBMERC_MAX - y * size
    miny = maxy - size
    return minx, miny, maxx, maxy


def lonlat_to_merc(lon: float, lat: float) -> Tuple[float, float]:
    lat = max(min(lat, MAX_LAT), -MAX_LAT)
    x = math.radians(lon) * 6378137.0
    y = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * 6378137.0
    return x, y


def merc_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    lon = math.degrees(x / 6378137.0)
    lat = math.degrees(2.0 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2.0)
    return lon, lat


def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int, int]:
    lat = max(min(lat, MAX_LAT), -MAX_LAT)
    n = num_tiles(z)
    xt = int((lon + 180.0) / 360.0 * n)
    latr = math.radians(lat)
    yt = int((1.0 - math.asinh(math.tan(latr)) / math.pi) / 2.0 * n)
    return min(max(xt, 0), n - 1), min(max(yt, 0), n - 1)


# ---------------------------------------------------------------------------
# Colormaps — pure-numpy lookup tables (256 x 3 uint8)
# ---------------------------------------------------------------------------

# Viridis anchor points sampled from the perceptually-uniform viridis map.
_VIRIDIS_ANCHORS = np.array([
    [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
    [41, 120, 142], [32, 144, 140], [34, 167, 133], [53, 183, 121],
    [90, 200, 103], [144, 215, 67], [210, 226, 27], [253, 231, 37],
], dtype=float)

_TERRAIN_ANCHORS = np.array([
    [51, 102, 153],    # deep water blue
    [136, 178, 204],   # shallow water
    [165, 214, 167],   # coastal green
    [76, 140, 74],     # lowland green
    [160, 140, 90],    # hills brown
    [140, 100, 70],    # mountain brown
    [200, 190, 180],   # high rock
    [255, 255, 255],   # snow
], dtype=float)

_IRON_OXIDE_ANCHORS = np.array([
    [13, 8, 8],        # near-black
    [89, 23, 13],      # dark hematite red
    [166, 54, 16],     # iron-oxide red
    [214, 112, 19],    # ochre orange
    [243, 185, 72],    # limonite yellow
    [252, 245, 224],   # pale gossan
], dtype=float)


def build_colormap(anchors: np.ndarray, n: int = 256) -> np.ndarray:
    """Linearly interpolate anchor colors into an n-entry LUT."""
    xs = np.linspace(0.0, 1.0, anchors.shape[0])
    xi = np.linspace(0.0, 1.0, n)
    lut = np.stack([np.interp(xi, xs, anchors[:, c]) for c in range(3)], axis=1)
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


COLORMAPS = {
    "viridis": build_colormap(_VIRIDIS_ANCHORS),
    "terrain": build_colormap(_TERRAIN_ANCHORS),
    "iron-oxide": build_colormap(_IRON_OXIDE_ANCHORS),
}


def apply_colormap(norm: np.ndarray, name: str = "viridis") -> np.ndarray:
    """Map normalized [0,1] array (NaN allowed) to RGBA uint8 image."""
    if name not in COLORMAPS:
        raise KeyError(f"unknown colormap '{name}'; available: {sorted(COLORMAPS)}")
    lut = COLORMAPS[name]
    valid = np.isfinite(norm)
    idx = np.clip(np.nan_to_num(norm, nan=0.0) * 255.0, 0, 255).astype(np.uint8)
    rgb = lut[idx]
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


# ---------------------------------------------------------------------------
# Raster registry + resampling
# ---------------------------------------------------------------------------

class RegisteredRaster:
    def __init__(self, grid: np.ndarray, bounds: Sequence[float],
                 colormap: str = "viridis", name: str = ""):
        self.id = uuid.uuid4().hex[:12]
        self.grid = grid.astype(float)
        self.bounds = tuple(float(b) for b in bounds)  # minx, miny, maxx, maxy (EPSG:3857)
        self.colormap = colormap if colormap in COLORMAPS else "viridis"
        self.name = name or f"raster-{self.id}"
        self.vmin = float(np.nanmin(grid))
        self.vmax = float(np.nanmax(grid))

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        span = self.vmax - self.vmin
        if span <= 0:
            return np.zeros_like(arr)
        return (arr - self.vmin) / span


RASTER_REGISTRY: Dict[str, RegisteredRaster] = {}


def register_raster(grid, bounds, colormap: str = "viridis", name: str = "") -> RegisteredRaster:
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError("grid must be a 2D array (nested list)")
    r = RegisteredRaster(arr, bounds, colormap, name)
    RASTER_REGISTRY[r.id] = r
    return r


def bilinear_sample(grid: np.ndarray, bounds: Sequence[float],
                    xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample grid (rows north->south) at Web-Mercator coords xs/ys via bilinear
    interpolation. Out-of-bounds -> NaN."""
    minx, miny, maxx, maxy = bounds
    nrows, ncols = grid.shape
    fx = (xs - minx) / (maxx - minx) * (ncols - 1)
    fy = (maxy - ys) / (maxy - miny) * (nrows - 1)

    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = fx - x0
    dy = fy - y0

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < ncols) & (y1 < nrows)
    x0c = np.clip(x0, 0, ncols - 1)
    x1c = np.clip(x1, 0, ncols - 1)
    y0c = np.clip(y0, 0, nrows - 1)
    y1c = np.clip(y1, 0, nrows - 1)

    v00 = grid[y0c, x0c]
    v01 = grid[y0c, x1c]
    v10 = grid[y1c, x0c]
    v11 = grid[y1c, x1c]
    val = (v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy) +
           v10 * (1 - dx) * dy + v11 * dx * dy)
    return np.where(valid, val, np.nan)


def render_tile_png(raster: RegisteredRaster, z: int, x: int, y: int,
                    size: int = 256, colormap: Optional[str] = None) -> bytes:
    """Resample the registered raster onto a size x size XYZ tile and encode PNG."""
    minx, miny, maxx, maxy = tile_bounds_merc(z, x, y)
    # pixel-center coordinates, rows top (north) to bottom (south)
    xs = minx + (np.arange(size) + 0.5) / size * (maxx - minx)
    ys = maxy - (np.arange(size) + 0.5) / size * (maxy - miny)
    gx, gy = np.meshgrid(xs, ys)
    vals = bilinear_sample(raster.grid, raster.bounds, gx, gy)
    rgba = apply_colormap(raster.normalize(vals), colormap or raster.colormap)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def validate_tile(z: int, x: int, y: int) -> bool:
    n = num_tiles(z)
    return 0 <= z <= 22 and 0 <= x < n and 0 <= y < n
