"""GeoLibre ``.geolibre.json`` project document builder (pure stdlib + json).

GeoLibre (opengeos/GeoLibre, MIT, PyPI ``geolibre``) is a *client-side* GIS
application (MapLibre GL JS + DuckDB-WASM + Whitebox WASM tools).  Its
portable interchange artefact is a JSON *project document* describing the
basemap, camera (center/zoom/bounds) and an ordered ``layers`` array whose
entries carry ``type`` / ``source`` / ``paint`` information in the same
spirit as MapLibre style layers.

This module AUTHORS those documents from MineralVision platform data.  It has
no dependency on the ``geolibre`` wheel (a ~50MB client bundle): the JSON
schema is documented public behaviour and authoring it server-side is the
platform's role.

Schema assumptions (documented honestly — GeoLibre's project format is a
thin wrapper over a MapLibre-style layer list):

* Top-level keys: ``format_version`` (int), ``generator`` (str),
  ``name``, ``basemap`` (id string), ``center`` ([lon, lat]),
  ``zoom`` (float), optional ``bounds`` ([w, s, e, n]),
  ``layers`` (array), ``legends`` (array), ``created`` (ISO-8601 UTC).
* Each layer: ``id``, ``type`` in
  {``xyz-tile``, ``geojson``, ``cog``, ``geoparquet``, ``heatmap``},
  ``source`` (type-specific), ``paint`` (renderer hints), ``opacity``
  (0..1), ``visible`` (bool).
* ``source`` shapes:
    - xyz-tile:   ``{"tiles": ["https://.../{z}/{x}/{y}"]}``
    - geojson:    ``{"data": <inline GeoJSON>}`` (GeoLibre caps inline
      GeoJSON around 50MB — prefer URL layers for big data)
    - cog:        ``{"url": "https://.../raster.tif"}``
    - geoparquet: ``{"url": "https://.../data.parquet"}``
    - heatmap:    ``{"data": <inline GeoJSON points>, "weightProperty": str}``

The builder performs structural validation only (required keys, value
ranges); it never fabricates data — callers supply real coordinates/URLs.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

FORMAT_VERSION = 1
GENERATOR = "mineralvision"

LAYER_TYPES = ("xyz-tile", "geojson", "cog", "geoparquet", "heatmap")

# Curated basemap catalog — mirrors geolibre.basemap_catalog() entries.
# Used when the geolibre package is not installed; when it IS installed the
# live catalog from the package is preferred (see routes.capabilities).
BASEMAP_CATALOG: Dict[str, Dict[str, str]] = {
    "osm": {"name": "OpenStreetMap", "style": "https://tile.openstreetmap.org/{z}/{x}/{y}.png"},
    "carto-positron": {"name": "Carto Positron", "style": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"},
    "carto-dark": {"name": "Carto Dark Matter", "style": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"},
    "esri-world-imagery": {"name": "Esri World Imagery", "style": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"},
    "opentopo": {"name": "OpenTopoMap", "style": "https://tile.opentopomap.org/{z}/{x}/{y}.png"},
}

COLOR_RAMPS = ("viridis", "plasma", "inferno", "magma", "terrain", "rainbow")


class ProjectBuilderError(ValueError):
    """Raised on structurally invalid project documents or layers."""


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def new_project(
    name: str,
    center: Sequence[float],
    zoom: float,
    bounds: Optional[Sequence[float]] = None,
    basemap: str = "osm",
) -> Dict[str, Any]:
    """Create an empty project document. ``center`` is [lon, lat]."""
    if len(center) != 2:
        raise ProjectBuilderError("center must be [lon, lat]")
    lon, lat = float(center[0]), float(center[1])
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ProjectBuilderError("center out of lon/lat range")
    if not (0.0 <= float(zoom) <= 24.0):
        raise ProjectBuilderError("zoom must be in [0, 24]")
    if basemap not in BASEMAP_CATALOG:
        raise ProjectBuilderError(
            f"unknown basemap '{basemap}'; available: {sorted(BASEMAP_CATALOG)}"
        )
    doc: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "generator": GENERATOR,
        "name": str(name),
        "basemap": basemap,
        "center": [lon, lat],
        "zoom": float(zoom),
        "layers": [],
        "legends": [],
        "created": datetime.now(timezone.utc).isoformat(),
    }
    if bounds is not None:
        if len(bounds) != 4:
            raise ProjectBuilderError("bounds must be [west, south, east, north]")
        doc["bounds"] = [float(v) for v in bounds]
    return doc


def _check_opacity(opacity: float) -> float:
    o = float(opacity)
    if not (0.0 <= o <= 1.0):
        raise ProjectBuilderError("opacity must be in [0, 1]")
    return o


def _append_layer(doc: Dict[str, Any], layer: Dict[str, Any]) -> Dict[str, Any]:
    if any(existing["id"] == layer["id"] for existing in doc["layers"]):
        raise ProjectBuilderError(f"duplicate layer id '{layer['id']}'")
    doc["layers"].append(layer)
    return doc


def add_tile_layer(
    doc: Dict[str, Any],
    layer_id: str,
    url_template: str,
    opacity: float = 1.0,
    attribution: Optional[str] = None,
    minzoom: int = 0,
    maxzoom: int = 22,
    visible: bool = True,
) -> Dict[str, Any]:
    """Add an XYZ raster tile layer. ``url_template`` must contain {z}/{x}/{y}."""
    if not all(tok in url_template for tok in ("{z}", "{x}", "{y}")):
        raise ProjectBuilderError("xyz-tile url_template must contain {z}/{x}/{y}")
    layer = {
        "id": layer_id,
        "type": "xyz-tile",
        "source": {"tiles": [url_template]},
        "paint": {"raster-opacity": _check_opacity(opacity)},
        "opacity": _check_opacity(opacity),
        "visible": bool(visible),
        "minzoom": int(minzoom),
        "maxzoom": int(maxzoom),
    }
    if attribution:
        layer["source"]["attribution"] = attribution
    return _append_layer(doc, layer)


def _validate_geojson(data: Dict[str, Any]) -> None:
    gtype = data.get("type")
    if gtype not in ("FeatureCollection", "Feature", "Point", "LineString",
                     "Polygon", "MultiPoint", "MultiPolygon", "GeometryCollection"):
        raise ProjectBuilderError(f"invalid GeoJSON type '{gtype}'")
    if gtype == "FeatureCollection" and not isinstance(data.get("features"), list):
        raise ProjectBuilderError("FeatureCollection requires a features array")


def add_geojson_layer(
    doc: Dict[str, Any],
    layer_id: str,
    geojson: Dict[str, Any],
    style: Optional[Dict[str, Any]] = None,
    opacity: float = 1.0,
    visible: bool = True,
) -> Dict[str, Any]:
    """Add an inline GeoJSON layer (vector points/lines/polygons)."""
    _validate_geojson(geojson)
    paint = {"circle-radius": 6, "circle-color": "#e74c3c",
             "line-color": "#2c3e50", "fill-opacity": 0.6}
    if style:
        paint.update(style)
    layer = {
        "id": layer_id,
        "type": "geojson",
        "source": {"data": copy.deepcopy(geojson)},
        "paint": paint,
        "opacity": _check_opacity(opacity),
        "visible": bool(visible),
    }
    return _append_layer(doc, layer)


def add_cog_layer(
    doc: Dict[str, Any],
    layer_id: str,
    url: str,
    opacity: float = 1.0,
    colormap: str = "viridis",
    rescale: Optional[Sequence[float]] = None,
    visible: bool = True,
) -> Dict[str, Any]:
    """Add a Cloud-Optimized GeoTIFF layer streamed by URL."""
    if not url:
        raise ProjectBuilderError("cog layer requires a url")
    source: Dict[str, Any] = {"url": url}
    if rescale is not None:
        if len(rescale) != 2:
            raise ProjectBuilderError("rescale must be [vmin, vmax]")
        source["rescale"] = [float(rescale[0]), float(rescale[1])]
    layer = {
        "id": layer_id,
        "type": "cog",
        "source": source,
        "paint": {"raster-opacity": _check_opacity(opacity), "colormap": colormap},
        "opacity": _check_opacity(opacity),
        "visible": bool(visible),
    }
    return _append_layer(doc, layer)


def add_geoparquet_layer(
    doc: Dict[str, Any],
    layer_id: str,
    url: str,
    style: Optional[Dict[str, Any]] = None,
    opacity: float = 1.0,
    visible: bool = True,
) -> Dict[str, Any]:
    """Add a GeoParquet layer loaded by URL (DuckDB-WASM on the client)."""
    if not url:
        raise ProjectBuilderError("geoparquet layer requires a url")
    layer = {
        "id": layer_id,
        "type": "geoparquet",
        "source": {"url": url},
        "paint": style or {"circle-color": "#2980b9", "circle-radius": 5},
        "opacity": _check_opacity(opacity),
        "visible": bool(visible),
    }
    return _append_layer(doc, layer)


def add_heatmap_layer(
    doc: Dict[str, Any],
    layer_id: str,
    geojson_points: Dict[str, Any],
    weight_property: str,
    radius: float = 25.0,
    color_ramp: str = "viridis",
    opacity: float = 0.8,
    visible: bool = True,
) -> Dict[str, Any]:
    """Add a heatmap layer (e.g. assay grades) over inline point GeoJSON.

    Every feature must carry the ``weight_property`` in properties.
    """
    _validate_geojson(geojson_points)
    if color_ramp not in COLOR_RAMPS:
        raise ProjectBuilderError(
            f"unknown color_ramp '{color_ramp}'; available: {COLOR_RAMPS}"
        )
    for feat in geojson_points.get("features", []):
        props = feat.get("properties") or {}
        if weight_property not in props:
            raise ProjectBuilderError(
                f"heatmap feature missing weight property '{weight_property}'"
            )
    layer = {
        "id": layer_id,
        "type": "heatmap",
        "source": {"data": copy.deepcopy(geojson_points),
                   "weightProperty": weight_property},
        "paint": {"heatmap-radius": float(radius), "heatmap-color-ramp": color_ramp},
        "opacity": _check_opacity(opacity),
        "visible": bool(visible),
    }
    return _append_layer(doc, layer)


def add_legend(
    doc: Dict[str, Any],
    layer_id: str,
    title: str,
    kind: str = "colorbar",
    color_ramp: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    unit: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach a legend/colorbar entry for a layer."""
    if kind not in ("colorbar", "categorical"):
        raise ProjectBuilderError("legend kind must be colorbar|categorical")
    if not any(l["id"] == layer_id for l in doc["layers"]):
        raise ProjectBuilderError(f"legend references unknown layer '{layer_id}'")
    entry: Dict[str, Any] = {"layer": layer_id, "title": str(title), "kind": kind}
    if color_ramp is not None:
        entry["color_ramp"] = color_ramp
    if vmin is not None:
        entry["vmin"] = float(vmin)
    if vmax is not None:
        entry["vmax"] = float(vmax)
    if unit is not None:
        entry["unit"] = unit
    doc["legends"].append(entry)
    return doc


# ---------------------------------------------------------------------------
# validation / summary
# ---------------------------------------------------------------------------


def validate_project(doc: Dict[str, Any]) -> List[str]:
    """Return a list of structural problems (empty list = valid)."""
    problems: List[str] = []
    if not isinstance(doc, dict):
        return ["document is not a JSON object"]
    for key in ("format_version", "generator", "name", "basemap", "center",
                "zoom", "layers"):
        if key not in doc:
            problems.append(f"missing required key '{key}'")
    if doc.get("generator") != GENERATOR:
        problems.append(f"generator is '{doc.get('generator')}', expected '{GENERATOR}'")
    center = doc.get("center")
    if not (isinstance(center, list) and len(center) == 2):
        problems.append("center must be [lon, lat]")
    zoom = doc.get("zoom")
    if not isinstance(zoom, (int, float)) or not (0 <= zoom <= 24):
        problems.append("zoom must be a number in [0, 24]")
    seen_ids = set()
    for i, layer in enumerate(doc.get("layers") or []):
        lid = layer.get("id")
        if not lid:
            problems.append(f"layer[{i}] missing id")
        elif lid in seen_ids:
            problems.append(f"duplicate layer id '{lid}'")
        seen_ids.add(lid)
        ltype = layer.get("type")
        if ltype not in LAYER_TYPES:
            problems.append(f"layer '{lid}' has unknown type '{ltype}'")
            continue
        src = layer.get("source") or {}
        if ltype == "xyz-tile" and not src.get("tiles"):
            problems.append(f"layer '{lid}' (xyz-tile) missing source.tiles")
        if ltype in ("cog", "geoparquet") and not src.get("url"):
            problems.append(f"layer '{lid}' ({ltype}) missing source.url")
        if ltype in ("geojson", "heatmap") and "data" not in src:
            problems.append(f"layer '{lid}' ({ltype}) missing source.data")
        op = layer.get("opacity", 1.0)
        if not isinstance(op, (int, float)) or not (0 <= op <= 1):
            problems.append(f"layer '{lid}' opacity out of range")
    layer_ids = {l.get("id") for l in doc.get("layers") or []}
    for legend in doc.get("legends") or []:
        if legend.get("layer") not in layer_ids:
            problems.append(f"legend references unknown layer '{legend.get('layer')}'")
    return problems


def describe_project(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and summarize a project document.

    Never raises on bad structure — returns ``{"valid": False, "problems": [...]}``.
    """
    problems = validate_project(doc)
    layers = doc.get("layers") or [] if isinstance(doc, dict) else []
    by_type: Dict[str, int] = {}
    inline_features = 0
    for layer in layers:
        by_type[layer.get("type", "?")] = by_type.get(layer.get("type", "?"), 0) + 1
        src = layer.get("source") or {}
        data = src.get("data")
        if isinstance(data, dict) and data.get("type") == "FeatureCollection":
            inline_features += len(data.get("features") or [])
    return {
        "valid": not problems,
        "problems": problems,
        "name": doc.get("name") if isinstance(doc, dict) else None,
        "generator": doc.get("generator") if isinstance(doc, dict) else None,
        "format_version": doc.get("format_version") if isinstance(doc, dict) else None,
        "basemap": doc.get("basemap") if isinstance(doc, dict) else None,
        "center": doc.get("center") if isinstance(doc, dict) else None,
        "zoom": doc.get("zoom") if isinstance(doc, dict) else None,
        "n_layers": len(layers),
        "layers_by_type": by_type,
        "layer_ids": [l.get("id") for l in layers],
        "inline_feature_count": inline_features,
        "n_legends": len(doc.get("legends") or []) if isinstance(doc, dict) else 0,
        "size_bytes": len(json.dumps(doc)) if isinstance(doc, dict) else 0,
    }
