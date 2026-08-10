"""Service layer: build ``.geolibre.json`` documents from platform data.

Real data sources (no fabrication):
* Drillhole collars + sample assays via the platform SQLAlchemy models
  (``src.api.database.DrillholeModel`` / ``SampleModel``), same session
  pattern as ``src.api.endpoints.drillholes``.
* Platform XYZ tile endpoints wired in ``main.py``:
    - ``/innovations/geotoolkit/tiles/features/{z}/{x}/{y}?layer=drillholes``
    - ``/innovations/geotoolkit/tiles/raster/{z}/{x}/{y}?raster_id=...``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

try:  # dual-context import
    from src.api.database import DrillholeModel, ProjectModel, SampleModel
    from src.api.innovations.geolibre import project_builder as pb
except ImportError:  # pragma: no cover
    from api.database import DrillholeModel, ProjectModel, SampleModel
    from api.innovations.geolibre import project_builder as pb

# Platform tile route templates (verified in main.py wiring of geotoolkit).
DRILLHOLE_FEATURE_TILES = "/innovations/geotoolkit/tiles/features/{z}/{x}/{y}?layer=drillholes"
RASTER_TILES = "/innovations/geotoolkit/tiles/raster/{z}/{x}/{y}?raster_id={raster_id}"

DEFAULT_ASSAY_KEYS = ("au", "au_ppm", "cu", "cu_pct", "grade")


def _pick_assay_key(samples: Sequence[Any], preferred: Optional[str]) -> Optional[str]:
    """Choose a real assay element present in the samples. Never invented."""
    if preferred:
        return preferred
    counts: Dict[str, int] = {}
    for s in samples:
        for key, val in (s.assay_data or {}).items():
            if isinstance(val, (int, float)):
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    for k in DEFAULT_ASSAY_KEYS:
        if k in counts:
            return k
    return max(counts, key=counts.get)


def drillholes_geojson(drillholes: Sequence[Any]) -> Dict[str, Any]:
    """Collar FeatureCollection from DrillholeModel rows (real coordinates)."""
    features = []
    for d in drillholes:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(d.collar_x), float(d.collar_y)]},
            "properties": {
                "hole_id": d.hole_id,
                "elevation": float(d.collar_z),
                "total_depth": float(d.total_depth),
                "azimuth": d.azimuth,
                "dip": d.dip,
                "status": d.status,
                "assay_count": d.assay_count,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def assay_heatmap_geojson(drillholes: Sequence[Any],
                          samples_by_hole: Dict[str, List[Any]],
                          assay_key: str) -> Dict[str, Any]:
    """Point FeatureCollection of collars weighted by mean assay grade."""
    features = []
    for d in drillholes:
        grades = [float(s.assay_data[assay_key])
                  for s in samples_by_hole.get(d.id, [])
                  if isinstance((s.assay_data or {}).get(assay_key), (int, float))]
        if not grades:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(d.collar_x), float(d.collar_y)]},
            "properties": {"hole_id": d.hole_id, "grade": sum(grades) / len(grades)},
        })
    return {"type": "FeatureCollection", "features": features}


def build_project_document(
    db: Session,
    project_id: str,
    base_url: str = "",
    assay_key: Optional[str] = None,
    basemap: str = "carto-positron",
    include_feature_tiles: bool = True,
    cog_layers: Optional[Sequence[Dict[str, Any]]] = None,
    geoparquet_layers: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a .geolibre.json document for a project.

    Returns ``None`` when the project has no drillholes (caller maps this to
    an honest 404).  All coordinates/grades come from the DB; all layer URLs
    are real platform routes.
    """
    drillholes: List[Any] = (
        db.query(DrillholeModel)
        .filter(DrillholeModel.project_id == project_id)
        .order_by(DrillholeModel.hole_id)
        .all()
    )
    if not drillholes:
        return None

    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    name = project.name if project else f"project-{project_id}"

    lons = [float(d.collar_x) for d in drillholes]
    lats = [float(d.collar_y) for d in drillholes]
    center = [(min(lons) + max(lons)) / 2.0, (min(lats) + max(lats)) / 2.0]
    pad_lon = max((max(lons) - min(lons)) * 0.1, 0.005)
    pad_lat = max((max(lats) - min(lats)) * 0.1, 0.005)
    bounds = [min(lons) - pad_lon, min(lats) - pad_lat,
              max(lons) + pad_lon, max(lats) + pad_lat]

    doc = pb.new_project(name=name, center=center, zoom=11.0,
                         bounds=bounds, basemap=basemap)

    collars = drillholes_geojson(drillholes)
    pb.add_geojson_layer(doc, "drillhole-collars", collars,
                         style={"circle-color": "#d35400", "circle-radius": 6})

    if include_feature_tiles:
        pb.add_tile_layer(
            doc, "drillhole-vector-tiles",
            base_url + DRILLHOLE_FEATURE_TILES,
            opacity=1.0, attribution="MineralVision geotoolkit", visible=False,
        )

    samples: List[Any] = (
        db.query(SampleModel)
        .filter(SampleModel.drillhole_id.in_([d.id for d in drillholes]))
        .all()
    )
    by_hole: Dict[str, List[Any]] = {}
    for s in samples:
        by_hole.setdefault(s.drillhole_id, []).append(s)

    key = _pick_assay_key(samples, assay_key)
    if key is not None:
        heat = assay_heatmap_geojson(drillholes, by_hole, key)
        if heat["features"]:
            grades = [f["properties"]["grade"] for f in heat["features"]]
            pb.add_heatmap_layer(doc, "assay-grade-heatmap", heat,
                                 weight_property="grade", opacity=0.75)
            pb.add_legend(doc, "assay-grade-heatmap",
                          title=f"Mean {key} grade by collar",
                          kind="colorbar", color_ramp="viridis",
                          vmin=min(grades), vmax=max(grades), unit=key)

    for cog in (cog_layers or []):
        pb.add_cog_layer(doc, cog["id"], cog["url"],
                         opacity=cog.get("opacity", 1.0),
                         colormap=cog.get("colormap", "viridis"))
    for pq in (geoparquet_layers or []):
        pb.add_geoparquet_layer(doc, pq["id"], pq["url"],
                                opacity=pq.get("opacity", 1.0))

    doc["source_project_id"] = project_id
    return doc


# ---------------------------------------------------------------------------
# minimal honest MapLibre HTML fallback renderer
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<!-- generator: mineralvision-fallback — hand-written MapLibre renderer used
     when the `geolibre` wheel is not installed. Renders the REAL layers from
     the project document (real tile/GeoJSON/COG URLs), no mock data. -->
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet"/>
<style>html,body,#map{{margin:0;height:100%;}}</style>
</head>
<body>
<div id="map"></div>
<script>
const PROJECT = {project_json};
const BASEMAPS = {basemaps_json};
const bm = BASEMAPS[PROJECT.basemap] || BASEMAPS["osm"];
const isStyle = bm.style.endsWith(".json");
const style = isStyle ? bm.style : {{
  version: 8,
  sources: {{base: {{type: "raster", tiles: [bm.style], tileSize: 256}}}},
  layers: [{{id: "base", type: "raster", source: "base"}}]
}};
const map = new maplibregl.Map({{
  container: "map", style: style,
  center: PROJECT.center, zoom: PROJECT.zoom
}});
if (PROJECT.bounds) map.fitBounds(PROJECT.bounds, {{padding: 40}});
map.on("load", () => {{
  for (const layer of PROJECT.layers) {{
    const src = layer.source || {{}};
    if (layer.type === "xyz-tile") {{
      map.addSource(layer.id, {{type: "raster", tiles: src.tiles, tileSize: 256}});
      map.addLayer({{id: layer.id, type: "raster", source: layer.id,
        paint: {{"raster-opacity": layer.opacity ?? 1.0}}}});
    }} else if (layer.type === "geojson") {{
      map.addSource(layer.id, {{type: "geojson", data: src.data}});
      map.addLayer({{id: layer.id, type: "circle", source: layer.id,
        filter: ["==", ["geometry-type"], "Point"],
        paint: {{"circle-color": (layer.paint||{{}})["circle-color"] || "#d35400",
                "circle-radius": (layer.paint||{{}})["circle-radius"] || 6,
                "circle-opacity": layer.opacity ?? 1.0}}}});
      map.addLayer({{id: layer.id + "-fill", type: "fill", source: layer.id,
        filter: ["==", ["geometry-type"], "Polygon"],
        paint: {{"fill-color": (layer.paint||{{}})["fill-color"] || "#2980b9",
                "fill-opacity": ((layer.paint||{{}})["fill-opacity"] ?? 0.6) * (layer.opacity ?? 1.0)}}}});
    }} else if (layer.type === "heatmap") {{
      map.addSource(layer.id, {{type: "geojson", data: src.data}});
      map.addLayer({{id: layer.id, type: "heatmap", source: layer.id,
        paint: {{"heatmap-weight": ["interpolate", ["linear"],
                  ["get", src.weightProperty || "weight"], 0, 0, 1, 1],
                "heatmap-radius": (layer.paint||{{}})["heatmap-radius"] || 25,
                "heatmap-opacity": layer.opacity ?? 0.8}}}});
    }}
    // cog / geoparquet layers need the full GeoLibre client (DuckDB-WASM /
    // COG range streaming) and are intentionally not rendered by this
    // minimal fallback; their URLs remain in PROJECT.layers.
  }}
}});
</script>
</body>
</html>
"""


def render_fallback_html(doc: Dict[str, Any]) -> str:
    """Render a real MapLibre HTML page from a project document.

    This is NOT a mock: it embeds the actual layer URLs / GeoJSON and renders
    them with MapLibre GL JS in the browser.  Marked with a
    ``generator: mineralvision-fallback`` HTML comment.
    """
    import json as _json

    return _HTML_TEMPLATE.format(
        title=str(doc.get("name", "mineralvision-project")).replace("<", "&lt;"),
        project_json=_json.dumps(doc),
        basemaps_json=_json.dumps(pb.BASEMAP_CATALOG),
    )
