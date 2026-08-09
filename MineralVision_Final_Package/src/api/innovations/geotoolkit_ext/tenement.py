"""Innovation 9 — tenement-guard.

Containment checks of entity points/drillholes against tenement polygons,
plus expiry-date obligation watch persisted to sqlite via SQLAlchemy.
"""

import os
import tempfile
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from shapely.geometry import shape, Point
from sqlalchemy import create_engine, String, Boolean, Date, Integer, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

try:
    from src.api.innovations.geotoolkit_ext.geo_common import fc_to_geometries
except ImportError:  # pragma: no cover
    from api.innovations.geotoolkit_ext.geo_common import fc_to_geometries

router = APIRouter()

# ---------------------------------------------------------------- database
_DB_PATH = os.environ.get(
    "GEOTOOLKIT_EXT_DB",
    os.path.join(tempfile.gettempdir(), "geotoolkit_ext_tenements.sqlite3"),
)
_engine = create_engine(f"sqlite:///{_DB_PATH}", echo=False)


class Base(DeclarativeBase):
    pass


class Tenement(Base):
    __tablename__ = "tenements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    expiry_date: Mapped[date] = mapped_column(Date)
    obligations: Mapped[List["Obligation"]] = relationship(
        back_populates="tenement", cascade="all, delete-orphan")


class Obligation(Base):
    __tablename__ = "tenement_obligations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenement_id: Mapped[int] = mapped_column(ForeignKey("tenements.id"))
    description: Mapped[str] = mapped_column(String)
    due_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    met: Mapped[bool] = mapped_column(Boolean, default=False)
    tenement: Mapped[Tenement] = relationship(back_populates="obligations")


Base.metadata.create_all(_engine)


def reset_engine(db_path: str) -> None:
    """Test helper: rebind storage to a fresh sqlite file."""
    global _engine, _DB_PATH
    _DB_PATH = db_path
    _engine = create_engine(f"sqlite:///{_DB_PATH}", echo=False)
    Base.metadata.create_all(_engine)


# ---------------------------------------------------------------- models
class CheckRequest(BaseModel):
    tenements: Dict[str, Any]                     # GeoJSON FeatureCollection (polygons)
    points: Optional[Dict[str, Any]] = None       # GeoJSON FeatureCollection (points)
    entities: Optional[List[Dict[str, Any]]] = None  # [{name/id, lon, lat}] shortcut


class ObligationIn(BaseModel):
    description: str
    due_date: Optional[str] = None  # ISO date
    met: bool = False


class ExpiryWatchRequest(BaseModel):
    name: str
    expiry_date: str  # ISO date
    obligations: List[ObligationIn] = []


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return datetime.fromisoformat(s).date()


# ---------------------------------------------------------------- endpoints
@router.post("/tenements/check")
def tenements_check(req: CheckRequest) -> Dict[str, Any]:
    ten_feats = fc_to_geometries(req.tenements)
    if not ten_feats:
        raise HTTPException(status_code=422, detail="no tenement polygons supplied")

    pts: List[Dict[str, Any]] = []
    if req.points is not None:
        for geom, props in fc_to_geometries(req.points):
            if geom.geom_type != "Point":
                raise HTTPException(status_code=422, detail="points layer must contain Point geometries")
            pts.append({"geometry": geom, "properties": props})
    if req.entities is not None:
        for e in req.entities:
            lon = e.get("lon", e.get("longitude"))
            lat = e.get("lat", e.get("latitude"))
            if lon is None or lat is None:
                raise HTTPException(status_code=422, detail="entity needs lon/lat")
            pts.append({"geometry": Point(float(lon), float(lat)),
                        "properties": {k: v for k, v in e.items()}})
    if not pts:
        raise HTTPException(status_code=422, detail="supply points FeatureCollection or entities list")

    results = []
    violations = []
    for i, p in enumerate(pts):
        g: Point = p["geometry"]
        containing = []
        for tgeom, tprops in ten_feats:
            if tgeom.covers(g):
                containing.append(tprops.get("name", tprops.get("id", "unnamed")))
        entry = {
            "index": i,
            "name": p["properties"].get("name", p["properties"].get("id", f"point_{i}")),
            "coordinates": [g.x, g.y],
            "within_tenements": containing,
            "compliant": len(containing) > 0,
        }
        results.append(entry)
        if not containing:
            violations.append({
                "type": "outside_tenement",
                "name": entry["name"],
                "coordinates": [g.x, g.y],
                "detail": "entity lies outside all supplied tenement polygons",
            })
    return {
        "checked": len(results),
        "results": results,
        "violations": violations,
        "violation_count": len(violations),
    }


@router.post("/tenements/expiry-watch", status_code=201)
def expiry_watch(req: ExpiryWatchRequest) -> Dict[str, Any]:
    exp = _parse_date(req.expiry_date)
    with Session(_engine) as s:
        existing = s.query(Tenement).filter_by(name=req.name).one_or_none()
        if existing is not None:
            s.delete(existing)
            s.flush()
        t = Tenement(name=req.name, expiry_date=exp)
        for ob in req.obligations:
            t.obligations.append(Obligation(
                description=ob.description,
                due_date=_parse_date(ob.due_date) if ob.due_date else None,
                met=ob.met,
            ))
        s.add(t)
        s.commit()
        s.refresh(t)
        return {
            "id": t.id,
            "name": t.name,
            "expiry_date": t.expiry_date.isoformat(),
            "days_until_expiry": (t.expiry_date - date.today()).days,
            "obligations": [
                {"description": o.description,
                 "due_date": o.due_date.isoformat() if o.due_date else None,
                 "met": o.met}
                for o in t.obligations
            ],
        }


@router.get("/tenements/alerts")
def tenement_alerts(within_days: int = Query(default=30, ge=0)) -> Dict[str, Any]:
    today = date.today()
    with Session(_engine) as s:
        alerts = []
        for t in s.query(Tenement).all():
            days = (t.expiry_date - today).days
            unmet = [
                {"description": o.description,
                 "due_date": o.due_date.isoformat() if o.due_date else None,
                 "overdue": bool(o.due_date and o.due_date < today)}
                for o in t.obligations if not o.met
            ]
            expiring = 0 <= days <= within_days
            if expiring or unmet:
                alerts.append({
                    "name": t.name,
                    "expiry_date": t.expiry_date.isoformat(),
                    "days_until_expiry": days,
                    "expiring_within_window": expiring,
                    "expired": days < 0,
                    "unmet_obligations": unmet,
                    "unmet_obligation_count": len(unmet),
                })
    return {"as_of": today.isoformat(), "within_days": within_days, "alerts": alerts}
