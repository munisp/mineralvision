"""Deterministic extraction of exploration entities from report text.

Regex/lexicon-based NLP (no ML): hole IDs (DH/RC/DD patterns), assay
intervals (from-to depths + value + unit + commodity), commodities, dates
(ISO / long / numeric), and coordinates (UTM / decimal degrees).

Every extraction carries a deterministic confidence score derived from how
many structural components of the pattern matched.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

COMMODITIES: Dict[str, str] = {
    "Au": "gold", "Ag": "silver", "Cu": "copper", "Pb": "lead", "Zn": "zinc",
    "Ni": "nickel", "Co": "cobalt", "Li": "lithium", "Fe": "iron", "U": "uranium",
    "Sn": "tin", "W": "tungsten", "Mo": "molybdenum", "Pt": "platinum",
    "Pd": "palladium", "REE": "rare earths",
}
_COMMODITY_NAMES = {name: sym for sym, name in COMMODITIES.items()}

ASSAY_UNITS = ("g/t", "ppm", "ppb", "%", "oz/t")

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

HOLE_ID_RE = re.compile(r"\b(DH|RC|DD)(\d{1,4})([A-Za-z]?)\b")

# "RC001: 45-46m @ 2.34 g/t Au" / "from 45.0 to 46.0 m ... 2.34 g/t Au"
_INTERVAL_FROMTO_RE = re.compile(
    r"(?P<hole>\b(?:DH|RC|DD)\d{1,4}[A-Za-z]?\b)?[^\n]{0,40}?"
    r"\bfrom\s+(?P<from>\d+(?:\.\d+)?)\s*(?:m|metres?)?\s*to\s+(?P<to>\d+(?:\.\d+)?)\s*(?:m|metres?)\b"
    r"[^\n]{0,30}?(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>g/t|ppm|ppb|%|oz/t)\s*(?P<commodity>[A-Za-z]{1,3})?",
    re.IGNORECASE,
)
_INTERVAL_DASH_RE = re.compile(
    r"(?P<hole>\b(?:DH|RC|DD)\d{1,4}[A-Za-z]?\b)?[^\n]{0,30}?"
    r"\b(?P<from>\d+(?:\.\d+)?)\s*[-–]\s*(?P<to>\d+(?:\.\d+)?)\s*(?:m|metres?)\b"
    r"\s*[@=:,]?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>g/t|ppm|ppb|%|oz/t)\s*(?P<commodity>[A-Za-z]{1,3})?",
    re.IGNORECASE,
)

_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_LONG_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# UTM: "50J 500500E 7500500N" / "zone 50, 500500 mE 7500500 mN"
_UTM_RE = re.compile(
    r"\b(?:(?:zone\s*)?(\d{1,2})([A-Za-z])?[\s,]+)?(\d{5,7}(?:\.\d+)?)\s*(?:m)?E[\s,]+(\d{6,8}(?:\.\d+)?)\s*(?:m)?N\b",
    re.IGNORECASE,
)
# Decimal degrees: "lat -23.4567, lon 119.5678" / "-23.4567, 119.5678"
_DECDEG_LABELED_RE = re.compile(
    r"\blat(?:itude)?\s*[:=]?\s*(-?\d{1,2}\.\d+)\s*[,;\s]+\s*lon(?:gitude)?\s*[:=]?\s*(-?\d{1,3}\.\d+)\b",
    re.IGNORECASE,
)
_DECDEG_BARE_RE = re.compile(r"(?<![\d.])(-?\d{1,2}\.\d{3,})\s*,\s*(-?\d{1,3}\.\d{3,})(?!\d)")


@dataclass
class Extraction:
    kind: str
    value: Dict[str, Any]
    confidence: float
    span: Tuple[int, int]


@dataclass
class ExtractionResult:
    hole_ids: List[Dict[str, Any]] = field(default_factory=list)
    intervals: List[Dict[str, Any]] = field(default_factory=list)
    commodities: List[Dict[str, Any]] = field(default_factory=list)
    dates: List[Dict[str, Any]] = field(default_factory=list)
    coordinates: List[Dict[str, Any]] = field(default_factory=list)


def _norm_commodity(token: Optional[str]) -> Optional[Dict[str, str]]:
    if not token:
        return None
    if token in COMMODITIES:
        return {"symbol": token, "name": COMMODITIES[token]}
    lower = token.lower()
    if lower in _COMMODITY_NAMES:
        return {"symbol": _COMMODITY_NAMES[lower], "name": lower}
    return None


def extract_hole_ids(text: str) -> List[Dict[str, Any]]:
    """Unique hole IDs in first-appearance order. Confidence 0.99 (strict pattern)."""
    seen: Dict[str, Tuple[int, int]] = {}
    for m in HOLE_ID_RE.finditer(text):
        hole = f"{m.group(1).upper()}{m.group(2)}{m.group(3)}"
        if hole not in seen:
            seen[hole] = (m.start(), m.end())
    return [{"hole_id": h, "confidence": 0.99, "span": list(span)} for h, span in seen.items()]


def extract_intervals(text: str) -> List[Dict[str, Any]]:
    """Assay intervals with from/to depths, value, unit, commodity.

    Confidence: 0.60 base for (from,to,value,unit) + 0.15 hole + 0.15
    commodity + 0.10 explicit 'from..to' phrasing.
    """
    results: List[Dict[str, Any]] = []
    consumed: List[Tuple[int, int]] = []
    for regex, phrasing_bonus in ((_INTERVAL_FROMTO_RE, 0.10), (_INTERVAL_DASH_RE, 0.0)):
        for m in regex.finditer(text):
            if any(m.start() >= a and m.end() <= b for a, b in consumed):
                continue
            from_m, to_m = float(m.group("from")), float(m.group("to"))
            if to_m <= from_m or to_m - from_m > 1000:
                continue
            value = float(m.group("value"))
            commodity = _norm_commodity(m.group("commodity"))
            hole = m.group("hole").upper() if m.group("hole") else None
            confidence = 0.60 + phrasing_bonus + (0.15 if hole else 0.0) + (0.15 if commodity else 0.0)
            results.append({
                "hole_id": hole,
                "from_m": from_m,
                "to_m": to_m,
                "value": value,
                "unit": m.group("unit"),
                "commodity": commodity,
                "confidence": round(confidence, 2),
                "span": [m.start(), m.end()],
            })
            consumed.append((m.start(), m.end()))
    results.sort(key=lambda r: r["span"][0])
    return results


def extract_commodities(text: str) -> List[Dict[str, Any]]:
    """Commodity mentions (symbol or full name), unique, with counts."""
    counts: Dict[str, int] = {}
    for symbol, name in COMMODITIES.items():
        n = len(re.findall(rf"\b{re.escape(symbol)}\b", text))
        n += len(re.findall(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE))
        if symbol == "REE":
            n += len(re.findall(r"\brare\s+earths?\b", text, flags=re.IGNORECASE))
        if n:
            counts[symbol] = n
    return [
        {"symbol": sym, "name": COMMODITIES[sym], "mentions": n, "confidence": 0.9}
        for sym, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def extract_dates(text: str) -> List[Dict[str, Any]]:
    """Dates normalized to ISO. Confidence: 0.98 ISO, 0.9 long, 0.7 numeric (D/M order)."""
    results: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []
    for m in _DATE_ISO_RE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            results.append({"date": f"{y:04d}-{mo:02d}-{d:02d}", "confidence": 0.98, "span": [m.start(), m.end()]})
            occupied.append((m.start(), m.end()))
    for m in _DATE_LONG_RE.finditer(text):
        d, mon, y = int(m.group(1)), _MONTHS.index(m.group(2).lower()[:3]) + 1, int(m.group(3))
        if 1 <= d <= 31:
            results.append({"date": f"{y:04d}-{mon:02d}-{d:02d}", "confidence": 0.9, "span": [m.start(), m.end()]})
            occupied.append((m.start(), m.end()))
    for m in _DATE_NUMERIC_RE.finditer(text):
        if any(m.start() >= a and m.end() <= b for a, b in occupied):
            continue
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            results.append({"date": f"{y:04d}-{mo:02d}-{d:02d}", "confidence": 0.7, "span": [m.start(), m.end()]})
    results.sort(key=lambda r: r["span"][0])
    return results


def extract_coordinates(text: str) -> List[Dict[str, Any]]:
    """UTM and decimal-degree coordinates.

    Confidence: 0.95 labeled UTM (with zone), 0.8 UTM without zone,
    0.95 labeled decdeg, 0.75 bare decdeg pair.
    """
    results: List[Dict[str, Any]] = []
    for m in _UTM_RE.finditer(text):
        east, north = float(m.group(3)), float(m.group(4))
        if not (100000 <= east <= 999999 and 0 <= north <= 10000000):
            continue
        zone = f"{m.group(1)}{m.group(2).upper() if m.group(2) else ''}" if m.group(1) else None
        results.append({
            "system": "UTM", "zone": zone, "east": east, "north": north,
            "confidence": 0.95 if zone else 0.8, "span": [m.start(), m.end()],
        })
    labeled_spans: List[Tuple[int, int]] = []
    for m in _DECDEG_LABELED_RE.finditer(text):
        lat, lon = float(m.group(1)), float(m.group(2))
        if abs(lat) <= 90 and abs(lon) <= 180:
            results.append({"system": "decdeg", "lat": lat, "lon": lon, "confidence": 0.95,
                            "span": [m.start(), m.end()]})
            labeled_spans.append((m.start(), m.end()))
    for m in _DECDEG_BARE_RE.finditer(text):
        if any(m.start() >= a and m.end() <= b for a, b in labeled_spans):
            continue
        lat, lon = float(m.group(1)), float(m.group(2))
        if abs(lat) <= 90 and abs(lon) <= 180:
            results.append({"system": "decdeg", "lat": lat, "lon": lon, "confidence": 0.75,
                            "span": [m.start(), m.end()]})
    results.sort(key=lambda r: r["span"][0])
    return results


def extract_all(text: str) -> ExtractionResult:
    return ExtractionResult(
        hole_ids=extract_hole_ids(text),
        intervals=extract_intervals(text),
        commodities=extract_commodities(text),
        dates=extract_dates(text),
        coordinates=extract_coordinates(text),
    )


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes via pypdf (lazy import).

    Raises RuntimeError("pypdf_unavailable") when pypdf is not installed —
    the route maps that to 501 Not Implemented.
    """
    try:
        from pypdf import PdfReader  # lazy: optional dependency
    except ImportError as exc:
        raise RuntimeError("pypdf_unavailable") from exc
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
