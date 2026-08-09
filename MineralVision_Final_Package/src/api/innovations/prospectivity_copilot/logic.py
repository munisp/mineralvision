"""
Deterministic natural-language exploration query parser (NO LLM).

Pipeline: tokenize -> lexicon extraction (commodities from the periodic table
plus a mineral->commodity map, deposit types, regions) -> numeric constraint
parsing ("gold > 2 g/t", "within 5km of ...") -> intent classification
(rank/list/count/explain) -> structured query (AST) -> SQLAlchemy execution
over ProjectModel / DrillholeModel / SampleModel.

The parser is fully deterministic: regular expressions + curated lexicons.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Periodic table: element name -> symbol (all 118, lowercase names).
PERIODIC_TABLE: Dict[str, str] = {
    "hydrogen": "H", "helium": "He", "lithium": "Li", "beryllium": "Be",
    "boron": "B", "carbon": "C", "nitrogen": "N", "oxygen": "O",
    "fluorine": "F", "neon": "Ne", "sodium": "Na", "magnesium": "Mg",
    "aluminium": "Al", "aluminum": "Al", "silicon": "Si", "phosphorus": "P",
    "sulfur": "S", "sulphur": "S", "chlorine": "Cl", "argon": "Ar",
    "potassium": "K", "calcium": "Ca", "scandium": "Sc", "titanium": "Ti",
    "vanadium": "V", "chromium": "Cr", "manganese": "Mn", "iron": "Fe",
    "cobalt": "Co", "nickel": "Ni", "copper": "Cu", "zinc": "Zn",
    "gallium": "Ga", "germanium": "Ge", "arsenic": "As", "selenium": "Se",
    "bromine": "Br", "krypton": "Kr", "rubidium": "Rb", "strontium": "Sr",
    "yttrium": "Y", "zirconium": "Zr", "niobium": "Nb", "molybdenum": "Mo",
    "technetium": "Tc", "ruthenium": "Ru", "rhodium": "Rh", "palladium": "Pd",
    "silver": "Ag", "cadmium": "Cd", "indium": "In", "tin": "Sn",
    "antimony": "Sb", "tellurium": "Te", "iodine": "I", "xenon": "Xe",
    "cesium": "Cs", "caesium": "Cs", "barium": "Ba", "lanthanum": "La",
    "cerium": "Ce", "praseodymium": "Pr", "neodymium": "Nd", "promethium": "Pm",
    "samarium": "Sm", "europium": "Eu", "gadolinium": "Gd", "terbium": "Tb",
    "dysprosium": "Dy", "holmium": "Ho", "erbium": "Er", "thulium": "Tm",
    "ytterbium": "Yb", "lutetium": "Lu", "hafnium": "Hf", "tantalum": "Ta",
    "tungsten": "W", "rhenium": "Re", "osmium": "Os", "iridium": "Ir",
    "platinum": "Pt", "gold": "Au", "mercury": "Hg", "thallium": "Tl",
    "lead": "Pb", "bismuth": "Bi", "polonium": "Po", "astatine": "At",
    "radon": "Rn", "francium": "Fr", "radium": "Ra", "actinium": "Ac",
    "thorium": "Th", "protactinium": "Pa", "uranium": "U", "neptunium": "Np",
    "plutonium": "Pu", "americium": "Am", "curium": "Cm", "berkelium": "Bk",
    "californium": "Cf", "einsteinium": "Es", "fermium": "Fm", "mendelevium": "Md",
    "nobelium": "No", "lawrencium": "Lr", "rutherfordium": "Rf", "dubnium": "Db",
    "seaborgium": "Sg", "bohrium": "Bh", "hassium": "Hs", "meitnerium": "Mt",
    "darmstadtium": "Ds", "roentgenium": "Rg", "copernicium": "Cn",
    "nihonium": "Nh", "flerovium": "Fl", "moscovium": "Mc", "livermorium": "Lv",
    "tennessine": "Ts", "oganesson": "Og",
}

# Symbol -> canonical element name (for echo/explanations).
SYMBOL_TO_NAME: Dict[str, str] = {}
for _name, _sym in PERIODIC_TABLE.items():
    SYMBOL_TO_NAME.setdefault(_sym, _name)

# Commodity group aliases -> member element names.
COMMODITY_GROUPS: Dict[str, List[str]] = {
    "rare earths": ["lanthanum", "cerium", "praseodymium", "neodymium",
                    "samarium", "europium", "gadolinium", "terbium",
                    "dysprosium", "holmium", "erbium", "thulium",
                    "ytterbium", "lutetium", "yttrium"],
    "rare earth elements": ["lanthanum", "cerium", "praseodymium", "neodymium"],
    "ree": ["lanthanum", "cerium", "neodymium"],
    "pgm": ["platinum", "palladium", "rhodium", "ruthenium", "iridium", "osmium"],
    "platinum group": ["platinum", "palladium"],
    "base metals": ["copper", "lead", "zinc", "nickel"],
    "battery metals": ["lithium", "cobalt", "nickel", "carbon", "manganese"],
    "precious metals": ["gold", "silver", "platinum", "palladium"],
}

# Ore/industrial minerals -> primary commodity (element name).
MINERAL_TO_COMMODITY: Dict[str, str] = {
    "chalcopyrite": "copper", "bornite": "copper", "chalcocite": "copper",
    "malachite": "copper", "azurite": "copper",
    "sphalerite": "zinc", "galena": "lead", "pentlandite": "nickel",
    "scheelite": "tungsten", "wolframite": "tungsten", "cassiterite": "tin",
    "bauxite": "aluminium", "uraninite": "uranium", "pitchblende": "uranium",
    "magnetite": "iron", "hematite": "iron", "haematite": "iron",
    "goethite": "iron", "ilmenite": "titanium", "rutile": "titanium",
    "spodumene": "lithium", "lepidolite": "lithium", "petalite": "lithium",
    "molybdenite": "molybdenum", "stibnite": "antimony", "cinnabar": "mercury",
    "cobaltite": "cobalt", "chromite": "chromium", "pyrolusite": "manganese",
    "monazite": "cerium", "bastnasite": "cerium", "xenotime": "yttrium",
    "vanadinite": "vanadium", "argentite": "silver", "sperrylite": "platinum",
    "diamond": "carbon",
    "graphite": "carbon", "apatite": "phosphorus", "sylvite": "potassium",
    "barite": "barium", "baryte": "barium", "fluorite": "fluorine",
}

DEPOSIT_TYPES: Dict[str, str] = {
    "porphyry": "porphyry", "orogenic": "orogenic", "vms": "vms",
    "volcanogenic": "vms", "vhms": "vms", "epithermal": "epithermal",
    "skarn": "skarn", "iocg": "iocg", "laterite": "laterite",
    "placer": "placer", "mvt": "mvt", "sedex": "sedex",
    "sediment-hosted": "sediment-hosted", "sediment hosted": "sediment-hosted",
    "kimberlite": "kimberlite", "pegmatite": "pegmatite",
    "bif": "bif", "banded iron": "bif", "iron formation": "bif",
    "magmatic": "magmatic", "unconformity": "unconformity",
    "carlin": "carlin", "carlin-type": "carlin", "greisen": "greisen",
    "roll-front": "roll-front", "roll front": "roll-front",
    "sandstone-hosted": "sandstone-hosted", "intrusion-related": "intrusion-related",
}

REGIONS: List[str] = [
    "western australia", "south australia", "northern territory",
    "new south wales", "queensland", "victoria", "tasmania",
    "pilbara", "yilgarn", "kalgoorlie", "goldfields", "gawler", "musgrave",
    "british columbia", "ontario", "quebec", "yukon", "labrador",
    "nevada", "arizona", "alaska", "idaho", "montana",
    "chile", "peru", "andes", "argentina", "brazil", "mexico",
    "south africa", "west africa", "ghana", "mali", "tanzania", "botswana",
    "namibia", "zambia", "drc", "congo",
    "finland", "sweden", "norway", "ireland", "spain", "portugal",
    "kazakhstan", "mongolia", "china", "india", "indonesia",
    "papua new guinea", "new zealand", "saudi arabia", "saudi", "antarctica",
]

# Unit conversion to g/t. 1 g/t == 1 ppm by mass; 1 % = 10 000 g/t;
# 1 troy oz per metric tonne = 31.1034768 g/t.
UNIT_TO_GPT: Dict[str, float] = {
    "g/t": 1.0, "gpt": 1.0, "gppm": 1.0, "ppm": 1.0,
    "ppb": 1e-3, "%": 1e4, "percent": 1e4, "pct": 1e4,
    "oz/t": 31.1034768, "opt": 31.1034768,
}

_OPS = [
    (r"greater than or equal to|at least|minimum of|no less than", ">="),
    (r"less than or equal to|at most|maximum of|no more than|up to", "<="),
    (r"greater than|more than|above|over|exceeds?|exceeding|higher than", ">"),
    (r"less than|below|under|lower than", "<"),
    (r"equal to|equals?|of exactly", "="),
]

INTENT_KEYWORDS = {
    "count": ["how many", "count", "number of", "total number"],
    "explain": ["explain", "why", "reasoning", "rationale", "justify"],
    "rank": ["rank", "ranking", "top", "best", "highest", "most prospective",
             "most promising", "prioritize", "prioritise"],
    "list": ["list", "show", "find", "get", "which", "give me", "display"],
}

ENTITY_KEYWORDS = {
    "drillholes": ["drillhole", "drillholes", "drill hole", "drill holes",
                   "holes", "hole"],
    "samples": ["sample", "samples", "assay", "assays", "intercept",
                "intercepts", "interval", "intervals"],
    "projects": ["project", "projects", "prospect", "prospects", "tenement",
                 "tenements", "property", "properties"],
}

_GRADE_RE = re.compile(
    # commodity limited to at most two words so leading intent/entity words
    # ("rank drillholes where gold > 2") are not swallowed into the match.
    r"(?P<commodity>(?:[a-z]+\s)?[a-z]+)\s*"
    r"(?P<opword>greater than or equal to|at least|minimum of|no less than|"
    r"less than or equal to|at most|maximum of|no more than|up to|"
    r"greater than|more than|above|over|exceeds?|exceeding|higher than|"
    r"less than|below|under|lower than|equal to|equals?|of exactly|"
    r">=|<=|>|<|=)\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>g/t|gpt|ppm|ppb|oz/t|opt|%|percent|pct)?"
    r"(?:\s*(?:of|in)\s*(?P<trail>[a-z]+))?",
    re.IGNORECASE,
)

_SYMBOL_OP_RE = re.compile(
    r"(?P<commodity>[a-z]{1,3})\s*(?P<op>>=|<=|>|<|=)\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>g/t|gpt|ppm|ppb|oz/t|opt|%|percent|pct)?",
    re.IGNORECASE,
)

# "grade at least 1 g/t" — commodity resolved from the rest of the phrase.
_GENERIC_GRADE_RE = re.compile(
    r"(?:grades?|assays?|values?|intercept)\s*"
    r"(?P<opword>greater than or equal to|at least|minimum of|no less than|"
    r"less than or equal to|at most|maximum of|no more than|up to|"
    r"greater than|more than|above|over|exceeds?|exceeding|higher than|"
    r"less than|below|under|lower than|equal to|equals?|of exactly|"
    r">=|<=|>|<|=)\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>g/t|gpt|ppm|ppb|oz/t|opt|%|percent|pct)?",
    re.IGNORECASE,
)

_DISTANCE_RE = re.compile(
    r"within\s+(?P<dist>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>kilometers|kilometres|kms|km|meters|metres|miles|mile|mi|m)\s+"
    r"of\s+(?P<ref>[a-z0-9][a-z0-9 .\-']*?)\s*(?=$|[,.;!?])",
    re.IGNORECASE,
)

_DIST_UNIT_TO_KM = {
    "km": 1.0, "kms": 1.0, "kilometers": 1.0, "kilometres": 1.0,
    "m": 0.001, "meters": 0.001, "metres": 0.001,
    "mi": 1.609344, "mile": 1.609344, "miles": 1.609344,
}


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass
class GradeConstraint:
    commodity: str          # canonical element name
    symbol: str
    op: str                 # one of >, >=, <, <=, =
    value: float            # raw value as written
    unit: str               # unit as written (default g/t)
    value_gpt: float        # normalized to g/t

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DistanceConstraint:
    distance_km: float
    reference: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedQuery:
    intent: str = "list"                     # rank | list | count | explain
    entity: str = "projects"                 # projects | drillholes | samples
    commodities: List[str] = field(default_factory=list)   # canonical names
    symbols: List[str] = field(default_factory=list)
    deposit_types: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    grade_constraints: List[GradeConstraint] = field(default_factory=list)
    distance_constraint: Optional[DistanceConstraint] = None
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "entity": self.entity,
            "commodities": list(self.commodities),
            "symbols": list(self.symbols),
            "deposit_types": list(self.deposit_types),
            "regions": list(self.regions),
            "grade_constraints": [c.to_dict() for c in self.grade_constraints],
            "distance_constraint": (
                self.distance_constraint.to_dict() if self.distance_constraint else None
            ),
            "raw": self.raw,
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.strip().lower()
    # protect "g/t" style tokens from the slash
    text = text.replace("g / t", "g/t")
    return text


# Two-letter English words that collide with element symbols; always read as
# stopwords, never commodities (write "indium", not "in").
SYMBOL_STOPWORDS = {
    "in", "at", "of", "to", "or", "an", "as", "is", "be", "by", "on",
    "no", "so", "up", "we", "it", "me", "he", "am",
}


def _canonical_commodity(token: str) -> Optional[str]:
    token = token.strip()
    if token in PERIODIC_TABLE:
        return SYMBOL_TO_NAME[PERIODIC_TABLE[token]]
    cap = token.capitalize()  # "au" -> "Au"
    if cap in SYMBOL_TO_NAME and len(token) <= 2 and token not in SYMBOL_STOPWORDS:
        return SYMBOL_TO_NAME[cap]
    if token in MINERAL_TO_COMMODITY:
        return MINERAL_TO_COMMODITY[token]
    return None


def parse_query(question: str) -> ParsedQuery:
    """Parse a natural-language exploration question into a structured AST.

    Deterministic: regex + lexicons only, no LLM calls.
    """
    raw = question
    text = _normalize(question)
    pq = ParsedQuery(raw=raw)

    # ---- distance constraint ("within 5 km of Kalgoorlie") ----------------
    m = _DISTANCE_RE.search(text)
    if m:
        dist_km = float(m.group("dist")) * _DIST_UNIT_TO_KM[m.group("unit")]
        ref = m.group("ref").strip(" .")
        pq.distance_constraint = DistanceConstraint(distance_km=dist_km, reference=ref)
        text = text[: m.start()] + " " + text[m.end():]

    # ---- grade constraints -------------------------------------------------
    consumed_spans: List[tuple] = []
    for m in _SYMBOL_OP_RE.finditer(text):
        com = _canonical_commodity(m.group("commodity"))
        if com is None:
            continue
        unit = (m.group("unit") or "g/t").lower()
        value = float(m.group("value"))
        pq.grade_constraints.append(GradeConstraint(
            commodity=com, symbol=PERIODIC_TABLE.get(com, SYMBOL_TO_NAME.get(
                PERIODIC_TABLE.get(com, ""), com)),
            op=m.group("op"), value=value, unit=unit,
            value_gpt=value * UNIT_TO_GPT.get(unit, 1.0),
        ))
        consumed_spans.append((m.start(), m.end()))

    for m in _GRADE_RE.finditer(text):
        # skip matches overlapping an already-consumed symbol-style match
        if any(m.start() < e and m.end() > s for s, e in consumed_spans):
            continue
        # commodity word(s) directly preceding the operator: take last token
        words = m.group("commodity").split()
        if not words:
            continue
        com = _canonical_commodity(words[-1])
        trailing = m.group("trail")
        if com is None and trailing:
            com = _canonical_commodity(trailing)
        if com is None:
            continue
        opword = m.group("opword")
        op = opword
        if not re.fullmatch(r">=|<=|>|<|=", opword):
            for pattern, sym in _OPS:
                if re.fullmatch(pattern, opword):
                    op = sym
                    break
        unit = (m.group("unit") or "g/t").lower()
        value = float(m.group("value"))
        pq.grade_constraints.append(GradeConstraint(
            commodity=com,
            symbol=PERIODIC_TABLE.get(com, com.upper()[:2]),
            op=op, value=value, unit=unit,
            value_gpt=value * UNIT_TO_GPT.get(unit, 1.0),
        ))
        consumed_spans.append((m.start(), m.end()))

    # ---- generic grade constraints ("grade at least 1 g/t") ---------------
    generic: List[GradeConstraint] = []
    for m in _GENERIC_GRADE_RE.finditer(text):
        if any(m.start() < e and m.end() > s for s, e in consumed_spans):
            continue
        opword = m.group("opword")
        op = opword
        if not re.fullmatch(r">=|<=|>|<|=", opword):
            for pattern, sym in _OPS:
                if re.fullmatch(pattern, opword):
                    op = sym
                    break
        unit = (m.group("unit") or "g/t").lower()
        value = float(m.group("value"))
        generic.append(GradeConstraint(
            commodity="", symbol="", op=op, value=value, unit=unit,
            value_gpt=value * UNIT_TO_GPT.get(unit, 1.0),
        ))
        consumed_spans.append((m.start(), m.end()))

    # fix symbol fields (canonical name -> symbol)
    for c in pq.grade_constraints:
        c.symbol = PERIODIC_TABLE.get(c.commodity, c.symbol)

    # remove consumed constraint text so commodity extraction is not confused
    if consumed_spans:
        chars = list(text)
        for s, e in consumed_spans:
            for i in range(s, e):
                chars[i] = " "
        text = "".join(chars)

    # ---- intent ------------------------------------------------------------
    for intent in ("count", "explain", "rank", "list"):
        if any(kw in text for kw in INTENT_KEYWORDS[intent]):
            pq.intent = intent
            break

    # ---- entity ------------------------------------------------------------
    explicit_entity = False
    for entity in ("drillholes", "samples", "projects"):
        if any(re.search(r"\b" + re.escape(kw) + r"\b", text)
               for kw in ENTITY_KEYWORDS[entity]):
            pq.entity = entity
            explicit_entity = True
            break
    # grade constraints with no explicit entity imply sample-level data
    if pq.grade_constraints and not explicit_entity:
        pq.entity = "samples"

    # ---- commodities (groups, minerals, elements) --------------------------
    found: List[str] = []
    for group, members in COMMODITY_GROUPS.items():
        if group in text:
            for member in members:
                if member not in found:
                    found.append(member)
            text = text.replace(group, " ")
    for token in re.findall(r"[a-z]+", text):
        com = _canonical_commodity(token)
        if com is not None and com not in found:
            found.append(com)
    # mineral names (multi-word pass first)
    for mineral, com in sorted(MINERAL_TO_COMMODITY.items(), key=lambda kv: -len(kv[0])):
        if mineral in text and com not in found:
            found.append(com)
    # commodities referenced only inside grade constraints still count
    for gc in pq.grade_constraints:
        if gc.commodity not in found:
            found.append(gc.commodity)
    # resolve generic "grade >= X" constraints against the phrase commodity
    if generic and found:
        for gc in generic:
            gc.commodity = found[0]
            gc.symbol = PERIODIC_TABLE.get(gc.commodity, "")
            pq.grade_constraints.append(gc)
    pq.commodities = found
    pq.symbols = [PERIODIC_TABLE.get(c, c) for c in found]

    # ---- deposit types ------------------------------------------------------
    for key in sorted(DEPOSIT_TYPES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", text):
            canonical = DEPOSIT_TYPES[key]
            if canonical not in pq.deposit_types:
                pq.deposit_types.append(canonical)

    # ---- regions (longest match first) --------------------------------------
    for region in sorted(REGIONS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(region) + r"\b", text):
            if not any(region in r or r in region for r in pq.regions):
                pq.regions.append(region)

    return pq


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

_OP_PHRASE = {">": "greater than", ">=": "at least", "<": "less than",
              "<=": "at most", "=": "equal to"}


def explain_query(pq: ParsedQuery) -> str:
    """Plain-language explanation of the parsed query."""
    parts = [f"{pq.intent.capitalize()} {pq.entity}"]
    if pq.commodities:
        parts.append("related to " + ", ".join(pq.commodities))
    if pq.deposit_types:
        parts.append("of deposit type " + ", ".join(pq.deposit_types))
    if pq.regions:
        parts.append("in " + ", ".join(pq.regions))
    for c in pq.grade_constraints:
        parts.append(
            f"where {c.commodity} is {_OP_PHRASE[c.op]} {c.value} {c.unit}"
            f" ({c.value_gpt:.4g} g/t)"
        )
    if pq.distance_constraint:
        d = pq.distance_constraint
        parts.append(f"within {d.distance_km:.3g} km of {d.reference}")
    return " ".join(parts) + "."


# ---------------------------------------------------------------------------
# Query execution over SQLAlchemy models
# ---------------------------------------------------------------------------

def _project_commodity_set(project) -> set:
    raw = project.commodities or []
    out = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        t = item.strip().lower()
        out.add(t)
        canon = _canonical_commodity(t)
        if canon:
            out.add(canon)
            out.add(PERIODIC_TABLE.get(canon, "").lower())
    return out


def _matches_commodities(pq: ParsedQuery, available: set) -> bool:
    wanted = set(pq.commodities) | {s.lower() for s in pq.symbols}
    if not wanted:
        return True
    return bool(wanted & available)


def _matches_regions(pq: ParsedQuery, *texts: Optional[str]) -> bool:
    if not pq.regions:
        return True
    hay = " ".join(t.lower() for t in texts if t)
    return any(r in hay for r in pq.regions)


def _sample_value(sample, commodity: str) -> Optional[float]:
    data = sample.assay_data or {}
    symbol = PERIODIC_TABLE.get(commodity, "")
    for key in (commodity, commodity.lower(), symbol, symbol.lower()):
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return None
    return None


def _op_ok(value: float, op: str, target: float) -> bool:
    if op == ">":
        return value > target
    if op == ">=":
        return value >= target
    if op == "<":
        return value < target
    if op == "<=":
        return value <= target
    return abs(value - target) < 1e-9


def _haversine_km(x1: float, y1: float, x2: float, y2: float) -> float:
    """Great-circle distance for lon/lat collars; planar fallback for grids."""
    # Treat coordinates as lon/lat when they look geographic, else planar km.
    if all(-180 <= v <= 180 for v in (x1, x2)) and all(-90 <= v <= 90 for v in (y1, y2)):
        r = 6371.0088
        p1, p2 = math.radians(y1), math.radians(y2)
        dp = math.radians(y2 - y1)
        dl = math.radians(x2 - x1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))
    return math.hypot(x2 - x1, y2 - y1) / 1000.0


def execute_query(session, pq: ParsedQuery, limit: int = 50) -> Dict[str, Any]:
    """Compile the AST to a query over Project/Drillhole/Sample models.

    SQLAlchemy handles relational filtering (region text, joins); JSON-valued
    columns (commodities list, assay_data dict) are post-filtered in Python,
    which is portable across SQLite/Postgres JSON implementations.
    """
    from ...database import ProjectModel, DrillholeModel, SampleModel

    limit = max(1, min(int(limit), 500))
    explanation = explain_query(pq)

    # ---- resolve distance reference to a collar centroid -------------------
    dist_ref_xy = None
    if pq.distance_constraint:
        ref = pq.distance_constraint.reference
        ref_projects = (
            session.query(ProjectModel)
            .filter(ProjectModel.name.ilike(f"%{ref}%") |
                    ProjectModel.location.ilike(f"%{ref}%"))
            .all()
        )
        collars = []
        for p in ref_projects:
            for h in p.drillholes:
                collars.append((h.collar_x, h.collar_y))
        if collars:
            dist_ref_xy = (sum(c[0] for c in collars) / len(collars),
                           sum(c[1] for c in collars) / len(collars))
        else:
            return {
                "results": [], "count": 0, "explanation": explanation,
                "warning": f"no project found matching distance reference '{ref}'",
            }

    def _within_distance(hole) -> bool:
        if dist_ref_xy is None:
            return True
        d = _haversine_km(hole.collar_x, hole.collar_y, *dist_ref_xy)
        return d <= pq.distance_constraint.distance_km

    results: List[Dict[str, Any]] = []

    if pq.entity == "projects":
        q = session.query(ProjectModel)
        for region in pq.regions:
            q = q.filter(ProjectModel.location.ilike(f"%{region}%"))
        projects = q.all()
        rows = []
        for p in projects:
            if not _matches_commodities(pq, _project_commodity_set(p)):
                continue
            holes = [h for h in p.drillholes if _within_distance(h)]
            if pq.distance_constraint and not holes:
                continue
            best = 0.0
            for h in holes:
                for s in h.samples:
                    for com in (pq.commodities or [None]):
                        v = _sample_value(s, com) if com else None
                        if v is not None:
                            best = max(best, v)
            row = {
                "id": p.id, "name": p.name, "location": p.location,
                "commodities": p.commodities or [], "status": p.status,
                "n_drillholes": len(holes), "best_grade_gpt": best,
            }
            if not _grade_constraints_satisfied(pq, holes):
                continue
            rows.append(row)
        if pq.intent == "rank":
            rows.sort(key=lambda r: (r["best_grade_gpt"], r["n_drillholes"]),
                      reverse=True)
        results = rows

    elif pq.entity == "drillholes":
        q = (session.query(DrillholeModel)
             .join(ProjectModel, DrillholeModel.project_id == ProjectModel.id))
        holes = q.all()
        rows = []
        for h in holes:
            p = h.project
            if not _matches_regions(pq, p.location, p.name):
                continue
            # grade constraints filter on assay values directly; only apply the
            # project commodity-tag filter when no grade constraint exists
            if not pq.grade_constraints and \
                    not _matches_commodities(pq, _project_commodity_set(p)):
                continue
            if not _within_distance(h):
                continue
            max_grade, accum = 0.0, 0.0
            ok = True
            for gc in pq.grade_constraints:
                vals = [v for v in (_sample_value(s, gc.commodity) for s in h.samples)
                        if v is not None]
                if not vals or not any(_op_ok(v, gc.op, gc.value_gpt) for v in vals):
                    ok = False
                    break
                max_grade = max(max_grade, max(vals))
            if not ok:
                continue
            for s in h.samples:
                for com in pq.commodities:
                    v = _sample_value(s, com)
                    if v is not None:
                        accum += v * max(s.to_depth - s.from_depth, 0.0)
                        max_grade = max(max_grade, v)
            rows.append({
                "id": h.id, "hole_id": h.hole_id, "project": p.name,
                "collar_x": h.collar_x, "collar_y": h.collar_y,
                "total_depth": h.total_depth, "max_grade_gpt": max_grade,
                "grade_thickness": accum, "n_samples": len(h.samples),
            })
        if pq.intent == "rank":
            rows.sort(key=lambda r: (r["max_grade_gpt"], r["grade_thickness"]),
                      reverse=True)
        results = rows

    else:  # samples
        q = (session.query(SampleModel)
             .join(DrillholeModel, SampleModel.drillhole_id == DrillholeModel.id)
             .join(ProjectModel, DrillholeModel.project_id == ProjectModel.id))
        samples = q.all()
        rows = []
        for s in samples:
            h = s.drillhole
            p = h.project
            if not _matches_regions(pq, p.location, p.name):
                continue
            if not pq.grade_constraints and \
                    not _matches_commodities(pq, _project_commodity_set(p)):
                continue
            if not _within_distance(h):
                continue
            ok = True
            matched_value = None
            for gc in pq.grade_constraints:
                v = _sample_value(s, gc.commodity)
                if v is None or not _op_ok(v, gc.op, gc.value_gpt):
                    ok = False
                    break
                matched_value = v if matched_value is None else max(matched_value, v)
            if not ok:
                continue
            rows.append({
                "id": s.id, "sample_id": s.sample_id, "hole_id": h.hole_id,
                "project": p.name, "from_depth": s.from_depth,
                "to_depth": s.to_depth, "assay_data": s.assay_data or {},
                "matched_grade_gpt": matched_value,
            })
        if pq.intent == "rank":
            rows.sort(key=lambda r: (r["matched_grade_gpt"] or 0.0), reverse=True)
        results = rows

    count = len(results)
    if pq.intent == "count":
        return {"results": [], "count": count, "explanation": explanation}
    return {"results": results[:limit], "count": count, "explanation": explanation}


def _grade_constraints_satisfied(pq: ParsedQuery, holes) -> bool:
    if not pq.grade_constraints:
        return True
    for gc in pq.grade_constraints:
        hit = False
        for h in holes:
            for s in h.samples:
                v = _sample_value(s, gc.commodity)
                if v is not None and _op_ok(v, gc.op, gc.value_gpt):
                    hit = True
                    break
            if hit:
                break
        if not hit:
            return False
    return True
