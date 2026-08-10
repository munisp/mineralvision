"""
Downhole well-log LAS (Log ASCII Standard 2.0) reader.

Pure-python, dependency-free (numpy only) parser for CWLS LAS 2.0 files:
~VERSION / ~WELL / ~CURVE / ~PARAMETER header sections plus the ~A ascii
data block. Handles multi-mnemonic curves, NULL substitution, WRAP mode
(conservatively rejected), and per-curve statistics on real parsed floats.

Honest policy: malformed files raise WellLogLASError; nothing is fabricated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


class WellLogLASError(ValueError):
    """Raised when a well-log LAS file cannot be parsed for real."""


@dataclass
class CurveInfo:
    """One ~CURVE mnemonic definition."""

    mnemonic: str
    unit: str = ""
    api_code: str = ""
    description: str = ""


@dataclass
class WellLogLAS:
    """Parsed LAS 2.0 well log."""

    version: Dict[str, str] = field(default_factory=dict)
    well: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, str] = field(default_factory=dict)
    curves: List[CurveInfo] = field(default_factory=list)
    data: Optional[np.ndarray] = None  # shape (n_rows, n_curves), NaN = null
    null_value: float = -999.25
    wrap: bool = False

    # -- helpers -----------------------------------------------------------
    def _curve_index(self, mnemonic: str) -> int:
        for i, c in enumerate(self.curves):
            if c.mnemonic.upper() == mnemonic.upper():
                return i
        raise WellLogLASError(f"curve '{mnemonic}' not present in file")

    @property
    def depth_mnemonic(self) -> str:
        if not self.curves:
            raise WellLogLASError("no curves parsed")
        return self.curves[0].mnemonic  # LAS spec: first curve is depth/index

    def curve_values(self, mnemonic: str) -> np.ndarray:
        """Real float values for a curve with NULLs replaced by NaN."""
        if self.data is None:
            raise WellLogLASError("no ~A data block parsed")
        col = self.data[:, self._curve_index(mnemonic)].astype(float)
        return np.where(np.isclose(col, self.null_value), np.nan, col)

    def curve_stats(self, mnemonic: str) -> Dict[str, Any]:
        """count/min/max/mean over non-null real values."""
        vals = self.curve_values(mnemonic)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return {"mnemonic": mnemonic, "count": 0,
                    "min": None, "max": None, "mean": None}
        return {
            "mnemonic": mnemonic,
            "count": int(vals.size),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
        }

    @property
    def depth_range(self) -> Dict[str, Optional[float]]:
        vals = self.curve_values(self.depth_mnemonic)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return {"start": None, "stop": None}
        return {"start": float(vals.min()), "stop": float(vals.max())}


# Header line:  MNEMONIC.UNIT  VALUE : DESCRIPTION
_HEADER_RE = re.compile(
    r"^\s*(?P<mnemonic>[^.\s:]+)\s*\.(?P<unit>[^\s:]*)"
    r"(?:\s+(?P<value>[^:]*?))?\s*(?::\s*(?P<desc>.*))?$"
)


def _parse_header_line(line: str) -> Dict[str, str]:
    m = _HEADER_RE.match(line)
    if not m:
        raise WellLogLASError(f"unparseable header line: {line!r}")
    return {
        "mnemonic": m.group("mnemonic").strip(),
        "unit": (m.group("unit") or "").strip(),
        "value": (m.group("value") or "").strip(),
        "description": (m.group("desc") or "").strip(),
    }


class WellLogLASReader:
    """Minimal-but-correct LAS 2.0 well-log reader."""

    def read(self, file_path: str) -> WellLogLAS:
        try:
            with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
                text = f.read()
        except OSError as exc:
            raise WellLogLASError(f"cannot read file: {exc}") from exc
        return self.parse_text(text)

    def parse_text(self, text: str) -> WellLogLAS:
        log = WellLogLAS()
        section: Optional[str] = None
        data_lines: List[str] = []
        saw_curve_section = False

        for raw in text.splitlines():
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("~"):
                tag = stripped[1:2].upper()
                if tag == "V":
                    section = "version"
                elif tag == "W":
                    section = "well"
                elif tag == "C":
                    section = "curve"
                    saw_curve_section = True
                elif tag == "P":
                    section = "parameter"
                elif tag == "A":
                    section = "ascii"
                else:
                    section = "other"
                continue

            if section == "version":
                item = _parse_header_line(line)
                log.version[item["mnemonic"].upper()] = item["value"]
                if item["mnemonic"].upper() == "WRAP":
                    log.wrap = item["value"].strip().upper() in ("YES", "Y")
            elif section == "well":
                item = _parse_header_line(line)
                log.well[item["mnemonic"].upper()] = item["value"]
                if item["mnemonic"].upper() == "NULL":
                    try:
                        log.null_value = float(item["value"])
                    except ValueError as exc:
                        raise WellLogLASError(
                            f"invalid NULL value: {item['value']!r}") from exc
            elif section == "curve":
                item = _parse_header_line(line)
                log.curves.append(CurveInfo(
                    mnemonic=item["mnemonic"],
                    unit=item["unit"],
                    api_code=item["value"],
                    description=item["description"],
                ))
            elif section == "parameter":
                item = _parse_header_line(line)
                log.parameters[item["mnemonic"].upper()] = item["value"]
            elif section == "ascii":
                data_lines.append(stripped)
            # 'other' sections are ignored by design

        if not saw_curve_section or not log.curves:
            raise WellLogLASError("missing ~CURVE section or no curves defined")
        if log.wrap:
            raise WellLogLASError(
                "WRAP=YES files are not supported by this reader "
                "(LAS 2.0 recommends WRAP=NO; re-export with wrapping disabled)")
        if not data_lines:
            raise WellLogLASError("missing ~A ascii data block")

        n_curves = len(log.curves)
        rows: List[List[float]] = []
        pending: List[float] = []
        for dline in data_lines:
            for tok in dline.split():
                try:
                    pending.append(float(tok))
                except ValueError as exc:
                    raise WellLogLASError(
                        f"non-numeric token in ~A data: {tok!r}") from exc
            while len(pending) >= n_curves:
                rows.append(pending[:n_curves])
                pending = pending[n_curves:]
        if pending:
            raise WellLogLASError(
                f"~A data block has {len(pending)} leftover values; "
                f"expected multiples of {n_curves} (one per curve)")
        if not rows:
            raise WellLogLASError("~A data block contains no complete rows")

        log.data = np.asarray(rows, dtype=float)
        return log


def read_welllog_las(file_path: str) -> WellLogLAS:
    """Convenience wrapper: parse a LAS 2.0 well-log file."""
    return WellLogLASReader().read(file_path)
