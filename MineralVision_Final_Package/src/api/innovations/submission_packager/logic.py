"""Deterministic submission bundle builder + validator.

Produces a regulatory tenement submission ZIP:
  - one file per template section (JSON / CSV renderers),
  - a manifest.json recording SHA-256 and size of every bundled file,
  - fully deterministic bytes: entries sorted by name, fixed DOS epoch
    (1980-01-01) timestamps, fixed compression settings.

The validator performs required-section completeness checks and fails on
missing or empty data.
"""

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .templates import TEMPLATES

# Fixed timestamp (DOS epoch minimum) → byte-identical archives across runs.
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_COMPRESSLEVEL = 9
MANIFEST_NAME = "manifest.json"


@dataclass
class ValidationIssue:
    section: str
    filename: str
    reason: str


@dataclass
class ValidationResult:
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)


@dataclass
class PackageResult:
    zip_bytes: bytes
    manifest: Dict[str, Any]
    validation: ValidationResult


def get_template(name: str) -> Dict[str, Any]:
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; have {sorted(TEMPLATES)}")
    return TEMPLATES[name]


def _section_data(data: Dict[str, Any], key: str) -> Any:
    value = data.get(key)
    if value is None or value == "" or value == [] or value == {}:
        return None
    return value


def validate_submission(template_name: str, data: Dict[str, Any]) -> ValidationResult:
    """Check required-section completeness. Fails on missing/empty data."""
    template = get_template(template_name)
    issues: List[ValidationIssue] = []
    for section in template["sections"]:
        if not section["required"]:
            continue
        value = _section_data(data, section["key"])
        if value is None:
            issues.append(
                ValidationIssue(
                    section=section["key"],
                    filename=section["filename"],
                    reason="required section missing or empty",
                )
            )
    return ValidationResult(valid=not issues, issues=issues)


def render_files(template_name: str, data: Dict[str, Any]) -> List[Tuple[str, bytes]]:
    """Render every present section to (filename, content-bytes)."""
    template = get_template(template_name)
    files: List[Tuple[str, bytes]] = []
    for section in template["sections"]:
        value = _section_data(data, section["key"])
        if value is None:
            continue
        content = section["renderer"](value)
        files.append((section["filename"], content.encode("utf-8")))
    return files


def build_manifest(template_name: str, files: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    """Manifest with SHA-256 and size for every bundled file (sorted)."""
    entries = [
        {
            "name": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        for name, content in sorted(files, key=lambda f: f[0])
    ]
    return {
        "template": template_name,
        "template_title": get_template(template_name)["title"],
        "generator": "mineralvision-submission-packager/1.0",
        "files": entries,
    }


def build_zip(files: List[Tuple[str, bytes]], manifest: Dict[str, Any]) -> bytes:
    """Deterministic ZIP: sorted entries, fixed 1980 timestamps."""
    all_files = sorted(files, key=lambda f: f[0])
    all_files.append((MANIFEST_NAME, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")))
    all_files.sort(key=lambda f: f[0])  # manifest.json sorts into place

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=_COMPRESSLEVEL) as zf:
        for name, content in all_files:
            info = zipfile.ZipInfo(filename=name, date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3  # unix, deterministic metadata
            zf.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=_COMPRESSLEVEL)
    return buf.getvalue()


def package_submission(
    template_name: str,
    data: Dict[str, Any],
    strict: bool = True,
) -> PackageResult:
    """Validate, render and bundle a submission.

    With ``strict=True`` (default) an incomplete submission raises ValueError
    listing the missing sections — packaging an invalid bundle is refused.
    """
    template = get_template(template_name)  # raises on unknown template
    validation = validate_submission(template_name, data)
    if strict and not validation.valid:
        missing = ", ".join(i.section for i in validation.issues)
        raise ValueError(f"incomplete submission for template {template_name!r}: {missing}")
    files = render_files(template_name, data)
    manifest = build_manifest(template_name, files)
    manifest["validation"] = {
        "valid": validation.valid,
        "issues": [{"section": i.section, "filename": i.filename, "reason": i.reason} for i in validation.issues],
    }
    zip_bytes = build_zip(files, manifest)
    return PackageResult(zip_bytes=zip_bytes, manifest=manifest, validation=validation)
