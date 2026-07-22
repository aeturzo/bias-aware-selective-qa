"""Ontology-grounded anchor provider.

The bias-correction anchor (the auditable external assumption) is read from the
DPP knowledge graph rather than hard-coded: attribute polarity (``direction``),
the expected group gap (``expectedGroupGap``), and the constraint tolerance
(``groupGapEpsilon``) come from ``ontology/dpp_anchors.ttl`` as RDF triples with
provenance. Parsing uses ``rdflib`` when available and falls back to a small
regex reader for the (controlled) anchor file so the package has no hard
dependency. If the ontology yields no anchors, we fall back to the built-in
constants so nothing breaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Built-in fallback (matches the historical constants).
_FALLBACK_GAP = {
    "carbon_kg_per_kwh": 5.0,
    "repairability_score": -0.30,
    "durability_score": -0.40,
}
_FALLBACK_DIRECTION = {
    "carbon_kg_per_kwh": "higherIsWorse",
    "repairability_score": "higherIsBetter",
    "durability_score": "higherIsBetter",
}
_FALLBACK_EPS = 0.35

_DEFAULT_TTL = Path(__file__).resolve().parents[1] / "ontology" / "dpp_anchors.ttl"


@dataclass
class AttributeAnchor:
    attribute_key: str
    direction: str            # "higherIsWorse" | "higherIsBetter"
    expected_group_gap: float
    epsilon: float
    provenance: str = ""      # source file / justification

    @property
    def sign(self) -> float:
        """+1 if higher is worse (adverse), -1 if higher is better (protective)."""
        return 1.0 if self.direction == "higherIsWorse" else -1.0


def _parse_ttl_rdflib(path: Path) -> dict[str, AttributeAnchor]:
    import rdflib  # type: ignore

    EX = rdflib.Namespace("http://example.com/dpp#")
    g = rdflib.Graph()
    g.parse(str(path), format="turtle")
    out: dict[str, AttributeAnchor] = {}
    for s in g.subjects(rdflib.RDF.type, EX.AttributeAnchor):
        key = g.value(s, EX.attributeKey)
        if key is None:
            continue
        direction = g.value(s, EX.direction)
        dir_name = str(direction).split("#")[-1] if direction is not None else "higherIsWorse"
        gap = g.value(s, EX.expectedGroupGap)
        eps = g.value(s, EX.groupGapEpsilon)
        comment = g.value(s, rdflib.RDFS.comment)
        out[str(key)] = AttributeAnchor(
            attribute_key=str(key), direction=dir_name,
            expected_group_gap=float(gap) if gap is not None else 0.0,
            epsilon=float(eps) if eps is not None else _FALLBACK_EPS,
            provenance=f"{path.name}: {comment}" if comment is not None else path.name,
        )
    return out


def _parse_ttl_regex(path: Path) -> dict[str, AttributeAnchor]:
    """Fallback parser for the controlled anchor file (no rdflib needed)."""
    text = path.read_text()
    out: dict[str, AttributeAnchor] = {}
    # split into subject blocks terminated by ' .' at end of a statement group
    for block in re.split(r"\.\s*\n(?=\S)", text):
        if "ex:AttributeAnchor" not in block:
            continue
        key = re.search(r'ex:attributeKey\s+"([^"]+)"', block)
        if not key:
            continue
        direction = re.search(r"ex:direction\s+ex:(\w+)", block)
        gap = re.search(r'ex:expectedGroupGap\s+"(-?[\d.]+)"', block)
        eps = re.search(r'ex:groupGapEpsilon\s+"(-?[\d.]+)"', block)
        comment = re.search(r'rdfs:comment\s+"([^"]+)"', block)
        out[key.group(1)] = AttributeAnchor(
            attribute_key=key.group(1),
            direction=direction.group(1) if direction else "higherIsWorse",
            expected_group_gap=float(gap.group(1)) if gap else 0.0,
            epsilon=float(eps.group(1)) if eps else _FALLBACK_EPS,
            provenance=f"{path.name}: {comment.group(1)}" if comment else path.name,
        )
    return out


class AnchorProvider:
    """Serves the KG-grounded bias-correction anchor (with constant fallback)."""

    def __init__(self, anchors: dict[str, AttributeAnchor], source: str) -> None:
        self.anchors = anchors
        self.source = source

    @classmethod
    def from_ontology(cls, ttl_path: str | Path | None = None) -> "AnchorProvider":
        path = Path(ttl_path) if ttl_path else _DEFAULT_TTL
        anchors: dict[str, AttributeAnchor] = {}
        source = str(path)
        if path.exists():
            try:
                anchors = _parse_ttl_rdflib(path)
                source += " (rdflib)"
            except Exception:
                try:
                    anchors = _parse_ttl_regex(path)
                    source += " (regex)"
                except Exception:
                    anchors = {}
        if not anchors:
            anchors = {
                k: AttributeAnchor(k, _FALLBACK_DIRECTION[k], v, _FALLBACK_EPS,
                                   "built-in fallback constant")
                for k, v in _FALLBACK_GAP.items()
            }
            source = "fallback-constants"
        return cls(anchors, source)

    # --- accessors used by the data-trust layer ---
    def policy_target(self) -> dict[str, float]:
        return {k: a.expected_group_gap for k, a in self.anchors.items()}

    def epsilon(self, default: float = _FALLBACK_EPS) -> float:
        vals = [a.epsilon for a in self.anchors.values()]
        return float(sum(vals) / len(vals)) if vals else default

    def direction(self, attribute_key: str) -> str:
        a = self.anchors.get(attribute_key)
        return a.direction if a else _FALLBACK_DIRECTION.get(attribute_key, "higherIsWorse")

    def sign(self, attribute_key: str) -> float:
        a = self.anchors.get(attribute_key)
        return a.sign if a else (1.0 if attribute_key not in _FALLBACK_DIRECTION
                                 or _FALLBACK_DIRECTION[attribute_key] == "higherIsWorse" else -1.0)

    def provenance(self) -> dict[str, str]:
        return {k: a.provenance for k, a in self.anchors.items()}
