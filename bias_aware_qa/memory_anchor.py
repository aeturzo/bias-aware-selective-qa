"""External-memory clean-reference anchor.

The strongest anchor is *measured truth*: a small set of audited/verified clean
facts (e.g. lab-certified attribute values for a few products) stored in
COMPASS's persistent external memory with provenance. Instead of an assumed
policy target, the correction is then pulled toward a group gap *derived from
these verified records*.

Facts are stored in the COMPASS memory content format so the same store is
usable by the real memory service:

    VERIFIED_CLEAN domain=<d> group=<EU|GS> feature=<f> value=<v> source=<s>

``MemoryCleanReference`` loads such facts from a JSONL store (``{content,
session_id, timestamp}`` lines, matching ``memory.meta.jsonl``) and/or from the
live COMPASS memory service, and derives per-feature clean group gaps to use as
the anchor. If too few verified facts exist for a feature, the caller falls back
to the ontology anchor, then to constants.

``seed_memory_from_domains`` simulates an audit: it samples a small fraction of
the clean records per domain/group/feature and writes them as verified facts.
In deployment these come from real certification/audit, not from the eval data.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_LINE_RE = re.compile(
    r"VERIFIED_CLEAN\s+domain=(?P<domain>\S+)\s+group=(?P<group>\S+)\s+"
    r"feature=(?P<feature>\S+)\s+value=(?P<value>-?[\d.]+)(?:\s+source=(?P<source>\S+))?"
)


@dataclass
class CleanFact:
    domain: str
    group: str          # "EU" | "GS"
    feature: str
    value: float
    source: str = "audit"

    def to_content(self) -> str:
        return (f"VERIFIED_CLEAN domain={self.domain} group={self.group} "
                f"feature={self.feature} value={self.value:.6g} source={self.source}")


def _domain_family(name: str) -> str:
    head, sep, tail = name.rpartition("_s")
    return head if sep and tail.isdigit() else name


class MemoryCleanReference:
    """Derives a clean group-gap anchor from verified facts in external memory."""

    def __init__(self, facts: list[CleanFact], source: str = "memory") -> None:
        self.facts = facts
        self.source = source

    # ---- loading ----
    @classmethod
    def from_jsonl(cls, path: str | Path) -> "MemoryCleanReference":
        path = Path(path)
        facts: list[CleanFact] = []
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                content = line
                try:
                    obj = json.loads(line)
                    content = obj.get("content", "")
                except Exception:
                    pass
                m = _LINE_RE.search(content)
                if m:
                    facts.append(CleanFact(
                        domain=m.group("domain"), group=m.group("group"),
                        feature=m.group("feature"), value=float(m.group("value")),
                        source=m.group("source") or "audit"))
        return cls(facts, source=f"memory-jsonl:{path.name}")

    @classmethod
    def from_compass_memory(cls, session_id: str = "verified_clean") -> "MemoryCleanReference":
        """Best-effort load from the live COMPASS memory service (optional)."""
        facts: list[CleanFact] = []
        try:
            import sys
            root = Path(__file__).resolve().parents[2]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from backend.services import memory_service  # type: ignore
            entries = memory_service.retrieve(session_id, "VERIFIED_CLEAN", top_k=10000)
            for e in entries:
                content = getattr(e, "content", "") or ""
                m = _LINE_RE.search(content)
                if m:
                    facts.append(CleanFact(m.group("domain"), m.group("group"),
                                           m.group("feature"), float(m.group("value")),
                                           m.group("source") or "audit"))
        except Exception:
            pass
        return cls(facts, source="compass-memory")

    # ---- anchor derivation ----
    def _vals(self, feature: str, group: str, domain: str | None) -> np.ndarray:
        out = [f.value for f in self.facts
               if f.feature == feature and f.group == group
               and (domain is None or _domain_family(f.domain) == domain)]
        return np.asarray(out, dtype=float)

    def group_gap(self, feature: str, domain: str | None = None, min_per_group: int = 3) -> float | None:
        g, _ = self.group_gap_with_n(feature, domain, min_per_group)
        return g

    def group_gap_with_n(self, feature: str, domain: str | None = None,
                         min_per_group: int = 3) -> tuple[float | None, int]:
        """Return (clean GS-EU gap, min facts per group) from verified records."""
        gs, eu = self._vals(feature, "GS", domain), self._vals(feature, "EU", domain)
        if gs.size < min_per_group or eu.size < min_per_group:
            return None, 0
        return float(gs.mean() - eu.mean()), int(min(gs.size, eu.size))

    def policy_target(self, features: Iterable[str], domain: str | None = None,
                      min_per_group: int = 3) -> dict[str, float]:
        out: dict[str, float] = {}
        for f in features:
            g = self.group_gap(f, domain=domain, min_per_group=min_per_group)
            if g is None and domain is not None:            # fall back to pooled facts
                g = self.group_gap(f, domain=None, min_per_group=min_per_group)
            if g is not None:
                out[f] = g
        return out

    @property
    def n_facts(self) -> int:
        return len(self.facts)


# ---- audit-sample seeding (for experiments; deployment uses real audits) ----
GAP_FEATURES = ["carbon_kg_per_kwh", "repairability_score", "durability_score"]


def seed_memory_from_domains(domains, frac: float = 0.08, seed: int = 0,
                             out_path: str | Path | None = None,
                             source: str = "sim_audit") -> "MemoryCleanReference":
    """Sample a small fraction of clean records per domain/group/feature as
    verified facts. Writes a COMPASS-format JSONL if out_path is given."""
    rng = np.random.default_rng(seed)
    facts: list[CleanFact] = []
    for dom in domains:
        clean = dom.clean_df
        gcol = dom.group_col
        feats = getattr(dom, "gap_features", None) or GAP_FEATURES
        for feat in feats:
            if feat not in clean.columns:
                continue
            for grp in ("EU", "GS"):
                sub = clean[clean[gcol] == grp][feat].dropna().to_numpy()
                if sub.size == 0:
                    continue
                k = max(3, int(round(frac * sub.size)))
                idx = rng.choice(sub.size, size=min(k, sub.size), replace=False)
                for v in sub[idx]:
                    facts.append(CleanFact(dom.name, grp, feat, float(v), source))
    ref = MemoryCleanReference(facts, source=f"sim-audit(frac={frac})")
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for f in facts:
                fh.write(json.dumps({"content": f.to_content(),
                                     "session_id": "verified_clean",
                                     "timestamp": time.time()}) + "\n")
        ref.source = f"memory-jsonl:{out.name} (sim-audit frac={frac}, n={len(facts)})"
    return ref
