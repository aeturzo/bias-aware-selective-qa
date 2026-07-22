"""Real COMPASS evidence adapter (with safe fallback to the mock).

Two roles:

1. ``CompassEvidenceQA`` -- for numeric attribute queries, optionally asks the
   real COMPASS answerer (``backend.api.answerer_ctx``) grounded on the observed
   group statistics, and parses a numeric estimate + confidence. If the backend
   or the OpenAI credentials are unavailable, or parsing fails, it transparently
   falls back to ``MockCompassQA`` so experiments always run.

2. ``load_compass_text_results`` -- ingests COMPASS's own exported per-question
   eval CSVs (logic/open/recall over the real corpora) and normalizes them into
   the unified record schema, so the paper's "all types" mix includes real
   text-QA alongside the bias-aware numeric layer.

To produce the real text-QA CSVs, run COMPASS's own eval harness
(``llmmain/run_eval_all.py``) with the credentials in ``bias_aware_qa/.env``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import Query
from .evidence_qa import EvidenceAnswer, MockCompassQA
from .env_loader import load_dotenv, openai_available

# Repo root = .../llmmain ; backend is importable from there.
_LLMMAIN_ROOT = Path(__file__).resolve().parents[2]


def _try_import_backend():
    """Best-effort import of the COMPASS answerer; returns callables or None."""
    if str(_LLMMAIN_ROOT) not in sys.path:
        sys.path.insert(0, str(_LLMMAIN_ROOT))
    try:
        from backend.api.answerer_ctx import answer_with_context_detailed  # type: ignore
        return answer_with_context_detailed
    except Exception:
        return None


class CompassEvidenceQA:
    """Evidence layer backed by the real COMPASS answerer when available."""

    def __init__(self, enable_llm: bool = True, min_group_evidence: int = 40) -> None:
        load_dotenv()
        self.mock = MockCompassQA(min_group_evidence=min_group_evidence)
        self._answer_fn = _try_import_backend() if enable_llm else None
        self.llm_ready = bool(self._answer_fn) and openai_available()
        self._cache: dict[str, EvidenceAnswer] = {}

    def _passages_from_query(self, q: Query) -> list[dict[str, Any]]:
        """Serialize the observed group statistics as retrievable passages."""
        return [{
            "text": (f"Domain {q.domain}. Attribute {q.feature}. "
                     f"Observed {q.kind} value from records = {q.observed_value:.4f}. "
                     f"EU records={q.n_evidence_eu}, GS records={q.n_evidence_gs}."),
            "source": f"{q.domain}:{q.feature}",
        }]

    def answer(self, q: Query) -> EvidenceAnswer:
        if q.qid in self._cache:
            return self._cache[q.qid]
        base = self.mock.answer(q)  # value/coverage model + naive stat CI
        if not self.llm_ready or not base.has_evidence:
            self._cache[q.qid] = base
            return base
        try:
            out = self._answer_fn(  # type: ignore[misc]
                f"What is the observed {q.kind} of {q.feature} in domain {q.domain}?",
                self._passages_from_query(q),
            )
            text = out.get("answer", "") if isinstance(out, dict) else str(out)
            m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
            value = float(m.group()) if m else base.value
            conf = float(out.get("confidence", base.evidence_confidence)) if isinstance(out, dict) else base.evidence_confidence
            ans = EvidenceAnswer(
                value=value, evidence_confidence=float(np.clip(conf, 0.0, 1.0)),
                has_evidence=True, stat_ci_lower=value - q.tolerance * 0.6,
                stat_ci_upper=value + q.tolerance * 0.6, n_evidence=base.n_evidence,
            )
            self._cache[q.qid] = ans
            return ans
        except Exception:
            self._cache[q.qid] = base
            return base  # any failure -> evidence stays as the mock's answer


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return bool(int(value))
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "correct", "success"}


def load_compass_text_results(csv_paths: list[str]) -> list[dict[str, Any]]:
    """Normalize COMPASS eval CSVs (logic/open/recall) to unified records.

    Expected columns: id, mode, type, domain, correct, confidence[, confidence_cal, split].
    Only the best COMPASS mode (ADAPTIVERAG/COMPASS) is kept if a `mode` column
    exists. Every text question is treated as *answered* with its confidence;
    selective abstention is evaluated downstream via the confidence ranking.
    """
    rows: list[dict[str, Any]] = []
    for path in csv_paths:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "mode" in df.columns:
            keep = df["mode"].astype(str).str.upper().isin({"ADAPTIVERAG", "COMPASS"})
            if keep.any():
                df = df[keep]
        conf_col = "confidence_cal" if "confidence_cal" in df.columns else "confidence"
        for _, r in df.iterrows():
            correct = _as_bool(r.get("correct", r.get("success", 0)))
            conf = float(pd.to_numeric(r.get(conf_col, np.nan), errors="coerce"))
            rows.append({
                "qid": str(r.get("id", "")), "domain": str(r.get("domain", "unknown")),
                "domain_profile": str(r.get("domain", "unknown")),
                "quantity": "", "kind": "text", "qtype": str(r.get("type", "text")),
                "is_biased": False, "is_undercovered": False,
                "clean_value": np.nan, "observed_value": np.nan,
                "answered": True, "value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "ci_covers_clean": False,
                "confidence": conf, "correct": correct,
                "system": "compass_text", "trust": float("nan"),
            })
    return rows
