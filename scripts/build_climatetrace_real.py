#!/usr/bin/env python
"""Build the real Climate TRACE arm input CSV.

The experiment loader expects a country-level CSV with:

    country,group,self_reported,climatetrace

This builder makes that file reproducibly from public sources:

* UNFCCC Detailed Data By Party mirror for party-reported national inventory
  totals, excluding LULUCF/LUCF where the two UNFCCC formats expose that row.
* Climate TRACE v7 API for independent country-level CO2e totals.
* OWID population grapher data to convert totals to tonnes CO2e per person.

The default year is "auto": choose the latest Climate TRACE-covered year with at
least ``--min-per-group`` countries in both Annex I and non-Annex I reported
inventory files. With the current UNFCCC mirror, that is 2015.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

UNFCCC_ANNEX_URL = (
    "https://raw.githubusercontent.com/openclimatedata/"
    "unfccc-detailed-data-by-party/main/data/"
    "detailed-data-by-country-annex-one.csv"
)
UNFCCC_ANNEX_FALLBACK_URL = (
    "https://cdn.jsdelivr.net/gh/openclimatedata/"
    "unfccc-detailed-data-by-party@main/data/"
    "detailed-data-by-country-annex-one.csv"
)
UNFCCC_NON_ANNEX_URL = (
    "https://raw.githubusercontent.com/openclimatedata/"
    "unfccc-detailed-data-by-party/main/data/"
    "detailed-data-by-country-non-annex-one.csv"
)
UNFCCC_NON_ANNEX_FALLBACK_URL = (
    "https://cdn.jsdelivr.net/gh/openclimatedata/"
    "unfccc-detailed-data-by-party@main/data/"
    "detailed-data-by-country-non-annex-one.csv"
)
OWID_POPULATION_URL = "https://ourworldindata.org/grapher/population.csv"
CLIMATETRACE_ADMINS_URL = "https://api.climatetrace.org/v7/admins?level=0&limit=300"
CLIMATETRACE_EMISSIONS_URL = "https://api.climatetrace.org/v7/sources/emissions"

ANNEX_CATEGORY = "Total GHG emissions without LULUCF"
NON_ANNEX_CATEGORY = "Total GHG emissions excluding LULUCF/LUCF"
GAS = "Aggregate GHGs"
FEATURE = "emissions_tco2e_per_capita"
UNIT = "tCO2e_per_capita"
USER_AGENT = "bias-aware-qa-research/0.1"

# Names in the UNFCCC CSV that do not exactly match common ISO3 name tables.
# None means a regional aggregate, not a country row.
PARTY_TO_ISO3: dict[str, str | None] = {
    "Bahamas": "BHS",
    "Bolivia (Plurinational State of)": "BOL",
    "Brunei Darussalam": "BRN",
    "Cabo Verde": "CPV",
    "Congo": "COG",
    "Cook Islands": "COK",
    "Cote d'Ivoire": "CIV",
    "Czechia": "CZE",
    "Democratic People's Republic of Korea": "PRK",
    "Democratic Republic of the Congo": "COD",
    "Eswatini": "SWZ",
    "European Union (Convention)": None,
    "European Union (KP)": None,
    "Gambia": "GMB",
    "Iran (Islamic Republic of)": "IRN",
    "Lao People's Democratic Republic": "LAO",
    "Micronesia (Federated States of)": "FSM",
    "Niue": "NIU",
    "North Macedonia": "MKD",
    "Republic of Korea": "KOR",
    "Republic of Moldova": "MDA",
    "Russian Federation": "RUS",
    "Sao Tome and Principe": "STP",
    "South Sudan": "SSD",
    "State of Palestine": "PSE",
    "Syrian Arab Republic": "SYR",
    "Timor-Leste": "TLS",
    "Turkey": "TUR",
    "United Kingdom of Great Britain and Northern Ireland": "GBR",
    "United Republic of Tanzania": "TZA",
    "United States of America": "USA",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Viet Nam": "VNM",
}


def _fetch(url: str | list[str], cache_path: Path, refresh: bool = False) -> bytes:
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urls = [url] if isinstance(url, str) else list(url)
    errors: list[str] = []
    for candidate in urls:
        req = Request(candidate, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        try:
            with urlopen(req, timeout=120) as resp:
                data = resp.read()
            cache_path.write_bytes(data)
            return data
        except (HTTPError, URLError, TimeoutError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("failed to fetch source; tried " + " | ".join(errors))


def _read_csv(url: str | list[str], cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_fetch(url, cache_path, refresh=refresh)))


def _read_json(url: str, cache_path: Path, refresh: bool = False) -> Any:
    return json.loads(_fetch(url, cache_path, refresh=refresh).decode("utf-8"))


def _norm_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"\([^)]*\)", " ", value.lower().replace("&", " and "))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _population_tables(pop: pd.DataFrame, year: int) -> tuple[dict[str, float], dict[str, str]]:
    pop_col = next(c for c in pop.columns if c not in {"Entity", "Code", "Year"})
    valid_codes = pop["Code"].astype(str).str.fullmatch(r"[A-Z]{3}", na=False)
    countries = pop[valid_codes].copy()
    by_name = {
        _norm_name(row.Entity): row.Code
        for row in countries[["Entity", "Code"]].drop_duplicates().itertuples(index=False)
    }
    year_rows = countries[countries["Year"].eq(year)]
    by_iso = {
        str(row.Code): float(getattr(row, pop_col))
        for row in year_rows[["Code", pop_col]].dropna().itertuples(index=False)
    }
    return by_iso, by_name


def _party_to_iso3(party: str, by_name: dict[str, str], admin_by_name: dict[str, str]) -> str | None:
    if party in PARTY_TO_ISO3:
        return PARTY_TO_ISO3[party]
    candidates = [_norm_name(party), _norm_name(party.replace("Republic of ", ""))]
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]
        if candidate in admin_by_name:
            return admin_by_name[candidate]
    return None


def _reported_rows(df: pd.DataFrame, *, category: str, group: str, year: int) -> list[dict[str, Any]]:
    year_col = str(year)
    if year_col not in df.columns:
        return []
    rows = df[df["Category"].eq(category) & df["Gas"].eq(GAS)].copy()
    rows[year_col] = pd.to_numeric(rows[year_col], errors="coerce")
    out: list[dict[str, Any]] = []
    for _, row in rows[rows[year_col].notna()].iterrows():
        unit = str(row["Unit"])
        if "kt" in unit or "Gg" in unit:
            total_tco2e = float(row[year_col]) * 1000.0
        else:
            raise ValueError(f"unexpected UNFCCC unit for {row['Party']}: {unit}")
        out.append({
            "party": str(row["Party"]),
            "group": group,
            "reported_total_tco2e": total_tco2e,
            "reported_unit": unit,
        })
    return out


def _choose_year(annex: pd.DataFrame, non_annex: pd.DataFrame, min_per_group: int) -> int:
    years = sorted({
        int(c) for c in annex.columns.intersection(non_annex.columns)
        if str(c).isdigit() and 2015 <= int(c) <= 2024
    }, reverse=True)
    for year in years:
        n_annex = len(_reported_rows(annex, category=ANNEX_CATEGORY, group="AnnexI", year=year))
        n_non = len(_reported_rows(non_annex, category=NON_ANNEX_CATEGORY, group="nonAnnexI", year=year))
        # Two Annex rows are EU aggregates; they are filtered after ISO mapping.
        n_annex_country = max(0, n_annex - 2)
        if n_annex_country >= min_per_group and n_non >= min_per_group:
            return year
    raise SystemExit(
        f"No Climate TRACE-covered year has at least {min_per_group} reported rows per group. "
        "Try --min-per-group 5."
    )


def _climatetrace_total_tco2e(iso3: str, year: int, cache_dir: Path,
                              refresh: bool, sleep_s: float) -> float:
    params = urlencode({"year": year, "gas": "co2e_100yr", "gadmId": iso3})
    cache_path = cache_dir / "climatetrace" / f"emissions_{year}_{iso3}.json"
    was_cached = cache_path.exists() and not refresh
    data = _read_json(f"{CLIMATETRACE_EMISSIONS_URL}?{params}", cache_path, refresh=refresh)
    if not was_cached and sleep_s > 0:
        time.sleep(sleep_s)
    summaries = data.get("totals", {}).get("summaries", [])
    for summary in summaries:
        if summary.get("gas") == "co2e_100yr":
            value = summary.get("emissionsQuantity")
            if value is not None and np.isfinite(float(value)):
                return float(value)
    raise ValueError(f"Climate TRACE response for {iso3} {year} has no co2e_100yr total")


def build(args: argparse.Namespace) -> pd.DataFrame:
    cache_dir = Path(args.cache_dir)
    annex = _read_csv([UNFCCC_ANNEX_URL, UNFCCC_ANNEX_FALLBACK_URL],
                      cache_dir / "unfccc_annex_one.csv", refresh=args.refresh)
    non_annex = _read_csv([UNFCCC_NON_ANNEX_URL, UNFCCC_NON_ANNEX_FALLBACK_URL],
                          cache_dir / "unfccc_non_annex_one.csv", refresh=args.refresh)

    year = _choose_year(annex, non_annex, args.min_per_group) if args.year == "auto" else int(args.year)
    pop = _read_csv(OWID_POPULATION_URL, cache_dir / "owid_population.csv", refresh=args.refresh)
    pop_by_iso, iso_by_name = _population_tables(pop, year)
    admins = _read_json(CLIMATETRACE_ADMINS_URL, cache_dir / "climatetrace_admins.json", refresh=args.refresh)
    admin_by_iso = {str(row["id"]): str(row["name"]) for row in admins}
    admin_by_name = {_norm_name(row["name"]): str(row["id"]) for row in admins}

    reported = (
        _reported_rows(annex, category=ANNEX_CATEGORY, group="AnnexI", year=year)
        + _reported_rows(non_annex, category=NON_ANNEX_CATEGORY, group="nonAnnexI", year=year)
    )
    output: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for item in reported:
        party = item["party"]
        iso3 = _party_to_iso3(party, iso_by_name, admin_by_name)
        if iso3 is None:
            skipped.append({"country": party, "reason": "no_country_iso3_or_regional_aggregate"})
            continue
        population = pop_by_iso.get(iso3)
        if population is None or population <= 0:
            skipped.append({"country": party, "iso3": iso3, "reason": "missing_population"})
            continue
        try:
            ct_total = _climatetrace_total_tco2e(
                iso3, year, cache_dir, refresh=args.refresh, sleep_s=args.sleep_s)
        except Exception as exc:  # noqa: BLE001 - keep batch running and report skipped countries.
            skipped.append({"country": party, "iso3": iso3, "reason": f"climatetrace_error: {exc}"})
            continue
        output.append({
            "country": party,
            "iso3": iso3,
            "climatetrace_name": admin_by_iso.get(iso3, ""),
            "group": item["group"],
            "self_reported": item["reported_total_tco2e"] / population,
            "climatetrace": ct_total / population,
            "self_reported_total_tco2e": item["reported_total_tco2e"],
            "climatetrace_total_tco2e": ct_total,
            "population": population,
            "year": year,
            "unit": UNIT,
            "feature": FEATURE,
            "source": "UNFCCC_Detailed_Data_By_Party__Climate_TRACE_API_v7__OWID_population",
            "self_reported_source": UNFCCC_ANNEX_URL if item["group"] == "AnnexI" else UNFCCC_NON_ANNEX_URL,
            "climatetrace_source": CLIMATETRACE_EMISSIONS_URL,
            "population_source": OWID_POPULATION_URL,
            "unfccc_reported_unit": item["reported_unit"],
        })

    df = pd.DataFrame(output).sort_values(["group", "country"]).reset_index(drop=True)
    counts = df.groupby("group").size().to_dict() if not df.empty else {}
    if counts.get("AnnexI", 0) < args.min_per_group or counts.get("nonAnnexI", 0) < args.min_per_group:
        raise SystemExit(
            "Generated data does not meet group-size threshold: "
            f"{counts}; skipped={skipped[:10]}"
        )
    df.attrs["year"] = year
    df.attrs["skipped"] = skipped
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="auto",
                    help="year to join, or 'auto' for latest year with enough UNFCCC rows")
    ap.add_argument("--min-per-group", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "real_data" / "climatetrace_real.csv"))
    ap.add_argument("--cache-dir", default=str(ROOT / "real_data" / "climatetrace_sources"))
    ap.add_argument("--refresh", action="store_true", help="redownload cached source files/API responses")
    ap.add_argument("--sleep-s", type=float, default=0.05,
                    help="polite delay after uncached Climate TRACE API calls")
    args = ap.parse_args()

    df = build(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    skipped = df.attrs.get("skipped", [])
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "year": df.attrs.get("year"),
        "rows": int(len(df)),
        "group_counts": {k: int(v) for k, v in df.groupby("group").size().to_dict().items()},
        "feature": FEATURE,
        "unit": UNIT,
        "sources": {
            "unfccc_annex": UNFCCC_ANNEX_URL,
            "unfccc_non_annex": UNFCCC_NON_ANNEX_URL,
            "climatetrace_api": CLIMATETRACE_EMISSIONS_URL,
            "owid_population": OWID_POPULATION_URL,
        },
        "skipped": skipped,
    }
    meta_path = out.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(df)} rows to {out}")
    print(f"Wrote provenance to {meta_path}")
    print(f"Year: {meta['year']} | unit: {UNIT} | groups: {meta['group_counts']}")
    if skipped:
        print(f"Skipped {len(skipped)} rows; first skipped entries: {skipped[:5]}", file=sys.stderr)


if __name__ == "__main__":
    main()
