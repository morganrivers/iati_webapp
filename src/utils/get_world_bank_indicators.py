#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from pathlib import Path
from datetime import date
from typing import Dict, List, Tuple, Optional

import pycountry

WGI_PATH = Path(__file__).parent.parent.parent / "data" / "world_bank_indicators.csv"

# simple in-module cache so we only parse the CSV once per process
_WGI_BY_ISO3: Optional[Dict[str, Dict[str, Dict[int, float]]]] = None
_WGI_NAME_INDEX: Optional[Dict[str, str]] = None

WGI_SERIES = {
    "CC.EST": "wgi_control_of_corruption_est",
    "GE.EST": "wgi_government_effectiveness_est",
    "PV.EST": "wgi_political_stability_est",
    "RQ.EST": "wgi_regulatory_quality_est",
    "RL.EST": "wgi_rule_of_law_est",
}

ISO2_TO_ISO3_OVERRIDES: Dict[str, str] = {
    "CI": "CIV",
    "CD": "COD",
    "TZ": "TZA",
    "CG": "COG",
    "CV": "CPV",
    "CZ": "CZE",
    "UK": "GBR",
    "EL": "GRC",
    "XK": "XKX",
}


def _norm_country_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "")).strip().lower()


def _iso2_to_iso3_guess(iso2: Optional[str]) -> Optional[str]:
    if not iso2:
        return None
    iso2 = iso2.strip().upper()
    if not iso2:
        return None

    if iso2 in ISO2_TO_ISO3_OVERRIDES:
        return ISO2_TO_ISO3_OVERRIDES[iso2]

    country = pycountry.countries.get(alpha_2=iso2)
    if country is not None and getattr(country, "alpha_3", None):
        return country.alpha_3

    return None


def load_world_bank_indicators(
    keep_series_codes: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, str]]:
    """
    Parse the World Bank-style indicators CSV into:

        by_iso3[ISO3][series_code][year] -> value (float)
        name_index[normalized country name] -> ISO3

    If keep_series_codes is provided, only those Series Code rows are retained.
    """
    global _WGI_BY_ISO3, _WGI_NAME_INDEX
    if _WGI_BY_ISO3 is not None and _WGI_NAME_INDEX is not None:
        return _WGI_BY_ISO3, _WGI_NAME_INDEX

    by_iso3: Dict[str, Dict[str, Dict[int, float]]] = {}
    name_idx: Dict[str, str] = {}

    keep = set(keep_series_codes) if keep_series_codes else None

    with WGI_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f, delimiter=",")
        headers = rdr.fieldnames or []

        year_cols: List[Tuple[int, str]] = []
        for h in headers:
            m = re.match(r"(\d{4})", h)
            if m:
                year_cols.append((int(m.group(1)), h))
        year_cols.sort()

        for row in rdr:
            iso3 = (row.get("Country Code") or "").strip().upper()
            cname = (row.get("Country Name") or "").strip()
            series_code = (row.get("Series Code") or "").strip()

            if not iso3 or not series_code:
                continue
            if keep is not None and series_code not in keep:
                continue

            year_map: Dict[int, float] = {}
            for year, col in year_cols:
                raw = (row.get(col) or "").strip()
                if not raw:
                    continue
                try:
                    year_map[year] = float(raw)
                except ValueError:
                    continue

            if not year_map:
                continue

            by_iso3.setdefault(iso3, {})[series_code] = year_map
            if cname:
                name_idx[_norm_country_name(cname)] = iso3

    _WGI_BY_ISO3 = by_iso3
    _WGI_NAME_INDEX = name_idx
    return by_iso3, name_idx


def _pick_series_for_country_year(
    iso3: str,
    series_code: str,
    year: int,
    wgi_by_iso3: Dict[str, Dict[str, Dict[int, float]]],
) -> Optional[Tuple[float, int]]:
    """
    Choose value for (iso3, series_code) for a reference year:
      - latest year <= reference year, else earliest available year.
    Returns (value, year_used) or None.
    """
    s_map = wgi_by_iso3.get(iso3, {}).get(series_code)
    if not s_map:
        return None

    years = list(s_map.keys())
    le_years = [y for y in years if y <= year]
    chosen_year = max(le_years) if le_years else min(years)

    val = s_map.get(chosen_year)
    if val is None:
        return None
    return val, chosen_year


def get_activity_wgi(
    meta_entry: dict,
    txn_first: Optional[date],
    wgi_by_iso3: Dict[str, Dict[str, Dict[int, float]]],
    wgi_name_index: Dict[str, str],
    series_codes: Optional[List[str]] = None,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[int]]]:
    """
    Weighted average WGI over recipient countries (same weighting behavior as CPIA helper).

    Returns:
      (scores_by_series_code, years_by_series_code)
    """
    start_date = (
        txn_first
        or meta_entry.get("actual_start_date")
        or meta_entry.get("planned_start_date")
    )
    if not isinstance(start_date, date):
        return {}, {}

    ref_year = start_date.year
    rc_entries = meta_entry.get("recipient_countries") or []
    if not rc_entries:
        return {}, {}

    codes = series_codes or list(WGI_SERIES.keys())

    collected: Dict[str, List[Tuple[float, Optional[float], int]]] = {c: [] for c in codes}

    for entry in rc_entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            continue
        code, name, pct = entry
        iso2 = (code or "").strip().upper() or None

        iso3 = _iso2_to_iso3_guess(iso2)
        if not iso3 and name:
            iso3 = wgi_name_index.get(_norm_country_name(name))

        if not iso3:
            continue

        for sc in codes:
            picked = _pick_series_for_country_year(iso3, sc, ref_year, wgi_by_iso3)
            if not picked:
                continue
            val, year_used = picked
            collected[sc].append((val, pct, year_used))

    scores_out: Dict[str, Optional[float]] = {}
    years_out: Dict[str, Optional[int]] = {}

    for sc, rows in collected.items():
        if not rows:
            scores_out[sc] = None
            years_out[sc] = None
            continue

        pos_weights = [pct for _, pct, _ in rows if pct is not None and pct > 0]
        if pos_weights:
            total = sum(pos_weights)
            weights = [(pct or 0.0) / total for _, pct, _ in rows] if total > 0 else [1.0 / len(rows)] * len(rows)
        else:
            weights = [1.0 / len(rows)] * len(rows)

        scores_out[sc] = sum(val * w for (val, _, _), w in zip(rows, weights))
        years_out[sc] = max(year_used for _, _, year_used in rows)

    return scores_out, years_out
