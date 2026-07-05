#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from pathlib import Path
from datetime import date
from typing import Dict, List, Tuple, Optional
import pycountry

_CPIA_BY_ISO3: Optional[Dict[str, Dict[int, float]]] = None
_CPIA_NAME_INDEX: Optional[Dict[str, str]] = None

CPIA_PATH = Path(__file__).parent.parent.parent / "data" / "cpia_scores.csv"

def _norm_country_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "")).strip().lower()


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


def _iso2_to_iso3_guess(
    iso2: Optional[str],
    cpia_by_iso3: Dict[str, Dict[int, float]],
) -> Optional[str]:
    if not iso2:
        return None
    iso2 = iso2.strip().upper()
    if not iso2:
        return None

    if iso2 in ISO2_TO_ISO3_OVERRIDES:
        return ISO2_TO_ISO3_OVERRIDES[iso2]

    country = pycountry.countries.get(alpha_2=iso2)
    if country is not None:
        alpha_3 = country.alpha_3
        if alpha_3:
            return alpha_3
    else:
        candidates = [iso3 for iso3 in cpia_by_iso3.keys() if iso3.startswith(iso2)]
        if len(candidates) == 1:
            return candidates[0]

    return None


def load_cpia_scores() -> Tuple[Dict[str, Dict[int, float]], Dict[str, str]]:
    """
    Parse a World Bank-style CPIA CSV into:

        by_iso3[ISO3][year] -> score (float)
        name_index[normalized country name] -> ISO3

    """
    global _CPIA_BY_ISO3, _CPIA_NAME_INDEX
    if _CPIA_BY_ISO3 is not None and _CPIA_NAME_INDEX is not None:
        return _CPIA_BY_ISO3, _CPIA_NAME_INDEX

    by_iso3: Dict[str, Dict[int, float]] = {}
    name_idx: Dict[str, str] = {}

    with CPIA_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames or []

        year_cols: List[Tuple[int, str]] = []
        for h in headers:
            m = re.match(r"(\d{4})", h)
            if m:
                year = int(m.group(1))
                year_cols.append((year, h))
        year_cols.sort()

        for row in rdr:
            iso3 = (row.get("Country Code") or "").strip().upper()
            cname = (row.get("Country Name") or "").strip()
            if not iso3:
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

            by_iso3[iso3] = year_map
            if cname:
                name_idx[_norm_country_name(cname)] = iso3

    _CPIA_BY_ISO3 = by_iso3
    _CPIA_NAME_INDEX = name_idx
    return by_iso3, name_idx

def _pick_cpia_for_country_year(
    iso3: str,
    year: int,
    cpia_by_iso3: Dict[str, Dict[int, float]],
) -> Optional[Tuple[float, int]]:
    """
    For a given country (ISO3) and reference year, choose the CPIA observation:
      - latest year <= reference year, else earliest available year.
    Returns (score, year_used) or None.
    """
    year_map = cpia_by_iso3.get(iso3)
    if not year_map:
        return None

    years = list(year_map.keys())
    le_years = [y for y in years if y <= year]
    if le_years:
        chosen_year = max(le_years)
    else:
        chosen_year = min(years)

    score = year_map.get(chosen_year)
    if score is None:
        return None
    return score, chosen_year
def get_activity_cpia(
    meta_entry: dict,
    txn_first: Optional[date],
    cpia_by_iso3: Dict[str, Dict[int, float]],
    cpia_name_index: Dict[str, str],
) -> Tuple[Optional[float], Optional[int]]:
    """
    Compute a CPIA score for an IATI activity as a *weighted average*
    over all recipient countries.

    Inputs expected in `meta_entry`:
      - "recipient_countries": list of (code, name, pct) as returned by
          _recipient_country_entries(act):
            code: ISO2 or None
            name: raw narrative or None
            pct:  float percentage (e.g. 23.84) or None
      - "actual_start_date": datetime.date or None
      - "planned_start_date": datetime.date or None

    `txn_first` is the first actual transaction date (or None).

    Steps:
      1) Reference year = year(txn_first or actual_start_date or planned_start_date)
      2) For each recipient country:
           - map ISO2 -> ISO3 (or use name via cpia_name_index)
           - pick CPIA(score, year_used) for that country
      3) Weighted average across countries:
           - if any pct>0: use those as weights (normalized)
           - else: equal weights across countries that have CPIA
      4) Return (weighted_score, max year_used among contributing countries)
    """
    start_date = (
        meta_entry.get("actual_start_date")
        or meta_entry.get("planned_start_date")
        or txn_first
    )
    if not isinstance(start_date, date):
        return None, None

    ref_year = start_date.year

    rc_entries = meta_entry.get("recipient_countries") or []

    country_scores: List[Tuple[float, Optional[float], int]] = []

    for entry in rc_entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            continue
        code, name, pct = entry
        iso2 = (code or "").strip().upper() or None

        iso3 = _iso2_to_iso3_guess(iso2, cpia_by_iso3)
        if not iso3 and name:
            iso3 = cpia_name_index.get(_norm_country_name(name))

        if not iso3:
            continue

        picked = _pick_cpia_for_country_year(iso3, ref_year, cpia_by_iso3)
        if not picked:
            continue
        score, year_used = picked
        country_scores.append((score, pct, year_used))

    if not country_scores:
        return None, None

    pos_weights = [pct for _, pct, _ in country_scores if pct is not None and pct > 0]
    if pos_weights:
        total = sum(pos_weights)
        if total <= 0:
            weights = [1.0 / len(country_scores)] * len(country_scores)
        else:
            weights = [(pct or 0.0) / total for _, pct, _ in country_scores]
    else:
        weights = [1.0 / len(country_scores)] * len(country_scores)

    weighted_score = sum(score * w for (score, _, _), w in zip(country_scores, weights))
    cpia_year = max(year_used for _, _, year_used in country_scores)

    return weighted_score, cpia_year