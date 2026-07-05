#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gdp_percap_lookup.py

Public API:
    get_gdp_percap_from_location(locations, recipient_countries, *, csv_path=None,
                                 default_value=None)

Returns a single float (USD) representing the (weighted) average GDP per capita
for the activity's location(s), or None if it cannot be reasonably inferred.

Signals used (in order of confidence):
  1) recipient-country entries: @code (ISO2), @percentage (weight), narrative
  2) location-id with vocabulary "A4": ISO2 directly
  3) human-readable names/descriptions ("Lao PDR", "Nairobi", sub-national regions):
       - map via alias table (e.g., Jharkhand -> IN)

Weights:
  - Percentages are used if present and > 0 (normalized to 1.0).
  - Otherwise, equal weights across unique inferred countries.

Data sources for GDP per capita (USD):
  A) Tidy CSV: iso2,gdp_per_capita_usd
  B) World Bank-style CSV: Country Code,1960,1961,...,<latest-year>
     (we take the latest non-empty year per row)

Configuration:
  - csv_path:      filepath to GDP per capita CSV (A or B). If None, use:
                   os.environ['GDP_PCAP_CSV'] or '../../data/gdp_per_capita_usd.csv'

Note:
  - We do not attempt sub-national GDP; sub-state names are only used to infer
    the correct country for country-level GDP per capita.

"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


# ======================
# In-memory caches
# ======================

_GDP_CACHE: Optional[Dict[str, float]] = None
_GDP_CACHE_KEY: Optional[str] = None


# ======================
# Alias & Sub-state maps
# ======================

# Country/region aliases → ISO2
_ALIAS_TO_ISO2: Dict[str, str] = {
    # Common country names/variants seen in IATI
    "lao pdr": "LA", "laos": "LA", "lao": "LA",
    "lao people’s democratic republic": "LA", "lao people’s dem. rep.": "LA",
    "mongolia": "MN",
    "egypt": "EG", "ägypten": "EG", "cairo": "EG",
    "kenia": "KE", "kenya": "KE", "nairobi": "KE",
    "india": "IN", "bharat": "IN",
    "nepal": "NP",
    "burkina faso": "BF",
    "colombia": "CO",
    "sierra leone": "SL",
    "bangladesh": "BD",
    "liberia": "LR",
    "malawi": "MW",
    "south africa": "ZA",
    "uganda": "UG",
    "tanzania, united republic of": "TZ", "tanzania": "TZ",
    "mozambique": "MZ",
    "zambia": "ZM",
    "algeria": "DZ", "algiers": "DZ",
    "indonesia": "ID",
    "nicaragua": "NI",
    "kazakhstan": "KZ",
    "armenia": "AM", "azerbaijan": "AZ", "moldova": "MD",
    "tajikistan": "TJ", "vietnam": "VN", "viet nam": "VN",
    "serbia": "RS",
    # Global/region tokens to ignore
    "global": "",
    "europe, regional": "",
}

# Sub-national “state/province” names → country ISO2 (extend as needed)
_SUBSTATE_TO_ISO2: Dict[str, str] = {
    # India
    "jharkhand": "IN", "chhattisgarh": "IN", "bihar": "IN",
    "rajasthan": "IN", "west bengal": "IN",
    "uttar pradesh": "IN", "odisha": "IN", "orissa": "IN",
    "betul": "IN",  # district example
    # Kenya (city example for robustness)
    "nairobi": "KE",
    # Egypt
    "cairo": "EG",
}


# ======================
# Utilities
# ======================

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _parse_percent(p: Optional[str]) -> Optional[float]:
    if not p:
        return None
    s = str(p).strip().rstrip("%")
    try:
        v = float(s)
        return v if abs(v) > 1e-12 else 0.0
    except Exception:
        return None


# ======================
# GDP table loading
# ======================

def _load_gdp_table(csv_path: Optional[Union[str, Path]] = None) -> Dict[str, float]:
    """
    Load GDP per capita (USD) into dict: ISO2 -> float.
    Supports:
      A) Tidy CSV: iso2,gdp_per_capita_usd
      B) World Bank-style CSV: Country Code,1960,...,<latest>
    """
    global _GDP_CACHE, _GDP_CACHE_KEY

    default_path = Path(__file__).parent.parent.parent / "data" / "gdp_per_capita_usd.csv"
    search_path = Path(
        csv_path or os.environ.get("GDP_PCAP_CSV", str(default_path))
    ).expanduser().resolve()

    if _GDP_CACHE is not None and _GDP_CACHE_KEY == str(search_path):
        return _GDP_CACHE

    table: Dict[str, float] = {}
    if not search_path.exists():
        # Do NOT cache a missing file: the file may appear later (e.g. copied in
        # after process start). Caching the empty table here would pin GDP to None
        # for the life of the process. Return empty without caching so the next
        # call re-checks.
        return table

    with search_path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        headers = [h.strip() for h in (rdr.fieldnames or [])]

        tidy = {"iso2", "gdp_per_capita_usd"}.issubset({h.lower() for h in headers})
        wb = ("Country Code" in headers) or ("country code" in [h.lower() for h in headers])

        if tidy:
            for row in rdr:
                iso2 = (row.get("iso2") or "").strip().upper()
                val = (row.get("gdp_per_capita_usd") or "").strip()
                if not iso2 or not val:
                    continue
                try:
                    table[iso2] = float(val)
                except Exception:
                    continue
        elif wb:
            code_col = "Country Code" if "Country Code" in headers else [h for h in headers if h.lower() == "country code"][0]
            year_cols = [h for h in headers if h.isdigit() and len(h) == 4]
            year_cols.sort()
            for row in rdr:
                iso2 = (row.get(code_col) or "").strip().upper()
                if not iso2:
                    continue
                val = None
                for yh in reversed(year_cols):
                    s = (row.get(yh) or "").strip()
                    if s:
                        try:
                            val = float(s)
                            break
                        except Exception:
                            continue
                if val is not None:
                    table[iso2] = val
        else:
            # Unknown format → leave empty
            table = {}

    _GDP_CACHE = table
    _GDP_CACHE_KEY = str(search_path)
    return table


# ======================
# ISO2 extraction
# ======================

def _extract_iso2_from_recipient(recips: Optional[Iterable[dict]]) -> List[Tuple[str, Optional[float]]]:
    out: List[Tuple[str, Optional[float]]] = []
    if not isinstance(recips, list):
        return out
    for rc in recips:
        code = (rc.get("@code") or "").strip().upper()
        pct = _parse_percent(rc.get("@percentage"))
        if code and len(code) == 2:
            out.append((code, pct))
        else:
            narr = rc.get("narrative") or []
            if isinstance(narr, list) and narr:
                alias = _ALIAS_TO_ISO2.get(_norm(narr[0].get("text()")), "")
                if alias:
                    out.append((alias, pct))
    return out


def _extract_iso2_from_location_ids(locations: Optional[Iterable[dict]]) -> List[str]:
    """
    Returns ISO2 codes from A4-vocabulary location-ids (2-letter codes).
    """
    iso2: List[str] = []
    if not isinstance(locations, list):
        return iso2

    for loc in locations:
        lids = loc.get("location-id") or []
        if isinstance(lids, list) and lids:
            item = lids[0]
            voc = (item.get("@vocabulary") or "").strip().upper()
            code = (item.get("@code") or "").strip()
            if voc == "A4" and len(code) == 2 and code.isalpha():
                iso2.append(code.upper())

    # de-dupe preserving order
    return list(dict.fromkeys(iso2))


def _extract_iso2_from_names_refs(locations: Optional[Iterable[dict]]) -> List[str]:
    """
    Use name/description/activity-description/@ref to deduce ISO2 via alias tables
    or @ref prefix 'AF-KAN' -> 'AF'.
    """
    out: List[str] = []
    if not isinstance(locations, list):
        return out

    for loc in locations:
        # names / descriptions
        for key in ("name", "description", "activity-description"):
            arr = loc.get(key) or []
            if isinstance(arr, list) and arr:
                narr = arr[0].get("narrative") or []
                if isinstance(narr, list) and narr:
                    txt = _norm(narr[0].get("text()"))
                    iso = _ALIAS_TO_ISO2.get(txt, "") or _SUBSTATE_TO_ISO2.get(txt, "")
                    if iso:
                        out.append(iso)

        # @ref two-letter prefix
        ref = (loc.get("@ref") or "").strip().upper()
        if len(ref) >= 2 and ref[:2].isalpha():
            out.append(ref[:2])

    # de-dupe preserving order
    return list(dict.fromkeys([c for c in out if c]))


# ======================
# Main public function
# ======================

def get_gdp_percap_from_location(
    locations: Optional[Iterable[dict]],
    recipient_countries: Optional[Iterable[dict]],
    *,
    csv_path: Optional[Union[str, Path]] = None,
    default_value: Optional[float] = None,
) -> Optional[float]:
    """
    Compute a single GDP per capita (USD) for the activity, or None.

    Strategy:
      1) recipient-country codes with % weights (normalized).
      2) location-id (A4) ISO2 codes.
      3) names/descriptions/@ref → alias/substate maps.
      4) Combine unique ISO2 with weights (percentages if available, else equal).
      5) Return weighted average of available GDP values; else None.

    Args:
        locations:           act.get("location")
        recipient_countries: act.get("recipient-country")
        csv_path:            GDP per capita CSV (tidy or WB-style). Optional.
        default_value:       returned only if you explicitly pass it and nothing matches;
                             otherwise the function returns None in that case.

    Returns:
        float (USD) or None
    """
    gdp = _load_gdp_table(csv_path)

    # 1) Recipient-country (possibly with weights)
    rc_pairs = _extract_iso2_from_recipient(recipient_countries)
    weighted: List[Tuple[str, Optional[float]]] = rc_pairs[:]  # [(ISO2, weight|None)]

    # 2) Location-id (A4 ISO2)
    iso2_from_ids = _extract_iso2_from_location_ids(locations)
    for iso in iso2_from_ids:
        if iso not in [c for c, _ in weighted]:
            weighted.append((iso, None))

    # 3) Names/descriptions/@ref → alias/substate
    iso2_from_names = _extract_iso2_from_names_refs(locations)
    for iso in iso2_from_names:
        if iso and iso not in [c for c, _ in weighted]:
            weighted.append((iso, None))

    # Remove empty/invalid tokens and dedupe
    clean_weighted: List[Tuple[str, Optional[float]]] = []
    seen = set()
    for iso, w in weighted:
        if not iso or len(iso) != 2:
            continue
        if iso not in seen:
            seen.add(iso)
            clean_weighted.append((iso.upper(), w))

    if not clean_weighted:
        return default_value if default_value is not None else None

    # Use % weights if present and positive; otherwise equal weights
    pos_weights = [(iso, w) for iso, w in clean_weighted if w is not None and w > 0]
    if pos_weights:
        total = sum(w for _, w in pos_weights)
        if total > 0:
            norm_weights = {iso: w / total for iso, w in pos_weights}
        else:
            norm_weights = {}
    else:
        norm_weights = {}

    # If we have normalized % weights, use them; otherwise equal weights
    if norm_weights:
        iso_weight_pairs = [(iso, norm_weights.get(iso, 0.0)) for iso, _ in clean_weighted if iso in norm_weights]
        # There can be additional iso’s without %—ignore them so that % weights remain 1.0
        if not iso_weight_pairs:
            # fall back to equal weights across all countries discovered
            eq = 1.0 / len(clean_weighted)
            iso_weight_pairs = [(iso, eq) for iso, _ in clean_weighted]
    else:
        eq = 1.0 / len(clean_weighted)
        iso_weight_pairs = [(iso, eq) for iso, _ in clean_weighted]

    # Compute weighted average over those with GDP values
    vals: List[Tuple[float, float]] = []
    for iso, w in iso_weight_pairs:
        val = gdp.get(iso.upper())
        if val is not None:
            vals.append((val, w))

    if not vals:
        return default_value if default_value is not None else None

    # Normalize weights of available entries (in case some countries lacked GDP values)
    totw = sum(w for _, w in vals)
    if totw <= 0:
        # equal weights among those that do have GDP values
        eqw = 1.0 / len(vals)
        return sum(v * eqw for v, _ in vals)

    return sum(v * (w / totw) for v, w in vals)


# ================
# CLI smoke test
# ================

if __name__ == "__main__":
    # Minimal self-test (will return None unless you provide a CSV with values for KE/UG)
    fake_locs = [
        {"location-id": [{"@vocabulary": "A4", "@code": "KE"}]},
        {"name": [{"narrative": [{"text()": "Uganda"}]}]},
        {"@ref": "TZ-123"},
    ]
    fake_rc = [
        {"@code": "KE", "@percentage": "60"},
        {"@code": "UG", "@percentage": "40"},
    ]
    val = get_gdp_percap_from_location(
        fake_locs, fake_rc,
        csv_path=os.environ.get("GDP_PCAP_CSV"),  # set if you want a real number
    )
    print(val)
