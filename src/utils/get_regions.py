#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map countries to World Bank regions and generate region fractions for activities.
"""

from typing import Dict, List, Tuple, Optional

# World Bank regional classification of ISO2 country codes
COUNTRY_TO_REGION = {
    # Africa East
    "BW": "AFE", "KE": "AFE", "LS": "AFE", "MG": "AFE", "MZ": "AFE",
    "NA": "AFE", "RW": "AFE", "SZ": "AFE", "TZ": "AFE", "UG": "AFE", "ZA": "AFE", "ZM": "AFE", "ZW": "AFE",

    # Africa West
    "BJ": "AFW", "BF": "AFW", "CV": "AFW", "CI": "AFW", "GM": "AFW", "GH": "AFW",
    "GN": "AFW", "GW": "AFW", "LR": "AFW", "ML": "AFW", "MR": "AFW", "NE": "AFW",
    "NG": "AFW", "SN": "AFW", "SL": "AFW", "TG": "AFW",

    # East Asia and Pacific
    "AG": "EAP", "AU": "EAP", "BD": "EAP", "BN": "EAP", "KH": "EAP", "CN": "EAP",
    "FJ": "EAP", "FM": "EAP", "HK": "EAP", "ID": "EAP", "JP": "EAP", "KR": "EAP",
    "KP": "EAP", "LA": "EAP", "MO": "EAP", "MY": "EAP", "MM": "EAP", "MH": "EAP",
    "MN": "EAP", "NP": "EAP", "NZ": "EAP", "PW": "EAP", "PH": "EAP", "SG": "EAP",
    "SB": "EAP", "TH": "EAP", "TL": "EAP", "VN": "EAP",

    # Europe and Central Asia
    "AL": "ECA", "AM": "ECA", "AT": "ECA", "AZ": "ECA", "BY": "ECA", "BE": "ECA",
    "BA": "ECA", "BG": "ECA", "HR": "ECA", "CY": "ECA", "CZ": "ECA", "DK": "ECA",
    "EE": "ECA", "FI": "ECA", "FR": "ECA", "GE": "ECA", "DE": "ECA", "GR": "ECA",
    "HU": "ECA", "IS": "ECA", "IE": "ECA", "IT": "ECA", "KZ": "ECA", "KG": "ECA",
    "LV": "ECA", "LI": "ECA", "LT": "ECA", "LU": "ECA", "MT": "ECA", "MD": "ECA",
    "ME": "ECA", "NL": "ECA", "NO": "ECA", "PL": "ECA", "PT": "ECA", "RO": "ECA",
    "RU": "ECA", "RS": "ECA", "SK": "ECA", "SI": "ECA", "ES": "ECA", "SE": "ECA",
    "CH": "ECA", "TJ": "ECA", "TR": "ECA", "TM": "ECA", "UA": "ECA", "GB": "ECA",
    "UZ": "ECA", "XK": "ECA",

    # Latin America and Caribbean
    "AI": "LAC", "AR": "LAC", "BB": "LAC", "BZ": "LAC", "BO": "LAC", "BR": "LAC",
    "CA": "LAC", "CL": "LAC", "CO": "LAC", "CR": "LAC", "CU": "LAC", "DM": "LAC",
    "DO": "LAC", "EC": "LAC", "SV": "LAC", "GD": "LAC", "GT": "LAC", "GY": "LAC",
    "HT": "LAC", "HN": "LAC", "JM": "LAC", "MX": "LAC", "NI": "LAC", "PA": "LAC",
    "PY": "LAC", "PE": "LAC", "KN": "LAC", "LC": "LAC", "VC": "LAC", "SR": "LAC",
    "TT": "LAC", "US": "LAC", "UY": "LAC", "VE": "LAC",

    # Middle East and North Africa
    "DZ": "MENA", "BH": "MENA", "EG": "MENA", "IR": "MENA", "IQ": "MENA", "IL": "MENA",
    "JO": "MENA", "KW": "MENA", "LB": "MENA", "LY": "MENA", "MA": "MENA", "OM": "MENA",
    "QA": "MENA", "SA": "MENA", "SY": "MENA", "TN": "MENA", "AE": "MENA", "WB": "MENA",
    "YE": "MENA",

    # South Asia
    "AF": "SAS", "BT": "SAS", "IN": "SAS", "MV": "SAS", "NP": "SAS", "PK": "SAS",
    "LK": "SAS",
}

# World Bank region 3-letter codes and full names
REGION_CODES = {
    "AFE": "Africa East",
    "AFW": "Africa West",
    "EAP": "East Asia and Pacific",
    "ECA": "Europe and Central Asia",
    "LAC": "Latin America and Caribbean",
    "MENA": "Middle East and North Africa",
    "SAS": "South Asia",
}

# List of all regions (for global activities)
ALL_REGION_CODES = list(REGION_CODES.keys())


def get_regions_for_activity(
    recipient_countries: Optional[List[Tuple[Optional[str], Optional[str], Optional[float]]]],
    activity_scope: Optional[str],
) -> str:
    """
    Generate region distribution string in format "REGION1:fraction|REGION2:fraction|..."
    where fractions sum to 1.

    Args:
        recipient_countries: List of (ISO2_code, country_name, percentage) tuples
        activity_scope: Activity scope code ("1" = global, "2" = regional, etc.)

    Returns:
        String like "AFE:0.33|EAP:0.33|LAC:0.34" or "AFE:1.0" for single region
    """
    if not recipient_countries:
        return ""

    # Check if global activity (scope code "1")
    scope_code = (activity_scope or "").strip()
    is_global = scope_code == "1"

    # Collect regions with their fractions
    region_fractions: Dict[str, float] = {}

    for iso2, country_name, pct in recipient_countries:
        if not iso2:
            continue

        iso2_upper = iso2.upper().strip()
        region = COUNTRY_TO_REGION.get(iso2_upper)

        if not region:
            # Unknown country, skip
            continue

        # Convert percentage to fraction (pct is 0-100 or None)
        if pct is not None and pct > 0:
            fraction = pct / 100.0
        else:
            # If no percentage specified, use equal weight
            fraction = 1.0

        region_fractions[region] = region_fractions.get(region, 0.0) + fraction

    if not region_fractions:
        # No valid regions found
        return ""

    # Normalize fractions to sum to 1
    total = sum(region_fractions.values())
    if total > 0:
        region_fractions = {r: f / total for r, f in region_fractions.items()}

    if is_global and not region_fractions:
        # Global activity with no identified countries - distribute equally
        n_regions = len(ALL_REGION_CODES)
        region_fractions = {r: 1.0 / n_regions for r in ALL_REGION_CODES}

    # Sort by region code for consistency
    sorted_regions = sorted(region_fractions.keys())

    # Format as "REGION:fraction|REGION:fraction|..."
    parts = [f"{region}:{fraction:.4f}" for region, fraction in zip(sorted_regions, [region_fractions[r] for r in sorted_regions])]

    return "|".join(parts)


def parse_regions_string(regions_str: Optional[str]) -> Dict[str, float]:
    """
    Parse regions string into dict: {region_code: fraction}
    Returns all 7 regions with 0.0 for missing ones.

    Args:
        regions_str: String like "AFE:0.5|LAC:0.5" or empty string

    Returns:
        Dict like {"AFE": 0.5, "AFW": 0.0, "EAP": 0.0, "ECA": 0.0, "LAC": 0.5, "MENA": 0.0, "SAS": 0.0}
    """
    # Initialize all regions to 0
    result = {region: 0.0 for region in ALL_REGION_CODES}

    if not regions_str or not isinstance(regions_str, str):
        return result

    # Parse each region:fraction pair
    for part in regions_str.split("|"):
        if ":" not in part:
            continue
        region, fraction_str = part.split(":", 1)
        region = region.strip()
        try:
            fraction = float(fraction_str.strip())
            if region in result:
                result[region] = fraction
        except ValueError:
            pass

    return result
