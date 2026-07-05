"""
Extract country-level features (GDP, CPIA, WGI) from location information.
Uses the same logic as H_extract_info_old_style.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime, date

from webapp_paths import ensure_src_paths
ensure_src_paths()

from utils import parse_location_string as _parse_loc_str_dicts
from get_gdppercap import get_gdp_percap_from_location
from cpia_lookup import load_cpia_scores, get_activity_cpia
from get_world_bank_indicators import load_world_bank_indicators, get_activity_wgi, WGI_SERIES
from get_regions import get_regions_for_activity, parse_regions_string

logger = logging.getLogger(__name__)


def parse_location_string(location_string: str) -> List[Tuple[Optional[str], Optional[str], Optional[float]]]:
    """
    Parse location string into recipient_countries format.

    Supports two formats:
    1. Simple: "KE" -> [("KE", None, 100.0)]
    2. Multi-country: "KE|50|UG|30|TZ|20" -> [("KE", None, 50.0), ("UG", None, 30.0), ("TZ", None, 20.0)]

    Returns:
        List of (iso2_code, country_name, percentage) tuples
    """
    return [(d["code"], None, float(d["pct"])) for d in _parse_loc_str_dicts(location_string)]


def extract_features_from_location(
    location_string: str,
    start_date: Optional[date] = None,
    activity_scope: Optional[str] = None,
    data_dir: Path = None,
) -> Dict[str, Optional[float]]:
    """
    Extract GDP per capita, CPIA score, WGI indicators, and regions from location string.
    Uses the same utilities as H_extract_info_old_style.py for consistency.

    Args:
        location_string: Country name, ISO code, or multi-country format
            - Simple: "Kenya" or "KE"
            - Multi-country: "KE|50|UG|30|TZ|20" (ISO2|pct|ISO2|pct|...)
        start_date: Activity start date (used for year-based lookups)
        activity_scope: Activity scope code ("1" = national, "2" = regional, etc.)
        data_dir: Path to data directory (defaults to ../../data)

    Returns:
        Dictionary with:
        - gdp_percap
        - cpia_score
        - governance_composite (mean of 5 WGI indicators)
        - wgi_any_missing (flag if any WGI missing)
        - region_AFE, region_AFW, region_EAP, region_ECA, region_LAC, region_MENA, region_SAS
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"

    result = {
        'gdp_percap': None,
        'cpia_score': None,
        'governance_composite': None,
        'wgi_any_missing': None,
        'region_AFE': 0.0,
        'region_AFW': 0.0,
        'region_EAP': 0.0,
        'region_ECA': 0.0,
        'region_LAC': 0.0,
        'region_MENA': 0.0,
        'region_SAS': 0.0,
    }

    try:
        recipient_countries = parse_location_string(location_string)
        logger.debug("parse_location_string(%r) -> %r", location_string, recipient_countries)

        if not recipient_countries:
            logger.debug("No recipient_countries parsed — returning empty result")
            return result

        recipient_countries_dicts = [
            {
                "@code": iso2 if iso2 else None,
                "@percentage": str(pct) if pct is not None else "100",
                "narrative": []
            }
            for iso2, name, pct in recipient_countries
        ]

        meta_entry = {
            "recipient_countries": recipient_countries,
            "planned_start_date": start_date,
            "actual_start_date": None,
        }

        logger.debug("GDP input: recipient_countries_dicts=%r", recipient_countries_dicts)
        try:
            gdp = get_gdp_percap_from_location(
                locations=None,
                recipient_countries=recipient_countries_dicts,
                csv_path=str(data_dir / "gdp_per_capita_usd.csv"),
            )
            result['gdp_percap'] = gdp
            logger.debug("GDP result: gdp_percap=%r", gdp)
        except Exception as e:
            logger.debug("GDP lookup failed: %s", e)

        logger.debug("CPIA input: recipient_countries=%r planned_start_date=%r",
                     meta_entry['recipient_countries'], meta_entry['planned_start_date'])
        try:
            cpia_by_iso3, cpia_name_index = load_cpia_scores()
            cpia_score, cpia_year = get_activity_cpia(
                meta_entry,
                txn_first=None,
                cpia_by_iso3=cpia_by_iso3,
                cpia_name_index=cpia_name_index
            )
            result['cpia_score'] = cpia_score
            logger.debug("CPIA result: cpia_score=%r year_used=%r", cpia_score, cpia_year)
        except Exception as e:
            logger.debug("CPIA lookup failed: %s", e)

        logger.debug("WGI input: recipient_countries=%r planned_start_date=%r",
                     meta_entry['recipient_countries'], meta_entry['planned_start_date'])
        try:
            wgi_by_iso3, wgi_name_index = load_world_bank_indicators(
                keep_series_codes=list(WGI_SERIES.keys())
            )
            wgi_scores, wgi_years = get_activity_wgi(
                meta_entry,
                txn_first=None,
                wgi_by_iso3=wgi_by_iso3,
                wgi_name_index=wgi_name_index,
                series_codes=list(WGI_SERIES.keys())
            )
            logger.debug("WGI result: wgi_scores=%r years_used=%r", wgi_scores, wgi_years)

            wgi_values = [
                wgi_scores.get('CC.EST'),
                wgi_scores.get('PV.EST'),
                wgi_scores.get('GE.EST'),
                wgi_scores.get('RQ.EST'),
                wgi_scores.get('RL.EST'),
            ]

            valid_wgi = [v for v in wgi_values if v is not None]
            if valid_wgi:
                result['governance_composite'] = sum(valid_wgi) / len(valid_wgi)
                result['wgi_any_missing'] = 1.0 if len(valid_wgi) < 5 else 0.0
            else:
                result['wgi_any_missing'] = 1.0
            logger.debug("governance_composite=%r wgi_any_missing=%r",
                         result['governance_composite'], result['wgi_any_missing'])

        except Exception as e:
            logger.debug("WGI lookup failed: %s", e)

        try:
            regions_str = get_regions_for_activity(
                recipient_countries,
                activity_scope
            )
            regions_dict = parse_regions_string(regions_str)
            for region, fraction in regions_dict.items():
                result[f'region_{region}'] = fraction
        except Exception as e:
            logger.warning("Could not extract regions: %s", e)

    except Exception as e:
        import traceback
        logger.error("Error extracting location features: %s\n%s", e, traceback.format_exc())

    return result


KEEP_REPORTING_ORGS = [
    "BMZ",
    "UK - Foreign, Commonwealth Development Office (FCDO)",
    "Asian Development Bank",
    "World Bank",
]

ORG_NAME_TO_DUMMY = {
    "UK - Foreign, Commonwealth Development Office (FCDO)": {"rep_org_0": 1, "rep_org_1": 0, "rep_org_2": 0},
    "Asian Development Bank": {"rep_org_0": 0, "rep_org_1": 1, "rep_org_2": 0},
    "World Bank": {"rep_org_0": 0, "rep_org_1": 0, "rep_org_2": 1},
    "BMZ": {"rep_org_0": 0, "rep_org_1": 0, "rep_org_2": 0},
}


def get_org_dummies(org_name: str) -> Dict[str, int]:
    return ORG_NAME_TO_DUMMY.get(org_name, {"rep_org_0": 0, "rep_org_1": 0, "rep_org_2": 0})


if __name__ == "__main__":
    print("Testing location feature extraction...")
    test_locations = ["KE", "BD"]
    for loc in test_locations:
        print(f"\n{loc}:")
        features = extract_features_from_location(loc)
        for k, v in features.items():
            print(f"  {k}: {v}")
