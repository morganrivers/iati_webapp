"""
Extract country-level features (GDP, CPIA, WGI) from location information.
Uses the same logic as H_extract_info_old_style.py
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime, date

# Add utils to path
UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))




from debug_utils import _print_ram

_print_ram("before from get_gdppercap import get_gdp_percap_from_location")
from get_gdppercap import get_gdp_percap_from_location
_print_ram("before from cpia_lookup import load_cpia_scores, get_activity_cpia")
from cpia_lookup import load_cpia_scores, get_activity_cpia
_print_ram("before from get_world_bank_indicators import load_world_bank_indicators, get_activity_wgi, WGI_SERIES")
from get_world_bank_indicators import load_world_bank_indicators, get_activity_wgi, WGI_SERIES
_print_ram("before from get_regions import get_regions_for_activity, parse_regions_string")
from get_regions import get_regions_for_activity, parse_regions_string


def parse_location_string(location_string: str) -> List[Tuple[Optional[str], Optional[str], Optional[float]]]:
    """
    Parse location string into recipient_countries format.

    Supports two formats:
    1. Simple: "Kenya" or "KE" -> [("KE", None, 100.0)]
    2. Multi-country: "KE|50|UG|30|TZ|20" -> [("KE", None, 50.0), ("UG", None, 30.0), ("TZ", None, 20.0)]

    Returns:
        List of (iso2_code, country_name, percentage) tuples
    """
    location_string = location_string.strip()
    if not location_string:
        return []

    # Check if it contains pipe separator (multi-country format)
    if "|" in location_string:
        parts = location_string.split("|")
        recipient_countries = []

        # Parse pairs of ISO2|percentage
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                iso2 = parts[i].strip().upper()
                try:
                    pct = float(parts[i + 1].strip())
                    recipient_countries.append((iso2, None, pct))
                    i += 2
                except ValueError:
                    # If percentage parsing fails, skip this pair
                    i += 2
            else:
                # Odd number of parts, skip the last one
                break

        return recipient_countries
    else:
        # Simple format: single country
        iso2 = None
        if len(location_string) == 2:
            # Looks like ISO2 code
            iso2 = location_string.upper()
        # Return with 100% allocation
        return [(iso2, None, 100.0)]


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

    import traceback
    with open("debug_location_calls.log", "a") as f:
        f.write("\n=== extract_features_from_location CALLED ===\n")
        f.write(f"location_string: {repr(location_string)}\n")
        f.write(f"start_date: {repr(start_date)}\n")
        f.write("STACK TRACE:\n")
        f.write("".join(traceback.format_stack()))
        f.write("\n")

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"

    # Initialize result with None values
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
        # Parse location string into recipient_countries format
        recipient_countries = parse_location_string(location_string)
        print(f"[LOC-FEAT-MODULE] parse_location_string({location_string!r}) → {recipient_countries!r}")
        # NOTE: this parse_location_string (in location_features.py) only recognises 2-char ISO2 codes.
        # The webapp stores ISO2 codes (KE, UG…) so single-country strings like "KE" work fine.
        # Multi-country strings like "KE|50|UG|50" also work.
        # If iso2 comes back as None for any entry, the downstream lookup will silently return None.

        if not recipient_countries:
            print(f"[LOC-FEAT-MODULE] No recipient_countries parsed — returning empty result")
            return result

        # Convert to format expected by utility functions
        # They expect dicts with "@code" and "@percentage" keys
        recipient_countries_dicts = [
            {
                "@code": iso2 if iso2 else None,
                "@percentage": str(pct) if pct is not None else "100",
                "narrative": []
            }
            for iso2, name, pct in recipient_countries
        ]

        # Create meta_entry dict for CPIA and WGI functions
        meta_entry = {
            "recipient_countries": recipient_countries,  # List of tuples format
            "planned_start_date": start_date,
            "actual_start_date": None,
        }

        # Try to get GDP per capita
        # NOTE: get_gdp_percap_from_location does NOT receive start_date — check if it uses a year internally
        print(f"[LOC-FEAT-MODULE] GDP input: recipient_countries_dicts={recipient_countries_dicts!r}  (no start_date passed)")
        try:
            gdp = get_gdp_percap_from_location(
                locations=None,
                recipient_countries=recipient_countries_dicts,
                csv_path=str(data_dir / "gdp_per_capita_usd.csv"),
            )
            result['gdp_percap'] = gdp
            print(f"[LOC-FEAT-MODULE] GDP result: gdp_percap={gdp!r}")
        except Exception as e:
            print(f"[LOC-FEAT-MODULE] GDP lookup FAILED: {e}")

        # Try to get CPIA score
        print(f"[LOC-FEAT-MODULE] CPIA input: recipient_countries={meta_entry['recipient_countries']!r}  planned_start_date={meta_entry['planned_start_date']!r}")
        try:
            cpia_by_iso3, cpia_name_index = load_cpia_scores()

            cpia_score, cpia_year = get_activity_cpia(
                meta_entry,
                txn_first=None,
                cpia_by_iso3=cpia_by_iso3,
                cpia_name_index=cpia_name_index
            )
            result['cpia_score'] = cpia_score
            print(f"[LOC-FEAT-MODULE] CPIA result: cpia_score={cpia_score!r}  year_used={cpia_year!r}")
        except Exception as e:
            print(f"[LOC-FEAT-MODULE] CPIA lookup FAILED: {e}")

        # Try to get WGI indicators
        print(f"[LOC-FEAT-MODULE] WGI input: recipient_countries={meta_entry['recipient_countries']!r}  planned_start_date={meta_entry['planned_start_date']!r}")
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
            print(f"[LOC-FEAT-MODULE] WGI result: wgi_scores={wgi_scores!r}  years_used={wgi_years!r}")

            # Extract the 5 WGI indicators
            wgi_values = [
                wgi_scores.get('CC.EST'),   # Control of Corruption
                wgi_scores.get('PV.EST'),   # Political Stability
                wgi_scores.get('GE.EST'),   # Government Effectiveness
                wgi_scores.get('RQ.EST'),   # Regulatory Quality
                wgi_scores.get('RL.EST'),   # Rule of Law
            ]

            # Calculate governance_composite as mean of 5 WGI indicators
            valid_wgi = [v for v in wgi_values if v is not None]
            if valid_wgi:
                result['governance_composite'] = sum(valid_wgi) / len(valid_wgi)
                result['wgi_any_missing'] = 1.0 if len(valid_wgi) < 5 else 0.0
            else:
                result['wgi_any_missing'] = 1.0
            print(f"[LOC-FEAT-MODULE] governance_composite={result['governance_composite']!r}  wgi_any_missing={result['wgi_any_missing']!r}")

        except Exception as e:
            print(f"[LOC-FEAT-MODULE] WGI lookup FAILED: {e}")

        # Try to get regions
        try:
            regions_str = get_regions_for_activity(
                recipient_countries,  # List of tuples format
                activity_scope
            )
            regions_dict = parse_regions_string(regions_str)

            # Update result with region fractions
            for region, fraction in regions_dict.items():
                result[f'region_{region}'] = fraction

        except Exception as e:
            print(f"Warning: Could not extract regions: {e}")

    except Exception as e:
        import traceback
        print(f"❌ Error extracting location features: {e}")
        print(traceback.format_exc())

    return result


# Reporting organizations to restrict to (from C_run_GLM_nobayes.py)
KEEP_REPORTING_ORGS = [
    "BMZ",
    "UK - Foreign, Commonwealth Development Office (FCDO)",
    "Asian Development Bank",
    "World Bank",
]

# Organization name to dummy variable mapping
ORG_NAME_TO_DUMMY = {
    "UK - Foreign, Commonwealth Development Office (FCDO)": {"rep_org_0": 1, "rep_org_1": 0, "rep_org_2": 0},
    "Asian Development Bank": {"rep_org_0": 0, "rep_org_1": 1, "rep_org_2": 0},
    "World Bank": {"rep_org_0": 0, "rep_org_1": 0, "rep_org_2": 1},
    "BMZ": {"rep_org_0": 0, "rep_org_1": 0, "rep_org_2": 0},
}


def get_org_dummies(org_name: str) -> Dict[str, int]:
    """
    Get organization dummy variables.

    Args:
        org_name: Full organization name

    Returns:
        Dictionary with rep_org_0, rep_org_1, rep_org_2
    """
    return ORG_NAME_TO_DUMMY.get(org_name, {"rep_org_0": 0, "rep_org_1": 0, "rep_org_2": 0})


if __name__ == "__main__":
    # Test
    print("Testing location feature extraction...")

    test_locations = ["Kenya", "KE", "Bangladesh", "BD"]

    for loc in test_locations:
        print(f"\n{loc}:")
        features = extract_features_from_location(loc)
        for k, v in features.items():
            print(f"  {k}: {v}")
