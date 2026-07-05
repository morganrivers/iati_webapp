#!/usr/bin/env python3
import pandas as pd
from collections import Counter

RANKED_CSV = "../../data/ranked_documents.csv"
INFO_CSV = "../../data/info_for_activity_forecasting.csv"


def parse_dac_codes(dac_str: str):
    """Parse '12345|67890' style DAC5 codes into a set of strings."""
    if not isinstance(dac_str, str) or not dac_str.strip():
        return set()
    parts = dac_str.split("|")
    return {p.strip() for p in parts if p.strip()}



def get_good_bad_and_target_codes():

    # --- Your DAC code sets (as strings) ---

    five_digit = [
        14010, 14015, 14020, 14021, 14022, 14032, 14050,
        21010, 21011, 21012, 12013, 23110, 23111, 23112, 23183,
        23210, 23220, 23230, 23231, 23232, 23240, 23250, 23260, 23270,
        23350, 23360, 23410, 23510, 23610, 23630, 23631, 23642,
        31130, 31192, 31210, 31220, 31260, 31281, 31282, 31291,
        32170, 32174, 32210, 41010, 41020, 41030, 41081, 41082, 43060
    ]
    three_digit = [
        140,   # Water Supply & Sanitation
        231,   # Energy Policy
        232,   # Energy generation, renewable sources
        234,   # Hybrid energy plants
        235,   # Nuclear energy plants
        312,   # Forestry
        410,   # General Environment Protection
    ]

    FIVE_DIG_SET = {str(c) for c in five_digit}
    THREE_DIG_SET = {str(c) for c in three_digit}

    # BAD codes you might drop
    BAD_CODES = {
        "14010",  # Water sector policy and administrative management
        "21010",  # Transport policy and administrative management
        "21011",  # public transport ish
        "21012",  # Public transport services
        "31192",  # Plant and post-harvest protection and pest control
        "32170",  # Non-ferrous metal industries
        "32210",  # Mineral/mining policy and administrative management
        "43060",  # Disaster Risk Reduction
    }

    # GOOD codes = everything in your five_digit/three_digit universe except the bad ones
    GOOD_CODES = (FIVE_DIG_SET | THREE_DIG_SET) - BAD_CODES

    TARGET_CODES = GOOD_CODES | BAD_CODES  # the full climate-ish selection you care about
    return GOOD_CODES, BAD_CODES, TARGET_CODES

def categorize_good_code(code: str) -> str:
    """
    Group GOOD DAC codes into a few coarse buckets:
    - Water & sanitation
    - Renewables & low-carbon energy
    - Energy policy & grid
    - Forests & land use
    - Environment & DRR
    - Other GOOD climate
    """
    prefix3 = code[:3]
    140, # Water Supply & Sanitation
    231, # Energy Policy
    232, # Energy generation, renewable sources
    234, # Hybrid energy plants
    235, # Nuclear energy plants
    312, # Forestry
    410 # General Environment Protection

    if prefix3 == "140":
        return "Water Supply and Sanitation"
    if prefix3 == "231":
        return "Improving Energy Policy"
    if prefix3 in {"232", "233", "234", "235", "236"}:
        return "Clean Energy Generation"
    if prefix3 in {"312", "311"}:
        return "Forestry & Sustainable Agriculture"
    if prefix3 in {"321","410"}:
        return "General Environmental Protection"
    print(f"ERROR: {code} couldnt be categorized")
    quit()
    # return "Other GOOD climate"

def get_any_good_and_only_bad(activities_with_target,activity_to_codes,GOOD_CODES,BAD_CODES):
    only_bad = set()
    any_good = set()

    for aid in activities_with_target:
        codes = activity_to_codes[aid]
        has_good = bool(codes & GOOD_CODES)
        has_bad = bool(codes & BAD_CODES)

        if has_good:
            any_good.add(aid)
        elif has_bad:
            only_bad.add(aid)
    return any_good, only_bad

def get_activity_to_codes():
    info = pd.read_csv(INFO_CSV)

    # Map activity_id -> set of DAC codes (merging across rows if needed)
    activity_to_codes = {}
    for _, row in info.iterrows():
        aid = row["activity_id"]
        codes = parse_dac_codes(row.get("dac5", ""))
        if not codes:
            continue
        activity_to_codes.setdefault(aid, set()).update(codes)

    return activity_to_codes

import json

MERGED_OVERALL_RATINGS = "../../data/merged_overall_ratings.jsonl"

def load_ids_from_jsonl(path: str) -> set[str]:
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            aid = obj.get("activity_id")
            if aid is not None:
                ids.add(str(aid))
    return ids
