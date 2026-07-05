#!/usr/bin/env python3
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations

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


def main():
    # --- Load ranked docs and get unique activity_ids ---
    ranked = pd.read_csv(RANKED_CSV)
    if "activity_id" not in ranked.columns:
        raise ValueError("ranked_docs CSV must contain an 'activity_id' column")

    ranked_ids = sorted(set(ranked["activity_id"].dropna()))
    merged_ids = load_ids_from_jsonl(MERGED_OVERALL_RATINGS)
    ranked_ids = [str(aid) for aid in ranked_ids]
    ranked_ids = [aid for aid in ranked_ids if aid in merged_ids]

    total_ranked = len(ranked_ids)

    print(f"Total unique activity_ids in merged_docs: {total_ranked}")

    # --- Load info CSV and restrict to ranked activity_ids ---
    info = pd.read_csv(INFO_CSV)
    if "activity_id" not in info.columns:
        raise ValueError("info_for_activity_forecasting.csv must contain an 'activity_id' column")
    if "dac5" not in info.columns:
        raise ValueError("info_for_activity_forecasting.csv must contain a 'dac5' column")

    info_sub = info[info["activity_id"].isin(ranked_ids)].copy()

    # Map activity_id -> set of DAC codes (merging across rows if needed)
    activity_to_codes = {}
    for _, row in info_sub.iterrows():
        aid = row["activity_id"]
        codes = parse_dac_codes(row.get("dac5", ""))
        if not codes:
            continue
        activity_to_codes.setdefault(aid, set()).update(codes)

    with_dac_info = len(activity_to_codes)
    missing = total_ranked - with_dac_info

    # get_activity_to_codes(INFO_CSV,)

    print(f"Of these, have at least one DAC5 code in info file: {with_dac_info}")
    print(f"Missing/without DAC info (or only empty dac5): {missing}")
    print()
    GOOD_CODES, BAD_CODES, TARGET_CODES = get_good_bad_and_target_codes()


    DAC_SECTORS = {
        11110: "Education policy and administrative management",            
        11120: "Education facilities and training",            
        11130: "Teacher training",            
        11182: "Educational research",            
        11220: "Primary education",            
        11230: "Basic life skills for adults",            
        11231: "Basic life skills for youth",            
        11232: "Primary education equivalent for adults",            
        11240: "Early childhood education",            
        11250: "School feeding",            
        11260: "Lower secondary education",            
        11320: "Upper Secondary Education (modified and includes data from 11322)",            
        11321: "Lower secondary education",            
        11322: "Upper secondary education",            
        11330: "Vocational training",            
        11420: "Higher education",            
        11430: "Advanced technical and managerial training",            
        12110: "Health policy and administrative management",            
        12181: "Medical education/training",            
        12182: "Medical research",            
        12191: "Medical services",            
        12196: "Health statistics and data",            
        12220: "Basic health care",            
        12230: "Basic health infrastructure",            
        12240: "Basic nutrition",            
        12250: "Infectious disease control",            
        12261: "Health education",            
        12262: "Malaria control",            
        12263: "Tuberculosis control",            
        12264: "COVID-19 control",            
        12281: "Health personnel development",            
        12310: "NCDs control, general",            
        12320: "Tobacco use control",            
        12330: "Control of harmful use of alcohol and drugs",            
        12340: "Promotion of mental health and well-being",            
        12350: "Other prevention and treatment of NCDs",            
        12382: "Research for prevention and control of NCDs",            
        13010: "Population policy and administrative management",            
        13020: "Reproductive health care",            
        13030: "Family planning",            
        13040: "STD control including HIV/AIDS",            
        13081: "Personnel development for population and reproductive health",            
        13096: "Population statistics and data",            
        14010: "Water sector policy and administrative management",            
        14015: "Water resources conservation (including data collection)",            
        14020: "Water supply and sanitation - large systems",            
        14021: "Water supply - large systems",            
        14022: "Sanitation - large systems",            
        14030: "Basic drinking water supply and basic sanitation",            
        14031: "Basic drinking water supply",            
        14032: "Basic sanitation",            
        14040: "River basins development",            
        14050: "Waste management/disposal",            
        14081: "Education and training in water supply and sanitation",            
        15110: "Public sector policy and administrative management",            
        15111: "Public finance management (PFM)",            
        15112: "Decentralisation and support to subnational government",            
        15113: "Anti-corruption organisations and institutions",            
        15114: "Domestic revenue mobilisation",            
        15116: "Tax collection",            
        15117: "Budget planning",            
        15118: "National audit",            
        15119: "Debt and aid management",            
        15120: "Public sector financial management",            
        15121: "Foreign affairs",            
        15122: "Diplomatic missions",            
        15123: "Administration of developing countries' foreign aid",            
        15124: "General personnel services",            
        15125: "Public Procurement",            
        15126: "Other general public services",            
        15127: "National monitoring and evaluation",            
        15128: "Local government finance",            
        15129: "Other central transfers to institutions",            
        15130: "Legal and judicial development",            
        15131: "Justice, law and order policy, planning and administration",            
        15132: "Police",            
        15133: "Fire and rescue services",            
        15134: "Judicial affairs",            
        15135: "Ombudsman",            
        15136: "Immigration",            
        15137: "Prisons",            
        15140: "Government administration",            
        15142: "Macroeconomic policy",            
        15143: "Meteorological services",            
        15144: "National standards development",            
        15150: "Democratic participation and civil society",            
        15151: "Elections",            
        15152: "Legislatures and political parties",            
        15153: "Media and free flow of information",            
        15154: "Executive office",            
        15155: "Tax policy and administration support",            
        15156: "Other non-tax revenue mobilisation",            
        15160: "Human rights",            
        15161: "Elections",            
        15162: "Human rights",            
        15163: "Free flow of information",            
        15164: "Women's equality organisations and institutions",            
        15170: "Women's rights organisations and movements, and government institutions",            
        15180: "Ending violence against women and girls",            
        15185: "Local government administration",            
        15190: "Facilitation of orderly, safe, regular and responsible migration and mobility",            
        15196: "Government and civil society statistics and data",            
        15210: "Security system management and reform",            
        15220: "Civilian peace-building, conflict prevention and resolution",            
        15230: "Participation in international peacekeeping operations",            
        15240: "Reintegration and SALW control",            
        15250: "Removal of land mines and explosive remnants of war",            
        15261: "Child soldiers (prevention and demobilisation)",            
        16010: "Social Protection",            
        16011: "Social protection and welfare services policy, planning and administration",            
        16012: "Social security (excl pensions)",            
        16013: "General pensions",            
        16014: "Civil service pensions",            
        16015: "Social services (incl youth development and women+ children)",            
        16020: "Employment creation",            
        16030: "Housing policy and administrative management",            
        16040: "Low-cost housing",            
        16050: "Multisector aid for basic social services",            
        16061: "Culture and recreation",            
        16062: "Statistical capacity building",            
        16063: "Narcotics control",            
        16064: "Social mitigation of HIV/AIDS",            
        16065: "Recreation and sport",            
        16066: "Culture",            
        16070: "Labour rights",            
        16080: "Social dialogue",            
        21010: "Transport policy and administrative management",            
        21011: "Transport policy, planning and administration",            
        21012: "Public transport services",            
        21013: "Transport regulation",            
        21020: "Road transport",            
        21021: "Feeder road construction",            
        21022: "Feeder road maintenance",            
        21023: "National road construction",            
        21024: "National road maintenance",            
        21030: "Rail transport",            
        21040: "Water transport",            
        21050: "Air transport",            
        21061: "Storage",            
        21081: "Education and training in transport and storage",            
        22010: "Communications policy and administrative management",            
        22011: "Communications policy, planning and administration",            
        22012: "Postal services",            
        22013: "Information services",            
        22020: "Telecommunications",            
        22030: "Radio/television/print media",            
        22040: "Information and communication technology (ICT)",            
        23010: "Energy policy and administrative management",            
        23020: "Power generation/non-renewable sources",            
        23030: "Power generation/renewable sources",            
        23040: "Electrical transmission/ distribution",            
        23050: "Gas distribution",            
        23061: "Oil-fired power plants",            
        23062: "Gas-fired power plants",            
        23063: "Coal-fired power plants",            
        23064: "Nuclear power plants",            
        23065: "Hydro-electric power plants",            
        23066: "Geothermal energy",            
        23067: "Solar energy",            
        23068: "Wind power",            
        23069: "Ocean power",            
        23070: "Biomass",            
        23081: "Energy education/training",            
        23082: "Energy research",            
        23110: "Energy policy and administrative management",            
        23111: "Energy sector policy, planning and administration",            
        23112: "Energy regulation",            
        23181: "Energy education/training",            
        23182: "Energy research",            
        23183: "Energy conservation and demand-side efficiency",            
        23210: "Energy generation, renewable sources - multiple technologies",            
        23220: "Hydro-electric power plants",            
        23230: "Solar energy for centralised grids",            
        23231: "Solar energy for isolated grids and standalone systems",            
        23232: "Solar energy - thermal applications",            
        23240: "Wind energy",            
        23250: "Marine energy",            
        23260: "Geothermal energy",            
        23270: "Biofuel-fired power plants",            
        23310: "Energy generation, non-renewable sources, unspecified",            
        23320: "Coal-fired electric power plants",            
        23330: "Oil-fired electric power plants",            
        23340: "Natural gas-fired electric power plants",            
        23350: "Fossil fuel electric power plants with carbon capture and storage (CCS)",            
        23360: "Non-renewable waste-fired electric power plants",            
        23410: "Hybrid energy electric power plants",            
        23510: "Nuclear energy electric power plants and nuclear safety",            
        23610: "Heat plants",            
        23620: "District heating and cooling",            
        23630: "Electric power transmission and distribution (centralised grids)",            
        23631: "Electric power transmission and distribution (isolated mini-grids)",            
        23640: "Retail gas distribution",            
        23641: "Retail distribution of liquid or solid fossil fuels",            
        23642: "Electric mobility infrastructures",            
        24010: "Financial policy and administrative management",            
        24020: "Monetary institutions",            
        24030: "Formal sector financial intermediaries",            
        24040: "Informal/semi-formal financial intermediaries",            
        24050: "Remittance facilitation, promotion and optimisation",            
        24081: "Education/training in banking and financial services",            
        25010: "Business policy and administration",            
        25020: "Privatisation",            
        25030: "Business development services",            
        25040: "Responsible business conduct",            
        31110: "Agricultural policy and administrative management",            
        31120: "Agricultural development",            
        31130: "Agricultural land resources",            
        31140: "Agricultural water resources",            
        31150: "Agricultural inputs",            
        31161: "Food crop production",            
        31162: "Industrial crops/export crops",            
        31163: "Livestock",            
        31164: "Agrarian reform",            
        31165: "Agricultural alternative development",            
        31166: "Agricultural extension",            
        31181: "Agricultural education/training",            
        31182: "Agricultural research",            
        31191: "Agricultural services",            
        31192: "Plant and post-harvest protection and pest control",            
        31193: "Agricultural financial services",            
        31194: "Agricultural co-operatives",            
        31195: "Livestock/veterinary services",            
        31210: "Forestry policy and administrative management",            
        31220: "Forestry development",            
        31261: "Fuelwood/charcoal",            
        31281: "Forestry education/training",            
        31282: "Forestry research",            
        31291: "Forestry services",            
        31310: "Fishing policy and administrative management",            
        31320: "Fishery development",            
        31381: "Fishery education/training",            
        31382: "Fishery research",            
        31391: "Fishery services",            
        32110: "Industrial policy and administrative management",            
        32120: "Industrial development",            
        32130: "Small and medium-sized enterprises (SME) development",            
        32140: "Cottage industries and handicraft",            
        32161: "Agro-industries",            
        32162: "Forest industries",            
        32163: "Textiles, leather and substitutes",            
        32164: "Chemicals",            
        32165: "Fertilizer plants",            
        32166: "Cement/lime/plaster",            
        32167: "Energy manufacturing (fossil fuels)",            
        32168: "Pharmaceutical production",            
        32169: "Basic metal industries",            
        32170: "Non-ferrous metal industries",            
        32171: "Engineering",            
        32172: "Transport equipment industry",            
        32173: "Modern biofuels manufacturing",            
        32174: "Clean cooking appliances manufacturing",            
        32182: "Technological research and development",            
        32210: "Mineral/mining policy and administrative management",            
        32220: "Mineral prospection and exploration",            
        32261: "Coal",            
        32262: "Oil and gas (upstream)",            
        32263: "Ferrous metals",            
        32264: "Nonferrous metals",            
        32265: "Precious metals/materials",            
        32266: "Industrial minerals",            
        32267: "Fertilizer minerals",            
        32268: "Offshore minerals",            
        32310: "Construction policy and administrative management",            
        33110: "Trade policy and administrative management",            
        33120: "Trade facilitation",            
        33130: "Regional trade agreements (RTAs)",            
        33140: "Multilateral trade negotiations",            
        33150: "Trade-related adjustment",            
        33181: "Trade education/training",            
        33210: "Tourism policy and administrative management",            
        41010: "Environmental policy and administrative management",            
        41020: "Biosphere protection",            
        41030: "Biodiversity",            
        41040: "Site preservation",            
        41050: "Flood prevention/control",            
        41081: "Environmental education/training",            
        41082: "Environmental research",            
        43010: "Multisector aid",            
        43030: "Urban development and management",            
        43031: "Urban land policy and management",            
        43032: "Urban development",            
        43040: "Rural development",            
        43041: "Rural land policy and management",            
        43042: "Rural development",            
        43050: "Non-agricultural alternative development",            
        43060: "Disaster Risk Reduction",            
        43071: "Food security policy and administrative management",            
        43072: "Household food security programmes",            
        43073: "Food safety and quality",            
        43081: "Multisector education/training",            
        43082: "Research/scientific institutions",            
        51010: "General budget support-related aid",            
        52010: "Food assistance",            
        53030: "Import support (capital goods)",            
        53040: "Import support (commodities)",            
        60010: "Action relating to debt",            
        60020: "Debt forgiveness",            
        60030: "Relief of multilateral debt",            
        60040: "Rescheduling and refinancing",            
        60061: "Debt for development swap",            
        60062: "Other debt swap",            
        60063: "Debt buy-back",            
        72010: "Material relief assistance and services",            
        72011: "Basic Health Care Services in Emergencies",            
        72012: "Education in emergencies",            
        72040: "Emergency food assistance",            
        72050: "Relief co-ordination and support services",            
        73010: "Immediate post-emergency reconstruction and rehabilitation",            
        74010: "Disaster prevention and preparedness",            
        74020: "Multi-hazard response preparedness",            
        91010: "Administrative costs (non-sector allocable)",            
        92010: "Support to national NGOs",            
        92020: "Support to international NGOs",            
        92030: "Support to local and regional NGOs",            
        93010: "Refugees/asylum seekers  in donor countries (non-sector allocable)",            
        93011: "Refugees/asylum seekers in donor countries - food and shelter",            
        93012: "Refugees/asylum seekers in donor countries - training",            
        93013: "Refugees/asylum seekers in donor countries - health",            
        93014: "Refugees/asylum seekers in donor countries - other temporary sustenance",            
        93015: "Refugees/asylum seekers in donor countries - voluntary repatriation",            
        93016: "Refugees/asylum seekers in donor countries - transport",            
        93017: "Refugees/asylum seekers in donor countries - rescue at sea",            
        93018: "Refugees/asylum seekers in donor countries - administrative costs",            
        99810: "Sectors not specified",            
        99820: "Promotion of development awareness (non-sector allocable)",
    }

    for code in sorted(GOOD_CODES, key=int):
        if len(code) == 5 and int(code) in DAC_SECTORS:
            print(rf"\item {code}: {DAC_SECTORS[int(code)]}")

    # --- Among ranked activity_ids, how many hit any of your TARGET_CODES? ---
    activities_with_target = {
        aid for aid, codes in activity_to_codes.items()
        if codes & TARGET_CODES
    }
    n_target = len(activities_with_target)

    pct_target_of_ranked = (n_target / total_ranked * 100) if total_ranked else 0.0
    pct_target_of_with_dac = (n_target / with_dac_info * 100) if with_dac_info else 0.0

    print(f"Activities with at least one TARGET DAC5 code (GOOD or BAD): {n_target}")
    print(f"  -> {pct_target_of_ranked:.1f}% of all merged activity_ids")
    print(f"  -> {pct_target_of_with_dac:.1f}% of merged activity_ids that have DAC info")
    print()

    # --- Good vs Bad breakdown; "only bad" = lost if you drop bad codes ---
    

    any_good, only_bad = get_any_good_and_only_bad(activities_with_target,activity_to_codes, GOOD_CODES,BAD_CODES)

    def pct(num, denom):
        return (num / denom * 100) if denom else 0.0

    n_any_good = len(any_good)
    n_only_bad = len(only_bad)

    print("Good vs bad coverage among merged activity_ids:")
    print(f"  Activities with at least one GOOD code: {n_any_good}")
    print(f"    -> {pct(n_any_good, total_ranked):.1f}% of all ids")
    print(f"    -> {pct(n_any_good, n_target):.1f}% of those with any TARGET code")

    print(f"  Activities whose only TARGET codes are BAD codes (would be lost if BAD dropped): {n_only_bad}")
    print(f"    -> {pct(n_only_bad, total_ranked):.1f}% of all ids")
    print(f"    -> {pct(n_only_bad, n_target):.1f}% of those with any TARGET code")
    print()

    # --- Counts of activity_ids per DAC5 code (descending) ---
    code_to_activities = defaultdict(set)
    for aid, codes in activity_to_codes.items():
        for code in codes:
            code_to_activities[code].add(aid)

    rows = []
    for code, aids in code_to_activities.items():
        flag = ""
        if code in GOOD_CODES:
            flag = "GOOD"
        elif code in BAD_CODES:
            flag = "BAD"
        rows.append((code, len(aids), flag))

    rows.sort(key=lambda x: x[1], reverse=True)

    print("Activity_id counts per DAC5 code (descending):")
    print("code,count,flag")
    for code, count, flag in rows:
        print(f"{code},{count},{flag}")
    print()

    # --- Focus on GOOD codes only (restricted to ranked) ---
    # Build per-activity GOOD code sets (including activities with zero GOOD codes)
    good_codes_per_activity = {}
    for aid in ranked_ids:
        codes = activity_to_codes.get(aid, set())
        good = codes & GOOD_CODES
        good_codes_per_activity[aid] = good

    total_good_codes = sum(len(c) for c in good_codes_per_activity.values())
    n_with_any_good = sum(1 for c in good_codes_per_activity.values() if c)

    avg_good_per_activity_all = (total_good_codes / total_ranked) if total_ranked else 0.0
    avg_good_per_activity_cond = (total_good_codes / n_with_any_good) if n_with_any_good else 0.0

    print("GOOD DAC code usage among merged ratings jsonl activities:")
    print(f"  Total GOOD DAC code assignments (activity,code pairs): {total_good_codes}")
    print(f"  Activities with at least one GOOD code: {n_with_any_good}")
    print(f"  Average GOOD codes per activity (all ranked): {avg_good_per_activity_all:.2f}")
    print(f"  Average GOOD codes per activity (only those with >=1 GOOD): {avg_good_per_activity_cond:.2f}")
    print()

    # --- Top 3 overlaps (double-codes): most common GOOD code pairs per activity ---
    pair_counter = Counter()
    for codes in good_codes_per_activity.values():
        if len(codes) < 2:
            continue
        for c1, c2 in combinations(sorted(codes), 2):
            pair_counter[(c1, c2)] += 1

    print("Top 3 overlapping GOOD DAC code pairs (double-codes):")
    for (c1, c2), count in pair_counter.most_common(3):
        print(f"  {c1} & {c2}: {count} activities")
    print()

    # --- Category counts for GOOD codes (for pie chart) ---
    category_counts = Counter()
    for codes in good_codes_per_activity.values():
        for code in codes:
            cat = categorize_good_code(code)
            category_counts[cat] += 1

    # Drop empty categories just in case
    category_counts = Counter({k: v for k, v in category_counts.items() if v > 0})

    print("GOOD DAC code category counts (merged jsonl activities only):")
    total_cat = sum(category_counts.values())
    for cat, cnt in category_counts.most_common():
        share = (cnt / total_cat * 100) if total_cat else 0.0
        print(f"  {cat}: {cnt} ({share:.1f}%)")
    print()

    # --- Pie chart of GOOD DAC categories ---
    if total_cat > 0:
        labels = list(category_counts.keys())
        sizes = list(category_counts.values())

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )
        plt.title("Distribution of Activity Topic Codes")
        plt.tight_layout()
        plt.show()
    else:
        print("No GOOD DAC codes found for ranked activities; skipping pie chart.")

# def plot_good_dac_category_pies_by_split(
#     split_to_ids: dict[str, set[str] | list[str] | pd.Index],
#     *,
#     info_csv: str = INFO_CSV,
#     title_prefix: str = "GOOD DAC categories",
# ):
#     info = pd.read_csv(info_csv, usecols=["activity_id", "dac5"])
#     info["activity_id"] = info["activity_id"].astype(str)

#     activity_to_codes: dict[str, set[str]] = {}
#     for _, row in info.iterrows():
#         aid = row["activity_id"]
#         codes = parse_dac_codes(row.get("dac5", ""))
#         if codes:
#             activity_to_codes.setdefault(aid, set()).update(codes)

#     GOOD_CODES, _, _ = get_good_bad_and_target_codes()

#     def _counts_for_ids(aids) -> Counter:
#         c = Counter()
#         for aid in aids:
#             codes = activity_to_codes.get(str(aid), set()) & GOOD_CODES
#             for code in codes:
#                 c[categorize_good_code(code)] += 1
#         return Counter({k: v for k, v in c.items() if v > 0})

#     # --- compute all split counts first (so we can lock category->color globally) ---
#     counts_by_split: dict[str, Counter] = {}
#     all_cats = set()
#     for split_name, ids in split_to_ids.items():
#         cnt = _counts_for_ids(ids)
#         counts_by_split[split_name] = cnt
#         all_cats |= set(cnt.keys())

#     if not all_cats:
#         return

#     # Stable category order + stable colors
#     cats = sorted(all_cats)  # consistent across splits/runs
#     cmap = plt.get_cmap("tab20")
#     color_map = {cat: cmap(i % cmap.N) for i, cat in enumerate(cats)}

#     for split_name, cnt in counts_by_split.items():
#         if not cnt:
#             continue

#         labels = [cat for cat in cats if cat in cnt]
#         sizes = [cnt[cat] for cat in labels]
#         colors = [color_map[cat] for cat in labels]

#         plt.figure(figsize=(6, 6))
#         plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
#         plt.title(f"{title_prefix}: {split_name}")
#         plt.tight_layout()
#         plt.show(block=False)

if __name__ == "__main__":
    main()