from collections import Counter
import re
import unicodedata
import pprint
from typing import Dict, Any, Iterable, List, Optional, Set
from pathlib import Path
import json
import glob

import numpy as np
import pandas as pd


ACTIVITY_SCOPES = {
    "global": 7,
    "regional": 6,
    "multi-national": 5,
    "national": 4,
    "sub-national: multi-first-level administrative areas": 3,
    "sub-national: single first-level administrative area": 2,
    "sub-national: single second-level administrative area": 1,
    "single location": 0,
}

VERBOSE = False

RATING_MAP = {
    'Highly Unsatisfactory': 0,
    'Unsatisfactory': 1,
    'Moderately Unsatisfactory': 2,
    'Moderately Satisfactory': 3,
    'Satisfactory': 4,
    'Highly Satisfactory': 5
}
RATING_MAP_LOWER = {k.lower(): v for k, v in RATING_MAP.items()}

RATING_MAP_INVERSE = {v: k for k, v in RATING_MAP.items()}

def get_success_measure_from_rating_value(rating_value, min_rating=None, max_rating=None, activity_id=None):
    RATING_MAP_LOCAL = {
        'highly unsatisfactory': 0,
        'unsatisfactory': 1,
        'moderately unsatisfactory': 2,
        'moderately satisfactory': 3,
        'satisfactory': 4,
        'highly satisfactory': 5,
    }

    SIMPLE_THREE_GRADES = {
        "high": 4,
        "medium": 2.5,
        "low": 1,
    }

    SUBSTANTIAL_GRADES = {
        "high": 4,
        "substantial": 3,
        "modest": 2,
        "negligible": 1,
    }

    SIMPLE_SUCCESS = {
        "highly successful": 5,
        "successful": 4,
        "unsuccessful": 1,
    }

    SIMPLE_EXPECTATIONS = {
        "exceeded expectations": 5,
        "met expectations": 3.5,
        "did not meet expectations": 1,
    }

    v = rating_value.lower()

    numeric_rating = RATING_MAP_LOCAL.get(v)
    if numeric_rating is not None:
        return numeric_rating

    simple_three = SIMPLE_THREE_GRADES.get(v)
    if simple_three is not None:
        return simple_three

    substantial_grades = SUBSTANTIAL_GRADES.get(v)
    if substantial_grades is not None:
        return substantial_grades

    simple_success = SIMPLE_SUCCESS.get(v)
    if simple_success is not None:
        return simple_success

    simple_expectations = SIMPLE_EXPECTATIONS.get(v)
    if simple_expectations is not None:
        return simple_expectations


    if "moderately unsatisfactory" in v:
        return 2
    elif "moderately satisfactory" in v:
        return 3
    elif 'highly unsatisfactory' in v:
        return 0
    elif 'highly satisfactory' in v:
        return 5
    elif 'unsatisfactory' in v:
        return 1
    elif 'satisfactory' in v:
        return 4

    if "excellent performance" in v:
        return 5
    if "poor performance" in v:
        return 1
    if "low performance" in v:
        return 1

    return None


_CANON_LABEL_TO_0_5 = {
    "highly unsatisfactory": 0,
    "unsatisfactory": 1,
    "moderately unsatisfactory": 2,
    "moderately satisfactory": 3,
    "satisfactory": 4,
    "highly satisfactory": 5,
}

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _norm_text(x) -> str:
    s = "" if x is None else str(x)
    s = _strip_accents(s).lower().strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\([^)]*\)", "", s).strip()
    s = s.strip(" .,:;*_\"'`")
    return s
_ALIAS_TO_CANON = {
    "hs": "highly satisfactory",
    "s": "satisfactory",
    "ms": "moderately satisfactory",
    "mu": "moderately unsatisfactory",
    "u": "unsatisfactory",
    "hu": "highly unsatisfactory",

    "highly sat": "highly satisfactory",
    "highly satisf.": "highly satisfactory",
    "mod sat": "moderately satisfactory",
    "mod satisf.": "moderately satisfactory",
    "mod unsat": "moderately unsatisfactory",
    "mod unsatisf.": "moderately unsatisfactory",
    "unsat": "unsatisfactory",
    "unsatisf.": "unsatisfactory",

    "tres satisfaisant": "highly satisfactory",
    "tres satisfaisante": "highly satisfactory",
    "tres insatisfaisant": "highly unsatisfactory",
    "tres insatisfaisante": "highly unsatisfactory",

    "satisfaisant": "satisfactory",
    "satisfaisante": "satisfactory",
    "moyennement satisfaisant": "moderately satisfactory",
    "moyennement satisfaisante": "moderately satisfactory",
    "partiellement satisfaisant": "moderately satisfactory",
    "partiellement satisfaisante": "moderately satisfactory",

    "insatisfaisant": "unsatisfactory",
    "insatisfaisante": "unsatisfactory",
    "plutot insatisfaisant": "moderately unsatisfactory",
    "plutot insatisfaisante": "moderately unsatisfactory",

    "excellent": "highly satisfactory",
    "tres bon": "highly satisfactory",
    "tres bonne": "highly satisfactory",
    "bon": "satisfactory",
    "bonne": "satisfactory",
    "assez bon": "moderately satisfactory",
    "assez bonne": "moderately satisfactory",
    "moyen": "moderately satisfactory",
    "moyenne": "moderately satisfactory",
    "faible": "unsatisfactory",
    "tres faible": "highly unsatisfactory",

    "reussi": "satisfactory",
    "reussie": "satisfactory",
    "non reussi": "unsatisfactory",
    "non reussie": "unsatisfactory",
    "atteint": "satisfactory",
    "atteinte": "satisfactory",
    "non atteint": "unsatisfactory",
    "non atteinte": "unsatisfactory",

    "altamente satisfactoria": "highly satisfactory",
    "altamente satisfactorio": "highly satisfactory",
    "muy satisfactorio": "highly satisfactory",
    "muy satisfactoria": "highly satisfactory",
    "satisfactorio": "satisfactory",
    "satisfactoria": "satisfactory",
    "moderadamente satisfactorio": "moderately satisfactory",
    "moderadamente satisfactoria": "moderately satisfactory",
    "parcialmente satisfactorio": "moderately satisfactory",
    "parcialmente satisfactoria": "moderately satisfactory",

    "insatisfactorio": "unsatisfactory",
    "insatisfactoria": "unsatisfactory",
    "moderadamente insatisfactorio": "moderately unsatisfactory",
    "moderadamente insatisfactoria": "moderately unsatisfactory",
    "altamente insatisfactorio": "highly unsatisfactory",
    "altamente insatisfactoria": "highly unsatisfactory",

    "insuficiente": "unsatisfactory",
    "deficiente": "unsatisfactory",
    "muy deficiente": "highly unsatisfactory",

    "excelente": "highly satisfactory",
    "muy bueno": "highly satisfactory",
    "muy buena": "highly satisfactory",
    "bueno": "satisfactory",
    "buena": "satisfactory",
    "regular": "moderately satisfactory",
    "aceptable": "moderately satisfactory",
    "malo": "unsatisfactory",
    "mala": "unsatisfactory",
    "muy malo": "highly unsatisfactory",
    "muy mala": "highly unsatisfactory",

    "logrado": "satisfactory",
    "lograda": "satisfactory",
    "no logrado": "unsatisfactory",
    "no lograda": "unsatisfactory",
    "alcanzado": "satisfactory",
    "alcanzada": "satisfactory",
    "no alcanzado": "unsatisfactory",
    "no alcanzada": "unsatisfactory",

    "muito satisfatorio": "highly satisfactory",
    "muito satisfatoria": "highly satisfactory",
    "altamente satisfatorio": "highly satisfactory",
    "altamente satisfatoria": "highly satisfactory",
    "satisfatorio": "satisfactory",
    "satisfatoria": "satisfactory",
    "moderadamente satisfatorio": "moderately satisfactory",
    "moderadamente satisfatoria": "moderately satisfactory",

    "insatisfatorio": "unsatisfactory",
    "insatisfatoria": "unsatisfactory",
    "moderadamente insatisfatorio": "moderately unsatisfactory",
    "moderadamente insatisfatoria": "moderately unsatisfactory",
    "muito insatisfatorio": "highly unsatisfactory",
    "muito insatisfatoria": "highly unsatisfactory",

    "excelente": "highly satisfactory",
    "muito bom": "highly satisfactory",
    "muito boa": "highly satisfactory",
    "bom": "satisfactory",
    "boa": "satisfactory",
    "razoavel": "moderately satisfactory",
    "regular": "moderately satisfactory",
    "fraco": "unsatisfactory",
    "fraca": "unsatisfactory",
    "muito fraco": "highly unsatisfactory",
    "muito fraca": "highly unsatisfactory",

    "atingido": "satisfactory",
    "atingida": "satisfactory",
    "nao atingido": "unsatisfactory",
    "nao atingida": "unsatisfactory",
    "alcancado": "satisfactory",
    "alcancada": "satisfactory",
    "nao alcancado": "unsatisfactory",
    "nao alcancada": "unsatisfactory",

    "sehr gut": "highly satisfactory",
    "ausgezeichnet": "highly satisfactory",
    "exzellent": "highly satisfactory",
    "hervorragend": "highly satisfactory",
    "gut": "satisfactory",
    "zufriedenstellend": "satisfactory",
    "eher gut": "moderately satisfactory",
    "teilweise zufriedenstellend": "moderately satisfactory",
    "mittel": "moderately satisfactory",
    "durchschnittlich": "moderately satisfactory",
    "akzeptabel": "moderately satisfactory",

    "unzureichend": "unsatisfactory",
    "schwach": "unsatisfactory",
    "mangelhaft": "unsatisfactory",
    "schlecht": "unsatisfactory",
    "sehr schlecht": "highly unsatisfactory",
    "ungenugend": "highly unsatisfactory",

    "erreicht": "satisfactory",
    "nicht erreicht": "unsatisfactory",
    "ziel erreicht": "satisfactory",
    "ziel nicht erreicht": "unsatisfactory",
    "stufe 2: erfolgreich": "satisfactory",
    "level 2 successful": "satisfactory",
    "sehr erfolgreich": "highly satisfactory",
    "erfolgreich": "satisfactory",
    "nicht erfolgreich": "unsatisfactory",

    "significantly exceeded expectations": "highly satisfactory",
    "substantially exceeded expectations": "highly satisfactory",
    "far exceeded expectations": "highly satisfactory",
    "exceeded expectations": "highly satisfactory",
    "exceeding expectations": "highly satisfactory",

    "meeting expectations": "satisfactory",
    "met expectations": "satisfactory",

    "below expectations": "moderately unsatisfactory",
    "did not meet expectations": "unsatisfactory",
    "not meeting expectations": "unsatisfactory",
    "failed to meet expectations": "unsatisfactory",

    "significantly exceeded objective": "highly satisfactory",
    "exceeded objective": "highly satisfactory",
    "steadily advancing, achieving remarkable results": "highly satisfactory",
    "achieved objective": "satisfactory",
    "did not achieve objective": "unsatisfactory",

    "green": "satisfactory",
    "amber": "moderately satisfactory",
    "yellow": "moderately satisfactory",
    "red": "unsatisfactory",

    "on track": "satisfactory",
    "fully implemented": "satisfactory",
    "on-track": "satisfactory",
    "on course": "satisfactory",
    "ahead of track": "highly satisfactory",
    "off track": "unsatisfactory",
    "off-track": "unsatisfactory",
    "delayed": "moderately unsatisfactory",
    "behind schedule": "moderately unsatisfactory",

    "achieved": "satisfactory",
    "not achieved": "unsatisfactory",
    "positive": "satisfactory",
    "negative": "unsatisfactory",
    "overall successful": "satisfactory",
    "overall unsuccessful": "moderately unsatisfactory",

    "mixed performance": "moderately satisfactory",
    "some improvements": "moderately satisfactory",
    "moderately successful": "moderately satisfactory",
    "moderate": "moderately satisfactory",
    "partially successful": "moderately satisfactory",
    "partly successful": "moderately satisfactory",
    "relatively satisfying": "moderately satisfactory",
    "successfully": "moderately satisfactory",

    "needs improvement": "moderately unsatisfactory",
    "less than successful": "moderately unsatisfactory",
    "less than effective": "moderately unsatisfactory",
    "limited progress": "moderately unsatisfactory",
    "insufficient progress": "unsatisfactory",

    "nearly all indicators are satisfactorily achieved": "satisfactory",
    "very effectively though there is surely room for improvements": "satisfactory",
    "well": "satisfactory",
    "performed well": "satisfactory",
    "good": "satisfactory",
    "fair": "moderately satisfactory",

    "very effectively": "highly satisfactory",
    "excellent performance": "highly satisfactory",
    "excellent position": "highly satisfactory",
    "excellently": "highly satisfactory",
    "extremely well": "highly satisfactory",
    "very positive": "highly satisfactory",
    "extraordinary benefits": "highly satisfactory",
    "strong, positive impact": "highly satisfactory",

    "moving ahead and on track to achieving its objectives": "satisfactory",
    "a success": "satisfactory",
    "successful": "satisfactory",
    "highly successful": "highly satisfactory",
    "unsuccessful": "unsatisfactory",
    "highly unsuccessful": "highly unsatisfactory",

    "very successful": "highly satisfactory",
    "moderately unsuccessful": "moderately unsatisfactory",
    "very unsuccessful": "highly unsatisfactory",

    "mostly unsuccessful": "unsatisfactory",
    "disappointing": "unsatisfactory",
    "poor": "unsatisfactory",
    "poorly": "unsatisfactory",
    "weak": "unsatisfactory",
    "very weak": "highly unsatisfactory",

    "strong": "satisfactory",
    "substantial progress": "satisfactory",
    "very high level": "highly satisfactory",
    "largely exceeded expectations, representing very impressive value for money": "highly satisfactory",
    "a - meets expectations": "satisfactory",
    "a – meets expectations": "satisfactory",
    "resultado satisfactorio": "satisfactory",

}

_DFID_GRADE_TO_0_5 = {
    "a++": 5.0,  # substantially exceeded expectations
    "a+":  4.5,  # exceeded expectations
    "a":   4.0,  # met expectations
    "b":   1.0,  # did not meet expectations (your rule)
    "c":   0.0,  # substantially did not meet
}



def _extract_numbers(x) -> list[float]:
    s = "" if x is None else str(x)
    s = s.replace(",", ".")
    return [float(m) for m in re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?", s)]

def _coerce_num(x) -> Optional[float]:
    ns = _extract_numbers(x)
    return ns[0] if ns else None

def _parse_percent(x) -> Optional[float]:
    s = "" if x is None else str(x)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", s)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))

def _is_dfid_grade_family(v: str, mn: str, mx: str, activity_id=None) -> bool:
    if activity_id is None:
        return False
    aid = str(activity_id).upper()
    g = (v.split() or [""])[0]
    return aid.startswith("GB-") and g in _DFID_GRADE_TO_0_5  # catches GB-GOV-*, etc.


_ALIAS_TO_CANON.update({
    "inadequate": "unsatisfactory",
    "acceptable": "moderately satisfactory",
    "impressive results": "satisfactory",
    "satisfied with most": "moderately satisfactory",
    "good overall progress": "moderately satisfactory",
    "good value for money": "satisfactory",
    "erfullt": "satisfactory",
    "rather successful": "moderately satisfactory",
    "eher erfolgreich": "moderately satisfactory",
    "parcialmente exitoso": "moderately satisfactory",
    "altamente exitoso": "highly satisfactory",
    "tres bon resultat": "highly satisfactory",
    "parcialmente insatisfactorio": "moderately unsatisfactory",
    "parcialmente insatisfactoria": "moderately unsatisfactory",
})

def get_from_number(rating_value, max_rating, min_rating, activity_id):
    s = "" if rating_value is None else str(rating_value).strip()
    m = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*/\s*([0-9]+(?:[.,][0-9]+)?)\s*$", s)
    if m:
        num = float(m.group(1).replace(",", "."))
        den = float(m.group(2).replace(",", "."))
        if den != 0 and (0 <= num <= den):
            return 5 * (num / den)

    if _is_number(rating_value) and _is_number(max_rating) and _is_number(min_rating):
        lo = float(min_rating)
        hi = float(max_rating)
        v  = float(rating_value)

        if hi == lo:
            return None
        if hi < lo:
            lo, hi = hi, lo

        if not (lo <= v <= hi):
            return None

        score = 5 * (v - lo) / (hi - lo)   # default: higher = better

        invert = (
            activity_id is not None
            and str(activity_id).startswith("DE-1")
            and lo == 1.0 and hi == 6.0 # assume it's always inverted regardless of lo/high if de-1
        )
        if invert:
            score = 5 - score              # DE-1*: lower = better (1 best, 6 worst)

        return score

def get_success_measure_from_rating_value_wrapped(rating_value, min_rating=None, max_rating=None, activity_id=None):

    v = _norm_text(rating_value)
    mn = _norm_text(min_rating)
    mx = _norm_text(max_rating)
    if activity_id.startswith("DE-1"):
        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s+von\s+16\s+punkten", v)
        if m:
            x = float(m.group(1).replace(",", "."))
            if 0 <= x <= 16:
                return 5 * (x / 16)

        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s+out\s+of\s+16\s+points", v)
        if m:
            x = float(m.group(1).replace(",", "."))
            if 0 <= x <= 16:
                return 5 * (x / 16)

        m = re.match(r"level\s+.\s*:\s*(.+?)\s*$", v, flags=re.IGNORECASE)
        if m:
            v = m.group(1).strip()

        m = re.match(r"nivel\s+.\s*:\s*(.+?)\s*$", v, flags=re.IGNORECASE)
        if m:
            v = m.group(1).strip()

        v = v.split(" als ")[-1].strip()

        lo = _coerce_num(min_rating)
        hi = _coerce_num(max_rating)
        nums = _extract_numbers(v)

    pct = _parse_percent(v)
    if pct is not None:
        judged_percent_multiplied_by_5 = 5.0 * max(0.0, min(100.0, pct)) / 100.0
        if VERBOSE:
            print(f"Rating as number: {judged_percent_multiplied_by_5} from {rating_value}, min: {mn}, max: {mx}, act: {activity_id}")

        return judged_percent_multiplied_by_5

    v = v.replace("a-plus", "a+").replace("a plus", "a+").replace("a–", "a-").replace("a -", "a-")
    v = v.replace("b-plus", "b+").replace("b plus", "b+").replace("b–", "b-").replace("b -", "b-")
    v = v.replace("c-plus", "c+").replace("c plus", "c+").replace("c–", "c-").replace("c -", "c-")
    v = v.replace("d-plus", "d+").replace("d plus", "d+").replace("d–", "d-").replace("d -", "d-")
    v = v.replace("f-plus", "f+").replace("f plus", "f+").replace("f–", "f-").replace("f -", "f-")
    if v.startswith("a") and "met expectation" in v:
        v = "a"
    if v.startswith("a+") and "exceed" in v:
        v = "a+"

    if _is_dfid_grade_family(v, mn, mx, activity_id=activity_id):
        g = v.split()[0]  # handles e.g. "a - meets expectations" -> "a"
        if g in _DFID_GRADE_TO_0_5:
            rating = _DFID_GRADE_TO_0_5[g]
            return rating

    raw = "" if rating_value is None else str(rating_value)

    cand_raw = raw
    cand_no_parens = re.sub(r"\([^)]*\)", " ", raw).strip()
    m = re.search(r"\(([^()]*)\)", raw)
    cand_in_parens = m.group(1).strip() if m else ""

    parts = re.split(r"\s+als\s+|\s+=\s+|\s*:\s*|\s*-\s*|\s*–\s*|\s*—\s*", raw, maxsplit=1)
    cand_before_split = parts[0].strip() if len(parts) > 1 else ""
    cand_after_split  = parts[1].strip() if len(parts) > 1 else ""

    for cand in (cand_raw, cand_no_parens, cand_in_parens, cand_before_split, cand_after_split):
        if not cand:
            continue
        attempt = _norm_text(cand)


        if activity_id.startswith("XI-IATI-IA"):
            # "EVALUABLE (SCORE 6.8)" -> 0..10 mapped to 0..5
            m = re.search(r"score\s*([0-9]+(?:[.,][0-9]+)?)", str(rating_value), flags=re.IGNORECASE)
            if m:
                sc = float(m.group(1).replace(",", "."))
                return 5.0 * max(0.0, min(10.0, sc)) / 10.0

            # "0.81 - SATISFACTORIO" -> prob 0..1 mapped to 0..5
            m = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*[-:]", str(rating_value))
            if m:
                p = float(m.group(1).replace(",", "."))
                if 0.0 <= p <= 1.0:
                    return 5.0 * p

            IADB_RATINGS = {
                "very probable": 5.0,
                "muy probable": 5.0,
                "probable": 5.0 * 2/3,
                "provavel": 5.0 * 2/3,      # "Provável (P)" -> _norm_text -> "provavel"
                "low probability": 5.0 * 1/3,
                "poco probable": 5.0 * 1/3, # "Poco Probable (PP)" -> _norm_text -> "poco probable"
                "improbable": 0.0,
            }


            result = IADB_RATINGS.get(attempt)
            if result is not None:
                return result

        canon = _ALIAS_TO_CANON.get(attempt)
        if canon is not None:
            rating = float(_CANON_LABEL_TO_0_5[canon])
            return rating


    for cand in (cand_raw, cand_no_parens, cand_in_parens, cand_before_split, cand_after_split):
        if not cand:
            continue
        attempt = _norm_text(cand)
        v0 = get_success_measure_from_rating_value(attempt, min_rating, max_rating, activity_id=activity_id)
        if v0 is not None:
            return v0

        if "highly successful" in attempt:
            return 5
        if "highly unsuccessful" in attempt:
            return 0
        if "moderately successful" in attempt:
            return 4
        if "moderately unsuccessful" in attempt:
            return 2


        if attempt.startswith("successfully"):
            return 4

        if attempt.startswith("satisfactorily"):
            return 4

        if attempt.endswith("very successful") and not "not" in attempt:
            return 5

    for cand in (cand_raw, cand_no_parens, cand_in_parens, cand_before_split, cand_after_split):        
        if not cand:
            continue
        attempt = _norm_text(cand)
        
        number_parsed = get_from_number(attempt,mn,mx,activity_id)
        if number_parsed is not None:
            if VERBOSE:
                print(f"Rating as number: {number_parsed} from {rating_value}, min: {mn}, max: {mx}, act: {activity_id}")
            return number_parsed

    if v.endswith("; successful"):
        return 4
    if v.endswith(", successful"):
        return 4

    return None

def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False

def pick_start_date(row: pd.Series):
    for c in ["actual_start_date","original_planned_start_date","txn_first_date"]:
        if c in row and pd.notna(row[c]):
            return row[c]
    return pd.NaT

def get_text_to_describe_rating_distribution(aid,ratings,rating_stats, num_options):
    aid = str(aid)

    stats = None
    if isinstance(rating_stats, dict) and "overall" in rating_stats:
        by_prefix = rating_stats.get("by_prefix") or {}

        if aid.startswith("44000-") and by_prefix.get("44000-") is not None:
            stats = by_prefix["44000-"]
        elif aid.startswith("DE-1") and by_prefix.get("DE-1") is not None:
            stats = by_prefix["DE-1"]
        else:
            stats = rating_stats["overall"]
    else:
        print("WARNING: USING BACKWARD COMPATIBLE INDEPENDENT OF ACTIVITY")
        stats = rating_stats

    prompt_lines = []

    use_six = isinstance(num_options, int) and num_options == 6
    use_thirds = False
    use_quartiles = False
    if not use_six:
        if isinstance(num_options, int) and num_options >= 3 and num_options % 2 == 1:
            use_thirds = True
        else:
            use_quartiles = True

    prompt_lines = []
    if stats:
        if use_six and stats.get("six_percents"):
            sp = stats["six_percents"]
            p_worst = sp.get(1, 0.0)
            p_2     = sp.get(2, 0.0)
            p_3     = sp.get(3, 0.0)
            p_4     = sp.get(4, 0.0)
            p_5     = sp.get(5, 0.0)
            p_best  = sp.get(6, 0.0)
            prompt_lines.append(
                "In the historical data, the overall outcomes are "
                "distributed as: \n"
                f"  Highly Satisfactory: {int(p_best)}%\n"
                f"  Satisfactory: {int(p_5)}%\n"
                f"  Moderately Satisfactory: {int(p_4)}%\n"
                f"  Moderately Unsatisfactory: {int(p_3)}%\n"
                f"  Unsatisfactory: {int(p_2)}%\n"
                f"  Highly Unsatisfactory: {int(p_worst)}%\n"
            )
            prompt_lines.append("")
        else:
            input("I have paused execution. there is an unexpected distribution being used. check helpers_for_ratings_and_final_activity_features.py")
    else:
        input("I have paused execution. there is an unexpected distribution being used. check helpers_for_ratings_and_final_activity_features.py")
    return prompt_lines

def get_ratings_text(final_result, rating_min=None, rating_max=None, activity_id=None):
    success_measure = get_success_measure_from_rating_value_wrapped(final_result,min_rating=rating_min,max_rating=rating_max,activity_id=activity_id)
    if success_measure is None:
        return None, None, None, None, None

    integer_rating = int(round(success_measure))

    idx_in_original = 5 - integer_rating
    scale_options = [
        "Highly Satisfactory",
        "Satisfactory",
        "Moderately Satisfactory",
        "Moderately Unsatisfactory",
        "Unsatisfactory",
        "Highly Unsatisfactory",
    ]  # best -> worst
    v_norm = scale_options[idx_in_original]

    worst_first = False

    if worst_first:
        worst_to_best = list(scale_options)
    else:
        worst_to_best = list(reversed(scale_options))

    num_options = len(worst_to_best)
    final_result_for_prompt = scale_options[idx_in_original]

    n = num_options
    mid_low_idx = (n - 1) // 2
    if n % 2 == 0:
        mid_high_idx = mid_low_idx + 1
        even_text_low = even_text_high = ""
    else:
        even_text_low = "on the low side of "
        even_text_high = "on the high side of "
        mid_high_idx = mid_low_idx


    midpoint_low_val = worst_to_best[mid_low_idx]
    midpoint_high_val = worst_to_best[mid_high_idx]


    midpoint_low_text = f"{even_text_low}{midpoint_low_val} or lower"
    midpoint_high_text = f"{even_text_high}{midpoint_high_val} or higher"

    options_text = ", ".join(f"'{o}'" for o in scale_options)
    return num_options, midpoint_low_text, midpoint_high_text, options_text, final_result_for_prompt


def _parse_options_list(options_text: str) -> List[str]:
    """
    Parse the options_text returned by get_ratings_text, e.g.:
        "'1', '2', '3'" -> ["1", "2", "3"]
        "'Highly Satisfactory', 'Satisfactory', ..."
    """
    if not options_text:
        return []
    parts = options_text.split(",")
    out: List[str] = []
    for p in parts:
        s = p.strip()
        if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
            s = s[1:-1]
        out.append(s.strip())
    return out

def get_equivalent_score_as_fraction(scale_info: Dict[str, Any]) -> Optional[float]:
    """
    Return f in [0,1] where 0 = worst, 1 = best

    Prefer the 'fraction' field (derived from get_success_measure_from_rating_value).
    Only fall back to positional logic if 'fraction' is missing.
    """
    frac = scale_info.get("fraction")
    if frac is not None:
        return float(frac)

    num_options = scale_info.get("num_options")
    options_text = scale_info.get("options_text") or ""
    final_label = (scale_info.get("final_result_for_prompt") or "").strip()

    if not num_options or num_options <= 0:
        return None

    opts = _parse_options_list(options_text)
    if not opts:
        return None

    final_norm = final_label.lower()
    idx = None
    for i, opt in enumerate(opts):
        if opt.lower() == final_norm:
            idx = i
            break

    if idx is None:
        for i, opt in enumerate(opts):
            if final_norm in opt.lower():
                idx = i
                break
    if idx is None:
        return None

    if num_options == 1:
        return 0.5

    worst_side_idx = num_options - 1
    best_side_idx = 0
    span = worst_side_idx - best_side_idx
    if span <= 0:
        print("ERROR: span <0!")
        return 0.5

    return (worst_side_idx - idx) / span


def compute_training_distribution_from_scales(
    ratings: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    aid_fraction: Dict[str, float] = {}
    six_counts = {i: 0 for i in range(1, 7)}
    quartile_counts = {i: 0 for i in range(1, 5)}
    third_counts = {i: 0 for i in range(1, 4)}

    aid_six: Dict[str, int] = {}
    aid_quartile: Dict[str, int] = {}
    aid_third: Dict[str, int] = {}

    for aid, info in ratings.items():
        aid_str = str(aid)
        scale_info = get_rating_scale_info(aid_str, ratings)
        if scale_info is None:
            continue

        f = scale_info.get("fraction")
        if f is None:
            f = get_equivalent_score_as_fraction(scale_info)
        if f is None:
            continue

        if f < 0.0 or f > 1.0:
            print("ERROR: fraction outside [0,1]", aid_str, f)
            continue

        aid_fraction[aid_str] = f

        six_idx = min(5, int(f * 6))  # 0..5
        six_cat = six_idx + 1
        six_counts[six_cat] += 1
        aid_six[aid_str] = six_cat

        q_idx = min(3, int(f * 4))  # 0..3
        q_cat = q_idx + 1
        quartile_counts[q_cat] += 1
        aid_quartile[aid_str] = q_cat

        if six_cat in (1, 2):
            t_cat = 1
        elif six_cat in (3, 4):
            t_cat = 2
        else:
            t_cat = 3
        third_counts[t_cat] += 1
        aid_third[aid_str] = t_cat

    total = len(aid_fraction) or 1
    six_percents = {k: v * 100.0 / total for k, v in six_counts.items()}
    quartile_percents = {k: v * 100.0 / total for k, v in quartile_counts.items()}
    third_percents = {k: v * 100.0 / total for k, v in third_counts.items()}

    return {
        "aid_fraction": aid_fraction,
        "aid_six": aid_six,
        "six_counts": six_counts,
        "six_percents": six_percents,
        "aid_quartile": aid_quartile,
        "quartile_counts": quartile_counts,
        "quartile_percents": quartile_percents,
        "aid_third": aid_third,
        "third_counts": third_counts,
        "third_percents": third_percents,
    }


def compute_training_distribution_by_prefix(
    ratings: Dict[str, Dict[str, Any]],
    prefixes: Iterable[str] = ("44000-", "DE-1"),
) -> Dict[str, Any]:
    overall = compute_training_distribution_from_scales(ratings)

    by_prefix = {}
    for p in prefixes:
        sub = {aid: info for aid, info in ratings.items() if str(aid).startswith(p)}
        by_prefix[p] = compute_training_distribution_from_scales(sub) if sub else None

    return {
        "overall": overall,
        "by_prefix": by_prefix,
    }

def get_rating_scale_info_from_rating_object(aid: str, rating_info):
    rating_value = rating_info.get("rating_value")
    rating_min = rating_info.get("min")
    rating_max = rating_info.get("max")

    num_options, midpoint_low_text, midpoint_high_text, options_text, final_result_for_prompt = \
        get_ratings_text(rating_value, rating_min, rating_max, activity_id=aid)

    if num_options is None:
        return None

    numeric_rating = get_success_measure_from_rating_value_wrapped(
        rating_value, rating_min, rating_max, activity_id=aid
    )


    if VERBOSE:
        print(f"Rating as number: {numeric_rating} from {rating_value}, min: {rating_min}, max: {rating_max}, act: {aid}")
    fraction = None
    if numeric_rating is not None:
        fraction = numeric_rating / 5.0

    return {
        "num_options": num_options,
        "midpoint_low_text": midpoint_low_text,
        "midpoint_high_text": midpoint_high_text,
        "options_text": options_text,
        "final_result_for_prompt": final_result_for_prompt,
        "rating_value_raw": rating_value,
        "rating_min": rating_min,
        "rating_max": rating_max,
        "numeric_rating": numeric_rating,  # 0–5
        "fraction": fraction,
    }

def get_rating_scale_info(aid: str, ratings: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rating_info = ratings.get(aid)
    if not rating_info:
        return None
    return get_rating_scale_info_from_rating_object(aid, rating_info)


def load_good_overall_ids(filepath: str) -> dict:
    """
    Activities that already have a usable overall rating.

    "Good" rating criteria:
      - rating object comes from:
          * response_text (JSON string or dict), else
          * from_gemini.overall_rating (dict)
      - rating_value is non-empty
      - description != 'NO RATING AVAILABLE' (case-insensitive)

    Returns:
      { activity_id: {"description": ..., "rating_value": ..., "min": ..., "max": ...}, ... }
    """
    ids = {}
    path = Path(filepath)
    if not path.exists():
        return ids

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except Exception:
                continue

            aid = str(obj.get("activity_id") or "").strip()
            if not aid:
                continue

            rating_obj = None

            # 1) response_text: JSON string or dict
            rt = obj.get("response_text")
            if isinstance(rt, str) and rt:
                try:
                    rating_obj = json.loads(rt)
                except Exception:
                    rating_obj = None
            elif isinstance(rt, dict):
                rating_obj = rt

            # 2) fallback: from_gemini.overall_rating
            if rating_obj is None:
                fg = obj.get("from_gemini")
                if isinstance(fg, dict):
                    maybe_overall = fg.get("overall_rating")
                    if isinstance(maybe_overall, dict):
                        rating_obj = maybe_overall

            if not isinstance(rating_obj, dict):
                continue

            desc = str(rating_obj.get("description") or "").strip()
            val  = str(rating_obj.get("rating_value") or "").strip()
            if not val:
                continue
            if desc.upper() == "NO RATING AVAILABLE":
                continue

            # keep min/max as-is (could be numeric/str); don't force .strip() on non-strings
            min_rating = rating_obj.get("min")
            max_rating = rating_obj.get("max")

            ids[aid] = {
                "description": desc,
                "rating_value": val,
                "min": min_rating,
                "max": max_rating,
            }

    return ids

from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import re
import numpy as np
import pandas as pd

_NUM_RE = re.compile(r"\d+")


def load_jsonl_text_by_activity_id(path: Path) -> Dict[str, dict]:
    m: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            aid = str(obj.get("activity_id", "")).strip()
            if not aid:
                continue
            m[aid] = {
                "text": obj.get("text", None),
                "section": obj.get("section", None),
                "model": obj.get("model", None),
            }
    return m


def load_ratings(filepath: str) -> pd.Series:
    """
    Load numeric ratings from merged_overall_ratings.jsonl.

    "Good" rating criteria:
      - rating object comes from:
          * response_text (JSON string or dict), else
          * from_gemini.overall_rating (dict)
      - rating_value is non-empty
      - description != 'NO RATING AVAILABLE' (case-insensitive)
    """
    ratings = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except Exception:
                continue

            aid = str(obj.get("activity_id") or "").strip()
            if not aid:
                continue

            rating_obj = None

            # 1) response_text: JSON string or dict
            rt = obj.get("response_text")
            if isinstance(rt, str) and rt:
                try:
                    rating_obj = json.loads(rt)
                except Exception:
                    rating_obj = None
            elif isinstance(rt, dict):
                rating_obj = rt

            # 2) fallback: from_gemini.overall_rating
            if rating_obj is None:
                fg = obj.get("from_gemini")
                if isinstance(fg, dict):
                    maybe_overall = fg.get("overall_rating")
                    if isinstance(maybe_overall, dict):
                        rating_obj = maybe_overall

            if not isinstance(rating_obj, dict):
                continue

            desc = str(rating_obj.get("description") or "").strip()
            rating_value = str(rating_obj.get("rating_value") or "").strip()
            if not rating_value:
                continue
            if desc.upper() == "NO RATING AVAILABLE":
                continue

            rating_min = rating_obj.get("min")
            rating_max = rating_obj.get("max")

            numeric_rating = get_success_measure_from_rating_value_wrapped(
                rating_value, rating_min, rating_max, activity_id=aid
            )
            if VERBOSE:
                print(f"Rating as number: {numeric_rating} from {rating_value}, min: {rating_min}, max: {rating_max}, act: {aid}")

            if numeric_rating is not None:
                ratings[aid] = numeric_rating

    return pd.Series(ratings, name="rating")




def load_world_bank_indicators(filepath):
    """Load activity scope from CSV and map to numeric codes."""
    df = pd.read_csv(filepath, usecols=[
        "activity_id",
        "cpia_score",
        "wgi_control_of_corruption_est",
        "wgi_government_effectiveness_est",
        "wgi_political_stability_est",
        "wgi_regulatory_quality_est",
        "wgi_rule_of_law_est",
    ])
    df["cpia_score"] = (
        df["cpia_score"]
        .astype(float)
    )
    df["wgi_control_of_corruption_est"] = (
        df["wgi_control_of_corruption_est"]
        .astype(float)
    )
    df["wgi_government_effectiveness_est"] = (
        df["wgi_government_effectiveness_est"]
        .astype(float)
    )
    df["wgi_political_stability_est"] = (
        df["wgi_political_stability_est"]
        .astype(float)
    )
    df["wgi_regulatory_quality_est"] = (
        df["wgi_regulatory_quality_est"]
        .astype(float)
    )
    df["wgi_rule_of_law_est"] = (
        df["wgi_rule_of_law_est"]
        .astype(float)
    )
    return df.set_index("activity_id")

import pandas as pd



def parse_orgs(org_str: str, role_filter: str | None = None) -> Set[str]:
    """
    Parse organisation strings like:
      'Canada (Funding); FAO - Food and Agriculture Organization (Implementing)'
    into a set of lowercased organisation names.

    If role_filter is given (e.g. 'Implementing'), only keep entries containing that.
    """
    if not isinstance(org_str, str) or not org_str.strip():
        return set()

    orgs: Set[str] = set()
    for part in org_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if role_filter and f"({role_filter})" not in part:
            continue
        if "(" in part:
            name = part.split("(", 1)[0].strip()
        else:
            name = part
        if name:
            orgs.add(name.lower())
    return orgs



import re

CANON_LABELS = [
    "Highly Satisfactory",
    "Satisfactory",
    "Moderately Satisfactory",
    "Moderately Unsatisfactory",
    "Unsatisfactory",
    "Highly Unsatisfactory",
]

_LABEL_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, sorted(CANON_LABELS, key=len, reverse=True))) + r")\b",
    flags=re.IGNORECASE,
)

_MD_STRIP_RE = re.compile(r"[*_`]+")




def parse_last_line_label_after_forecast(content, record=None):
    """
    Content is long text; last non-empty line contains the label,
    e.g. "FORECAST: Moderately Satisfactory" or "4. **Forecast:** Successful".

    Returns a numeric rating on your canonical 0–5 scale
    using get_success_measure_from_rating_value.
    """
    text = str(content)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        print("ERROR: failed to parse last line result (no non-empty lines).")
        print("failed text:")
        pprint.pprint(lines)
        return None

    last = lines[-1]

    line = re.sub(r'^\s*(?:[-*+]|\d+[\).\]])\s*', '', last)

    line = line.replace("**", "").replace("__", "")
    line = line.strip(" *_")

    m = re.search(r'forecast\s*[:\-–—]?\s*(.+)$', line, flags=re.IGNORECASE)
    if not m:
        print("ERROR: last line does not contain 'FORECAST'")
        print("last line:", last)
        return None

    label = m.group(1).strip()

    label = label.strip(" *.?_\"'`-")

    rating_min = None
    rating_max = None
    if isinstance(record, dict):
        rating_min = record.get("min")
        rating_max = record.get("max")

    activity_id = None
    if isinstance(record, dict):
        activity_id = record.get("activity_id")
    numeric = get_success_measure_from_rating_value_wrapped(
        label,
        min_rating=rating_min,
        max_rating=rating_max,
        activity_id=activity_id,
    )

    if numeric is None:
        print("ERROR: failed to map forecast label to numeric rating")
        print("last line:", last)
        print("parsed label:", label)
        return None

    return numeric






