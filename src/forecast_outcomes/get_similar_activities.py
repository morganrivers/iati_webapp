#!/usr/bin/env python3
import json
import pprint
import math
import sys
from typing import Set
from pathlib import Path
import numpy as np
import pandas as pd
VERBOSE = False
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
CSV_PATH = "../../data/info_for_activity_forecasting.csv"
EMBEDDINGS_PATH = Path("../../data/activity_text_embeddings_gemini.jsonl")
_EMBEDDINGS_CACHE = None



def find_similar_activities_bm25(
    activity_id: str,
    df: pd.DataFrame,
    corpus: dict,
    query_text: str,
    top_n: int = 20,
    allowed_ids=None,
) -> tuple:
    import re
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError("rank_bm25 is required for BM25 KNN. pip install rank-bm25") from e

    aid_set = set(df["activity_id"].astype(str))
    query_row = df[df["activity_id"].astype(str) == activity_id].iloc[0] if activity_id in aid_set else pd.Series()

    if not query_text:
        return pd.DataFrame(columns=list(df.columns) + ["similarity"]), query_row

    q_start = q_end = None
    if activity_id in aid_set:
        qrow = df[df["activity_id"].astype(str) == activity_id].iloc[0]
        q_start = qrow.get("start_date")
        q_end = qrow.get("end_date")

    if allowed_ids is not None:
        allowed_ids = set(str(a) for a in allowed_ids)

    candidate_ids = []
    for _, row in df.iterrows():
        aid = str(row.get("activity_id", ""))
        if aid == activity_id:
            continue
        if aid not in corpus:
            continue
        if allowed_ids is not None and aid not in allowed_ids:
            continue
        if q_start is not None and pd.notna(q_start):
            if not passes_quartile_constraint(q_start, q_end, row.get("start_date"), row.get("end_date")):
                continue
        candidate_ids.append(aid)

    if not candidate_ids:
        return pd.DataFrame(columns=list(df.columns) + ["similarity"]), query_row

    def _tokenize(text: str) -> list:
        return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()

    tokenized_corpus = [_tokenize(corpus[aid]) for aid in candidate_ids]
    bm25 = BM25Okapi(tokenized_corpus)
    raw_scores = bm25.get_scores(_tokenize(query_text))

    order = np.argsort(-raw_scores)[:top_n]
    top_ids = [candidate_ids[i] for i in order]
    top_scores = raw_scores[order]

    max_s = float(top_scores[0]) if len(top_scores) > 0 and top_scores[0] > 0 else 1.0
    top_sims = top_scores / max_s

    aid_to_idx = {str(row["activity_id"]): idx for idx, row in df.iterrows()}
    valid = [(aid, sim) for aid, sim in zip(top_ids, top_sims) if aid in aid_to_idx]
    if not valid:
        return pd.DataFrame(columns=list(df.columns) + ["similarity"]), query_row

    idxs = [aid_to_idx[aid] for aid, _ in valid]
    sims = [float(sim) for _, sim in valid]
    result = df.loc[idxs].copy()
    result["similarity"] = sims

    cols_front = [
        "activity_id", "similarity", "activity_title", "activity_scope",
        "country_location", "gdp_percap", "dac5", "reporting_orgs",
        "participating_orgs", "start_date", "end_date",
    ]
    cols_front = [c for c in cols_front if c in result.columns]
    other_cols = [c for c in result.columns if c not in cols_front]
    return result[cols_front + other_cols], query_row


def load_activity_embeddings_sqlite(db_path) -> dict[str, np.ndarray]:
    """
    Load embeddings from the 'embeddings' table in webapp.db.
    Stores float32 BLOBs (~27 MB) instead of JSON text (~99 MB).
    Shares _EMBEDDINGS_CACHE with load_activity_embeddings so only one is ever loaded.
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    import sqlite3
    embs: dict[str, np.ndarray] = {}
    with sqlite3.connect(str(db_path)) as conn:
        for aid, blob in conn.execute("SELECT activity_id, embedding FROM embeddings"):
            v = np.frombuffer(blob, dtype="float32").copy()
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            embs[str(aid)] = v

    _EMBEDDINGS_CACHE = embs
    return embs


def load_activity_embeddings(path: Path = EMBEDDINGS_PATH) -> dict[str, np.ndarray]:
    """
    Load embeddings from JSONL into a dict: activity_id -> L2-normalised np.ndarray.
    Assumes each line is like:
      {"activity_id": "...", "embedding": [float, ...], ...}
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    embs: dict[str, np.ndarray] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Bad JSON in {path} at line {line_no}: {e}"
                ) from e

            aid = obj.get("activity_id")
            vec = obj.get("embedding")  # key name matches your file

            if aid is None or vec is None:
                continue

            v = np.asarray(vec, dtype="float32")
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm

            embs[str(aid)] = v

    _EMBEDDINGS_CACHE = embs
    return embs



def parse_countries(country_location: str) -> Set[str]:
    """Extract a set of 2-letter country codes from the country_location field."""
    if not isinstance(country_location, str) or not country_location.strip():
        return set()

    text = country_location

    if "Recipient countries:" in text:
        text = text.split("Recipient countries:", 1)[1]

    text = text.split("|", 1)[0]

    tokens = (
        text.replace(",", " ")
        .replace(";", " ")
        .replace("%", " ")
        .split()
    )

    codes = set()
    for tok in tokens:
        tok = tok.strip().upper()
        if len(tok) == 2 and tok.isalpha():
            codes.add(tok)
    return codes


def parse_dac_codes(dac_str: str) -> Set[str]:
    """Parse '12345|67890' style DAC5 codes into a set of strings."""
    if not isinstance(dac_str, str) or not dac_str.strip():
        return set()
    parts = dac_str.split("|")
    return {p.strip() for p in parts if p.strip()}


def parse_orgs(org_str: str, role_filter: str | None = None) -> Set[str]:
    """
    Parse organisation strings like:
      'Canada (Funding); FAO - Food and Agriculture Organization (Implementing)'
    into a set of lowercased organisation names.

    If role_filter is given (e.g. 'Implementing'), only keep entries containing that.
    """
    if not isinstance(org_str, str) or not org_str.strip():
        return set()

    orgs = set()
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


def cosine_similarity_sets(a: Set[str], b: Set[str]) -> float:
    """Cosine similarity between two sets treated as binary indicator vectors."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / math.sqrt(len(a) * len(b))


def gdp_similarity(g1, g2) -> float:
    """Similarity based on GDP per capita (log-distance)."""
    if pd.isna(g1) or pd.isna(g2):
        return 0.0
    return 1.0 / (1.0 + abs(math.log1p(float(g1)) - math.log1p(float(g2))))

def recency_similarity(q_start, cand_end) -> float:
    """How close (in time) the candidate's end is to the query start."""
    if pd.isna(q_start) or pd.isna(cand_end):
        return 0.0
    gap_days = (q_start - cand_end).days
    gap_years = abs(gap_days) / 365.25
    return 1.0 / (1.0 + gap_years)


def pick_start_date(row: pd.Series):
    """Choose a start date from available columns."""
    for col in ["actual_start_date", "original_planned_start_date", "txn_first_date"]:
        if col in row and pd.notna(row[col]):
            return row[col]
    return pd.NaT


def pick_end_date(row: pd.Series):
    """Choose an end date from available columns."""
    for col in ["actual_close_date", "original_planned_close_date","txn_last_date"]:
        if col in row and pd.notna(row[col]):
            return row[col]
    return pd.NaT



def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    return add_derived_columns(pd.read_csv(csv_path))


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add scope_code and generic start_date/end_date columns to a raw activity
    dataframe. Shared by the CSV loader (prepare_dataframe) and the SQLite-backed
    webapp loader so both produce identical similarity inputs.
    """
    df = df.copy()

    df["activity_scope_norm"] = (
        df["activity_scope"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["scope_code"] = df["activity_scope_norm"].map(ACTIVITY_SCOPES)

    date_cols = [
        "original_planned_start_date",
        "original_planned_close_date",
        "actual_start_date",
        "actual_end_date",
        "actual_close_date",
        "txn_first_date",
        "txn_last_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["start_date"] = df.apply(pick_start_date, axis=1)
    df["end_date"] = df.apply(pick_end_date, axis=1)

    return df


def print_top_feature_matches(
    candidates: pd.DataFrame,
    activity_id: str,
    q_scope,
    q_countries: Set[str],
    q_dac: Set[str],
    q_report_orgs: Set[str],
    q_impl_orgs: Set[str],
    q_gdp,
):
    top_scope = []
    top_countries = []
    top_dac = []
    top_rep = []
    top_impl = []
    top_gdp = []

    for _, row in candidates.iterrows():
        if row.get("activity_id") == activity_id:
            continue

        aid = row.get("activity_id")

        s_scope = row.get("scope_code")
        if pd.notna(q_scope) and pd.notna(s_scope):
            scope_sim = 1.0 / (1.0 + abs(int(q_scope) - int(s_scope)))
            top_scope.append((scope_sim, aid, row.get("activity_scope")))

        c_countries = parse_countries(row.get("country_location", ""))
        if q_countries and c_countries:
            inter = len(q_countries & c_countries)
            union = len(q_countries | c_countries)
            if union:
                top_countries.append((inter / union, aid, row.get("country_location")))

        dac_sim = cosine_similarity_sets(q_dac, parse_dac_codes(row.get("dac5", "")))
        if dac_sim > 0:
            top_dac.append((dac_sim, aid, row.get("dac5")))

        rep_sim = cosine_similarity_sets(
            q_report_orgs, parse_orgs(row.get("reporting_orgs", ""))
        )
        if rep_sim > 0:
            top_rep.append((rep_sim, aid, row.get("reporting_orgs")))

        impl_sim = cosine_similarity_sets(
            q_impl_orgs,
            parse_orgs(row.get("participating_orgs", ""), role_filter="Implementing"),
        )
        if impl_sim > 0:
            top_impl.append((impl_sim, aid, row.get("participating_orgs")))


        if not pd.isna(q_gdp) and not pd.isna(row.get("gdp_percap")):
            gsim = gdp_similarity(q_gdp, row.get("gdp_percap"))
            if gsim > 0:
                top_gdp.append((gsim, aid, row.get("gdp_percap")))

    print("\n\n\n\n TOP FEATURE MATCHES")
    def show(label, items):
        items = sorted(items, key=lambda x: x[0], reverse=True)[:5]
        pprint.pprint(label)
        for sim, aid, val in items:
            pprint.pprint((round(sim, 3), aid, val))

    show("top_scope", top_scope)
    show("top_countries", top_countries)
    show("top_dac", top_dac)
    show("top_report_orgs", top_rep)
    show("top_impl_orgs", top_impl)
    show("top_gdp", top_gdp)

def get_dac_sim(query_dac, cand_dac, hierarchy_param: float = 0.5) -> float:
    """
    Similarity between two sets of DAC5 codes, hierarchical by digit.

    hierarchy_param in [0,1]:
      0.0 -> only care about first digit (broad sector)
      1.0 -> mostly care about deeper digits (exact / very specific match)
      intermediate -> some mix (higher => more weight on deeper digits)

    2nd digit only contributes if 1st matches, 3rd only if first 2 match, etc.
    """
    if not query_dac or not cand_dac:
        return 0.0

    IGNORE = {"99810"}
    LOW = {"60062", "60040", "43010", "91010", "99820"}

    try:
        p = float(hierarchy_param)
    except Exception:
        p = 0.5
    p = max(0.0, min(1.0, p))

    def norm(code: str):
        s = "".join(ch for ch in str(code) if ch.isdigit())
        if len(s) < 5:
            return None
        return s[:5]

    w1 = 1.0 - p
    rest = p
    w_rest = rest / 4.0 if rest > 0 else 0.0
    weights = [w1, w_rest, w_rest, w_rest, w_rest]
    total_w = sum(weights) or 1.0
    weights = [w / total_w for w in weights]

    def code_sim(a: str, b: str) -> float:
        if a in IGNORE or b in IGNORE:
            return 0.0

        sa = norm(a)
        sb = norm(b)
        if sa is None or sb is None:
            return 0.0

        score = 0.0
        for i in range(5):
            if sa[i] != sb[i]:
                break
            score += weights[i]

        if a in LOW or b in LOW:
            score /= 3.0

        return score

    def dir_sim(src, dst) -> float:
        if not src or not dst:
            return 0.0
        total = 0.0
        for a in src:
            best = 0.0
            for b in dst:
                s = code_sim(a, b)
                if s > best:
                    best = s
            total += best
        return total / len(src)

    s1 = dir_sim(query_dac, cand_dac)
    s2 = dir_sim(cand_dac, query_dac)
    return 0.5 * (s1 + s2)


def passes_quartile_constraint(q_start, q_end, c_start, c_end) -> bool:
    """Return False if candidate 3/4 mark is after query 1/4 mark."""
    if pd.isna(q_start) or pd.isna(q_end) or pd.isna(c_start) or pd.isna(c_end):
        return False
    q_25 = q_start + (q_end - q_start) * 0.25
    c_75 = c_start + (c_end - c_start) * 0.75
    return c_75 <= q_25

def find_similar_activities(activity_id: str,
                            csv_path: str = CSV_PATH,
                            top_n: int = 20,
                            allowed_ids=None,
                            feature_weightings_hyperparams=None) -> pd.DataFrame:
    df = prepare_dataframe(csv_path)

    if "activity_id" not in df.columns:
        raise ValueError("CSV must contain an 'activity_id' column")

    if activity_id not in set(df["activity_id"]):
        raise ValueError(f"Activity ID '{activity_id}' not found in CSV")

    query_row = df[df["activity_id"] == activity_id].iloc[0]

    q_scope = query_row.get("scope_code")
    q_countries = parse_countries(query_row.get("country_location", ""))
    q_dac = parse_dac_codes(query_row.get("dac5", ""))
    q_report_orgs = parse_orgs(query_row.get("reporting_orgs", ""))
    q_impl_orgs = parse_orgs(query_row.get("participating_orgs", ""), role_filter="Implementing")
    q_gdp = query_row.get("gdp_percap")
    q_start = query_row.get("start_date")
    q_end = query_row.get("end_date")

    search_item = df[df["activity_id"] == activity_id].copy()
    candidates = df
    if allowed_ids is not None:
        allowed_ids = set(allowed_ids)
    df_ids = set(df["activity_id"])

    if pd.notna(q_start):
        candidates_notna = candidates[(pd.notna(candidates["end_date"]))]
        cand_ids = set(candidates_notna["activity_id"])
        candidates = candidates_notna
    else:
        print("ERROR: the query start q_start is invalid!")
    cand_ids = set(candidates["activity_id"])
    if VERBOSE:
        print_top_feature_matches(
            candidates,
            activity_id,
            q_scope,
            q_countries,
            q_dac,
            q_report_orgs,
            q_impl_orgs,
            q_gdp,
        )


    scores = []
    total_number = 0
    total_number_checking_score = 0
    before_the_date_filter = 0
    for idx, row in candidates.iterrows():
        if allowed_ids is not None:
            if row.get("activity_id") not in allowed_ids:
                continue

        before_the_date_filter += 1
        if not passes_quartile_constraint(
            q_start,
            q_end,
            row.get("start_date"),
            row.get("end_date"),
        ):
            continue
        total_number += 1
        s_scope = row.get("scope_code")
        if pd.isna(q_scope) or pd.isna(s_scope):
            scope_sim = 0.0
        else:
            diff = abs(int(q_scope) - int(s_scope))
            scope_sim = 1.0 / (1.0 + diff)

        c_countries = parse_countries(row.get("country_location", ""))
        if q_countries and c_countries:
            inter = len(q_countries & c_countries)
            union = len(q_countries | c_countries)
            jacc = inter / union if union > 0 else 0.0
        else:
            jacc = 0.0

        if jacc > 0:
            geo_sim = jacc
        else:
            geo_sim = gdp_similarity(q_gdp, row.get("gdp_percap"))
        dac_sim = get_dac_sim(q_dac,parse_dac_codes(row.get("dac5", "")),0.75)

        rep_sim = cosine_similarity_sets(
            q_report_orgs,
            parse_orgs(row.get("reporting_orgs", ""))
        )

        impl_sim = cosine_similarity_sets(
            q_impl_orgs,
            parse_orgs(row.get("participating_orgs", ""), role_filter="Implementing")
        )

        rec_sim = recency_similarity(q_start, row.get("end_date"))
        if feature_weightings_hyperparams is not None:
            score = (
                feature_weightings_hyperparams["scope"] * scope_sim +
                feature_weightings_hyperparams["geo"] * geo_sim +
                feature_weightings_hyperparams["dac"] * dac_sim +
                feature_weightings_hyperparams["rep"] * rep_sim +
                feature_weightings_hyperparams["impl"] * impl_sim +
                feature_weightings_hyperparams["rec"] * rec_sim
            )
        else:
            score = (
                0.081* scope_sim +
                0.187* geo_sim +
                0.144* dac_sim +
                0.200* rep_sim +
                0.261* impl_sim +
                0.127* rec_sim
            )

        total_number_checking_score += 1
        if score > 0:
            scores.append((idx, score))
    if not scores:
        return pd.DataFrame(columns=list(df.columns) + ["similarity"])

    idxs, sim_vals = zip(*scores)
    result = candidates.loc[list(idxs)].copy()
    result["similarity"] = sim_vals
    result = result.sort_values("similarity", ascending=False)

    cols_front = [
        "activity_id",
        "similarity",
        "activity_title",
        "activity_scope",
        "country_location",
        "gdp_percap",
        "dac5",
        "reporting_orgs",
        "participating_orgs",
        "start_date",
        "end_date",
    ]
    cols_front = [c for c in cols_front if c in result.columns]
    other_cols = [c for c in result.columns if c not in cols_front]

    def describe_scope_code(code):
        for name, val in ACTIVITY_SCOPES.items():
            if val == code:
                return name
        return None

    top5 = result.head(5)

    if VERBOSE:
        print("\nQuery activity:", activity_id)
        print("  title     :", str(query_row.get("activity_title", ""))[:120])
        print("  scope_code:", q_scope, "->", describe_scope_code(q_scope))
        print("  countries :", sorted(q_countries) if q_countries else [])
        print("  dac5      :", sorted(q_dac) if q_dac else [])
        print("  rep_orgs  :", sorted(q_report_orgs) if q_report_orgs else [])
        print("  impl_orgs :", sorted(q_impl_orgs) if q_impl_orgs else [])
        print("  gdp_pc    :", q_gdp)
        print("  start/end :", str(q_start), "->", query_row.get("end_date"))

    for i, (_, row) in enumerate(top5.iterrows(), start=1):
        aid = row.get("activity_id")
        c_scope = row.get("scope_code")
        c_countries = parse_countries(row.get("country_location", ""))
        c_dac = parse_dac_codes(row.get("dac5", ""))
        c_rep = parse_orgs(row.get("reporting_orgs", ""))
        c_impl = parse_orgs(row.get("participating_orgs", ""), role_filter="Implementing")
        c_gdp = row.get("gdp_percap")

        if pd.isna(q_scope) or pd.isna(c_scope):
            scope_sim = 0.0
        else:
            scope_sim = 1.0 / (1.0 + abs(int(q_scope) - int(c_scope)))

        if q_countries and c_countries:
            inter = len(q_countries & c_countries)
            union = len(q_countries | c_countries)
            jacc = inter / union if union else 0.0
        else:
            jacc = 0.0
        if jacc > 0.5:
            geo_sim = jacc
        else:
            geo_sim = gdp_similarity(q_gdp, c_gdp)/2 # divide by 2 to make it so even if has same gdp, not as good match as being the same country

        dac_sim = cosine_similarity_sets(q_dac, c_dac)
        rep_sim = cosine_similarity_sets(q_report_orgs, c_rep)
        impl_sim = cosine_similarity_sets(q_impl_orgs, c_impl)
        gdp_sim = gdp_similarity(q_gdp, c_gdp)

        if VERBOSE:

            print(f"\n[{i}] activity_id={aid}  sim={row['similarity']:.3f}")
            print("  title     :", str(row.get("activity_title", ""))[:120])
            print("  scope     :", c_scope, "->", describe_scope_code(c_scope),
                  f"(sim={scope_sim:.3f})")
            print("  countries :", sorted(c_countries) if c_countries else [],
                  f"(Jaccard={jacc:.3f}, geo_sim={geo_sim:.3f})")
            print("  dac5      :", sorted(c_dac) if c_dac else [],
                  f"(sim={dac_sim:.3f})")
            print("  rep_orgs  :", sorted(c_rep) if c_rep else [],
                  f"(sim={rep_sim:.3f})")
            print("  impl_orgs :", sorted(c_impl) if c_impl else [],
                  f"(sim={impl_sim:.3f})")
            print("  gdp_pc    :", c_gdp, f"(sim={gdp_sim:.3f})")
            print("  start/end :", row.get("start_date"), "->", row.get("end_date"))
    return result[cols_front + other_cols].head(top_n), search_item




def find_similar_activities_semantic(
    activity_id: str,
    df: pd.DataFrame,
    embeddings: dict,
    top_n: int = 20,
    allowed_ids=None,
):
    query_row = df[df["activity_id"] == activity_id].iloc[0]

    if "activity_id" not in df.columns:
        raise ValueError("dataframe must contain an 'activity_id' column")

    aid_to_idx = {}
    for idx, row in df.iterrows():
        aid = row.get("activity_id")
        if isinstance(aid, str) and aid:
            aid_to_idx[aid] = idx

    if activity_id not in aid_to_idx:
        raise ValueError(f"Activity ID '{activity_id}' not found in metadata dataframe")

    embs = embeddings

    if activity_id not in embs:
        raise ValueError(f"Activity ID '{activity_id}' not found in provided embeddings")

    q_vec = embs[activity_id]

    q_start = query_row.get("start_date")
    q_end = query_row.get("end_date")


    if allowed_ids is not None:
        allowed_ids = set(str(a) for a in allowed_ids)

    candidates = df

    if pd.notna(q_start):
        candidates_notna = candidates[(pd.notna(candidates["end_date"]))]
        cand_ids = set(candidates_notna["activity_id"])
        candidates = candidates_notna
    else:
        print("ERROR: the query start q_start is invalid!")

    candidate_ids = []
    for idx, row in candidates.iterrows():
        aid = row.get("activity_id")
        if aid not in embs.keys():
            continue

        if allowed_ids is not None:
            if aid not in allowed_ids:
                continue

        if not passes_quartile_constraint(
            q_start,
            q_end,
            row.get("start_date"),
            row.get("end_date"),
        ):
            continue

        candidate_ids.append(aid)


    if not candidate_ids:
        return (
            pd.DataFrame(columns=list(df.columns) + ["similarity"]),
            df[df["activity_id"] == activity_id].copy(),
        )

    cand_vecs = np.stack([embs[aid] for aid in candidate_ids], axis=0)
    sims = cand_vecs @ q_vec
    order = np.argsort(-sims)
    order = order[:top_n]

    top_ids = [candidate_ids[i] for i in order]
    top_sims = sims[order]

    idxs = [aid_to_idx[aid] for aid in top_ids]
    result = df.loc[idxs].copy()
    result["similarity"] = top_sims

    cols_front = [
        "activity_id",
        "similarity",
        "activity_title",
        "activity_scope",
        "country_location",
        "gdp_percap",
        "dac5",
        "reporting_orgs",
        "participating_orgs",
        "start_date",
        "end_date",
    ]
    cols_front = [c for c in cols_front if c in result.columns]
    other_cols = [c for c in result.columns if c not in cols_front]

    result = result[cols_front + other_cols]

    return result, query_row

def main(activity_id: str):
    print("\nFIRST, FEATURE BASED:\n\n")
    df_sim, search_item = find_similar_activities(activity_id)
    pprint.pprint(df_sim.to_csv(index=False))
    pprint.pprint("df_sim")
    pprint.pprint(df_sim)
    pprint.pprint("search_item")
    pprint.pprint(search_item)

    print("")
    print("")
    print("\nAND NOW, SEMANTIC VECTOR SIMILARITY:\n\n")
    df_sim, search_item = find_similar_activities_semantic(
        activity_id,
        df=prepare_dataframe(CSV_PATH),
        embeddings=load_activity_embeddings(),
    )
    pprint.pprint(df_sim.to_csv(index=False))
    pprint.pprint("df_sim")
    pprint.pprint(df_sim)
    pprint.pprint("search_item")
    pprint.pprint(search_item)

