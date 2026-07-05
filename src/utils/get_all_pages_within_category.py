#!/usr/bin/env python3
"""
Filter PDF scoring rows by section/categories and a minimum *score* threshold.

- Reads only ../../data/pdf_categories_scores.csv (one row per page).
- No use of ranked_documents.csv, rank, or grade.
- Returns rows that match the provided section and any given category/subcategories,
  optionally requiring score >= MIN_SCORE.
- Sorts results by:
    1) how many of the provided category/subcategory fields match (more first)
    2) score descending
    3) page number ascending (tie-breaker)

How to use:
- Edit the constants at the bottom (SECTION, CATEGORY, SUBCAT_A, SUBCAT_B, MIN_SCORE)
  and run the script. It will print the matching records.
- No CLI flags—keep it simple.
"""

from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import pprint
from typing import Iterable, Union
import re
import pandas as pd
from pathlib import Path
PDF_SCORES_CSV = Path("../../data/pdf_categories_scores.csv")


import os, sys
import pandas as pd
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

def _category_columns(section: str) -> Dict[str, str]:
    """Return the column names for category/subcats based on section."""
    sec = (section or "").strip().lower()
    if sec == "baseline":
        return {
            "category": "baseline_category_1",
            "sub_a": "baseline_category_2",
            "sub_b": "baseline_category_3",
        }
    elif sec == "outcome":
        return {
            "category": "outcome_category_1",
            "sub_a": "outcome_category_2",
            "sub_b": "outcome_category_3",
        }
    else:
        raise ValueError("section must be 'Baseline' or 'Outcome'")


def _norm(s: Any) -> str:
    return ("" if s is None else str(s)).strip().lower()

def load_and_filter_rows(
    section: str,
    category: Union[str, Iterable[str], None] = None,
    subcategory_a: Union[str, Iterable[str], None] = None,
    subcategory_b: Union[str, Iterable[str], None] = None,
    min_score: Optional[float] = None,
    csv_path: Path = PDF_SCORES_CSV,
    top_k_per_activity: Optional[int] = 20,  # NEW
    get_surrounding_if_not_enough: Optional[bool] = False,  # NEW
    get_at_least_top_k: Optional[bool] = False,  # NEW
) -> List[Dict[str, Any]]:
    """
    Load the scores CSV and return matching rows as a list of dicts.

    Filters:
      - section: "Baseline" or "Outcome" (required)
      - category / subcategory_a / subcategory_b:
          * may be a single string OR an iterable of strings
          * OR semantics within each field (any value may match)
          * case-insensitive
          * matching mode controlled by `match_mode`:
              - "exact": equality (default)
              - "contains": substring match
              - "regex": values are regex patterns combined with OR
      - min_score: keep rows where score >= min_score (if provided)

    Returned rows include a computed 'match_count' used for sorting:
      number of fields (among category, subcategory_a, subcategory_b) that matched.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing PDF scores CSV: {csv_path}")

    def _norm(s: Any) -> str:
        return ("" if s is None else str(s)).strip().lower()

    def _to_lower_set(x: Union[str, Iterable[str], None]) -> set[str] | None:
        if x is None:
            return None
        if isinstance(x, (str, bytes)):
            val = _norm(x)
            return {val} if val else None
        try:
            vals = {_norm(v) for v in x if _norm(v)}
            return vals or None
        except TypeError:
            val = _norm(x)
            return {val} if val else None

    def _match_series(series: pd.Series, want: set[str] | None) -> pd.Series:
        if want is None:
            return pd.Series(True, index=series.index)
        return series.isin(want)

    df = pd.read_csv(csv_path)

    df["section"] = df["section"].astype(str).str.strip()
    cols = _category_columns(section)
    cat_col, suba_col, subb_col = cols["category"], cols["sub_a"], cols["sub_b"]

    df = df[df["section"].str.lower() == section.strip().lower()].copy()
    if df.empty:
        return []

    for c in (cat_col, suba_col, subb_col):
        df[c] = df[c].astype(str).str.strip().str.lower()

    want_cat = _to_lower_set(category)
    want_a   = _to_lower_set(subcategory_a)
    want_b   = _to_lower_set(subcategory_b)

    m_cat = _match_series(df[cat_col], want_cat)
    m_a   = _match_series(df[suba_col], want_a)
    m_b   = _match_series(df[subb_col], want_b)

    mask = pd.Series(True, index=df.index)

    if want_cat is not None:
        mask &= m_cat

    if (want_a is not None) or (want_b is not None):
        sub_mask = None
        if want_a is not None and want_b is not None:
            sub_mask = m_a | m_b
        elif want_a is not None:
            sub_mask = m_a
        else:
            sub_mask = m_b
        mask &= sub_mask

    mask_before_score = mask.copy()

    per_act_cat_sub = (
        mask_before_score.groupby(df["activity_id"]).sum()
    )
    if min_score is not None:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        score_mask = df["score"] >= float(min_score)
        mask &= score_mask
    filtered = df[mask].copy()

    all_acts = set(df["activity_id"].unique())
    acts_with_rows = set(filtered["activity_id"].unique())
    missing_acts = sorted(all_acts - acts_with_rows)

    reason_counts = {
        "no_cat_or_sub_match": 0,
        "matched_cat_or_sub_but_all_below_min_score": 0,
        "other": 0,
    }

    for aid in missing_acts:
        base_idx = df[df["activity_id"] == aid].index
        n_total = len(base_idx)
        n_cat_sub = int(mask_before_score.loc[base_idx].sum())
        n_final = int(mask.loc[base_idx].sum())

        if n_cat_sub == 0:
            reason_counts["no_cat_or_sub_match"] += 1
        elif n_final == 0:
            reason_counts["matched_cat_or_sub_but_all_below_min_score"] += 1
        else:
            reason_counts["other"] += 1


    if filtered.empty:
        return []

    def _count_match(row) -> int:
        count = 0
        if want_cat is not None and row[cat_col] in want_cat:
            count += 1 if row[cat_col] in want_cat else 0
        if want_a is not None:
            count += 1 if row[suba_col] in want_a else 0
        if want_b is not None:    
            count += 1 if row[subb_col] in want_b else 0
        return count

    filtered["match_count"] = filtered.apply(_count_match, axis=1)

    filtered["score"] = pd.to_numeric(filtered["score"], errors="coerce")
    filtered["page_start"] = pd.to_numeric(filtered.get("page_start", 0), errors="coerce").fillna(0).astype(int)

    if top_k_per_activity is not None:
        seeds = (
            filtered
            .sort_values(["activity_id", "score", "page_start"],
                         ascending=[True, False, True],
                         kind="mergesort")
            .groupby("activity_id", group_keys=True)
            .head(top_k_per_activity)
        )

        acts_with_filtered = set(filtered["activity_id"].unique())
        acts_with_seeds = set(seeds["activity_id"].unique())
        acts_with_no_seeds = sorted(acts_with_filtered - acts_with_seeds)


        if not get_surrounding_if_not_enough and not get_at_least_top_k:
            filtered = seeds
        else:
            df_section = df.copy()
            df_section["page_start"] = pd.to_numeric(
                df_section.get("page_start", 0),
                errors="coerce"
            ).fillna(0).astype(int)
            df_section["doc_index_int"] = pd.to_numeric(
                df_section.get("doc_index_int"),
                errors="coerce"
            ).astype("Int64")
            df_section["score"] = pd.to_numeric(
                df_section.get("score"),
                errors="coerce"
            )

            out = []
            for aid, g in seeds.groupby("activity_id", group_keys=False):
                k = top_k_per_activity
                chosen = g.copy()

                chosen_keys = set(
                    map(
                        tuple,
                        chosen[["doc_index_int", "page_start"]].itertuples(
                            index=False, name=None
                        )
                    )
                )

                pool = df_section[df_section["activity_id"] == aid]

                if get_surrounding_if_not_enough:
                    def grab_after_once(chosen_df: pd.DataFrame):
                        added = False
                        for _, s in chosen_df.sort_values(
                            ["score", "page_start"],
                            ascending=[False, True]
                        ).iterrows():
                            nxt = (
                                pool[
                                    (pool["doc_index_int"] == s["doc_index_int"]) &
                                    (pool["page_start"] > s["page_start"])
                                ]
                                .sort_values("page_start", ascending=True)
                                .head(1)
                            )
                            if not nxt.empty:
                                key = (
                                    int(nxt.iloc[0]["doc_index_int"]),
                                    int(nxt.iloc[0]["page_start"]),
                                )
                                if key not in chosen_keys:
                                    chosen_df = pd.concat(
                                        [chosen_df, nxt],
                                        ignore_index=True
                                    )
                                    chosen_keys.add(key)
                                    return chosen_df, True
                        return chosen_df, added

                    def grab_before_once(chosen_df: pd.DataFrame):
                        added = False
                        for _, s in chosen_df.sort_values(
                            ["score", "page_start"],
                            ascending=[False, True]
                        ).iterrows():
                            prv = (
                                pool[
                                    (pool["doc_index_int"] == s["doc_index_int"]) &
                                    (pool["page_start"] < s["page_start"])
                                ]
                                .sort_values("page_start", ascending=False)
                                .head(1)
                            )
                            if not prv.empty:
                                key = (
                                    int(prv.iloc[0]["doc_index_int"]),
                                    int(prv.iloc[0]["page_start"]),
                                )
                                if key not in chosen_keys:
                                    chosen_df = pd.concat(
                                        [chosen_df, prv],
                                        ignore_index=True
                                    )
                                    chosen_keys.add(key)
                                    return chosen_df, True
                        return chosen_df, added

                    while len(chosen) < k:
                        progressed = False
                        while len(chosen) < k:
                            chosen, added = grab_after_once(chosen)
                            if not added:
                                break
                            progressed = True
                        if len(chosen) >= k:
                            break
                        while len(chosen) < k:
                            chosen, added = grab_before_once(chosen)
                            if not added:
                                break
                            progressed = progressed or added
                        if not progressed:
                            break

                if get_at_least_top_k and len(chosen) < k:
                    remaining = pool[
                        ~pool[["doc_index_int", "page_start"]]
                        .apply(tuple, axis=1)
                        .isin(chosen_keys)
                    ]

                    if not remaining.empty:
                        remaining = remaining.sort_values(
                            ["score", "page_start"],
                            ascending=[False, True],
                            kind="mergesort"
                        )
                        for _, row in remaining.iterrows():
                            key = (int(row["doc_index_int"]), int(row["page_start"]))
                            if key in chosen_keys:
                                continue
                            chosen = pd.concat(
                                [chosen, row.to_frame().T],
                                ignore_index=True
                            )
                            chosen_keys.add(key)
                            if len(chosen) >= k:
                                break

                out.append(chosen.head(k))

            filtered = pd.concat(out, ignore_index=True)

    filtered = filtered.sort_values(
        by=["match_count", "score", "page_start"],
        ascending=[False, False, True],
        kind="mergesort",
    )

    return filtered.to_dict(orient="records")



"""
baseline_enum_1 = [
    "table_of_contents",
    "blank_page",
    "glossary",
    "references",
    "core_activities",
    "theory_of_change",
    "targets",
    "broader_context",
    "preliminary_results",
    "other",
]

baseline_enum_2 = [
    "condensed_summary",
    "sub_activities_outlined",
    "detailed_implementation_plans",
    "broad_objectives",
    "possible_outcomes",
    "quantitative_targets",
    "qualitative_targets",
    "risks_as_word_or_numeric",
    "risks_or_dangers_generally",
    "plans_to_address_key_risks",
    "positive_indicators",
    "progress_reports",
    "similar_cases_outcomes",
    "implementation_context_country",
    "contextual_challenges",
    "financing_details",
    "budget_and_legal",
    "who_implements",
    "whether_part_of_larger_program",
    "partner_identity_or_skill",
    "whether_skin_in_the_game",
    "other_stakeholder_engagement",
    "activity_monitoring_details",
]


evaluation_enum_1 = [
    "glossary",
    "blank_page",
    "table_of_contents",
    "outcome_evaluation",
    "activity_description",
    "references",
    "other",
]
evaluation_enum_2 = [
    "expected_outcomes",
    "deviation_from_plans",
    "preliminary_results",
    "final_outcomes",
    "delays_or_early_completion",
    "over_or_under_spending",
    "overview_as_was_planned",
    "unrelated_to_evaluation",
]

"""

