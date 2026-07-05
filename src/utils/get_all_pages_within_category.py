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
# ---------- Input path (edit if your layout differs) ----------
PDF_SCORES_CSV = Path("../../data/pdf_categories_scores.csv")


import os, sys
import pandas as pd
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

# from prompt_bundle_pdf import view_rows_as_pdf

from view_categorization_scores import view_rows_as_pdf

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
            # Not iterable (e.g., numpy scalar); treat as single
            val = _norm(x)
            return {val} if val else None

    def _match_series(series: pd.Series, want: set[str] | None) -> pd.Series:
        if want is None:
            # If no filter provided for this field, it's a pass-through True mask
            return pd.Series(True, index=series.index)
        return series.isin(want)

    df = pd.read_csv(csv_path)

    # Normalize and filter by section
    df["section"] = df["section"].astype(str).str.strip()
    cols = _category_columns(section)
    cat_col, suba_col, subb_col = cols["category"], cols["sub_a"], cols["sub_b"]

    df = df[df["section"].str.lower() == section.strip().lower()].copy()
    if df.empty:
        return []
    # print(f"[DEBUG] Section={section!r}, rows in section: {len(df)}")
    # print(f"[DEBUG] Activities in section: {sorted(df['activity_id'].unique())}")

    # Normalize comparison columns
    for c in (cat_col, suba_col, subb_col):
        df[c] = df[c].astype(str).str.strip().str.lower()

    # Build OR sets
    want_cat = _to_lower_set(category)
    want_a   = _to_lower_set(subcategory_a)
    want_b   = _to_lower_set(subcategory_b)

    # Field-wise masks (OR within each field)
    m_cat = _match_series(df[cat_col], want_cat)
    m_a   = _match_series(df[suba_col], want_a)
    m_b   = _match_series(df[subb_col], want_b)

    # Combine across fields with AND for provided fields only
    # (i.e., if you pass only subcategory_b, filter by B only)
    # mask = pd.Series(True, index=df.index)
    # if want_cat is not None:
    #     mask &= m_cat
    # if want_a is not None:
    #     mask &= m_a
    # if want_b is not None:
    #     mask &= m_b
    # Combine across fields:
    # category must match (if provided) AND (subcategory_a OR subcategory_b) must match if either is provided
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

    # print(f"[DEBUG] want_cat={want_cat}, want_a={want_a}, want_b={want_b}")

    # Save a copy of the mask before score threshold so we can diagnose later
    mask_before_score = mask.copy()

    per_act_cat_sub = (
        mask_before_score.groupby(df["activity_id"]).sum()
    )
    # print("[DEBUG] Rows per activity after category/subcategory filters (before score):")
    # print(per_act_cat_sub.to_dict())
    # Score threshold
    if min_score is not None:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        score_mask = df["score"] >= float(min_score)
        # print(f"[DEBUG] min_score={min_score}")
        # print("[DEBUG] Rows per activity that pass score alone:",
              # score_mask.groupby(df["activity_id"]).sum().to_dict())
        mask &= score_mask
    filtered = df[mask].copy()

    all_acts = set(df["activity_id"].unique())
    acts_with_rows = set(filtered["activity_id"].unique())
    missing_acts = sorted(all_acts - acts_with_rows)

    # print(f"[DEBUG] Activities with at least one row after all filters: {sorted(acts_with_rows)}")
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

    print("[DEBUG] Missing activities reason counts:", reason_counts)
    print("[DEBUG] Total activities:", len(df["activity_id"].unique()))
    print("[DEBUG] Activities with at least one row:", len(acts_with_rows))
    print("[DEBUG] Activities with no rows:", len(missing_acts))

    if filtered.empty:
        return []

    # Compute how many of the (provided) fields matched for sorting
    def _count_match(row) -> int:
        count = 0
        if want_cat is not None and row[cat_col] in want_cat:
            # For contains/regex, recompute with the same rule
            count += 1 if row[cat_col] in want_cat else 0
        if want_a is not None:
            count += 1 if row[suba_col] in want_a else 0
        if want_b is not None:    
            count += 1 if row[subb_col] in want_b else 0
        return count

    filtered["match_count"] = filtered.apply(_count_match, axis=1)

    # Ensure numeric fields for sorting
    filtered["score"] = pd.to_numeric(filtered["score"], errors="coerce")
    filtered["page_start"] = pd.to_numeric(filtered.get("page_start", 0), errors="coerce").fillna(0).astype(int)

    # # Keep top-N by score within each activity_id (optional)
    # if top_k_per_activity is not None:
    #     filtered = (
    #         filtered
    #         .sort_values(["activity_id", "score", "page_start"],
    #                      ascending=[True, False, True],
    #                      kind="mergesort")
    #         .groupby("activity_id", group_keys=False)
    #         .head(top_k_per_activity)
    #     )
    # Keep top-N by score within each activity_id (optional)
    # if top_k_per_activity is not None:
    #     # initial seeds (true matches only)
    #     seeds = (
    #         filtered
    #         .sort_values(["activity_id", "score", "page_start"],
    #                      ascending=[True, False, True],
    #                      kind="mergesort")
    #         .groupby("activity_id", group_keys=True)
    #         .head(top_k_per_activity)
    #     )

    #     if not get_surrounding_if_not_enough:
    #         filtered = seeds
    #     else:
    #         # candidates for neighbors are *all rows in this section* (even non-matches)
    #         # make sure their keys are numeric for ordering
    #         df_section = df.copy()
    #         df_section["page_start"] = pd.to_numeric(df_section.get("page_start", 0), errors="coerce").fillna(0).astype(int)
    #         df_section["doc_index_int"] = pd.to_numeric(df_section.get("doc_index_int"), errors="coerce").astype("Int64")

    #         out = []
    #         for aid, g in seeds.groupby("activity_id", group_keys=False):
    #             k = top_k_per_activity
    #             chosen = g.copy()
    #             # quick set of (doc,page) we've already included
    #             chosen_keys = set(map(tuple, chosen[["doc_index_int","page_start"]].itertuples(index=False, name=None)))
    #             pool = df_section[df_section["activity_id"] == aid]

    #             # helper to try grabbing one neighbor "after" per chosen row (round-robin)
    #             def grab_after_once(chosen):
    #                 added = False
    #                 for _, s in chosen.sort_values(["score","page_start"], ascending=[False, True]).iterrows():
    #                     nxt = (
    #                         pool[(pool["doc_index_int"] == s["doc_index_int"]) & (pool["page_start"] > s["page_start"])]
    #                         .sort_values("page_start", ascending=True)
    #                         .head(1)
    #                     )
    #                     if not nxt.empty:
    #                         key = (int(nxt.iloc[0]["doc_index_int"]), int(nxt.iloc[0]["page_start"]))
    #                         if key not in chosen_keys:
    #                             chosen = pd.concat([chosen, nxt], ignore_index=True)
    #                             chosen_keys.add(key)
    #                             return chosen, True
    #                 return chosen, added

    #             # helper to try grabbing one neighbor "before" per chosen row
    #             def grab_before_once(chosen):
    #                 added = False
    #                 for _, s in chosen.sort_values(["score","page_start"], ascending=[False, True]).iterrows():
    #                     prv = (
    #                         pool[(pool["doc_index_int"] == s["doc_index_int"]) & (pool["page_start"] < s["page_start"])]
    #                         .sort_values("page_start", ascending=False)
    #                         .head(1)
    #                     )
    #                     if not prv.empty:
    #                         key = (int(prv.iloc[0]["doc_index_int"]), int(prv.iloc[0]["page_start"]))
    #                         if key not in chosen_keys:
    #                             chosen = pd.concat([chosen, prv], ignore_index=True)
    #                             chosen_keys.add(key)
    #                             return chosen, True
    #                 return chosen, added

    #             # fill with AFTER pages first, then BEFORE pages, until k or stuck
    #             while len(chosen) < k:
    #                 progressed = False
    #                 while len(chosen) < k:
    #                     chosen, added = grab_after_once(chosen)
    #                     if not added: break
    #                     progressed = True
    #                 if len(chosen) >= k: break
    #                 while len(chosen) < k:
    #                     chosen, added = grab_before_once(chosen)
    #                     if not added: break
    #                     progressed = progressed or added
    #                 if not progressed:
    #                     break

    #             out.append(chosen.head(k))

    #         filtered = pd.concat(out, ignore_index=True)
    # Keep top-N by score within each activity_id (optional)
    if top_k_per_activity is not None:
        # initial seeds (true matches only)
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
        print(f"[DEBUG] Activities with filtered rows but no seeds: {acts_with_no_seeds}")


        # no neighbors, no padding: old behavior
        if not get_surrounding_if_not_enough and not get_at_least_top_k:
            filtered = seeds
        else:
            # candidates for neighbors / padding are *all rows in this section*
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

                # 1) existing neighbor behavior (if requested)
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

                    # fill with AFTER pages first, then BEFORE pages, until k or stuck
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

                # 2) if still not enough and get_at_least_top_k=True,
                #    pad with remaining pages of this activity (even non-matches),
                #    preferring higher score.
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
            # print("[DEBUG] Final activities returned:",
            #       sorted(set(filtered["activity_id"].unique())))

            filtered = pd.concat(out, ignore_index=True)

    # Sort: more matched fields first, higher score first, earlier page first
    filtered = filtered.sort_values(
        by=["match_count", "score", "page_start"],
        ascending=[False, False, True],
        kind="mergesort",
    )

    return filtered.to_dict(orient="records")

def print_activity_coverage(
    rows: List[Dict[str, Any]],
    csv_path: Path = PDF_SCORES_CSV,
) -> tuple[int, int, float, Optional[float], Optional[float]]:
    """
    Print coverage of unique activity_ids represented in `rows`, plus:
      - average number of unique documents per activity (within `rows`)
      - average number of pages per document (within `rows`)

    If all rows share the same `section`, the denominator is the number of
    unique activities in that section of `csv_path`. Otherwise, the denominator
    is the number of unique activities across the entire CSV.

    Returns:
        (matched_count, denom_count, fraction, avg_docs_per_activity, avg_pages_per_document)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing PDF scores CSV: {csv_path}")

    # ---- Load denominator universe -------------------------------------------------
    df_all = pd.read_csv(csv_path)

    given_sections = {
        ("" if r.get("section") is None else str(r.get("section")).strip().lower())
        for r in rows
        if r.get("section") is not None
    }
    if len(given_sections) == 1:
        only_section = next(iter(given_sections))
        df_den = df_all[df_all["section"].astype(str).str.strip().str.lower() == only_section]
        denom_scope = f"section='{only_section}'"
    else:
        df_den = df_all
        denom_scope = "all sections"

    denom_ids = {
        x for x in df_den.get("activity_id", pd.Series(dtype=object)).dropna().unique()
    }
    denom_count = len(denom_ids)

    # ---- Numerator from provided rows ---------------------------------------------
    matched_ids = {r.get("activity_id") for r in rows if r.get("activity_id") is not None}
    matched_in_scope = matched_ids & denom_ids
    matched_count = len(matched_in_scope)
    fraction = (matched_count / denom_count) if denom_count else 0.0

    # ---- Averages computed within the provided rows --------------------------------
    avg_docs_per_activity: Optional[float] = None
    avg_pages_per_document: Optional[float] = None

    if rows:
        df_rows = pd.DataFrame(rows)

        # Resolve document and page fields with sensible fallbacks
        doc_field = next(
            (c for c in ["doc_index_int", "doc_index", "document_id", "cached_file"] if c in df_rows.columns),
            None,
        )
        page_field = "page_start" if "page_start" in df_rows.columns else ("page" if "page" in df_rows.columns else None)

        # Normalize types a bit
        if doc_field is not None:
            # preserve uniqueness even if mixed types; cast to string for stable nunique
            df_rows[doc_field] = df_rows[doc_field].astype(str)
        if page_field is not None:
            df_rows[page_field] = pd.to_numeric(df_rows[page_field], errors="coerce")

        # Compute averages only if we have the necessary fields
        if ("activity_id" in df_rows.columns) and (doc_field is not None):
            docs_per_activity = (
                df_rows.dropna(subset=["activity_id"])
                       .groupby("activity_id")[doc_field]
                       .nunique()
            )
            if len(docs_per_activity) > 0:
                avg_docs_per_activity = float(docs_per_activity.mean())

            if page_field is not None:
                pages_per_doc = (
                    df_rows.dropna(subset=["activity_id", page_field])
                           .groupby(["activity_id", doc_field])[page_field]
                           .nunique()
                )
                if len(pages_per_doc) > 0:
                    avg_pages_per_document = float(pages_per_doc.mean())

    # ---- Pretty print --------------------------------------------------------------
    print("Activity coverage")
    print(f"  Scope:                           {denom_scope}")
    print(f"  Matched unique activities:       {matched_count}")
    print(f"  Denominator (unique activities): {denom_count}")
    print(f"  Fraction:                        {matched_count}/{denom_count} = {fraction:.3f} ({fraction*100:.1f}%)")

    if avg_docs_per_activity is not None:
        print(f"  Avg unique docs per activity:    {avg_docs_per_activity:.3f}  (within provided rows)")
    else:
        print(f"  Avg unique docs per activity:    n/a (missing columns)")

    if avg_pages_per_document is not None:
        print(f"  Avg pages per document:          {avg_pages_per_document:.3f} (within provided rows)")
    else:
        print(f"  Avg pages per document:          n/a (missing columns)")

    return matched_count, denom_count, fraction, avg_docs_per_activity, avg_pages_per_document



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
if __name__ == "__main__":

    print("Loading finance / budget pages...")

    rows_financial = load_and_filter_rows(
        section="Baseline",
        subcategory_a=["financing_details", "budget_and_legal"],
        subcategory_b=["financing_details", "budget_and_legal"],
        min_score=7,
        top_k_per_activity=5,
        get_surrounding_if_not_enough=True,
        get_at_least_top_k=True,
    )

    print(f"Loaded {len(rows_financial)} rows")

    if not rows_financial:
        print("No rows found, exiting.")
    else:

        # Optional: quick coverage sanity check
        activity_ids = {r.get("activity_id") for r in rows_financial if r.get("activity_id")}
        print(f"Unique activities covered: {len(activity_ids)}")

        # This opens Evince with page slices + metadata overlays
        view_rows_as_pdf(rows_financial)
        print_activity_coverage(rows_summary)


    print("this is a utility module, not intended to be called directly.")
    print("Some ways to use it can be found in the comments below this printout in the code.")
    # # delays
    # # Activity coverage
    # #   Scope:                           section='outcome'
    # #   Matched unique activities:       197
    # #   Denominator (unique activities): 589
    # #   Fraction:                        197/589 = 0.334 (33.4%)
    # #   Avg unique docs per activity:    1.152  (within provided rows)
    # #   Avg pages per document:          1.326 (within provided rows)
    # # OVERVIEW: looks like if I supply a boolean was_delayed, an optional activity_end_date (year and optionally month), or an optional amount_delayed (optional years and optional months)

    # rows_summary = load_and_filter_rows(
    #     section="Outcome",
    #     subcategory_a=["delays_or_early_completion"],
    #     subcategory_b=["delays_or_early_completion"],
    #     top_k_per_activity=5,  # adjust as needed; set to None to disable
    #     min_score=7,
    # )
    # view_rows_as_pdf(rows_summary)
    # print("delays")
    # print_activity_coverage(rows_summary)


    # # overspending underspending
    # # Activity coverage
    # #   Scope:                           section='outcome'
    # #   Matched unique activities:       283
    # #   Denominator (unique activities): 589
    # #   Fraction:                        283/589 = 0.480 (48.0%)
    # #   Avg unique docs per activity:    1.350  (within provided rows)
    # #   Avg pages per document:          1.442 (within provided rows)

    # rows_summary = load_and_filter_rows(
    #     section="Outcome",
    #     subcategory_a=["over_or_under_spending"],
    #     subcategory_b=["over_or_under_spending"],
    #     top_k_per_activity=5,  # adjust as needed; set to None to disable
    #     min_score=7,
    # )
    
    # # plan deviatons
    # # Activity coverage
    # #   Scope:                           section='outcome'
    # #   Matched unique activities:       512
    # #   Denominator (unique activities): 589
    # #   Fraction:                        512/589 = 0.869 (86.9%)
    # #   Avg unique docs per activity:    1.709  (within provided rows)
    # #   Avg pages per document:          2.680 (within provided rows)
    # # view_rows_as_pdf(rows_summary)
    # print("\n\noverspending underspending")
    # print_activity_coverage(rows_summary)

    # rows_summary = load_and_filter_rows(
    #     section="Outcome",
    #     subcategory_a=["deviation_from_plans"],
    #     subcategory_b=["deviation_from_plans"],
    #     top_k_per_activity=5,  # adjust as needed; set to None to disable
    #     min_score=7,
    # )
    # # view_rows_as_pdf(rows_summary)
    # print("\n\nplan deviatons")
    # print_activity_coverage(rows_summary)

    # bundle_prompt_and_results_view(
    #     prompt=prompt_text,
    #     uploads_or_paths=[str(full_pdf)],
    #     out_path=str(out_path),
    #     docs=docs_meta,
    #     # results=results_block,
    #     results=None,
    #     section_title="IATI CATEGORIZATION VIEW",
    #     open_with_evince_flag=True,   # open right away
    #     # wait=False,              # don't block
    # )
    # # quit()



    # # The rows_summary below seems to be quite good for identifying pages which summarize key objectives and activities. Sometimes, the next page would be useful as it gets cut off.
    # rows_summary = load_and_filter_rows(
    #     section="Baseline",
    #     subcategory_a=["condensed_summary","broad_objectives"],
    #     subcategory_b=["condensed_summary","broad_objectives"],
    #     top_k_per_activity=5,  # adjust as needed; set to None to disable
    #     min_score=7,
    # )
    # _print_records(rows_summary)
    # print_activity_coverage(rows_summary)
    # # view_rows_as_pdf(rows_summary)
    # # Scope:                           section='baseline'
    # # Matched unique activities:       576
    # # Denominator (unique activities): 626
    # # Fraction:                        576/626 = 0.920 (92.0%)
    # # Avg unique docs per activity:    1.580  (within provided rows)
    # # Avg pages per document:          2.211 (within provided rows)

    # # quit()



    # # Overall, I was happy, the majority seem relevant for assessing involvment or skill levels/experience
    # rows_partner_abilities = load_and_filter_rows(
    #     section="Baseline",
    #     subcategory_a=["partner_identity_or_skill","whether_skin_in_the_game", "other_stakeholder_engagement","who_implements",],
    #     subcategory_b=["partner_identity_or_skill","whether_skin_in_the_game", "other_stakeholder_engagement","who_implements",],
    #     # subcategory_a=["who_implements"],
    #     # subcategory_b=["who_implements"],
    #     top_k_per_activity=3,  # adjust as needed; set to None to disable
    #     min_score=7,
    # )
    # _print_records(rows_partner_abilities)
    # # print_activity_coverage(rows_partner_abilities)
    # # Activity coverage
    # #   Scope:                           section='baseline'
    # #   Matched unique activities:       471
    # #   Denominator (unique activities): 626
    # #   Fraction:                        471/626 = 0.752 (75.2%)
    # #   Avg unique docs per activity:    1.223  (within provided rows)
    # #   Avg pages per document:          2.009 (within provided rows)
    # view_rows_as_pdf(rows_partner_abilities)
    # quit()

    # generally looks good, unsurprisingly
    # rows_projections = load_and_filter_rows(
    #     section="Baseline",
    #     subcategory_a=["possible_outcomes","quantitative_targets","qualitative_targets"],
    #     subcategory_b=["possible_outcomes","quantitative_targets","qualitative_targets"],
    #     top_k_per_activity=5,  # adjust as needed; set to None to disable
    #     min_score=7,
    # )
    # print_activity_coverage(rows_projections)
    # # view_rows_as_pdf(rows_projections)
    # # Scope:                           section='baseline'
    # # Matched unique activities:       439
    # # Denominator (unique activities): 626
    # # Fraction:                        439/626 = 0.701 (70.1%)
    # # Avg unique docs per activity:    1.146  (within provided rows)
    # # Avg pages per document:          3.288 (within provided rows)

    # quit()

    # rows_risks = load_and_filter_rows(
    #     section="Baseline",
    #     subcategory_a=["risks_as_word_or_numeric", "risks_or_dangers_generally","contextual_challenges"],
    #     subcategory_b=["risks_as_word_or_numeric", "risks_or_dangers_generally","contextual_challenges"],
    #     top_k_per_activity=5,  # adjust as needed; set to None to disable
    #     min_score=7,
    # )
    # print("\n\nrows_risks")
    # print_activity_coverage(rows_risks)
    # # rows_risks
    # # Activity coverage
    # #   Scope:                           section='baseline'
    # #   Matched unique activities:       589
    # #   Denominator (unique activities): 626
    # #   Fraction:                        589/626 = 0.941 (94.1%)
    # #   Avg unique docs per activity:    1.630  (within provided rows)
    # #   Avg pages per document:          2.478 (within provided rows)


    # rows_country_context = load_and_filter_rows(
    #     section="Baseline",
    #     subcategory_a="implementation_context_country",
    #     subcategory_b="implementation_context_country",
    #     min_score=7,
    #     top_k_per_activity=5,  # adjust as needed; set to None to disable
    # )
    # print("\n\nrows_country_context")
    # print_activity_coverage(rows_country_context)
    # # quit()
    # # rows_country_context
    # # Activity coverage
    # #   Scope:                           section='baseline'
    # #   Matched unique activities:       598
    # #   Denominator (unique activities): 626
    # #   Fraction:                        598/626 = 0.955 (95.5%)
    # #   Avg unique docs per activity:    1.599  (within provided rows)
    # #   Avg pages per document:          2.537 (within provided rows)

    # rows_possibilities = load_and_filter_rows(
    #     section="Baseline",
    #     subcategory_a=["positive_indicators", "plans_to_address_key_risks"],
    #     subcategory_b=["positive_indicators", "plans_to_address_key_risks"],
    #     min_score=7,
    #     top_k_per_activity=3,  # adjust as needed; set to None to disable
    # )

    # # these looked good, all looked at seemed like key risks
    # print("\n\nrows_possibilities")
    # print_activity_coverage(rows_possibilities)
    # view_rows_as_pdf(rows_possibilities)
    # rows_possibilities
    # Activity coverage
    #   Scope:                           section='baseline'
    #   Matched unique activities:       427
    #   Denominator (unique activities): 626
    #   Fraction:                        427/626 = 0.682 (68.2%)
    #   Avg unique docs per activity:    1.274  (within provided rows)
    #   Avg pages per document:          1.781 (within provided rows)
    # quit()

    # rows_similar_cases = load_and_filter_rows(
    #     section="Baseline",
    #     subcategory_a=["similar_cases_outcomes","progress_reports"],
    #     subcategory_b=["similar_cases_outcomes","progress_reports"],
    #     top_k_per_activity=5,  # adjust as needed; set to None to disable
    #     min_score=3,
    # )
    # _print_records(rows_similar_cases)
    # print_activity_coverage(rows_similar_cases)
    # view_rows_as_pdf(rows_similar_cases)
    # Activity coverage
    #   Scope:                           section='baseline'
    #   Matched unique activities:       85
    #   Denominator (unique activities): 626
    #   Fraction:                        85/626 = 0.136 (13.6%)
    #   Avg unique docs per activity:    1.047  (within provided rows)
    #   Avg pages per document:          1.281 (within provided rows)
    # switching to 3 score instead of 7 improves to 19%.
    # adding in "progress reports", set grade to 5 minimum
    # About half of these are not very clear or require a lot of context to understand whether they would help forecasts
    # another half are reasonably helpful
    # but I could see an LLM being confused by some of these tables of financial statements - those should be excluded
    # Generally, makes sense to combine progress reports and similar cases. Progress reports seems geenrally more problematic?
    # Activity coverage
    #   Scope:                           section='baseline'
    #   Matched unique activities:       227
    #   Denominator (unique activities): 626
    #   Fraction:                        227/626 = 0.363 (36.3%)
    #   Avg unique docs per activity:    1.084  (within provided rows)

    # _print_records(rows)
