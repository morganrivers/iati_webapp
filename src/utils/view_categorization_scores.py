#!/usr/bin/env python3
"""
Build simple viewer PDFs for IATI categorizations.

- Loads ../../data/pdf_category_scores.csv
- For each unique (activity_id, cached_file), it:
    * Prepend pages with a compact "SCRATCHPAD" and "GRADES/SCORES" summary,
    * Appends the FULL original PDF,
    * Saves to ../../data/viewer_categorizations/{activity_id}-{doc_title}.pdf
    * Opens the result in Evince.

Requires:
  micromamba install reportlab pypdf pandas
  (and the helper module `prompt_bundle_quick.py` available on PYTHONPATH)

Note:
  The original PDFs are expected under ../../data/iati_all_pdfs/
"""

import os, sys
import pandas as pd
from pathlib import Path
import pprint

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))


from prompt_bundle_pdf import bundle_prompt_and_results_view

# ---- config (relative to this script) ----
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / "../../data").resolve()
CSV_PATH = DATA_DIR / "pdf_categories_scores.csv"
PDFS_DIR = DATA_DIR / "iati_all_pdfs"
OUT_DIR = DATA_DIR / "viewer_categorizations"

def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "-").replace("\\", "-")
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_", " ", ".", ","))
    s = "_".join(s.split())
    return s[:120] if len(s) > 120 else s

def build_prompt_block(rows: pd.DataFrame) -> str:
    """
    Build the text that goes before the PDF:
      - A compact "SECTION TITLE"
      - A 'PROMPT' (we'll use a quick meta header)
      - We'll leverage the function's 'results' for a JSON-like grade table.
    """
    r0 = rows.iloc[0]
    header = []
    header.append(f"ACTIVITY: {r0.get('activity_id','')}")
    header.append(f"TITLE: {r0.get('activity_title','')}")
    header.append(f"DOCUMENT: {r0.get('doc_title','')}")
    header.append(f"CACHED FILE: {r0.get('cached_file','')}")
    header.append("")
    header.append("SCRATCHPAD NOTES (per row):")
    header.append("-" * 60)

    # include scratchpad snippets per row (most recent first)
    for _, r in rows.sort_values("run_timestamp_iso", ascending=False).iterrows():
        tag = f"[{r.get('run_timestamp_iso','')}] section={r.get('section','')} p{r.get('page_start','')}–{r.get('page_end','')}, model={r.get('model_name','')}"
        scratch_val = r.get("scratchpad")
        if pd.isna(scratch_val):
             scratch = "(no scratchpad)"
        else:
            scratch = str(scratch_val).strip() if scratch_val else "(no scratchpad)"
        header.append(tag)
        header.append(scratch)
        header.append("-" * 60)

    return "\n".join(header)

def build_results_block(rows: pd.DataFrame):
    """
    A compact dict/list that the helper will pretty-print on 'RESULTS' pages.
    We capture grades/scores and category picks.
    """
    results = []
    for _, r in rows.iterrows():
        results.append({
            "run_timestamp_iso": r.get("run_timestamp_iso"),
            "section": r.get("section"),
            "page_range": f"{r.get('page_start','')}–{r.get('page_end','')}",
            "model": r.get("model_name"),
            "score": r.get("score"),
            "baseline_cats": [r.get("baseline_category_1"), r.get("baseline_category_2"), r.get("baseline_category_3")],
            "outcome_cats": [r.get("outcome_category_1"), r.get("outcome_category_2"), r.get("outcome_category_3")],
        })
    return results

def doc_pages_total(rows: pd.DataFrame) -> int:
    # prefer the non-null first value
    vals = rows["pdf_pages_total"].dropna()
    return int(vals.iloc[0]) if not vals.empty else 0
# --- add this helper near your other helpers ---
def build_per_page_infos(rows: pd.DataFrame, total_pages: int):
    """
    Returns a list of length total_pages (1-based pages), where each entry is
    either a dict of per-page info or None. We pick the most recent row per page
    if duplicates exist.
    """
    # Keep only rows that clearly map to a single page
    r = rows.copy()
    # Prefer most recent run per page
    r = r.sort_values("run_timestamp_iso", ascending=False)

    # Map page -> dict
    page_map = {}
    for _, x in r.iterrows():
        try:
            pstart = int(x.get("page_start"))
            pend = int(x.get("page_end"))
        except Exception:
            continue
        if pstart != pend:  # skip ranges; this viewer is per-page
            continue
        if pstart < 1 or pstart > total_pages:
            continue
        if pstart in page_map:
            continue  # already took the most recent for this page

        scratch_val = x.get("scratchpad")
        scratchpad_text = "" if pd.isna(scratch_val) else str(scratch_val).strip()

        page_map[pstart] = {
            "page_number": pstart,
            "section": x.get("section"),
            "model": x.get("model_name"),
            "score": x.get("score"),
            "baseline_category_1": x.get("baseline_category_1"),
            "baseline_category_2": x.get("baseline_category_2"),
            "baseline_category_3": x.get("baseline_category_3"),
            "outcome_category_1": x.get("outcome_category_1"),
            "outcome_category_2": x.get("outcome_category_2"),
            "outcome_category_3": x.get("outcome_category_3"),
            "has_quantitative_targets": x.get("has_quantitative_targets"),
            "has_quantitative_outcomes": x.get("has_quantitative_outcomes"),
            "has_overall_ratings": x.get("has_overall_ratings"),
            "scratchpad": scratchpad_text,
        }

    # Build dense list 1..total_pages
    out = []
    for i in range(1, total_pages + 1):
        out.append(page_map.get(i))  # None where missing
    return out


def view_rows_as_pdf(rows):
    df = pd.DataFrame(rows)
    # quit()
    # Group by each unique full PDF (per activity/doc file)
    group_cols = ["activity_id","doc_title","cached_file"]
    total_documents_viewed = 0
    if len(rows)>0:
        for (activity_id, doc_title, cached_file), rows in df.groupby(group_cols):
            # pprint.pprint("rows")
            total_documents_viewed += 1
            full_pdf = (PDFS_DIR / str(cached_file)).resolve()
            if not full_pdf.exists():
                raise FileNotFoundError(f"PDF not found: {full_pdf}")

            # Build preface pages content
            prompt_text = build_prompt_block(rows)
            results_block = build_results_block(rows)

            # # Build preface pages content
            # prompt_text = build_prompt_block(rows)
            # results_block = build_results_block(rows)

            total_pages = doc_pages_total(rows)
            per_page_infos = build_per_page_infos(rows, total_pages)

            print("\n\ndoc_title")
            print(doc_title)
            print("cached_file")
            print(cached_file)
            print("activity_id")
            print(activity_id)
            # print("activity_description")
            # print(activity_description)

            # Minimal doc info for the Info page just before the full PDF
            docs_meta = [{
                "doc_title": str(doc_title) if pd.notna(doc_title) else "(untitled)",
                "language": "unknown",
                "codes": "",
                "pages": total_pages,
                "is_duplicate": False,
                # >>> this is the key your bundler looks for <<<
                "per_page_infos": per_page_infos,
            }]


            # # Minimal doc info for the Info page just before the full PDF
            # docs_meta = [{
            #     "doc_title": str(doc_title) if pd.notna(doc_title) else "(untitled)",
            #     "language": "unknown",
            #     "codes": "",  # no IATI codes provided here
            #     "pages": doc_pages_total(rows),
            #     "is_duplicate": False,
            # }]

            # Output path
            out_name = f"{activity_id}-{safe_name(str(doc_title) or 'doc')}.pdf"
            out_path = (OUT_DIR / out_name).resolve()

            # Build the bundle with:
            #   Section title, Prompt (scratchpad pages), then Info+FULL PDF, then Results (grades/scores)
            bundle_prompt_and_results_view(
                prompt=prompt_text,
                uploads_or_paths=[str(full_pdf)],
                out_path=str(out_path),
                docs=docs_meta,
                # results=results_block,
                results=None,
                section_title="IATI CATEGORIZATION VIEW",
                open_with_evince_flag=True,   # open right away
                show_unannotated_pages=False,   # NEW
            )
            input("press enter to continue")

            if total_documents_viewed >= 10:
                quit()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Ensure essential cols exist (kept minimal and explicit)
    needed = [
        "activity_id","activity_title","doc_title","cached_file","pdf_pages_total",
        "run_timestamp_iso","section","page_start","page_end","model_name",
        "scratchpad","score",
        "baseline_category_1","baseline_category_2","baseline_category_3",
        "outcome_category_1","outcome_category_2","outcome_category_3",
    ]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    # Group by each unique full PDF (per activity/doc file)
    group_cols = ["activity_id","doc_title","cached_file"]
    total_documents_viewed = 0
    for (activity_id, doc_title, cached_file), rows in df.groupby(group_cols):
        total_documents_viewed += 1
        full_pdf = (PDFS_DIR / str(cached_file)).resolve()
        if not full_pdf.exists():
            raise FileNotFoundError(f"PDF not found: {full_pdf}")

        # Build preface pages content
        prompt_text = build_prompt_block(rows)
        results_block = build_results_block(rows)

        # # Build preface pages content
        # prompt_text = build_prompt_block(rows)
        # results_block = build_results_block(rows)

        total_pages = doc_pages_total(rows)
        per_page_infos = build_per_page_infos(rows, total_pages)

        # Minimal doc info for the Info page just before the full PDF
        docs_meta = [{
            "doc_title": str(doc_title) if pd.notna(doc_title) else "(untitled)",
            "language": "unknown",
            "codes": "",
            "pages": total_pages,
            "is_duplicate": False,
            # >>> this is the key your bundler looks for <<<
            "per_page_infos": per_page_infos,
        }]


        # # Minimal doc info for the Info page just before the full PDF
        # docs_meta = [{
        #     "doc_title": str(doc_title) if pd.notna(doc_title) else "(untitled)",
        #     "language": "unknown",
        #     "codes": "",  # no IATI codes provided here
        #     "pages": doc_pages_total(rows),
        #     "is_duplicate": False,
        # }]

        # Output path
        out_name = f"{activity_id}-{safe_name(str(doc_title) or 'doc')}.pdf"
        out_path = (OUT_DIR / out_name).resolve()

        # Build the bundle with:
        #   Section title, Prompt (scratchpad pages), then Info+FULL PDF, then Results (grades/scores)
        bundle_prompt_and_results_view(
            prompt=prompt_text,
            uploads_or_paths=[str(full_pdf)],
            out_path=str(out_path),
            docs=docs_meta,
            # results=results_block,
            results=None,
            section_title="IATI CATEGORIZATION VIEW",
            open_with_evince_flag=True,   # open right away
            # wait=False,              # don't block
        )

        if total_documents_viewed >= 10:
            quit()

if __name__ == "__main__":
    main()
