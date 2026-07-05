#!/usr/bin/env python3
"""
Quick, flexible bundler to visualize exactly what you're sending to the model:

- Page 1..N: the full prompt (wrapped).
- Then, for each attachment:
    * A short info page (uses `docs[i]` if provided; otherwise generic),
    * Followed by the attached PDF slice pages.

Call with either:
  - a list of local PDF slice paths, OR
  - the Gemini `uploaded_files` objects (the function will try to extract a local path).

Dependencies:
  micromamba install reportlab pypdf

Usage:
  from prompt_bundle_quick import bundle_prompt_view
  bundle_prompt_view(prompt, uploaded_files_or_paths, "/tmp/bundle.pdf", docs=docs, open_with_evince=True)
"""
# --- add near the top (imports) ---
import json
from typing import Union




from typing import List, Dict, Any, Iterable, Optional
import os
import shutil
import subprocess
import sys
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit

import json
import pprint
import sys
import asyncio
from datetime import datetime
from typing import Optional, Set, Dict, Any, Tuple
from pathlib import Path

# UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
# if str(UTILS_DIR) not in sys.path:
#     sys.path.insert(0, str(UTILS_DIR))

# from get_all_pages_within_category import load_and_filter_rows

# Optional human-readable IATI category labels
_CATEGORY_LABELS = {
    "A01": "Pre- and post-project impact appraisal",
    "A02": "Objectives / Purpose of activity",
    "A03": "Intended ultimate beneficiaries",
    "A04": "Conditions",
    "A05": "Budget",
    "A06": "Summary information about contract",
    "A07": "Review of project performance and evaluation",
    "A08": "Results, outcomes and outputs",
    "A09": "Memorandum of understanding",
    "A10": "Tender",
    "A11": "Contract",
    "A12": "Activity web page",
}




# --- NEW: per-page info helper ---
def _build_page_info_text(doc_idx: int, page_idx: int, info: Optional[Dict[str, Any]]) -> str:
    """
    Build a short text block to insert before a given page.
    `info` can be any dict; common keys might include:
      - score
      - baseline_category_1/2/3
      - outcome_category_1/2/3
      - any other metadata you want to see
    """
    header = [f"Doc {doc_idx} — Page {page_idx} info"]
    if not info:
        header.append("(no per-page info provided)")
        return "\n".join(header)

    # Reuse make_string() if present, else format simple key: value lines
    try:
        body = make_string(info)
    except Exception:
        body = "".join([f"    {k}: {v}\n" for k, v in (info or {}).items()])

    return "\n".join(header + ["", body.strip()])


def _normalize_results_text(results: Union[str, dict, list]) -> str:
    """Pretty-print results (JSON, list, dict) or pass through string."""
    if isinstance(results, (dict, list)):
        return json.dumps(results, indent=2, ensure_ascii=False)
    if isinstance(results, str):
        s = results.strip()
        # Pretty try: if it's JSON but given as string
        try:
            obj = json.loads(s)
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return s
    return json.dumps(results, indent=2, ensure_ascii=False)

def bundle_prompt_and_results_view(
    prompt: str,
    uploads_or_paths: Iterable[Any],
    out_path: str,
    docs: Optional[List[Dict[str, Any]]] = None,
    results: Union[str, dict, list, None] = None,
    section_title: Optional[str] = None,
    open_with_evince_flag: bool = False,
    wait: bool = False,
    show_unannotated_pages: bool = True,   # NEW
) -> str:
    """
    Like bundle_prompt_view, but also appends a RESULTS section page(s).
    Structure:
      - Optional SECTION TITLE page (if provided),
      - PROMPT pages,
      - For each attachment: INFO page(s) + slice pages,
      - RESULTS page(s) (pretty-printed).
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    slice_paths = _extract_local_paths(list(uploads_or_paths))
    if not slice_paths:
        raise ValueError("Could not extract any local file paths from uploads_or_paths")

    if docs is not None and len(docs) != len(slice_paths):
        raise ValueError(f"docs length ({len(docs)}) must match number of files ({len(slice_paths)})")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    writer = PdfWriter()
    tmp_dir = os.path.dirname(os.path.abspath(out_path))

    def _render_text_to_temp(name: str, text: str) -> str:
        p = os.path.join(tmp_dir, name)
        _wrap_text_to_pdf(text, p)
        return p

    # Optional section title
    if section_title:
        title_pdf = _render_text_to_temp("_section_title_tmp.pdf", section_title)
        for page in PdfReader(title_pdf).pages:
            writer.add_page(page)

    # 1) Prompt
    prompt_pdf = _render_text_to_temp("_prompt_text_tmp.pdf", prompt)
    for page in PdfReader(prompt_pdf).pages:
        writer.add_page(page)

    # # 2) Each doc: info + slice
    # for i, p in enumerate(slice_paths, start=1):
    #     info_text = _build_doc_info_text(i, docs[i - 1] if docs else None)
    #     info_pdf = _render_text_to_temp(f"_doc_{i:03d}_info_tmp.pdf", info_text)
    #     for page in PdfReader(info_pdf).pages:
    #         writer.add_page(page)
    #     for page in PdfReader(p).pages:
    #         writer.add_page(page)
    # 2) Each doc: info + slice
    for i, p in enumerate(slice_paths, start=1):
        info_text = _build_doc_info_text(i, docs[i - 1] if docs else None)
        info_pdf = _render_text_to_temp(f"_doc_{i:03d}_info_tmp.pdf", info_text)
        for page in PdfReader(info_pdf).pages:
            writer.add_page(page)

        # --- CHANGED: insert a tiny info sheet before every page, if provided ---
        per_page_infos = None
        if docs and docs[i - 1] and isinstance(docs[i - 1], dict):
            per_page_infos = docs[i - 1].get("per_page_infos")

        pdf_reader = PdfReader(p)
        for page_idx, page in enumerate(pdf_reader.pages, start=1):
            one_info = None
            if isinstance(per_page_infos, list) and len(per_page_infos) >= page_idx:
                one_info = per_page_infos[page_idx - 1]
            if one_info is None and not show_unannotated_pages:
                continue
            if one_info is not None:
                page_info_pdf = _render_text_to_temp(
                    f"_doc_{i:03d}_page_{page_idx:04d}_info_tmp.pdf",
                    _build_page_info_text(i, page_idx, one_info),
                )
                for ip in PdfReader(page_info_pdf).pages:
                    writer.add_page(ip)
            writer.add_page(page)

    # 3) Results
    if results is not None:
        results_text = "RESULTS (model output):\n\n" + _normalize_results_text(results)
        results_pdf = _render_text_to_temp("_results_tmp.pdf", results_text)
        for page in PdfReader(results_pdf).pages:
            writer.add_page(page)

    # Write and clean
    with open(out_path, "wb") as f_out:
        writer.write(f_out)

    # Best-effort cleanup of small temps we created
    for name in ["_section_title_tmp.pdf", "_prompt_text_tmp.pdf", "_results_tmp.pdf"]:
        try: os.remove(os.path.join(tmp_dir, name))
        except Exception: pass
    for i in range(1, len(slice_paths) + 1):
        try: os.remove(os.path.join(tmp_dir, f"_doc_{i:03d}_info_tmp.pdf"))
        except Exception: pass

    if open_with_evince_flag:
        open_with_evince(out_path, wait=wait)
    return out_path


def _wrap_text_to_pdf(text: str, out_path: str, pagesize=A4, margin=50, font="Helvetica", fontsize=10, leading=1.4):
    """Render arbitrary text into one or more PDF pages and write to out_path."""
    c = canvas.Canvas(out_path, pagesize=pagesize)
    width, height = pagesize
    max_width = width - 2 * margin
    y = height - margin
    c.setFont(font, fontsize)

    for paragraph in text.split("\n"):
        wrapped = simpleSplit(paragraph, font, fontsize, max_width)
        if not wrapped:
            wrapped = [""]
        for line in wrapped:
            if y < margin + fontsize * 2:
                c.showPage()
                c.setFont(font, fontsize)
                y = height - margin
            c.drawString(margin, y, line)
            y -= fontsize * leading
    c.showPage()
    c.save()

def _truthy_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y")
    return bool(v)

def _codes_to_labels(codes_str: str) -> str:
    codes = [c.strip().upper() for c in (codes_str or "").split(",") if c.strip()]
    labels = [_CATEGORY_LABELS.get(c) for c in codes if _CATEGORY_LABELS.get(c)]
    return ", ".join(labels) if labels else "unknown"

def _build_doc_info_text(idx: int, doc: Optional[Dict[str, Any]]) -> str:
    if not doc:
        return (
            f"{idx}. TITLE: (unknown)\n"
            f"   LANGUAGE: unknown\n"
            f"   CATEGORY LABELS: unknown\n"
            f"   PAGES: \n"
            f"   IS_DUPLICATE: False\n"
        )
    title = doc.get("doc_title") or "(untitled)"
    lang = doc.get("language") or "unknown"
    pages = doc.get("pages")
    try:
        pages = int(pages) if pages is not None else ""
    except Exception:
        pages = str(pages or "")
    labels = _codes_to_labels(doc.get("codes", "") or "")
    is_dup = _truthy_bool(doc.get("is_duplicate", False))
    return (
        f"{idx}. TITLE: {title}\n"
        f"   LANGUAGE: {lang}\n"
        f"   CATEGORY LABELS: {labels}\n"
        f"   PAGES: {pages}\n"
        f"   IS_DUPLICATE: {is_dup}\n"
    )

def _extract_local_paths(items: Iterable[Any]) -> List[str]:
    """
    Accepts a list of:
      - strings (paths),
      - file-like objects with `.name`,
      - objects/dicts with `.path` or ['path'].
    Returns list of existing file paths.
    """
    paths: List[str] = []
    for it in items:
        p = None
        if isinstance(it, str):
            p = it
        elif hasattr(it, "path"):
            try:
                p = getattr(it, "path")
            except Exception:
                p = None
        elif hasattr(it, "name"):
            try:
                p = getattr(it, "name")
            except Exception:
                p = None
        elif isinstance(it, dict):
            p = it.get("path") or it.get("name")
        # Normalize and check
        if p:
            p = os.path.abspath(str(p))
            if os.path.exists(p):
                paths.append(p)
    return paths

def open_with_evince(path: str, wait: bool = False) -> None:
    """Open a PDF with Evince (or fall back to xdg-open)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    evince = shutil.which("evince")
    opener = evince or shutil.which("xdg-open")
    if not opener:
        raise RuntimeError("Neither 'evince' nor 'xdg-open' is available on PATH.")
    if wait:
        subprocess.run([opener, path], check=False)
    else:
        subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def make_string(dictionary):
    final_string = ""
    for key, value in dictionary.items():
        final_string += "    " + key + ": " + value + "\n"
    return final_string
