from __future__ import annotations
# modules/rag_bm25.py
# Retrieval for RAG synthesis, with two backends:
#
#   USE_VECTOR_RAG = False  →  BM25 (default; no API cost, no heavy deps)
#   USE_VECTOR_RAG = True   →  Gemini embedding-001 cosine similarity
#                              (better semantic matching; needs GEMINI_API_KEY)
#
# Toggle via env var:  RAG_USE_VECTOR=1  or set USE_VECTOR_RAG directly below.
# All heavy imports (rank_bm25, google-genai, numpy) are lazy — zero module-load cost.

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Backend selector ──────────────────────────────────────────────────────────
USE_VECTOR_RAG: bool = False

# PDF→txt conversion (lightweight, no model weights).
# extract_pdfs_as_txt must be on sys.path (UTILS_DIR is added by run_rag_forecast.py).
from extract_pdfs_as_txt import normalized_basename, process_one

# ── Regex constants (copied verbatim from short_well_or_badly_rag.py) ─────────

_PHRASE_LABEL_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[-*•]\s*)?(?:PHRASE|SEARCH\s*PHRASE|QUERY\s*PHRASE|P)\s*([1-5])\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)
_NUM_ITEM_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(?:[-*•]\s*)?([1-5])\s*[\.\)\-:]\s*(.+?)\s*$")
_SECTION_HDR_RE = re.compile(
    r"^\s*#{0,3}\s*(SEARCH\s+PHRASES|QUERY\s+PHRASES|PHRASES)\s*:?\s*$",
    re.IGNORECASE,
)
_SKIP_PREFIX_RE = re.compile(
    r"^\s*(?:reasoning|explanation|justification|rationale)\s*:\s*",
    re.IGNORECASE,
)
_SEARCH_PHRASE_INLINE_RE = re.compile(r"^\s*search\s*phrase\s*:\s*(.+?)\s*$", re.IGNORECASE)


# ── Small helpers ─────────────────────────────────────────────────────────────

def is_de1_activity(activity_id: str) -> bool:
    s = str(activity_id)
    return s.startswith("DE-1-") or s.startswith("DE-1")




def _clean_phrase(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*[-*•]\s+", "", s).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    s = re.sub(r"\s*#{2,}\s*$", "", s).strip()
    s = re.sub(r"^\*{1,3}(.+?)\*{1,3}$", r"\1", s).strip()
    s = re.sub(r"^\_{1,3}(.+?)\_{1,3}$", r"\1", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_txt_for_pdf(pdf_path: Path, txt_output_dir: Path, allow_ocr: bool = True, activity_id: str = "") -> Path:
    txt_output_dir.mkdir(parents=True, exist_ok=True)
    base = normalized_basename(pdf_path)
    # If multiple uploads share the same filename (e.g. "uploaded.pdf"), prefix with
    # the activity_id so each gets its own .txt file instead of all reusing the first.
    if activity_id:
        base = f"{activity_id}__{base}"
    txt_path = txt_output_dir / f"{base}.txt"

    def is_empty(p: Path) -> bool:
        if not p.exists():
            return True
        if p.stat().st_size < 50:
            return True
        return not p.read_text(encoding="utf-8", errors="ignore").strip()

    if is_empty(txt_path):
        if txt_path.exists():
            txt_path.unlink()
        status, msg = process_one(pdf_path, txt_output_dir, allow_ocr, only_newer=False, out_name=base)
        print(msg)
        if status != "OK":
            print(f"Warning: conversion for {pdf_path} returned status={status}")

    return txt_path


# ── Phrase parsing (copied verbatim from short_well_or_badly_rag.py) ──────────

def parse_phrases(text: str, *, n: int = 5) -> List[str]:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    ps: Dict[int, str] = {}

    for idx, line in enumerate(lines):
        m = _PHRASE_LABEL_RE.match(line)
        if not m:
            continue
        i = int(m.group(1))
        raw = _clean_phrase(m.group(2))
        if raw:
            ps.setdefault(i, raw)
            continue
        look = []
        for j in range(idx + 1, min(idx + 6, len(lines))):
            cand = lines[j].strip()
            if not cand:
                continue
            if _SKIP_PREFIX_RE.match(cand):
                break
            msp = _SEARCH_PHRASE_INLINE_RE.match(cand)
            if msp:
                cand = msp.group(1)
            cand = _clean_phrase(cand)
            if not cand:
                continue
            look.append(cand)
            break
        if look:
            ps.setdefault(i, look[0])

    if all(k in ps for k in range(1, n + 1)):
        return [ps[k] for k in range(1, n + 1)]

    hdr_idxs = [i for i, line in enumerate(lines) if _SECTION_HDR_RE.match(line)]
    for hpos_i, hpos in enumerate(hdr_idxs):
        end = hdr_idxs[hpos_i + 1] if (hpos_i + 1) < len(hdr_idxs) else len(lines)
        block = lines[hpos + 1 : end]
        ps2: Dict[int, str] = {}
        for bl in block:
            m = _NUM_ITEM_RE.match(bl)
            if not m:
                continue
            i = int(m.group(1))
            val = _clean_phrase(m.group(2))
            if val:
                ps2.setdefault(i, val)
        if all(k in ps2 for k in range(1, n + 1)):
            return [ps2[k] for k in range(1, n + 1)]

    ps3: Dict[int, str] = {}
    for line in lines:
        m = _NUM_ITEM_RE.match(line)
        if not m:
            continue
        i = int(m.group(1))
        val = _clean_phrase(m.group(2))
        if not val:
            continue
        if i == 1 and ps3:
            ps3 = {}
        ps3.setdefault(i, val)
        if all(k in ps3 for k in range(1, n + 1)):
            return [ps3[k] for k in range(1, n + 1)]

    bullets: List[str] = []
    for line in lines:
        if re.match(r"^\s*[-*•]\s+", line):
            v = _clean_phrase(line)
            if v:
                bullets.append(v)
    if len(bullets) >= n:
        return bullets[:n]

    if ps:
        return [ps[k] for k in sorted(ps.keys())][:n]

    return []


# ── BM25 retrieval ────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    if not text.strip():
        return []
    step = max(1, chunk_size - overlap)
    return [
        text[i : i + chunk_size]
        for i in range(0, len(text), step)
        if text[i : i + chunk_size].strip()
    ]


def _bm25_retrieve(
    chunks: List[str],
    phrases: List[str],
    top_k: int = 6,
    snippet_chars: int = 900,
) -> Dict[str, List[str]]:
    """BM25 keyword retrieval over text chunks. rank_bm25 is lazy-imported."""
    if not chunks:
        return {ph: [] for ph in phrases}

    from rank_bm25 import BM25Okapi  # lazy: no cost at module load time

    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    out: Dict[str, List[str]] = {}
    for ph in phrases:
        scores = bm25.get_scores(ph.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        out[ph] = [
            f"[baseline] {chunks[i][:snippet_chars]}"
            for i in top_idx
            if scores[i] > 0
        ]
    return out


def _gemini_retrieve(
    chunks: List[str],
    phrases: List[str],
    top_k: int = 6,
    snippet_chars: int = 900,
) -> Dict[str, List[str]]:
    """
    Gemini embedding-001 cosine-similarity retrieval.
    Falls back to BM25 if GEMINI_API_KEY is missing or any API error occurs.

    Chunk embeddings are batched (up to 100 at a time) to minimise round-trips.
    Phrase embeddings are individual calls (only 5 phrases).
    """
    if not chunks:
        return {ph: [] for ph in phrases}

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("WARNING: RAG_USE_VECTOR=1 but GEMINI_API_KEY not set; falling back to BM25")
        return _bm25_retrieve(chunks, phrases, top_k=top_k, snippet_chars=snippet_chars)

    try:
        import numpy as np
        from google import genai as _genai
        from llm_tracing import wrap_genai_client

        client = wrap_genai_client(_genai.Client(api_key=api_key))
        model = "gemini-embedding-001"

        def _embed(texts: List[str]) -> "np.ndarray":
            """Batch-embed up to 100 texts per call; fall back to a loop on error."""
            BATCH = 100
            all_vecs: List[List[float]] = []
            for start in range(0, len(texts), BATCH):
                batch = texts[start : start + BATCH]
                try:
                    result = client.models.embed_content(model=model, contents=batch)
                    all_vecs.extend(e.values for e in result.embeddings)
                except Exception:
                    # SDK version may not support list contents — fall back to loop
                    for t in batch:
                        r = client.models.embed_content(model=model, contents=t)
                        all_vecs.append(r.embeddings[0].values)
            return np.array(all_vecs, dtype=np.float32)

        print(f"Gemini RAG: embedding {len(chunks)} chunks ...")
        chunk_vecs = _embed(chunks)                          # (n_chunks, dim)
        norms = np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
        chunk_vecs = chunk_vecs / np.maximum(norms, 1e-9)   # L2 normalise

        out: Dict[str, List[str]] = {}
        for ph in phrases:
            ph_result = client.models.embed_content(model=model, contents=ph)
            ph_vec = np.array(ph_result.embeddings[0].values, dtype=np.float32)
            ph_norm = np.linalg.norm(ph_vec)
            if ph_norm > 0:
                ph_vec /= ph_norm
            scores = chunk_vecs @ ph_vec                     # cosine similarities
            top_idx_list = np.argsort(-scores)[:top_k].tolist()
            out[ph] = [
                f"[baseline] {chunks[i][:snippet_chars]}"
                for i in top_idx_list
                if scores[i] > 0.1
            ]
        return out

    except Exception as exc:
        print(f"WARNING: Gemini vector retrieval failed ({exc}); falling back to BM25")
        return _bm25_retrieve(chunks, phrases, top_k=top_k, snippet_chars=snippet_chars)


# ── Synthesis message builder ─────────────────────────────────────────────────

def build_synthesis_messages(
    activity_id: str,
    phrases: List[str],
    phrase_to_snips: Dict[str, List[str]],
) -> Tuple[str, str]:
    early_only = is_de1_activity(activity_id)
    system_msg = (
        "You are an experienced international aid decision maker with a quantitative mindset. "
        "Provide information related to forecasting the activity outcomes based on the query phrases below using only the evidence excerpts provided. Do not exclude any relevant information."
    )
    if early_only:
        system_msg += " Only include facts that would be knowable at the start of the activity; ignore later progress or results."

    ev_lines: List[str] = []
    for ph, snips in phrase_to_snips.items():
        ev_lines.append(f"\nPHRASE: {ph}")
        if snips:
            ev_lines.extend(f"- {s}" for s in snips)
        else:
            ev_lines.append("- (no matches)")

    user_prompt = f"""
ACTIVITY ID: {activity_id}


PHRASES:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(phrases))}

EVIDENCE EXCERPTS:
{chr(10).join(ev_lines)}

Respond with relevant information about the activity from the excerpts that would be useful to forecasting the eventual success of the activity, without losing any relevant information or context. Focus on providing information relevant to the 5 phrases above.

Respond only in English.
""".strip()

    return system_msg, user_prompt


# ── End-to-end builder ────────────────────────────────────────────────────────

def build_rag_evidence_and_prompt(
    activity_id: str,
    docs_log_csv: Path,
    txt_output_dir: Path,
    persist_root: Path,        # accepted but unused: BM25 needs no persistence
    phrases: List[str],
    *,
    allow_ocr: bool = True,
    top_k: int = 6,
    override_pdf_paths: Optional[List[Path]] = None,
) -> Tuple[Dict, object]:
    if override_pdf_paths is not None:
        pdf_paths_with_meta: List[Tuple[Path, Dict[str, str]]] = [
            (Path(p), {"type": "baseline"}) for p in override_pdf_paths if Path(p).exists()
        ]
    else:
        # For non-webapp use: collect PDFs from the docs log.
        # We don't replicate list_activity_docs here; pass override_pdf_paths instead.
        print(f"WARNING: rag_bm25 has no docs_log_csv support; no docs for {activity_id}")
        pdf_paths_with_meta = []

    print("pdf_paths_with_meta")
    print(pdf_paths_with_meta)

    # Convert PDFs to text
    all_text_parts: List[str] = []
    for pdf_path, meta in pdf_paths_with_meta:
        txt_path = ensure_txt_for_pdf(pdf_path, txt_output_dir, allow_ocr, activity_id=str(activity_id))
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        print(f"TXT {txt_path.name}: {len(text)} chars")
        all_text_parts.append(text)

    diag_base = {
        "activity_id": str(activity_id),
        "n_pdfs": len(pdf_paths_with_meta),
        "n_txt": len(all_text_parts),
    }

    if not all_text_parts:
        return diag_base, ("", "")

    all_text = "\n\n".join(all_text_parts)
    chunks = _chunk_text(all_text)
    backend = "Gemini vector" if USE_VECTOR_RAG else "BM25"
    print(f"{backend}: {len(chunks)} chunks from {len(all_text)} chars")

    if USE_VECTOR_RAG:
        phrase_to_snips = _gemini_retrieve(chunks, phrases, top_k=top_k)
    else:
        phrase_to_snips = _bm25_retrieve(chunks, phrases, top_k=top_k)

    diag = {
        **diag_base,
        "snips_per_phrase": {k: len(v) for k, v in phrase_to_snips.items()},
    }
    return diag, phrase_to_snips


def build_synthesis_prompt_from_phrasegen_text(
    activity_id: str,
    phrasegen_text: str,
    docs_log_csv: Path,
    txt_output_dir: Path,
    persist_root: Path,
    *,
    allow_ocr: bool = True,
    top_k: int = 6,
    override_pdf_paths: Optional[List[Path]] = None,
) -> Tuple[Dict, Tuple[str, str], List[str], Dict[str, List[str]]]:
    """
    Drop-in replacement for short_well_or_badly_rag.build_synthesis_prompt_from_phrasegen_text.
    Uses BM25 or Gemini vector retrieval depending on USE_VECTOR_RAG (env: RAG_USE_VECTOR=1).
    """
    phrases = parse_phrases(phrasegen_text, n=5)
    if len(phrases) < 5:
        print(f"ERROR: phrase extraction failed for activity_id={activity_id} (got {len(phrases)}/5). Skipping.")
        diag = {"activity_id": str(activity_id), "phrase_parse_ok": False, "n_phrases": len(phrases)}
        return diag, ("", ""), phrases, {}

    diag, phrase_to_snips_or_msgs = build_rag_evidence_and_prompt(
        activity_id=activity_id,
        docs_log_csv=docs_log_csv,
        txt_output_dir=txt_output_dir,
        persist_root=persist_root,
        phrases=phrases,
        allow_ocr=allow_ocr,
        top_k=top_k,
        override_pdf_paths=override_pdf_paths,
    )

    if not phrase_to_snips_or_msgs or isinstance(phrase_to_snips_or_msgs, tuple):
        return diag, ("", ""), phrases, {}

    phrase_to_snips = phrase_to_snips_or_msgs

    total_snips = sum(len(v) for v in phrase_to_snips.values())
    if total_snips == 0:
        diag = dict(diag)
        diag["no_retrieval_hits"] = True
        return diag, ("", ""), phrases, {}

    system_msg, user_prompt = build_synthesis_messages(activity_id, phrases, phrase_to_snips)

    diag = dict(diag)
    diag["phrase_parse_ok"] = True
    diag["n_phrases"] = len(phrases)

    return diag, (system_msg, user_prompt), phrases, phrase_to_snips
