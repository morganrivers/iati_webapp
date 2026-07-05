#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pprint
import json
import ast
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple

from google import genai

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from repo_paths import DATA_DIR

INPUT_PATH = DATA_DIR / "outputs_targets.jsonl"
OUT_PATH = DATA_DIR / "outputs_targets_embeddings.jsonl"
GEMINI_EMBED_MODEL = "gemini-embedding-001"

SEP_RE = re.compile(r"\n\s*---\s*\n", re.M)

def _try_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def _try_literal(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def _canon(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def split_on_separators(text: str):
    return [c.strip() for c in SEP_RE.split(text)]

def remove_no_response_tail(chunks):
    out = []
    for c in chunks:
        if c.strip().lower() == "no response":
            break
        out.append(c)
    return out

def dedupe_chunks(chunks):
    seen = set()
    kept = []
    for c in chunks:
        if not c:
            continue
        key = _canon(c)
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return kept

def normalize_response_text(raw) -> str:
    if raw is None:
        return ""

    # If it's already a list, normalize each item and join
    if isinstance(raw, list):
        parts = [normalize_response_text(x) for x in raw]
        parts = [p.strip() for p in parts if p.strip()]
        return "\n\n".join(parts).strip()

    # Coerce to string
    s = raw if isinstance(raw, str) else str(raw)
    s = s.strip()

    # Peel encoding layers iteratively, but NEVER recurse on non-list
    for _ in range(10):
        changed = False

        j = _try_json(s)
        if isinstance(j, list):
            return normalize_response_text(j)
        if isinstance(j, str) and j != s:
            s = j.strip()
            changed = True

        lit = _try_literal(s)
        if isinstance(lit, list):
            return normalize_response_text(lit)
        if isinstance(lit, str) and lit != s:
            s = lit.strip()
            changed = True

        if not changed:
            break

    # Strip one layer of wrapping quotes if present
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        inner = s[1:-1].strip()
        if inner:
            s = inner

    # Unescape common sequences
    s = s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\"", "\"").replace("\\'", "'")

    # Remove ALL asterisks
    s = s.replace("*", "")

    # Remove any lines beginning with: {'type': 'string',
    s = re.sub(r"(?m)^\s*\{'type':\s*'string',.*\n?", "", s)

    # Normalize excessive blank lines
    lines = [ln.rstrip() for ln in s.splitlines()]
    out_lines = []
    blank = 0
    for ln in lines:
        if ln.strip() == "":
            blank += 1
            if blank <= 2:
                out_lines.append("")
        else:
            blank = 0
            out_lines.append(ln)
    s = "\n".join(out_lines).strip()

    # Handle '---' separators, drop NO RESPONSE tail, dedupe chunks
    chunks = split_on_separators(s)
    chunks = remove_no_response_tail(chunks)

    # If no explicit separator, split by paragraphs for dedupe
    if len(chunks) == 1:
        chunks = [c.strip() for c in re.split(r"\n{2,}", chunks[0])]

    chunks = [c.strip() for c in chunks if c.strip()]
    chunks = remove_no_response_tail(chunks)
    chunks = dedupe_chunks(chunks)

    return "\n\n".join(chunks).strip()

def load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            aid = obj.get("activity_id")
            if isinstance(aid, str) and aid:
                ids.add(aid)
    return ids

def iter_targets(path: Path) -> Iterable[Tuple[str, str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            aid = (obj.get("activity_id") or "").strip()
            if not aid:
                continue
            section = (obj.get("section") or "").strip()
            cleaned = normalize_response_text(obj.get("response_text", ""))
            if not cleaned:
                continue
            yield aid, section, cleaned
