#!/usr/bin/env python3
"""
Extract text from all files in ./iati_all_pdfs and write .txt files to
./iati_all_pdfs_txt_format using the same base filename.

- PDFs:
    1) Use 'pdftotext' (Poppler) if available (fast, accurate).
    2) Else use pdfminer.six if installed.
    3) Else use pypdf as a last resort.
    4) Optional OCR: if text is empty/very short and both 'tesseract' and 'pdftoppm'
       are available, perform page-by-page OCR at 300 DPI.

- DOCX:
    1) Use docx2txt if installed.
    2) Else use python-docx if installed.
    3) Else try LibreOffice ('soffice' or 'libreoffice') headless conversion to .txt.

- Legacy .doc:
    Try LibreOffice headless conversion to .txt if 'soffice'/'libreoffice' exists.

- Plain .txt: copied (normalized) to output.

No network/API calls — all local.

Usage:
    python3 extract_iati_texts.py
Optional flags:
    --workers N          (default: sensible based on CPU; caps at 8)
    --ocr                enable OCR fallback for PDFs with no text (requires tesseract + pdftoppm)
    --recursive          recurse into subdirectories of the input folder
    --only-newer         skip if output exists and is newer than input
    --progress-every N   print a progress line every N files (default 200)

Directory layout (as provided by the user):
    Input:  ./iati_all_pdfs
    Output: ./iati_all_pdfs_txt_format
"""
# --- add near other imports ---
from __future__ import annotations
import signal

# --- add somewhere near helpers ---
def _per_file_timeout_handler(signum, frame):
    raise TimeoutError("per-file timeout (20s)")

import argparse
import concurrent.futures as futures
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# -------- Configuration (paths per the user's setup) --------
INPUT_DIR = Path("./iati_all_pdfs").resolve()
OUTPUT_DIR = Path("./iati_all_pdfs_txt_format").resolve()

# -------- Optional dependency checks (we import lazily in workers) --------
def cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None

HAVE_PDFTOTEXT = cmd_exists("pdftotext")
HAVE_PDFTOPPM = cmd_exists("pdftoppm")
HAVE_TESSERACT = cmd_exists("tesseract")
HAVE_SOFFICE = cmd_exists("soffice") or cmd_exists("libreoffice")

# -------- Helpers --------
PDF_MAGIC = b"%PDF-"
OLE_MAGIC = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"  # legacy .doc
ZIP_MAGIC = b"PK\x03\x04"  # docx (OOXML) is a zip
HTML_MAGIC_PREFIXES = (b'<!DOCT', b'<!doc', b'<html', b'<HTML')
def sniff_type(path: Path) -> str:
    try:
        with path.open("rb") as f:
            raw = f.read(1024)
    except Exception:
        return "unknown"
    header = raw.lstrip(b"\xef\xbb\xbf \t\r\n")[:8]

    if header.startswith(PDF_MAGIC):
        return "pdf"
    if header.startswith(OLE_MAGIC):
        return "doc"
    if header.startswith(ZIP_MAGIC):
        return "zip"  # don't guess yet; inspect later
    if header.startswith(b'Rar!\x1a'):
        return "rar"
    if any(header.startswith(p) for p in HTML_MAGIC_PREFIXES):
        return "html"

    # fall back to extension hints
    lower = path.name.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith(".docx"):
        return "docx"
    if lower.endswith(".doc"):
        return "doc"
    if lower.endswith((".txt", ".md")):
        return "txt"
    return "unknown"

def normalized_basename(path: Path) -> str:
    """
    Strip repeated known document extensions from the filename tail and return the base.
    e.g., 'foo.pdf.pdf' -> 'foo', 'bar.DOCX' -> 'bar'
    """
    base = path.name
    # Remove multiple trailing doc-ish extensions
    pattern = re.compile(r"(\.(pdf|docx|doc|rtf|html|htm|txt|md))+$", re.IGNORECASE)
    base = pattern.sub("", base)
    # Remove any trailing dots/spaces
    base = base.rstrip(". ").strip()
    if not base:
        base = path.stem  # fallback
    return base

def write_text_atomic(out_path: Path, text: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Temporary file in the same directory to ensure atomic rename on same filesystem
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(out_path.parent), delete=False) as tmp:
        tmp.write(text)
        tmp_name = tmp.name
    os.replace(tmp_name, out_path)

def normalize_whitespace(text: str) -> str:
    # Keep line breaks but collapse extreme whitespace runs
    # Also normalize Windows/Mac line endings to '\n'
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines to just 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip() + "\n"

def copy_txt_as_is(in_path: Path) -> str:
    try:
        data = in_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        # As binary, best effort decode
        data = in_path.read_bytes().decode("utf-8", errors="replace")
    return normalize_whitespace(data)

# -------- PDF extraction strategies --------
def pdf_to_text_pdftotext(in_path: Path,keep_pagebreaks=False) -> Optional[str]:
    if not HAVE_PDFTOTEXT:
        return None
    try:
        # Use layout to preserve reading order better for tabular docs; -nopgbrk to avoid ^L
        if keep_pagebreaks:
            result = subprocess.run(
                ["pdftotext", "-layout", "-enc", "UTF-8", str(in_path), "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        else:
            result = subprocess.run(
                ["pdftotext", "-layout", "-enc", "UTF-8", "-nopgbrk", str(in_path), "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        if result.returncode != 0:
            return None
        text = result.stdout.decode("utf-8", errors="replace")
        # pdftotext sometimes emits nothing for image-only pages
        return text if text.strip() else ""
    except Exception:
        return None

def pdf_to_text_pdfminer(in_path: Path) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:
        return None
    try:
        text = extract_text(str(in_path)) or ""
        return text
    except Exception:
        return None

def pdf_to_text_pypdf(in_path: Path) -> Optional[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # fallback older name
        except Exception:
            return None
    try:
        reader = PdfReader(str(in_path))
        buf = io.StringIO()
        for page in getattr(reader, "pages", []):
            try:
                buf.write(page.extract_text() or "")
                buf.write("\n")
            except Exception:
                continue
        return buf.getvalue()
    except Exception:
        return None

def pdf_ocr_tesseract(in_path: Path, dpi: int = 300) -> Optional[str]:
    """
    OCR a PDF by rasterizing to PNG with pdftoppm, then tesseract each page to text.
    Requires pdftoppm + tesseract installed.
    """
    print("tesseract running...")
    if not (HAVE_PDFTOPPM and HAVE_TESSERACT):
        print("no tesseract")
        return None
    tmpdir = None
    try:
        tmpdir = Path(tempfile.mkdtemp(prefix="ocr_"))
        # Convert pages to images
        prefix = tmpdir / "page"
        conv = subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", str(in_path), str(prefix)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if conv.returncode != 0:
            print("noinzero code")
            return None
        # Gather generated images in sorted order
        imgs = sorted(tmpdir.glob("page-*.png"))
        if not imgs:
            print("not image")
            return None
        parts = []
        for img in imgs:
            try:
                print("trying image")
                ocr = subprocess.run(
                    ["tesseract", str(img), "stdout", "-l", "eng"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if ocr.returncode != 0:
                    continue
                parts.append(ocr.stdout.decode("utf-8", errors="replace"))
            except Exception:
                print("failed image")
                continue
        if not parts:
            print("returned none...")
        return "\n\n".join(parts) if parts else None
    finally:
        if tmpdir and tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
def extract_pdf_text(in_path: Path, allow_ocr: bool = False, keep_pagebreaks: bool = False) -> str:
    # Priority: pdftotext -> pdfminer -> pypdf -> OCR (if allowed)
    if keep_pagebreaks:
        # try pdftotext first, but KEEP page breaks
        text = pdf_to_text_pdftotext(in_path, keep_pagebreaks=True)
        if text is not None:
            text = text or ""
            if text.strip():
                return normalize_whitespace(text)
        # fall through to the normal backup chain below
        return ""
    for fn in (pdf_to_text_pdftotext, pdf_to_text_pdfminer, pdf_to_text_pypdf):
        text = fn(in_path)
        if text is None:
            continue
        text = text or ""
        if text.strip():
            return normalize_whitespace(text)

    if allow_ocr:
        text = pdf_ocr_tesseract(in_path)
        if text:
            return normalize_whitespace(text)

    return ""


# -------- DOCX / DOC extraction --------
def extract_docx_text(in_path: Path) -> Optional[str]:
    # Try docx2txt first
    try:
        import docx2txt  # type: ignore
        text = docx2txt.process(str(in_path)) or ""
        return normalize_whitespace(text)
    except Exception:
        pass
    # Try python-docx
    try:
        import docx  # type: ignore
        doc = docx.Document(str(in_path))
        paras = [p.text for p in doc.paragraphs]
        text = "\n".join(paras)
        return normalize_whitespace(text)
    except Exception:
        pass
    # Fallback: try soffice conversion
    return soffice_convert_to_text(in_path)

def soffice_convert_to_text(in_path: Path) -> Optional[str]:
    if not HAVE_SOFFICE:
        return None
    soffice_cmd = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice_cmd:
        return None
    tmpdir = Path(tempfile.mkdtemp(prefix="soffice_"))
    try:
        # Convert to UTF-8 txt
        # LibreOffice normally picks UTF-8 for text output; we’ll read as UTF-8.
        proc = subprocess.run(
            [soffice_cmd, "--headless", "--convert-to", "txt:Text", "--outdir", str(tmpdir), str(in_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            return None
        # Find the generated TXT
        out_txt = None
        for p in tmpdir.iterdir():
            if p.suffix.lower() == ".txt":
                out_txt = p
                break
        if not out_txt or not out_txt.exists():
            return None
        data = out_txt.read_text(encoding="utf-8", errors="replace")
        return normalize_whitespace(data)
    except Exception:
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
import zipfile

def zip_inner_kind(p: Path) -> Optional[str]:
    try:
        with zipfile.ZipFile(p) as z:
            names = set(z.namelist())
        if any(n.startswith("word/") for n in names):
            return "docx"
        if any(n.startswith("ppt/") for n in names):
            return "pptx"
        if any(n.startswith("xl/") for n in names):
            return "xlsx"
        return None
    except Exception:
        return None
def html_to_text(in_path: Path) -> str:
    try:
        from bs4 import BeautifulSoup  # pip install beautifulsoup4
        html = in_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        # Drop scripts/styles
        for tag in soup(["script", "style", "noscript"]): tag.extract()
        text = soup.get_text("\n")
        return normalize_whitespace(text)
    except Exception:
        # Very crude fallback if bs4 not available
        data = in_path.read_text(encoding="utf-8", errors="replace")
        # Strip tags roughly
        text = re.sub(r"<[^>]+>", " ", data)
        return normalize_whitespace(text)
def is_valid_pdf(path: Path) -> bool:
    if not cmd_exists("pdfinfo"):
        return True  # can’t check
    r = subprocess.run(["pdfinfo", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode == 0

def process_one(path: Path, out_dir: Path, allow_ocr: bool, only_newer: bool, keep_pagebreaks=False, out_name: str = "") -> Tuple[str, str]:
    print("called process_one")
    """
    Returns (status, message)
    status in {"OK", "SKIP", "ERR"}
    out_name: if provided, use this as the output stem instead of normalized_basename(path).
    """
    # --- BEGIN timeout guard (main thread only — signal.SIGALRM is not available in subthreads) ---
    import threading as _threading
    _use_signal = _threading.current_thread() is _threading.main_thread()
    old_handler = None
    if _use_signal:
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _per_file_timeout_handler)
        signal.alarm(20)
    # --- END timeout guard ---

    try:
        print("helper_scripts/extract_pdfs_as_txt.py: processing pdf...")
        ftype = sniff_type(path)
        base = out_name if out_name else normalized_basename(path)
        out_path = out_dir / f"{base}.txt"

        if only_newer and out_path.exists() and out_path.stat().st_mtime >= path.stat().st_mtime:
            return ("SKIP", f"{path.name} -> (up-to-date)")
        elif ftype == "pdf":
            if not is_valid_pdf(path):
                return ("ERR", f"{path.name} -> invalid/unsupported PDF (pdfinfo)")
            # before extract_pdf_text call
            if allow_ocr and _use_signal:
                signal.alarm(60)  # or dynamically based on filesize/pagecount
            text = extract_pdf_text(path, allow_ocr=allow_ocr, keep_pagebreaks=keep_pagebreaks)
        elif ftype == "docx":
            text = extract_docx_text(path)
        elif ftype == "doc":
            text = soffice_convert_to_text(path)
        elif ftype == "zip":
            kind = zip_inner_kind(path)
            if kind == "docx":
                text = extract_docx_text(path)
            else:
                text = None  # unsupported zip
        elif ftype == "html":
            text = html_to_text(path)  # add a tiny extractor (see below)
        elif ftype == "txt":
            text = copy_txt_as_is(path)

        else:
            # Try by extension hints in case sniff was ambiguous
            lower = path.name.lower()
            if lower.endswith(".pdf"):
                if allow_ocr and _use_signal:
                    signal.alarm(60)  # or dynamically based on filesize/pagecount
                text = extract_pdf_text(path, allow_ocr=allow_ocr)
            elif lower.endswith(".docx"):
                text = extract_docx_text(path)
            elif lower.endswith(".doc"):
                text = soffice_convert_to_text(path)
            elif lower.endswith(".txt") or lower.endswith(".md"):
                text = copy_txt_as_is(path)
            else:
                text = None

        if text is None:
            return ("ERR", f"{path.name} -> unsupported type or no extractor")
        if not text.strip():
            write_text_atomic(out_path, "")
            return ("OK", f"{path.name} -> (no extractable text; wrote empty)")

        print("helper_scripts/extract_pdfs_as_txt.py: text processed!")
        write_text_atomic(out_path, text)
        return ("OK", f"{path.name} -> {out_path.name}")

    except TimeoutError as e:
        return ("ERR", f"{path.name} -> {e}")  # clearly mark timed-out files
    except Exception as e:
        return ("ERR", f"{path.name} -> {e.__class__.__name__}: {e}")
    finally:
        if _use_signal:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

# -------- Main --------
def gather_files(root: Path, recursive: bool) -> list[Path]:
    if not recursive:
        return [p for p in root.iterdir() if p.is_file()]
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            files.append(Path(dirpath) / name)
    return files

def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs/DOCX in ./iati_all_pdfs to ./iati_all_pdfs_txt_format as .txt files.")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))),
                        help="Number of parallel workers (default: up to 8 based on CPU)")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR fallback for image-only PDFs (needs tesseract + pdftoppm)")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories of input folder")
    parser.add_argument("--only-newer", action="store_true", help="Skip files whose output exists and is newer than input")
    parser.add_argument("--progress-every", type=int, default=200, help="Print progress every N files (default 200)")
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        print(f"ERROR: Input folder not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Extract IATI Texts ===")
    print(f"Input:   {INPUT_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Workers: {args.workers}")
    print(f"Use OCR: {'yes' if args.ocr else 'no'}")
    print(f"pdftotext: {'yes' if HAVE_PDFTOTEXT else 'no'} | pdfminer: lazy | pypdf: lazy")
    print(f"tesseract: {'yes' if HAVE_TESSERACT else 'no'} | pdftoppm: {'yes' if HAVE_PDFTOPPM else 'no'}")
    print(f"LibreOffice(headless): {'yes' if HAVE_SOFFICE else 'no'}")
    print("Scanning files...")

    files = gather_files(INPUT_DIR, recursive=args.recursive)
    total = len(files)
    print(f"Found {total} files.")

    ok = 0
    skipped = 0
    err = 0

    # Use a process pool for mixed CPU/IO tasks; keep it moderate by default
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        # Submit jobs
        jobs = {pool.submit(process_one, p, OUTPUT_DIR, args.ocr, args.only_newer): p for p in files}

        for i, fut in enumerate(futures.as_completed(jobs), 1):
            status, msg = fut.result()
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skipped += 1
            else:
                err += 1
            if i % max(1, args.progress_every) == 0:
                print(f"[{i}/{total}] OK={ok} SKIP={skipped} ERR={err}  Last: {msg}")

    print(f"Done. Processed: {total}. OK={ok}, SKIP={skipped}, ERR={err}")

if __name__ == "__main__":
    main()
