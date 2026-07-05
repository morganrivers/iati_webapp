import os
import io
from typing import Dict, Any, List

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

# Adjust if you want to inject this instead
LOCATION_PDFS = "../../data/iati_all_pdfs"

MAX_TITLE_CHARS = 40


def _big_error(msg: str) -> None:
    print(f"ERROR: {msg.upper()}")
    raise ValueError(msg)


def add_in_page_info_top_left(
    bundle: Dict[str, Any],
    slice_path_unedited: str,
    section: str,
) -> str:
    """
    Given:
      - bundle: a dict with key "items", each item having:
          - cached_file: original pdf filename
          - doc_title: title of the original document
          - page_start: 1-based page index in the original pdf
      - slice_path_unedited: path to the combined PDF created from those sampled pages
      - section: string to prefix, e.g. "Baseline" or "Outcome"

    Returns:
      - path to a new PDF with text added at top-left of each page:
        "{section} | {doc_title} | page {page_start}/{total_pages_for_that_doc}"

    Prints BIG ALL CAPS ERRORS and raises if assumptions are violated.
    """
    if "items" not in bundle or not bundle["items"]:
        _big_error("BUNDLE MISSING NON-EMPTY 'items'")

    items: List[Dict[str, Any]] = bundle["items"]

    if not os.path.exists(slice_path_unedited):
        _big_error(f"SLICE_PATH_UNEDITED DOES NOT EXIST: {slice_path_unedited}")

    # Sort items exactly like you did when creating the combined slice
    items_sorted: List[Dict[str, Any]] = sorted(
        items,
        key=lambda x: (x["cached_file"], int(x["page_start"]))
    )

    # Read the combined slice we just created
    slice_reader = PdfReader(slice_path_unedited)
    combined_page_count = len(slice_reader.pages)

    if combined_page_count != len(items_sorted):
        _big_error(
            f"COMBINED PDF PAGE COUNT ({combined_page_count}) != "
            f"NUMBER OF ITEMS ({len(items_sorted)})"
        )

    # Cache original PDFs (per cached_file) so we don't reopen them every time
    original_readers: dict[str, PdfReader] = {}
    original_page_counts: dict[str, int] = {}

    def _get_original_info(cached_file: str) -> int:
        if not cached_file:
            _big_error("ITEM HAS EMPTY OR MISSING cached_file")
        if cached_file not in original_readers:
            original_pdf_path = os.path.join(LOCATION_PDFS, cached_file)
            if not os.path.exists(original_pdf_path):
                _big_error(f"ORIGINAL PDF NOT FOUND: {original_pdf_path}")
            reader = PdfReader(original_pdf_path)
            total_pages = len(reader.pages)
            if total_pages <= 0:
                _big_error(f"ORIGINAL PDF HAS NO PAGES: {original_pdf_path}")
            original_readers[cached_file] = reader
            original_page_counts[cached_file] = total_pages
        return original_page_counts[cached_file]

    # Prepare writer for annotated pages
    writer = PdfWriter()

    # For each page in the combined slice, overlay text with per-doc page info
    for idx, page in enumerate(slice_reader.pages):
        item = items_sorted[idx]

        # Get dimensions for THIS specific page
        media_box = page.mediabox
        page_width = float(media_box.width)
        page_height = float(media_box.height)

        cached_file = item.get("cached_file")
        if cached_file is None:
            _big_error(f"ITEM MISSING cached_file: {item}")

        doc_title = item.get("doc_title")
        if not doc_title:
            _big_error(f"ITEM MISSING doc_title: {item}")

        # Truncate overly long titles
        if len(doc_title) > MAX_TITLE_CHARS:
            doc_title_display = doc_title[: MAX_TITLE_CHARS - 3] + "..."
        else:
            doc_title_display = doc_title

        try:
            original_page_num = int(item["page_start"])
        except Exception:
            _big_error(f"INVALID page_start IN ITEM: {item}")

        total_pages = _get_original_info(cached_file)

        if original_page_num < 1 or original_page_num > total_pages:
            _big_error(
                f"page_start {original_page_num} OUT OF RANGE FOR ORIGINAL PDF "
                f"WITH {total_pages} PAGES (ITEM: {item})"
            )

        label_text = f"{section} | {doc_title_display} | page {original_page_num}/{total_pages}"

        # Make a one-page PDF in memory with the label text, same page size as THIS page
        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=(page_width, page_height))
        c.setFont("Helvetica-Bold", 12)
        # small margin from top-left
        margin_x = 18
        margin_y = 18
        c.drawString(margin_x, page_height - margin_y, label_text)
        c.save()
        packet.seek(0)

        overlay_pdf = PdfReader(packet)
        overlay_page = overlay_pdf.pages[0]

        # Merge overlay into page
        page.merge_page(overlay_page)
        writer.add_page(page)

    # Write out a new labeled PDF next to the unedited slice
    if slice_path_unedited.lower().endswith(".pdf"):
        slice_path_labeled = slice_path_unedited[:-4] + "_labeled.pdf"
    else:
        slice_path_labeled = slice_path_unedited + "_labeled.pdf"

    with open(slice_path_labeled, "wb") as f_out:
        writer.write(f_out)

    return slice_path_labeled
