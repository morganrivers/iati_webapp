from debug_utils import _print_ram

_print_ram("before all extracting_and_grading_imports ")
import contextlib

import io
import csv
import time
import json
import pprint
import sys
import asyncio
from datetime import datetime
from typing import Optional, Set, Dict, Any, Tuple, List
from pathlib import Path
import os


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
_print_ram("before import genai extracting_and_grading_imports ")
from google import genai          # loads the package once
_print_ram("after import genai extracting_and_grading_imports ")


import traceback  # <-- add this


import re



_print_ram("other random import genai extracting_and_grading_imports ")
import os
import asyncio
import tempfile
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from functools import partial  # <-- add this
try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    PdfReader = None
    PdfWriter = None
_print_ram("after other random import genai extracting_and_grading_imports ")

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

_print_ram("before some more genai extracting_and_grading_imports ")

from label_sampled_pages import add_in_page_info_top_left
_print_ram("after label_sampled_pages")
from prompt_bundle_pdf import open_with_evince
_print_ram("after prompt_bundle")


from get_all_pages_within_category import load_and_filter_rows
_print_ram("after load_and_filter_rows")
# from score_page_relevance import load_docs, load_acts_map, load_activity_counts, filter_usable, make_genai_client, activity_title, iter_page_batches, write_pdf_slice, desc, activity_title

# NEW: at top
from concurrent.futures import ThreadPoolExecutor
_print_ram("after some more genai extracting_and_grading_imports ")

# NEW: pass an execpool down (don’t rely on the loop’s default one)
def make_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=CONCURRENCY, thread_name_prefix="genai")

PRINT_PROMPT_OPENAI = True
PRINT_PROMPT_BEFORE_UPLOAD = True
OPEN_UPLOADED_WITH_EVINCE = False

AIRPLANE_MODE = False


# Toggle between live calls and Gemini Batch API
BATCH_MODE = False  # False when you want live calls

if BATCH_MODE:
    os.makedirs("../../data/batch_requests", exist_ok=True)

# BATCH_JSONL = "../../data/batch_requests/summaries_batch_requests.jsonl"

CONCURRENCY = 5  # run up to 5 activities at once
# CONCURRENCY = 3  # run up to 5 activities at once

LOCATION_PDFS = "../../data/iati_all_pdfs"
MODEL_NAME        = "gemini-2.5-flash"
# MODEL_NAME        = "gemini-3-pro-preview" # can be overwritten
TIMEOUT_SECONDS = 300

def get_key(row):
    return row.get("activity_id")
    # return (obj.get("activity_id"), obj.get("cached_file"), obj.get("page_start"))

# ---------- Gemini client & structured schemas ----------
def make_genai_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY_GEMINI")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY (or GOOGLE_API_KEY_GEMINI).")
    return genai.Client(api_key=api_key)

# Prompt + response schema builders for Gemini structured output.
# One prompt per CSV row: if both quantitative outcomes and ratings are flagged,
# we combine into a single prompt & a combined schema.
def wait_file_active(client, uploaded, *, timeout=60, interval=1.0):
    """Poll the uploaded file until ACTIVE, or raise on FAILED/timeout."""
    t0 = time.time()
    # Some client versions expose 'name' (e.g., 'files/abc123'), others 'id'
    file_name = getattr(uploaded, "name", None) or getattr(uploaded, "id")
    while True:
        f = client.files.get(name=file_name)  # raises if not found
        state = getattr(f, "state", getattr(f, "display_state", None))
        if state == "ACTIVE":
            return f
        if state == "FAILED":
            raise RuntimeError(f"Uploaded file failed to process on server: {file_name}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Timed out waiting for file to become ACTIVE: {file_name} (last state={state})")
        time.sleep(interval)

async def assemble_and_upload_activity_pdf(bundle: Dict[str, Any], client, executor,section=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        slice_path_unedited = os.path.join(tmpdir, "combined.pdf")
        # pprint.pprint(bundle)
        try:
            writer = PdfWriter()

            # SORT BY FILE NAME AND PAGE, BEFORE MERGING INTO A DOCUMENT
            items_sorted = sorted(
                bundle["items"],
                key=lambda x: (x["cached_file"], int(x["page_start"]))
            )

            for item in items_sorted:
                abs_path = os.path.join(LOCATION_PDFS, item["cached_file"])
                page_idx0 = int(item["page_start"]) - 1
                reader = PdfReader(abs_path)
                writer.add_page(reader.pages[page_idx0])


            # for item in bundle["items"]:
            #     abs_path = os.path.join(LOCATION_PDFS, item["cached_file"])
            #     page_idx0 = int(item["page_start"]) - 1
            #     reader = PdfReader(abs_path)
            #     writer.add_page(reader.pages[page_idx0])


            with open(slice_path_unedited, "wb") as fout:
                writer.write(fout)

            # --- NEW: decide section + always run add_in_page_info_top_left ---
            if section is None:
                # infer from keys on the bundle
                if "index_activity" in bundle:
                    section = "Activity Information Document"
                elif "index_evaluation" in bundle:
                    section = "Activity Evaluation Document"
                else:
                    # "index" / generic upload case
                    section = "Uploaded Document"
                    
            slice_path = add_in_page_info_top_left(bundle, slice_path_unedited, section)

            if OPEN_UPLOADED_WITH_EVINCE:
                # print("Showing another acitity pdf to upload:")
                # pprint.pprint(item)
                # NEW: preview the combined slice in Evince before upload
                open_with_evince(slice_path, wait=True)
                # input("hit enter!!!")
        except Exception as e:
            err = {"ERROR": f"slice_error: {e}"}
            print(err["ERROR"])
            return err

        try:
            loop = asyncio.get_running_loop()

            # wrap to preserve keyword args while using our executor
            def _upload():
                return client.files.upload(file=slice_path)

            def _wait_active(uploaded_file):
                return wait_file_active(client, uploaded_file, timeout=120, interval=0.5)

            # upload and poll ACTIVE BEFORE leaving the tempdir
            uploaded = await asyncio.wait_for(
                loop.run_in_executor(executor, _upload),
                timeout=TIMEOUT_SECONDS,
            )
            await asyncio.wait_for(
                loop.run_in_executor(executor, lambda: _wait_active(uploaded)),
                timeout=TIMEOUT_SECONDS + 10,
            )
        except asyncio.TimeoutError:
            return {"ERROR": "upload_timeout"}
        except Exception as e:
            err = {"ERROR": f"upload_error: {e}"}
            print(err["ERROR"])
            return err

    print("uploaded!")
    return uploaded


async def run_one_row(response_schema, prompt, row, client, seen_keys, output_jsonl, execpool, model):
    obj: Dict[str, Any] = {}

    # Skip if this row was already processed before
    key = get_key(row)
    if key in seen_keys:
        # print(f"key was seen: {key}. Skipping.")
        return None
    seen_keys.add(key)

    if "num_pages" in row.keys():
        # Build base obj
        obj.update({
            "activity_id": row["activity_id"],
            "section": row.get("section"),
            "num_pages": len(row.get("items", [])),
        })
    else:
        obj.update({
            "activity_id": row["activity_id"],
        })

    system_instruction = None
    text_prompt = prompt
    prompt_type = ""
    if isinstance(prompt, dict):
        system_instruction = prompt.get("system_msg")
        text_prompt = prompt.get("prompt")
        prompt_type = prompt.get("prompt_type","")

    if PRINT_PROMPT_BEFORE_UPLOAD:
        pprint.pprint(f"prompt for activity id {row['activity_id']}")
        pprint.pprint(prompt)

    # pprint.pprint("row")
    # pprint.pprint(row)

    uploaded = None
    if row.get("items"):
        # Upload / assemble
        uploaded = await assemble_and_upload_activity_pdf(row, client, execpool)
        if isinstance(uploaded, dict) and "ERROR" in uploaded:
            # propagate the uploader's error shape
            return {**obj, **uploaded}


    activity_upload = None
    evaluation_upload = None

    # Upload activity pages
    if row.get("activity_items"):
        bundle = {"items": row["activity_items"]}
        activity_upload = await assemble_and_upload_activity_pdf(bundle, client, execpool,section="Activity Information Document")
        if isinstance(activity_upload, dict) and "ERROR" in activity_upload:
            return {**obj, **activity_upload}

    # Upload evaluation pages
    if row.get("evaluation_items"):
        bundle = {"items": row["evaluation_items"]}
        evaluation_upload = await assemble_and_upload_activity_pdf(bundle, client, execpool,section="Activity Evaluation Document")
        if isinstance(evaluation_upload, dict) and "ERROR" in evaluation_upload:
            return {**obj, **evaluation_upload}

    # --- NEW: Batch mode branch (no live model call) ---
    if BATCH_MODE:
        # Build "parts" exactly like in the ranker: text prompt + fileData blocks
        parts = []

        if system_instruction is not None:
            parts.append({
                "text": f"SYSTEM INSTRUCTION:\n{system_instruction}"
            })

        parts.append({"text": text_prompt})

        # Attach combined activity / evaluation / generic bundles
        if activity_upload:
            parts.append({"text": "ACTIVITY DOCUMENTS:"})
            parts.append({
                "fileData": {
                    "fileUri": activity_upload.uri,
                    "mimeType": activity_upload.mime_type,
                }
            })

        if evaluation_upload:
            parts.append({"text": "EVALUATION DOCUMENTS:"})
            parts.append({
                "fileData": {
                    "fileUri": evaluation_upload.uri,
                    "mimeType": evaluation_upload.mime_type,
                }
            })

        if uploaded and not activity_upload and not evaluation_upload:
            parts.append({"text": "ACTIVITY DOCUMENTS:"})
            parts.append({
                "fileData": {
                    "fileUri": uploaded.uri,
                    "mimeType": uploaded.mime_type,
                }
            })

        request_obj: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
        }

        if response_schema is not None:
            request_obj["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseJsonSchema": response_schema,
            }

        batch_key = f"{obj['activity_id']}::{obj.get('section') or 'NA'}"

        line = {
            "key": batch_key,
            "request": request_obj,
        }

        out_path = Path(output_jsonl)
        batch_dir = Path("../../data/batch_requests")
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_path = batch_dir / f"{out_path.stem}_batch{out_path.suffix}"

        with open(batch_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line))
            f.write("\n")

        print(f"Wrote batch request for {obj['activity_id']} / {obj.get('section')} with key={batch_key}")
        return None

    # model call via our execpool
    try:
        loop = asyncio.get_running_loop()
        def _sync_call():
            # contents = [text_prompt, uploaded] if uploaded else [text_prompt]
            contents = [text_prompt]

            # Add labeled activity documents
            if activity_upload:
                contents.append("ACTIVITY DOCUMENTS:")
                contents.append({
                    "fileData": {
                        "fileUri": activity_upload.uri,
                        "mimeType": activity_upload.mime_type,
                    }
                })

            # Add labeled evaluation documents
            if evaluation_upload:
                contents.append("EVALUATION DOCUMENTS:")
                contents.append({
                    "fileData": {
                        "fileUri": evaluation_upload.uri,
                        "mimeType": evaluation_upload.mime_type,
                    }
                })

            # Fallback: for generic bundles that only have "items" (your KNN script),
            # attach that single combined PDF as well, but only if we didn't already
            # add activity/evaluation docs.
            if uploaded and not activity_upload and not evaluation_upload:
                contents.append("ACTIVITY DOCUMENTS:")
                contents.append({
                    "fileData": {
                        "fileUri": uploaded.uri,
                        "mimeType": uploaded.mime_type,
                    }
                })
            if system_instruction is not None:
                gen_config = {
                    "system_instruction": system_instruction,
                }
            else:
                gen_config = {}

            if response_schema is not None:
                # Only enable JSON mode + structured output when you actually pass a schema
                gen_config["response_mime_type"] = "application/json"
                gen_config["response_schema"] = response_schema

            return client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config,
            )

        fut = loop.run_in_executor(execpool, _sync_call)
        response = await asyncio.wait_for(fut, timeout=TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        obj["ERROR"] = "asyncio.exceptions.TimeoutError"
        return obj
    except Exception as e:
        obj["ERROR"] = f"{type(e).__name__}: {e}"
        return obj

    obj["prompt_type"] = prompt_type

    # ---- Token usage + metadata ----
    usage = getattr(response, "usage_metadata", None)
    if usage:
        obj["token_usage"] = {
            "prompt_token_count": getattr(usage, "prompt_token_count", None),
            "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
            "candidates_token_count": getattr(usage, "candidates_token_count", None),
            "total_token_count": getattr(usage, "total_token_count", None),
            # optional breakdown if your SDK provides it
            "prompt_tokens_details": [
                {"modality": getattr(d, "modality", None), "token_count": getattr(d, "token_count", None)}
                for d in (getattr(usage, "prompt_tokens_details", []) or [])
            ],
        }

    # optional: keep a couple of handy IDs alongside usage
    obj["model_version"] = getattr(response, "model_version", None)


    pprint.pprint(f"\nResponse from model {model}")
    pprint.pprint(response)
    # ---- Got a response; store serializable bits and validate JSON ----
    text = getattr(response, "text", str(response))
    obj["response_text"] = text

    # pprint.pprint("obj")
    # pprint.pprint(obj)
    return obj

async def run_one_row_openai(prompt, row, client, seen_keys, output_jsonl, execpool, model):
    if(type(prompt) == dict):
        system_msg = prompt["system_msg"]
        prompt = prompt["prompt"]
    else:
        system_msg = (
            "You are a careful, experienced international aid decision maker.\n"
            "Only reply with an integer between 0 and 100 (no extra text)"
        )

    if model == "gemini":
        model = MODEL_NAME

    obj: Dict[str, Any] = {}

    # Skip if this row was already processed before
    key = get_key(row)
    if key in seen_keys:
        print(f"key was seen: {key}. Skipping.")
        return None
    seen_keys.add(key)

    # Build base obj
    obj.update({
        "activity_id": row["activity_id"],
        "section": row.get("section"),
    })
    print("about to call")
    # model call via our execpool
    try:
        loop = asyncio.get_running_loop()
        def _sync_call():
            print("calling..")
            if PRINT_PROMPT_OPENAI:
                pprint.pprint("system_msg")
                pprint.pprint(system_msg)
                pprint.pprint("prompt")
                pprint.pprint(prompt)
            extra = {}
            if model == "deepseek-reasoner":
                extra = {"thinking": {"type": "enabled"}}

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                extra_body=extra,
            )
            # print("QUITTING")
            # quit()  # Commented out for debugging
            if PRINT_PROMPT_OPENAI:
                pprint.pprint("resp")
                pprint.pprint(resp)
            return resp #.choices[0].message.content.strip()
        fut = loop.run_in_executor(execpool, _sync_call)
        response = await asyncio.wait_for(fut, timeout=TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        obj["ERROR"] = "asyncio.exceptions.TimeoutError"
        return obj
    except Exception as e:
        obj["ERROR"] = f"{type(e).__name__}: {e}"
        return obj

    # pprint.pprint("obj")
    # pprint.pprint(obj)
    # geshdsga
    # quit()
    # optional: keep a couple of handy IDs alongside usage
    # obj["model_version"] = "gpt-3.5"

    # --- serialize only what you want ---
    c0 = response.choices[0]
    msg = c0.message

    obj["response"] = {
        "id": response.id,  # drop this if you don't want it
        "content": msg.content,
        "role": getattr(msg, "role", "assistant"),
        "finish_reason": c0.finish_reason,
        "usage": {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        },
    }
    # remove Nones to keep it tidy
    obj["response"]["usage"] = {k: v for k, v in obj["response"]["usage"].items() if v is not None}

    # pprint.pprint("response")
    # pprint.pprint(response)
    # obj["response_text"] = response

    # pprint.pprint("obj")
    # pprint.pprint(obj)
    return obj



# was: -> Set[Tuple[...]]
def load_seen_keys(output_jsonl_path: str) -> Set[str]:
    seen: Set[str] = set()
    p = Path(output_jsonl_path)
    if not p.exists():
        return seen
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = obj.get("activity_id")
            if not aid:
                continue
            # only consider success as "seen"
            if obj.get("ERROR"):
                continue
            seen.add(str(aid))
    return seen


def read_last_success_row(output_jsonl_path) -> Dict[str, Any]:
    """Return the last successful row from a loop_over_rows_to_call_model output.

    The loop appends rows, so a file may contain errored rows from prior or
    failed attempts before the successful one. A success is any row without an
    ERROR key, matching load_seen_keys() semantics. Consumers extract their own
    content field (response_text, response, etc.) from the returned row.
    """
    p = Path(output_jsonl_path)
    if not p.exists():
        raise RuntimeError(f"No output generated at {p}")
    success = None
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("ERROR"):
                continue
            success = obj
    if success is None:
        raise RuntimeError(f"No successful row in output: {p}")
    return success


# async def loop_over_rows_to_call_model(output_jsonl, rows, prompts, response_schema=None,execpool=None,model="gemini"):
#     # quit()
#     if not execpool:
#         execpool = make_executor()
#     output_jsonl_path = Path(output_jsonl)
#     output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
#     if model == "gemini" or "gemini" in model.lower():
#         client = make_genai_client()
#     elif model == "chatgpt" or "gpt" in model.lower():
#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     else:
#         print("error: can only call gemini or chatgpt")
#         quit()
async def loop_over_rows_to_call_model(output_jsonl, rows, prompts, response_schema=None, execpool=None, model="gemini"):
    is_gemini = (model == "gemini" or "gemini" in model.lower() or model.startswith("projects/"))
    is_deepseek = model.startswith("deepseek-")

    # is_gemini = (model == "gemini" or "gemini" in model.lower() or model.startswith("projects/"))
    if not execpool:
        execpool = make_executor()

    output_jsonl_path = Path(output_jsonl)
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)


    client = None
    if not AIRPLANE_MODE:
        if is_gemini:
            client = make_genai_client()
            model = MODEL_NAME if model == "gemini" else model

        elif is_deepseek:
            client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )

        elif model == "chatgpt" or "gpt" in model.lower():
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        else:
            print("error: can only call gemini, deepseek-*, or chatgpt/gpt")
            quit()


    # we may want transaction bounds for later.
    # txn_bounds = load_txn_bounds()
    # act_bounds = load_activity_bounds_from_subset()


    MAX_ATTEMPTS = 2
    CONSECUTIVE_TIMEOUT_LIMIT = 3
    current_execpool = execpool

    # Build planned set once — rows and prompts don't change between attempts
    planned = set()
    for r in rows:
        if get_key(r) in prompts.keys():
            planned.add(get_key(r))

    for attempt in range(MAX_ATTEMPTS):
        if attempt > 0:
            current_execpool = make_executor()  # fresh pool; old threads drain safely
            print(f"[RETRY] Attempt {attempt + 1} — retrying errored rows...")

        seen_keys = load_seen_keys(output_jsonl)
        # Run rows concurrently with a semaphore; write outputs line-by-line without locks
        sem = asyncio.Semaphore(CONCURRENCY)
        consecutive_timeouts = [0]
        circuit_open = asyncio.Event()

        tasks = []  # defined before _guard so cancellation can reach all tasks

        # Default-parameter binding captures per-attempt state at definition time
        async def _guard(prompt_and_row: list,
                         _seen=seen_keys, _sem=sem,
                         _ct=consecutive_timeouts, _co=circuit_open,
                         _ep=current_execpool, _tasks=tasks):
            if _co.is_set():
                return
            async with _sem:
                if _co.is_set():
                    return
                try:
                    if AIRPLANE_MODE:
                        from dummy_response_text_generator import get_dummy_response_text
                        activity_id = prompt_and_row[1].get("activity_id", "unknown")
                        dummy_text = get_dummy_response_text(
                            response_schema, prompt_and_row[0], activity_id
                        )
                        dummy_obj = {"activity_id": activity_id, "response_text": dummy_text}
                        with open(output_jsonl_path, "a", encoding="utf-8", buffering=1) as f_out:
                            f_out.write(json.dumps(dummy_obj, ensure_ascii=False) + "\n")
                            f_out.flush()
                        return

                    if is_gemini:
                        obj = await asyncio.wait_for(
                            run_one_row(response_schema, prompt_and_row[0], prompt_and_row[1],
                                        client, _seen, output_jsonl, _ep, model),
                            timeout=TIMEOUT_SECONDS + 30
                        )
                    else:
                        obj = await asyncio.wait_for(
                            run_one_row_openai(prompt_and_row[0], prompt_and_row[1], client, _seen, output_jsonl, _ep, model),
                            timeout=TIMEOUT_SECONDS + 30
                        )
                except asyncio.TimeoutError:
                    obj = {"ERROR": "Overall operation timeout",
                           "activity_id": prompt_and_row[1].get("activity_id")}
                if not obj:
                    return
                is_timeout_err = "timeout" in str(obj.get("ERROR", "")).lower()
                if obj.get("ERROR"):
                    print(f"[ERROR] {obj.get('activity_id')} {obj['ERROR']}")
                    if is_timeout_err:
                        _ct[0] += 1
                        print(f"[CIRCUIT BREAKER] consecutive timeouts: {_ct[0]}/{CONSECUTIVE_TIMEOUT_LIMIT}")
                        if _ct[0] >= CONSECUTIVE_TIMEOUT_LIMIT:
                            print(f"[CIRCUIT BREAKER] threshold reached — cancelling {sum(1 for t in _tasks if not t.done())} remaining tasks")
                            _co.set()
                            for t in _tasks:
                                if not t.done():
                                    t.cancel()
                    else:
                        _ct[0] = 0
                else:
                    _ct[0] = 0

                with open(output_jsonl_path, "a", encoding="utf-8", buffering=1) as f_out:
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    f_out.flush()  # ensure it hits disk even when stdout is redirected

        for r in rows:
            if get_key(r) in prompts.keys():
                prompt_and_row = (prompts[get_key(r)], r)
                tasks.append(asyncio.create_task(_guard(prompt_and_row)))
            else:
                print(f"[DEBUG] no prompt for activity_id={get_key(r)} (skipping row)")

        done = load_seen_keys(output_jsonl)  # only successes, by your definition
        missing_done = sorted(planned - done)
        print(f"[DEBUG] attempt={attempt + 1} planned={len(planned)} success_done={len(done)} missing_success={len(missing_done)}")
        Path("planned_but_not_success.txt").write_text("\n".join(missing_done) + "\n", encoding="utf-8")
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                    print("Task error:", r)
        print(f"[attempt {attempt + 1}] Processed {len(tasks)} rows, output: {output_jsonl_path}.")

        if attempt > 0:
            current_execpool.shutdown(wait=False)  # clean up retry executor; caller owns attempt-0 pool

        done_after = load_seen_keys(output_jsonl)
        still_missing = planned - done_after
        if not still_missing:
            print("[RETRY] All planned rows succeeded.")
            break
        if attempt < MAX_ATTEMPTS - 1:
            print(f"[RETRY] {len(still_missing)} rows still missing — retrying in 10s...")
            await asyncio.sleep(10)  # let network stabilize before retry




def consolidate_rows_by_activity(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse one-page rows into a single bundle per activity_id.

    Input rows are assumed to have:
      - 'activity_id' (present)
      - 'section' (present)
      - 'cached_file' (present)
      - 'page_start' (1-based; present)

    Returns a list of dicts like:
      {
        "activity_id": "...",
        "section": "Baseline" | "Outcome",
        "items": [
            {"cached_file": "...pdf", "page_start": 7},   # 1-based page
            {"cached_file": "...pdf", "page_start": 9},
            ...
        ],
      }

    Order of items follows the input order (which you already sorted).
    """
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        k = (str(r["activity_id"]), str(r.get("section", "")), str(r.get("activity_title", "")))
        buckets[k].append({
            "cached_file": r["cached_file"],
            "doc_title": r["doc_title"],
            "page_start": int(r["page_start"]),
        })

    bundles: List[Dict[str, Any]] = []
    for (activity_id, section, title), items in buckets.items():
        bundles.append({
            "activity_id": activity_id,
            "title": title,
            "section": section,
            "items": items,
        })
    return bundles


def load_summaries_from_chatgpt_into_bundles(
    bundles: List[Dict[str, Any]],
    jsonl_path: Path | str = "../../data/outputs_summaries.jsonl",
) -> List[Dict[str, Any]]:
    """
    Read JSONL summaries keyed by activity_id and attach a 'chatgpt_description'
    to each bundle with a matching activity_id.

    Expects each JSONL line to at least contain:
      - 'activity_id': str
      - one text field among: 'chatgpt_description','text','response','content',
        'output','answer','summary','completion','message'

    Returns the same list (mutated) for convenience.
    """
    return load_generic_jsonl_and_put_into_bundles(bundles,jsonl_path, key_to_add_in="response_text",response_key_name="chatgpt_description")

def load_generic_jsonl_and_put_into_bundles(
    bundles: List[Dict[str, Any]],
    jsonl_path: Path | str = "../../data/outputs_summaries.jsonl",
    key_to_add_in: str = "response_text",
    response_key_name: str = "chatgpt_description"
) -> List[Dict[str, Any]]:
    """
    Read JSONL summaries keyed by activity_id and attach a 'chatgpt_description'
    to each bundle with a matching activity_id.

    Expects each JSONL line to at least contain:
      - 'activity_id': str
      - one text field among: 'chatgpt_description','text','response','content',
        'output','answer','summary','completion','message'

    Returns the same list (mutated) for convenience.
    """
    jsonl_path = Path(jsonl_path)
    summaries: Dict[str, str] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # pprint.pprint("obj")
            # pprint.pprint(obj)
            # print("key_to_add_in")
            # print(key_to_add_in)
            if "activity_id" in obj:
                if "ERROR" in obj.keys():
                    continue
                # pprint.pprint("obj")
                # pprint.pprint(obj)
                # print("getting rid of printouts..")
                # break
                aid = str(obj["activity_id"])
                # print("obj keys")
                # print(obj.keys())
                if key_to_add_in != "response_text":
                    response_obj = json.loads(obj["response_text"])

                    if key_to_add_in in response_obj.keys():
                        response = response_obj[key_to_add_in]
                    else:
                        continue
                else:
                    if "response_text" in obj.keys():
                        response = obj["response_text"]
                    else:
                        response = obj["response"]["content"]
                    # if key_to_add_in != "response_text":

                # response = str(obj[key_to_add_in])
                # val = next((str(obj[k]) for k in text_keys if k in obj and obj[k] is not None), None)
                # if val is not None:
                summaries[aid] = response

    for b in bundles:
        aid = str(b.get("activity_id"))
        if aid in summaries:
            b[response_key_name] = summaries[aid]

    return bundles




####################  fallback for if we can't find some of the ids from our first attempt, just get some early highly ranked pages instead ##########


# Only exclude these (exactly as requested)
EXCLUDED_CAT_PAGES = {
    "glossary",
    "blank_page",
    "table_of_contents",
    "references",
}


def _norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_cat(cat: str) -> str:
    return (cat or "").strip().lower().replace(" ", "_")



def _load_page_cats_by_cached_file(categories_csv: Path) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """
    (activity_id, cached_file) -> [(page_start, outcome_category_1_norm), ...]
    Falls back to (activity_id, doc_title_norm) if cached_file is missing in the CSV.
    """
    cats_cf: dict[tuple[str, str], list[tuple[int, str]]] = {}
    cats_title: dict[tuple[str, str], list[tuple[int, str]]] = {}

    if not categories_csv.exists():
        return {"_cf": cats_cf, "_title": cats_title}

    with categories_csv.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            aid = (r.get("activity_id") or "").strip()
            if not aid:
                continue
            try:
                p = int(r.get("page_start") or 0)
            except Exception:
                continue
            if p <= 0:
                continue

            cat = _norm_cat(r.get("outcome_category_1") or "")

            cached_file = (r.get("cached_file") or "").strip()
            if cached_file:
                cats_cf.setdefault((aid, cached_file), []).append((p, cat))
            else:
                # fallback if pdf_categories_scores.csv doesn't have cached_file
                t = _norm_title(r.get("doc_title") or "")
                if t:
                    cats_title.setdefault((aid, t), []).append((p, cat))

    for k in cats_cf:
        cats_cf[k].sort(key=lambda x: x[0])
    for k in cats_title:
        cats_title[k].sort(key=lambda x: x[0])

    return {"_cf": cats_cf, "_title": cats_title}


def _pick_first_pages_avoiding_excluded_v2(
    *,
    aid: str,
    cached_file: str,
    doc_title: str,
    pages_total: int,
    page_cats_maps: dict,
    max_pages: int = 10,
) -> list[int]:
    """
    Prefer earliest pages whose category is NOT excluded, if categorization exists.
    Match by (aid, cached_file) primarily; fall back to (aid, doc_title_norm).
    """
    if pages_total <= 0:
        return []

    cats_cf = page_cats_maps.get("_cf", {})
    cats_title = page_cats_maps.get("_title", {})

    # 1) best: match by cached_file
    key_cf = (aid, cached_file)
    if cached_file and key_cf in cats_cf:
        candidates = [p for (p, cat) in cats_cf[key_cf] if cat not in EXCLUDED_CAT_PAGES]
        candidates = sorted(set(p for p in candidates if 1 <= p <= pages_total))
        if candidates:
            return candidates[:max_pages]

    # 2) fallback: match by title
    tnorm = _norm_title(doc_title)
    key_t = (aid, tnorm)
    if tnorm and key_t in cats_title:
        candidates = [p for (p, cat) in cats_title[key_t] if cat not in EXCLUDED_CAT_PAGES]
        candidates = sorted(set(p for p in candidates if 1 <= p <= pages_total))
        if candidates:
            return candidates[:max_pages]

    # 3) last resort: just take first pages
    return list(range(1, min(max_pages, pages_total) + 1))






def _norm_lang(s: str) -> str:
    # keep it simple: "en", "en-us" -> "en"
    s = (s or "").strip().lower()
    if not s:
        return ""
    return s.split("-")[0].split("_")[0]

def _load_final_lookup_by_title(final_csv: Path) -> dict[tuple[str, str, str, str, int], dict]:
    """
    (aid, section, norm_title, norm_lang, pages) -> final row (includes cached_file)
    """
    lut = {}
    with final_csv.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            aid = (r.get("activity_id") or "").strip()
            section = (r.get("section") or "").strip()
            title = _norm_title(r.get("doc_title") or "")
            lang = _norm_lang(r.get("language") or "")
            cached = (r.get("cached_file") or "").strip()

            try:
                pages = int(r.get("pages") or 0)
            except Exception:
                pages = 0

            if not aid or section not in ("Baseline", "Outcome"):
                continue
            if not cached or pages <= 0 or not title:
                continue

            key = (aid, section, title, lang, pages)
            # If duplicates exist, first one wins (fine for your "unique enough" assumption)
            lut.setdefault(key, r)
    return lut


from collections import Counter

DEBUG_AID = "44000-P157571"
debug_miss_reasons = Counter()

def _load_top_ranked_doc_by_activity_strict_title_join(
    ranked_csv: Path,
    final_csv: Path,
    *,
    allow_de1_outcome_fallback: bool = True,
) -> dict[str, dict]:
    """
    aid -> {cached_file, doc_title, section, page_count/pages, assigned_rank, activity_title?}

    Join ranked->final by (aid, section, doc_title, language, pages/page_count).
    """
    lut = _load_final_lookup_by_title(final_csv)

    best_baseline = {}
    best_outcome = {}

    n_ranked = 0
    n_join_ok = 0
    n_join_miss = 0

    with ranked_csv.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            aid = (r.get("activity_id") or "").strip()
            section = (r.get("section") or "").strip()
            if not aid or section not in ("Baseline", "Outcome"):
                continue

            # skip scratchpad rows (they don't have rank/doc fields consistently)
            if section == "scratchpad":
                continue

            try:
                rank = int(r.get("assigned_rank") or -999)
            except Exception:
                continue
            if rank < 1:
                continue

            title_norm = _norm_title(r.get("doc_title") or "")
            lang_norm = _norm_lang(r.get("language") or "")
            try:
                pages = int(r.get("page_count") or 0)
            except Exception:
                pages = 0

            if not title_norm or pages <= 0:
                continue

            n_ranked += 1
            key = (aid, section, title_norm, lang_norm, pages)
            fr = lut.get(key)
            
            if not fr:
                n_join_miss += 1
                if aid == DEBUG_AID:
                    # collect candidates that share (aid, section)
                    cands = [k for k in lut.keys() if k[0] == aid and k[1] == section]

                    print("\n[DEBUG JOIN MISS] ranked row -> key not found")
                    print("  aid:", aid)
                    print("  section:", section)
                    print("  raw title:", repr(r.get("doc_title") or ""))
                    print("  title_norm:", repr(title_norm))
                    print("  raw lang:", repr(r.get("language") or ""))
                    print("  lang_norm:", repr(lang_norm))
                    print("  ranked page_count:", repr(r.get("page_count")))
                    print("  pages int:", pages)
                    print("  candidates in lut for (aid,section):", len(cands))

                    # show how candidates differ
                    same_title = [k for k in cands if k[2] == title_norm]
                    same_lang  = [k for k in cands if k[3] == lang_norm]
                    same_pages = [k for k in cands if k[4] == pages]
                    print("  candidates with same title_norm:", len(same_title))
                    print("  candidates with same lang_norm:", len(same_lang))
                continue
            n_join_ok += 1
            row = {
                "activity_id": aid,
                "section": section,
                "cached_file": (fr.get("cached_file") or "").strip(),
                "doc_title": (r.get("doc_title") or fr.get("doc_title") or "").strip(),
                "assigned_rank": rank,
                "activity_title": (fr.get("activity_title") or fr.get("title") or "").strip(),
                # keep around if useful later:
                "language": (r.get("language") or fr.get("language") or "").strip(),
                "pages": int(fr.get("pages") or pages),
            }
            if not row["cached_file"]:
                # should be impossible if lut is built correctly, but keep it safe
                continue

            target = best_baseline if section == "Baseline" else best_outcome
            cur = target.get(aid)
            if cur is None or rank < cur["assigned_rank"]:
                target[aid] = row

    print(f"[DEBUG] ranked rows eligible (rank>=1): {n_ranked}")
    print(f"[DEBUG] join ok: {n_join_ok}")
    print(f"[DEBUG] join miss: {n_join_miss}")
    print("[DEBUG] join-miss reasons (for debug aid):", debug_miss_reasons)

    # Return Outcome docs (for outcome fallback), not Baseline
    # But allow DE-1 activities to fall back to Baseline if no Outcome exists
    out = dict(best_outcome)

    if allow_de1_outcome_fallback:
        for aid, row in best_baseline.items():
            if aid.startswith("DE-1") and aid not in out:
                out[aid] = row

    return out








def add_fallback_rows_for_missing_activities_strict_baseline(
    rows_summary: list[dict],
    data_dir: Path,
    max_pages: int = 10,
    rated_ids = None
) -> list[dict]:
    """
    For activities that got 0 rows from load_and_filter_rows(...),
    add fallback rows from the *top-ranked Baseline doc* (strict),
    except for DE-1 where Outcome is allowed if Baseline doesn't exist.

    Does NOT require categorization; categorization is only used opportunistically
    to skip excluded categories.
    """
    ranked_csv = data_dir / "ranked_documents.csv"
    cats_csv = data_dir / "pdf_categories_scores.csv"
    pdf_dir = data_dir / "iati_all_pdfs"
    ranked_csv = data_dir / "ranked_documents.csv"
    final_csv  = data_dir / "activity_docs_log_final_restrictive.csv"

    topdoc = _load_top_ranked_doc_by_activity_strict_title_join(
        ranked_csv,
        final_csv,
        allow_de1_outcome_fallback=True,
    )
    print(f"[DEBUG] topdoc size: {len(topdoc)}")

    page_cats_maps = _load_page_cats_by_cached_file(cats_csv)

    have = {str(r.get("activity_id") or "").strip() for r in rows_summary if (r.get("activity_id") or "").strip()}
    if rated_ids is None:
        merged_path = data_dir / "merged_overall_ratings.jsonl"
        buf = io.StringIO()
        print("loading ratings...")
        from helpers_for_ratings_and_final_activity_features import load_ratings 
        print("done loading ratings...")

        with contextlib.redirect_stdout(buf):
            ratings = load_ratings(str(merged_path))
        ids_rated = set(ratings.index)
    else:
        ids_rated = rated_ids

    # 1-line change: only consider rated missing
    missing = [aid for aid in topdoc.keys() if aid in ids_rated and aid not in have]

    fallback_rows = []
    skip_de1 = 0
    skip_missing_file = 0
    print("missing")
    print(missing)
    for aid in missing:
        d = topdoc[aid]

        # enforce strict rule: non-DE-1 must be Baseline
        if not aid.startswith("DE-1") and d.get("section") != "Baseline":
            skip_de1 += 1
            continue

        cached_file = d["cached_file"]
        pdf_path = pdf_dir / cached_file
        if not pdf_path.exists():
            skip_missing_file += 1
            continue

        # try:
        #     reader = PdfReader(str(pdf_path))
        #     pages_total = len(reader.pages)
        # except Exception:
        #     continue

        pages_total = d.get("pages")

        pages = _pick_first_pages_avoiding_excluded_v2(
            aid=aid,
            cached_file=cached_file,
            doc_title=d.get("doc_title",""),
            pages_total=pages_total,
            page_cats_maps=page_cats_maps,
            max_pages=max_pages,
        )
        for p in pages:
            fallback_rows.append({
                "activity_id": aid,
                "activity_title": d.get("activity_title", "") or "",
                "section": "Baseline",
                "cached_file": cached_file,
                "doc_title": d.get("doc_title", "") or "",
                "page_start": p,
            })

    if fallback_rows:
        aids_added = {r["activity_id"] for r in fallback_rows}
        print(f"[FALLBACK] Added {len(fallback_rows)} page-rows covering {len(aids_added)} missing activities.")
    print("skip_missing_file")
    print(skip_missing_file)
    print("skip_de1")
    print(skip_de1)
    script_dir = Path(__file__).resolve().parent
    
    # keep only rated from the rows you already have # commented out bc do this earlier; faster
    # fallback_rows = [r for r in fallback_rows if str(r.get("activity_id","")).strip() in ids_rated]

    return rows_summary + fallback_rows
