"""RAG Forecast Viewer - shows all intermediate pipeline steps and final forecast."""
import io


import json
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st
from utils import notify_telegram, LLM_SESSION_CAP

WEBAPP_DIR = Path(__file__).resolve().parent.parent

LLM_FORECASTS_DIR = Path(__file__).resolve().parent.parent / "llm_forecasts"
from webapp_paths import DATA_DIR

_TAG = "deepseek_val"
_TAG_S3 = "deepseek_val_forced_rf"
_VARIANT = "exactly_like_halawi_et_al_better_model_rag_added_forced_rf"

# Switch: True = show ex-post outcome summary for each KNN neighbour instead of mock forecast.
SHOW_EXPOST_SUMMARY = True


class _LineCapture(io.TextIOBase):
    """Captures stdout line-by-line into a list, also echoing to the original stdout."""
    def __init__(self, logs_list, original_stdout=None, on_line=None):
        self._logs = logs_list
        self._original = original_stdout
        self._on_line = on_line
        self._buf = ""

    def write(self, s):
        if self._original:
            self._original.write(s)
            self._original.flush()
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self._logs.append(line)
            if self._on_line:
                self._on_line(line)
        return len(s)

    def flush(self):
        if self._original:
            self._original.flush()


def _run_rag_inprocess(activity_dir: Path, forecast_state: dict):
    """Run the RAG forecast in-process (no subprocess) to avoid re-importing packages.

    sys.stdout is temporarily redirected so all print/[RAM] output is captured in
    forecast_state['logs'] and also echoed to the original terminal stdout.
    """
    import run_rag_forecast as _rrm  # Python caches in sys.modules after first import

    def _on_line(line):
        """Parse [STAGE X/Y] markers so the progress bar stays up to date."""
        if "[STAGE " in line and "/" in line:
            try:
                stage_part = line.split("[STAGE ")[1].split("]")[0]
                forecast_state['current_stage'] = int(stage_part.split("/")[0])
                rest = line.split("]", 1)
                if len(rest) > 1:
                    forecast_state['stage_desc'] = rest[1].strip()
            except Exception:
                pass

    old_stdout = sys.stdout
    capture = _LineCapture(forecast_state['logs'], original_stdout=old_stdout, on_line=_on_line)
    sys.stdout = capture
    try:
        _rrm.main(activity_dir_override=activity_dir)
        forecast_state['returncode'] = 0
    except RuntimeError as e:
        forecast_state['error'] = str(e)
        forecast_state['returncode'] = 1
    except Exception as e:
        import traceback as _tb
        forecast_state['error'] = f"{e}\n{_tb.format_exc()}"
        forecast_state['returncode'] = 1
    finally:
        sys.stdout = old_stdout
        forecast_state['done'] = True


def _read_jsonl_for_aid(path: Path, activity_id: str) -> dict | None:
    """Return the first record in a JSONL file whose activity_id matches."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(obj.get("activity_id", "")) == activity_id:
                return obj
    return None


_expost_cache: dict[str, str] | None = None

def _load_expost_summaries() -> dict[str, str]:
    """Load postactivity_summaries.jsonl into {activity_id: summary_text}."""
    global _expost_cache
    if _expost_cache is not None:
        return _expost_cache
    path = DATA_DIR / "postactivity_summaries.jsonl"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = str(obj.get("activity_id", ""))
            resp = obj.get("response", {})
            if isinstance(resp, dict):
                txt = resp.get("content") or resp.get("text") or ""
            else:
                txt = ""
            if not txt:
                txt = obj.get("response_text", "")
            if aid and txt:
                out[aid] = txt
    _expost_cache = out
    return out


def _parse_examples_from_knn_prompt(prompt_text: str) -> list[dict]:
    """
    Extract the few-shot examples embedded in the KNN summary prompt.
    Returns a list of dicts with 'title' and 'forecast' keys.
    """
    examples = []
    parts = re.split(r"(EXAMPLE \d+:)", prompt_text)
    label_positions = [(i, parts[i]) for i in range(len(parts)) if re.match(r"EXAMPLE \d+:", parts[i])]

    seen_nums = {}
    for idx, label in label_positions:
        m = re.match(r"EXAMPLE (\d+):", label)
        if not m:
            continue
        num = int(m.group(1))
        content = parts[idx + 1] if idx + 1 < len(parts) else ""
        if num not in seen_nums:
            title_m = re.search(r"ACTIVITY TITLE:\s*(.+?)(?:\n|$)", content)
            title = title_m.group(1).strip() if title_m else f"Example {num}"
            seen_nums[num] = {"title": title, "activity_text": content.strip(), "forecast": ""}
        else:
            seen_nums[num]["forecast"] = content.strip()

    for num in sorted(seen_nums.keys()):
        examples.append(seen_nums[num])
    return examples


def _section(label: str, content: str | None, key: str):
    """Render a labelled expander. Shows a notice if content is missing."""
    with st.expander(label, expanded=False):
        if not content:
            st.info("⏳ Not yet generated — run the pipeline to populate this step.")
        else:
            st.text_area(" ", content, height=400, disabled=True,
                         label_visibility="collapsed", key=key)


def render_rag_forecast_page():
    st.title("Generate a Narrative Forecast")
    st.markdown(
        "Shows every intermediate step produced by a narrative forecast for the selected activity."
    )

    if not st.session_state.get("selected_project_folder"):
        st.warning("⚠️ No activity selected. Go to **Activity Forecasting** and select an activity first.")
        return

    activity_id = st.session_state.selected_project_folder
    st.info(f"**Activity:** {st.session_state.get('project_name', activity_id)}  |  ID: `{activity_id}`")

    activity_dir = WEBAPP_DIR / "extracted_pdf_data" / activity_id

    # ── In-progress: lock the page and show live progress ──────────────────────
    if st.session_state.rag_forecast_in_progress:
        st.error(
            "⏳ **Forecast in progress — please do not navigate away.** "
            "Page switching is not available until the forecast completes."
        )
        fs = st.session_state.rag_forecast_state

        log_box = st.empty()
        progress_box = st.empty()

        while not fs.get("done"):
            current_logs = list(fs.get("logs", []))
            current_stage = fs.get("current_stage", 0)
            total_stages = 8

            if current_stage:
                progress_box.progress(
                    min(current_stage / total_stages, 1.0),
                    text=f"Stage {current_stage}/{total_stages}: {fs.get('stage_desc', '')}",
                )

            log_box.code(
                "\n".join(current_logs) if current_logs else "(waiting for progress updates...)",
                language="text",
            )

            time.sleep(1)

        # Done
        returncode = fs.get("returncode", -1)
        error = fs.get("error")

        st.session_state.rag_last_log = list(fs.get("logs", []))

        if error:
            st.session_state.rag_last_status = f"exception:{error}"
        elif returncode == 0:
            st.session_state.rag_last_status = "ok"
        else:
            st.session_state.rag_last_status = f"error:{returncode}"

        st.session_state.rag_forecast_in_progress = False
        st.session_state.rag_forecast_state = None
        st.rerun()

    # ── Run button ─────────────────────────────────────────────────────────────
    st.warning(
        "⚠️ **Note:** The statistical model (Random Forest) prediction from the **Activity Forecasting** page "
        "is used as an input to this narrative forecast. Run and save that prediction first for best results."
    )

    if not st.session_state.get('llm_authenticated', False):
        st.warning("🔒 LLM access required — authenticate on the Activity Forecasting page to generate forecasts.")
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("▶ Generate Forecasts", type="primary", width='stretch',
                                disabled=not st.session_state.get('llm_authenticated', False))
    with col_status:
        last_status = st.session_state.get("rag_last_status", "")
        if last_status == "ok":
            st.success("✅ Pipeline complete!")
        elif last_status.startswith("error:"):
            code = last_status.split(":", 1)[1]
            st.error(f"❌ Pipeline exited with code {code}")
        elif last_status.startswith("exception:"):
            msg = last_status.split(":", 1)[1]
            st.error(f"❌ Failed to run pipeline: {msg}")

    if run_clicked:
        st.session_state.rag_forecast_confirm_pending = True
        st.rerun()

    # ── Confirmation dialog ────────────────────────────────────────────────────
    if st.session_state.get('rag_forecast_confirm_pending'):
        st.divider()
        st.warning(
            "⚠️ **Are you sure?**\n\n"
            "Generating the narrative forecast takes **several minutes** (multiple LLM API calls). "
            "During this time **page switching is not possible** — the app will be locked to this page "
            "until the forecast completes."
        )
        col_yes, col_no, _ = st.columns([1, 1, 4])
        if col_yes.button("✅ Yes, proceed", type="primary"):
            if st.session_state.llm_call_count >= LLM_SESSION_CAP:
                st.error(f"❌ Session limit reached ({LLM_SESSION_CAP} LLM calls). Refresh the page to reset.")
                st.stop()
            st.session_state.rag_forecast_confirm_pending = False
            forecast_state = {
                'logs': [],
                'done': False,
                'current_stage': 0,
                'stage_desc': '',
                'returncode': None,
                'error': None,
            }
            st.session_state.rag_forecast_state = forecast_state
            st.session_state.rag_forecast_in_progress = True
            st.session_state.llm_call_count += 1
            notify_telegram(
                f"🔔 RAG forecast triggered\n"
                f"Session calls so far: {st.session_state.llm_call_count}\n"
                f"Activity: {activity_id}"
            )
            st.session_state.rag_last_status = ""
            t = threading.Thread(
                target=_run_rag_inprocess,
                args=(activity_dir, forecast_state),
                daemon=True,
            )
            t.start()
            st.rerun()
        if col_no.button("❌ Cancel"):
            st.session_state.rag_forecast_confirm_pending = False
            st.rerun()
        st.stop()

    # Show persistent log from the most recent run
    if st.session_state.get("rag_last_log"):
        with st.expander("📋 Pipeline log (last run)", expanded=False):
            st.code("\n".join(st.session_state["rag_last_log"]), language="text")

    # ── Read all available pipeline files ──────────────────────────────────────
    files = {
        "knn_dryrun":      LLM_FORECASTS_DIR / f"dryrun_knn_summary_{_TAG}_call_1.jsonl",
        "knn_out":         LLM_FORECASTS_DIR / f"outputs_knn_summary_{_TAG}_call_1.jsonl",
        "phrase_dryrun":   LLM_FORECASTS_DIR / f"dryrun_phrasegen_{_VARIANT}_{_TAG}_call_1.jsonl",
        "phrase_out":      LLM_FORECASTS_DIR / f"outputs_phrasegen_{_TAG}_call_1.jsonl",
        "rag_dryrun":      LLM_FORECASTS_DIR / f"dryrun_prompts_rag_synthesis_{_VARIANT}_call_1.jsonl",
        "rag_out":         LLM_FORECASTS_DIR / f"outputs_raganswers_variant_{_VARIANT}_{_TAG}_call_1.jsonl",
        "s1_dryrun":       LLM_FORECASTS_DIR / f"dryrun_prompts_{_VARIANT}_{_TAG_S3}_s1_call_1.jsonl",
        "s1_out":          LLM_FORECASTS_DIR / f"outputs_{_VARIANT}_{_TAG_S3}_s1_call_1.jsonl",
        "s2_dryrun":       LLM_FORECASTS_DIR / f"dryrun_prompts_{_VARIANT}_{_TAG_S3}_s2_call_1.jsonl",
        "s2_out":          LLM_FORECASTS_DIR / f"outputs_{_VARIANT}_{_TAG_S3}_s2_call_1.jsonl",
        "s3_dryrun":       LLM_FORECASTS_DIR / f"dryrun_prompts_{_VARIANT}_{_TAG_S3}_s3_call_1.jsonl",
        "s3_out":          LLM_FORECASTS_DIR / f"outputs_{_VARIANT}_{_TAG_S3}_s3_call_1.jsonl",
    }

    records = {k: _read_jsonl_for_aid(v, activity_id) for k, v in files.items()}

    def _prompt(key):
        r = records.get(key)
        return r.get("prompt", "") if r else None

    def _sysmsg(key):
        r = records.get(key)
        return r.get("system_msg", "") if r else None

    def _output(key):
        r = records.get(key)
        if not r:
            return None
        resp = r.get("response")
        if isinstance(resp, dict):
            txt = resp.get("content") or resp.get("text") or ""
        else:
            txt = ""
        return txt or r.get("response_text") or None

    # ── Final Forecast ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Final Forecast")
    s3_text = _output("s3_out")
    if s3_text:
        st.success(s3_text)
    else:
        st.info("⏳ Final forecast not yet generated.")

    # ── KNN Neighbors / Mock Forecasts ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Three similar activities from the database and their outcomes")
    st.markdown(
        "The most similar activities as measured by vector similarity of the activity description with a rating of `Highly Unsatisfactory`, `Unsatisfactory`, or `Moderately Unsatisfactory`. "
    )


    knn_dryrun_rec = records.get("knn_dryrun")
    if knn_dryrun_rec and knn_dryrun_rec.get("prompt"):
        examples = _parse_examples_from_knn_prompt(knn_dryrun_rec["prompt"])
        neighbor_ids = knn_dryrun_rec.get("knn_neighbor_ids", [])
        expost = _load_expost_summaries() if SHOW_EXPOST_SUMMARY else {}
        if examples:
            for i, ex in enumerate(examples, 1):
                with st.expander(f"Example {i}: {ex['title']}", expanded=(i == 1)):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Activity Info**")
                        st.text_area(" ", ex["activity_text"], height=300, disabled=True,
                                     label_visibility="collapsed",
                                     key=f"knn_ex_{i}_activity")
                    with col_b:
                        if SHOW_EXPOST_SUMMARY:
                            st.markdown("**Ex-Post Outcome Summary**")
                            nb_id = neighbor_ids[i - 1] if i - 1 < len(neighbor_ids) else None
                            expost_text = expost.get(nb_id, "") if nb_id else ""
                            if not expost_text and nb_id:
                                expost_text = f"(No ex-post summary found for {nb_id})"
                            elif not expost_text:
                                expost_text = "(Neighbor ID not recorded — re-run the pipeline to capture it)"
                            st.text_area(" ", expost_text, height=300, disabled=True,
                                         label_visibility="collapsed",
                                         key=f"knn_ex_{i}_expost")
                        else:
                            st.markdown("**Mock Forecast**")
                            st.text_area(" ", ex["forecast"] if ex["forecast"] else "No mock forecast found",
                                         height=300, disabled=True,
                                         label_visibility="collapsed",
                                         key=f"knn_ex_{i}_forecast")
        else:
            st.info("Could not parse examples from 3 Similar Activities prompt.")
    else:
        st.info("⏳ 3 Similar Activities dryrun prompt not yet generated.")

    # ── Stage: 3 Similar Activities Summary ────────────────────────────────────
    st.markdown("---")
    st.subheader("Stage 1: 3 Similar Activities Summary")
    _section("Prompt", _prompt("knn_dryrun"), "knn_prompt")
    _section("LLM Output", _output("knn_out"), "knn_out")

    # ── Stage: Phrase Generation ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Stage 2: RAG Query Phrase Generation")
    _section("Prompt", _prompt("phrase_dryrun"), "phrase_prompt")
    _section("LLM Output", _output("phrase_out"), "phrase_out")

    # ── Stage: RAG Synthesis ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Stage 3: RAG Synthesis (PDF evidence retrieval)")
    _section("Prompt", _prompt("rag_dryrun"), "rag_prompt")
    _section("LLM Output", _output("rag_out"), "rag_out")

    # ── Stage: s1, s2, s3 Forecasts ───────────────────────────────────────────
    for stage, label in [("s1", "Why it might go badly"), ("s2", "Why it might go well"), ("s3", "Final Forecast")]:
        st.markdown("---")
        st.subheader(f"{label}")
        _section("Prompt", _prompt(f"{stage}_dryrun"), f"{stage}_prompt")
        _section("LLM Output", _output(f"{stage}_out"), f"{stage}_out_display")
