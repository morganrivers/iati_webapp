"""Single source of truth for LangSmith tracing.

Wraps LLM clients (OpenAI-compatible and google-genai) so that inference calls,
their inputs, outputs, and latency are captured by LangSmith when
LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) is set. No-op otherwise.

All callers that create an LLM client should route the client through
wrap_openai_client() or wrap_genai_client() so instrumentation lives in one
place.
"""

import contextlib
import contextvars
import os
import sys


_INIT_STATE = None
_CURRENT_PHASE_KEY = contextvars.ContextVar("iati_llm_phase_key", default=None)


def init_langsmith():
    global _INIT_STATE
    if _INIT_STATE is not None:
        return _INIT_STATE
    ls_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not ls_key:
        sys.stderr.write("no LANGSMITH/LANGCHAIN_API_KEY in env; langsmith tracing disabled\n")
        _INIT_STATE = False
        return False
    os.environ.setdefault("LANGCHAIN_API_KEY", ls_key)
    os.environ.setdefault("LANGSMITH_API_KEY", ls_key)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    proj = os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT")
    if proj:
        os.environ.setdefault("LANGCHAIN_PROJECT", proj)
        os.environ.setdefault("LANGSMITH_PROJECT", proj)
    try:
        import langsmith  # noqa: F401
    except ImportError:
        sys.stderr.write("langsmith not installed; tracing disabled\n")
        _INIT_STATE = False
        return False
    sys.stderr.write(f"langsmith tracing enabled, project={proj or 'default'}\n")
    _INIT_STATE = True
    return True


def wrap_openai_client(client):
    assert client is not None, "client is required"
    if not init_langsmith():
        return client
    from langsmith.wrappers import wrap_openai
    return wrap_openai(client)


class _TracedGenaiModels:
    """Proxy over google.genai Client.models that traces inference calls.

    __getattr__ fires only when normal lookup fails, so entries placed in
    __dict__ (_models, _traced_cache) resolve directly and never recurse.
    """

    _TRACED = {
        "generate_content": "llm",
        "generate_content_stream": "llm",
        "embed_content": "embedding",
        "count_tokens": "tool",
    }

    def __init__(self, models):
        assert models is not None, "models is required"
        self.__dict__["_models"] = models
        self.__dict__["_traced_cache"] = {}

    def __getattr__(self, name):
        if name in self._TRACED:
            cache = self.__dict__["_traced_cache"]
            if name not in cache:
                from langsmith import traceable
                inner = getattr(self._models, name)
                cache[name] = traceable(
                    name=f"gemini.{name}",
                    run_type=self._TRACED[name],
                )(inner)
            return cache[name]
        return getattr(self._models, name)


class _TracedGenaiClient:
    """Proxy over google.genai.Client whose .models is a traced proxy."""

    def __init__(self, client):
        assert client is not None, "client is required"
        self.__dict__["_client"] = client
        self.__dict__["_traced_models"] = _TracedGenaiModels(client.models)

    @property
    def models(self):
        return self._traced_models

    def __getattr__(self, name):
        return getattr(self._client, name)


def wrap_genai_client(client):
    assert client is not None, "client is required"
    if not init_langsmith():
        return client
    return _TracedGenaiClient(client)


@contextlib.contextmanager
def thread_scope(thread_id, **extra_metadata):
    """Group every LLM call inside the block under one LangSmith thread.

    LangSmith requires thread_id metadata on every run (including child runs)
    for filtering/token/cost aggregation to work. Entering this context makes
    langsmith's own tracing_context propagate that metadata to descendants,
    so wrap_openai_client() and wrap_genai_client() calls inside inherit it
    automatically. Extra kwargs are merged into the propagated metadata.
    """
    assert thread_id, "thread_id is required"
    if not init_langsmith():
        yield
        return
    from langsmith.run_helpers import tracing_context
    tid = str(thread_id)
    md = {"thread_id": tid, "session_id": tid}
    md.update(extra_metadata)
    with tracing_context(metadata=md):
        yield


@contextlib.contextmanager
def activity_phase_scope(activity_id, phase):
    """Thread scope keyed by one processing phase for a single activity.

    Each *invocation* gets its own thread (UUIDv7-suffixed) so re-running a
    phase (e.g. grading twice) produces two distinct threads. Nested re-entry
    for the same (activity_id, phase) inside one invocation is a no-op so the
    inner call shares the outer thread rather than starting a new one.

    All threads for one activity share activity_id + phase metadata, so the
    LangSmith UI can group them by filter.
    """
    assert activity_id, "activity_id is required"
    assert phase in {"extract", "grade", "forecast"}, f"unknown phase: {phase}"
    if not init_langsmith():
        yield
        return
    key = (str(activity_id), phase)
    if _CURRENT_PHASE_KEY.get() == key:
        yield
        return
    from langsmith import uuid7
    run_id = str(uuid7())
    thread_id = f"{activity_id}:{phase}:{run_id}"
    token = _CURRENT_PHASE_KEY.set(key)
    try:
        with thread_scope(
            thread_id,
            activity_id=str(activity_id),
            phase=phase,
            run_id=run_id,
        ):
            yield
    finally:
        _CURRENT_PHASE_KEY.reset(token)
