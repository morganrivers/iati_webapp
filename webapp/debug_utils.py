"""Centralised RAM-debug helper.

Set VERBOSE = True locally when profiling memory; leave False for production.
When False, _print_ram is a no-op and psutil is never imported.
"""

VERBOSE = False

if VERBOSE:
    import os
    import psutil as _psutil

    def _print_ram(label: str) -> None:
        mb = _psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"[RAM] {label}: {mb:.1f} MB", flush=True)
else:
    def _print_ram(label: str) -> None:
        pass


# Gates the [LOCDBG] traces for project-switch / activity-location debugging.
# Leave False in production; set True to trace project load + location widget state.
LOC_DEBUG = False

if LOC_DEBUG:
    def _loc_debug(msg: str) -> None:
        print(f"[LOCDBG] {msg}", flush=True)
else:
    def _loc_debug(msg: str) -> None:
        pass


# Gates the [GDP-TRACE] traces for country-feature (GDP/CPIA/WGI) write/read debugging.
# Leave False in production; set True to trace gdp_percap through extraction,
# widget-init, save/load, and field_edited resync.
GDP_DEBUG = False

if GDP_DEBUG:
    def _gdp_debug(msg: str) -> None:
        print(f"[GDP-TRACE] {msg}", flush=True)
else:
    def _gdp_debug(msg: str) -> None:
        pass
