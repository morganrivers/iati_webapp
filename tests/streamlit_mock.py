"""Single shared Streamlit mock for the test suite.

Every test module binds the same webapp modules (ui_components, project_manager,
...). Those modules capture ``streamlit`` at import time, so all test files must
share ONE mock object. If each file built its own mock, writes from one file's
tests would land in a different ``session_state`` than the imported functions
read. Import this module before importing any webapp code.
"""
import sys
from unittest.mock import MagicMock


class SessionState(dict):
    """dict with attribute access — mirrors st.session_state API."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        try:
            del self[k]
        except KeyError:
            raise AttributeError(k)


st_mock = MagicMock()
st_mock.session_state = SessionState()
st_mock.error = MagicMock()
st_mock.success = MagicMock()
st_mock.warning = MagicMock()
st_mock.info = MagicMock()
st_mock.rerun = MagicMock()
st_mock.cache_resource = lambda f: f

sys.modules["streamlit"] = st_mock
