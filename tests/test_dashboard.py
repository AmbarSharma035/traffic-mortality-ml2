"""Smoke test: every dashboard page renders without raising. Needs `python run_pipeline.py` first."""
import pytest

from src import config as cfg

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["Project status", "Severity model", "Hotspots", "Risk (prototype)", "Predict a crash"]


@pytest.mark.skipif(not cfg.METRICS_FILE.exists(), reason="run `python run_pipeline.py` first")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(cfg.ROOT / "app" / "dashboard.py"), default_timeout=90).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
