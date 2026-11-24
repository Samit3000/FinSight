"""Development-fold curves: thinning keeps shape, the reproduction check is strict, and the test hours are never read."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight import dev_curves as dc  # noqa: E402


def test_thin_keeps_endpoints_and_shape():
    x = np.linspace(0, 1, 10_001)
    pts = dc.thin(x, x ** 2)
    assert pts[0] == [0.0, 0.0] and pts[-1] == [1.0, 1.0]
    assert 50 < len(pts) < 1000
    assert all(abs(py - px ** 2) < 1e-4 for px, py in pts)


def test_curves_agree_with_sklearn_scalars():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    s = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
    c = dc.curves(y, s)
    from sklearn.metrics import average_precision_score, roc_auc_score
    assert c["average_precision"] == round(average_precision_score(y, s), 4) and c["roc_auc"] == round(roc_auc_score(y, s), 4)
    assert sum(b["rows"] for b in c["reliability"]) == len(y)
    roc = np.array(c["roc"])
    assert roc[0].tolist() == [0.0, 0.0] and roc[-1].tolist() == [1.0, 1.0] and np.all(np.diff(roc[:, 0]) >= 0)


def test_reproduction_check_is_exact():
    rec = {"F3": {"E4": (0.653, 0.9768)}}
    assert dc.check_reproduction({"E4": {"F3": {"average_precision": 0.653, "roc_auc": 0.9768}}}, rec) == []
    assert dc.check_reproduction({"E4": {"F3": {"average_precision": 0.6531, "roc_auc": 0.9768}}}, rec)
    assert dc.check_reproduction({}, rec) == ["E4 F3: not computed"]


def test_recorded_metrics_cover_all_folds():
    rec = dc.recorded_metrics(Path("."))
    assert set(rec) == {"F1", "F2", "F3"} and set(rec["F3"]) == {"E0", "E1", "E2", "E3", "E4"}


def test_test_hours_are_never_loaded():
    src = Path("src/finsight/dev_curves.py").read_text(encoding="utf-8")
    assert "load_development_data" in src and "assert_development_steps" in src
    assert "load_sealed" not in src and "unlock" not in src.lower()
    assert all(f.test_max_step < 651 for f in __import__("finsight.evaluate", fromlist=["x"]).WALK_FORWARD_FOLDS)


def test_run_refuses_test_hours():
    import pandas as pd
    df = pd.DataFrame({"step": [700], "isFraud": [0]})
    with pytest.raises(Exception, match="651|final|locked|sealed"):
        dc.run(df, labels=["E4"], folds=["F3"])
