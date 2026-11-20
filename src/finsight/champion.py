"""
FinSight — the active champion model (P1.6).

`models/champion.json` names the model FinSight uses and the one it replaced. Load and score
the champion only through this module:

* `load_champion()` reads the registry, checks the artifact's SHA-256 and returns the bundle
  with its training configuration.
* `score_frame()` scores a full chronological frame. The recipient feature counts payments
  received in earlier steps, so the frame must start at step 1 (or the caller must accept
  truncated history explicitly); history is computed before any slicing.

Scores are risk scores from class-weighted training, not calibrated probabilities.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from finsight.train import MODEL_CONFIGS, add_history_features, build_features, file_sha256  # noqa: E402

REGISTRY_PATH = Path("models/champion.json")


class ChampionRegistryError(RuntimeError):
    """Raised when the registry, the artifact or the input frame is inconsistent."""


def read_registry(registry: Path = REGISTRY_PATH) -> Dict[str, Any]:
    return json.loads(Path(registry).read_text(encoding="utf-8"))


def _load_entry(entry: Dict[str, Any], root: Path) -> Dict[str, Any]:
    path = root / entry["artifact"]
    if not path.exists():
        raise ChampionRegistryError(f"{path} does not exist")
    sha = file_sha256(path)
    if sha != entry["artifact_sha256"]:
        raise ChampionRegistryError(f"{path} sha256 {sha} does not match the registry")
    bundle = joblib.load(path)
    cfg = MODEL_CONFIGS[entry["config_name"]]
    return {"entry": entry, "bundle": bundle, "config": cfg, "model": bundle["model"],
            "feature_names": list(bundle["feature_names"])}


def load_champion(registry: Path = REGISTRY_PATH, which: str = "active") -> Dict[str, Any]:
    """Load the active (default) or previous champion, verifying its hash."""
    reg = read_registry(registry)
    if which not in ("active", "previous"):
        raise ValueError("which must be 'active' or 'previous'")
    root = Path(registry).resolve().parent.parent
    return _load_entry(reg[which], root)


def score_frame(champion: Dict[str, Any], df: pd.DataFrame, allow_truncated_history: bool = False) -> np.ndarray:
    """Risk scores for every row of `df`, in order.

    `df` must hold step, type, amount and (for history features) nameDest. History is computed
    over the whole frame passed in, so pass every step from 1 up to the rows you need.
    """
    cfg = champion["config"]
    if cfg.get("history_features"):
        if "nameDest" not in df.columns and "name_dest" not in df.columns:
            raise ChampionRegistryError("the champion needs the recipient column to compute its history feature")
        if int(df["step"].min()) != 1 and not allow_truncated_history:
            raise ChampionRegistryError(
                "the frame must start at step 1 so recipient history is complete "
                "(pass allow_truncated_history=True to accept shorter history)"
            )
        df = add_history_features(df, cfg)
    X = build_features(df, cfg)[champion["feature_names"]]
    return champion["model"].predict_proba(X)[:, 1]


def operating_threshold(champion: Dict[str, Any], alerts_per_day: Optional[int] = None) -> float:
    entry = champion["entry"]
    if alerts_per_day is None:
        return float(entry["operating_point"]["threshold"])
    return float(entry["thresholds_by_alerts_per_day"][str(alerts_per_day)])
