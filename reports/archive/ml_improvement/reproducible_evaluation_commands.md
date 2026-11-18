# FinSight — Reproducible Evaluation Commands

This document provides exact, deterministic CLI commands to execute the audit and improvement sprint.

---

## 1. Environment & Setup

Ensure the virtual environment is active and `PYTHONPATH` includes `src`:

```bash
# Set PYTHONPATH for bash / zsh
export PYTHONPATH=src

# Or in PowerShell
$env:PYTHONPATH="src"
```

---

## 2. Baseline Model Frozen Evaluation

To re-score the frozen champion artifact without retraining:

```bash
python -m finsight.models --evaluate-frozen --model-path models/hist_gradient_boosting_no_day.joblib --data-path data/processed/transactions.csv.gz --reports-dir reports/remediation
```

---

## 3. Full ML Improvement Sprint Execution

To execute all 5 phases of the ML improvement sprint (Baseline audit, Fair benchmarking, Feature/imbalance experiments, Temporal robustness, and Deliverables generation):

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from finsight.ml_experiments import execute_full_ml_sprint
results = execute_full_ml_sprint()
print('Sprint status:', results['status'])
print('Gate decision:', results['gate_decision'])
"
```

---

## 4. Test Suite Execution

To run the comprehensive test suite validating models, features, leakage isolation, and risk rules:

```bash
pytest tests/ -v
```
