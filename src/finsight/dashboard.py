"""
FinSight — static HTML/CSS dashboard (Milestone 7, replaces the planned Power BI report).

`collect(conn)` reads the analytical views in a READ-ONLY session; `render(data)` turns that snapshot into one
self-contained HTML page (inline CSS, no external libraries, light and dark themes); `main()` writes
`dashboard/index.html` and `dashboard/data.json`.

What the page shows, and what it deliberately does not:

* Model evaluation comes only from `vw_risk_detection_benchmark` (development partitions scored by code; the
  sealed-test rows are the recorded one-time P1.5 result). No new performance figures are computed here.
* Workload, alert mix and reason panels use the label-free views 05–08.
* Dataset panels use the descriptive views 01 and 03 (labelled fraud counts of the dataset itself).
* Model attribution is labelled as not evidence of fraud; rule conditions are labelled as context.

Usage (database chosen by DB_HOST / DB_PORT / DB_NAME; nothing is written to it):
    py src/finsight/dashboard.py --out dashboard
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

logger = logging.getLogger("finsight.dashboard")

PERIODS = ("training", "calibration", "sealed_test_reported")
PERIOD_LABEL = {"training": "Training hours (1–550)", "calibration": "Calibration hours (551–650)",
                "sealed_test_reported": "Sealed-test hours (651–743)"}
PERIOD_VAR = {"training": "--series-1", "calibration": "--series-2", "sealed_test_reported": "--series-3"}
HOUR_BANDS = ("00-05", "06-08", "09-17", "18-23")
AMOUNT_BANDS = ("<100k", "100k-500k", "500k-1M", ">=1M")
FEATURE_TEXT = {"amount": "Amount", "transaction_hour": "Hour of day", "recipient_prior_in_count": "Recipient history",
                "type_CASH_OUT": "Type is CASH_OUT", "type_TRANSFER": "Type is TRANSFER", "type_PAYMENT": "Type (PAYMENT flag)",
                "type_CASH_IN": "Type (CASH_IN flag)", "type_DEBIT": "Type (DEBIT flag)"}
RULE_TEXT = {"RULE_001_HIGH_VALUE": "High value (≥ $200k)", "RULE_002_DRAINAGE_RAIL": "Drainage rail (TRANSFER / CASH_OUT)",
             "RULE_003_OVERNIGHT_DRAINAGE": "Overnight (01–05, ≥ $10k)", "RULE_004_LARGE_ROUND_AMOUNT": "Round amount (≥ $50k, × $5k)",
             "none": "No rule met"}
PARTITION_ORDER = ("validation", "development_holdout", "sealed_final_test")
MODEL_NOTE = ("Model attribution shows how inputs moved a model score; it is not evidence that a transaction is fraudulent.")
RULE_NOTE = "Rule conditions are deterministic checks shown for context; they do not raise alerts."


# ------------------------------------------------------------------------------
# Data (read-only)
# ------------------------------------------------------------------------------
def _rows(cur, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    out = []
    for r in cur.fetchall():
        rec = {}
        for c, v in zip(cols, r):
            if hasattr(v, "__float__") and not isinstance(v, (int, bool)):
                v = float(v)
            rec[c] = v
        out.append(rec)
    return out


def collect(conn, champion_entry: Dict[str, Any], alert_queue_limit: int = 25) -> Dict[str, Any]:
    """Snapshot of every number the page shows. The session is set read-only first."""
    conn.set_session(readonly=True)
    sha = champion_entry["artifact_sha256"]
    with conn.cursor() as cur:
        cur.execute("SHOW transaction_read_only")
        assert cur.fetchone()[0] == "on"
        d: Dict[str, Any] = {}
        d["dataset"] = _rows(cur, "SELECT SUM(transaction_count)::bigint AS transactions, SUM(gross_transaction_value) AS value, "
                                  "SUM(labelled_fraud_transaction_count)::bigint AS labelled_fraud, COUNT(*) AS days "
                                  "FROM vw_daily_transaction_kpis")[0]
        d["rails"] = _rows(cur, "SELECT payment_rail, transaction_count, rail_volume_share_pct, labelled_fraud_count, "
                                "labelled_fraud_rate_pct FROM vw_payment_rail_performance ORDER BY transaction_count DESC")
        d["benchmark"] = _rows(cur, "SELECT evaluation_partition, system_name, operating_point, alerts_per_day, precision, recall, "
                                    "average_precision, true_positives, false_positives, false_negatives "
                                    "FROM vw_risk_detection_benchmark ORDER BY evaluation_partition, system_name")
        d["workload"] = _rows(cur, "SELECT simulated_day, evaluation_period, hours_covered, is_full_day, transactions_scored, alerts, "
                                   "first_time_recipient_alerts FROM vw_alert_workload_daily WHERE model_artifact_sha256 = %s "
                                   "ORDER BY simulated_day, evaluation_period", (sha,))
        d["mix"] = _rows(cur, "SELECT evaluation_period, payment_type, hour_band, amount_band, alerts FROM vw_alert_mix "
                              "WHERE model_artifact_sha256 = %s", (sha,))
        d["reasons"] = _rows(cur, "SELECT evaluation_period, reason_source, reason, alerts FROM vw_alert_reason_summary "
                                  "WHERE model_artifact_sha256 = %s", (sha,))
        d["queue"] = _rows(cur, "SELECT transaction_id, step, simulated_day, transaction_hour, type, amount, evaluation_period, "
                                "risk_score, model_summary, rule_ids, rule_tier FROM vw_alert_queue WHERE model_artifact_sha256 = %s "
                                "ORDER BY step DESC, risk_score DESC LIMIT %s", (sha, alert_queue_limit))
        d["runs"] = _rows(cur, "SELECT count(*) AS runs, COALESCE(SUM(rows_scored), 0)::bigint AS rows_scored, "
                               "COALESCE(SUM(alerts), 0)::bigint AS alerts, MAX(completed_at)::text AS last_run "
                               "FROM scoring_run_log WHERE model_artifact_sha256 = %s", (sha,))[0]
    conn.rollback()
    for q in d["queue"]:
        q["rule_ids"] = list(q["rule_ids"] or [])
    d["champion"] = {k: champion_entry[k] for k in ("artifact", "artifact_sha256", "model_name", "operating_point", "evidence")}
    return d


# ------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------
def _e(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _int(v: Any) -> str:
    return f"{int(round(float(v))):,}"


def _pct(v: Optional[float], nd: int = 1) -> str:
    return "–" if v is None else f"{100 * float(v):.{nd}f}%"


CSS = """
:root{color-scheme:light;--surface-0:#f4f3f0;--surface-1:#fcfcfb;--border:#e2e0da;--grid:#ecebe7;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#76756f;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--seq-0:#cde2fb;--seq-1:#256abf;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--surface-0:#121211;
--surface-1:#1a1a19;--border:#34332f;--grid:#2a2a27;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9d9c93;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--seq-0:#1e2a3a;--seq-1:#5598e7;}}
:root[data-theme="dark"]{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;--grid:#2a2a27;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9d9c93;--series-1:#3987e5;--series-2:#d95926;
--series-3:#199e70;--seq-0:#1e2a3a;--seq-1:#5598e7;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:0 0 4px}
.sub{color:var(--text-secondary);margin:0 0 20px}.note{color:var(--text-muted);font-size:12px;margin:8px 0 0}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr))}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:16px;min-width:0}
.wide{grid-column:1/-1}
.tiles{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr));margin-bottom:16px}
.tile .v{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums}.tile .l{color:var(--text-secondary);font-size:13px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--text-secondary);font-weight:600;font-size:12px}td.n,th.n{text-align:right}
.scroll{overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0;font-size:12px;color:var(--text-secondary)}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.cols{position:relative;display:flex;align-items:flex-end;gap:2px;height:200px;border-bottom:1px solid var(--border);margin-top:12px}
.col{flex:1;display:flex;flex-direction:column-reverse;gap:2px;height:100%;justify-content:flex-start;min-width:0}
.seg{border-radius:4px 4px 0 0;min-height:1px}.seg.partial{opacity:.45}
.col:hover .seg{outline:2px solid var(--text-primary);outline-offset:1px}
.ref{position:absolute;left:0;right:0;border-top:1px dashed var(--text-muted)}
.ref span{position:absolute;right:0;top:-18px;font-size:11px;color:var(--text-muted);background:var(--surface-1);padding:0 4px}
.xaxis{display:flex;gap:2px;font-size:10px;color:var(--text-muted)}.xaxis span{flex:1;text-align:center;min-width:0}
tr.hl td{font-weight:650;background:color-mix(in oklab, var(--seq-1) 12%, var(--surface-1))}
.hbar{display:grid;grid-template-columns:minmax(120px,45%) 1fr auto;gap:8px;align-items:center;margin:6px 0;font-size:13px}
.hbar .t{height:12px;background:var(--grid);border-radius:4px;overflow:hidden}
.hbar .f{height:100%;background:var(--seq-1);border-radius:0 4px 4px 0}
.hbar:hover .f{outline:2px solid var(--text-primary)}
.heat td.c{text-align:right;color:var(--text-primary)}
.tag{display:inline-block;padding:1px 6px;border-radius:999px;font-size:11px;border:1px solid var(--border);
color:var(--text-secondary);white-space:nowrap}
.empty{color:var(--text-muted);padding:24px 0;text-align:center}
.disclaimer{background:var(--surface-1);border:1px solid var(--border);border-left:4px solid var(--text-muted);
border-radius:8px;padding:12px 16px;margin:16px 0;color:var(--text-secondary)}
"""


def _tiles(d: Dict[str, Any]) -> str:
    ds, runs, champ = d["dataset"], d["runs"], d["champion"]
    sealed = {r["system_name"]: r for r in d["benchmark"] if r["evaluation_partition"] == "sealed_final_test"}
    e4 = next((r for n, r in sealed.items() if "E4" in n), None)
    prev = next((r for n, r in sealed.items() if "previous champion" in n), None)
    tiles = [
        (_int(ds["transactions"]), "Transactions in the dataset"),
        (f"{_int(ds['labelled_fraud'])} ({100 * ds['labelled_fraud'] / ds['transactions']:.3f}%)", "Labelled fraud in the dataset"),
        (_int(runs["rows_scored"]), "Transactions scored by the champion"),
        (_int(runs["alerts"]), "Alerts stored, all hours"),
    ]
    if e4 and prev:
        tiles.append((f"{e4['average_precision']:.3f}", f"Champion average precision on the sealed test (previous model {prev['average_precision']:.3f})"))
    tiles.append((f"{champ['operating_point']['threshold']:.4f}", "Alert threshold (≈ 100 alerts/day on calibration hours)"))
    return '<div class="tiles">' + "".join(
        f'<div class="card tile"><div class="v">{_e(v)}</div><div class="l">{_e(l)}</div></div>' for v, l in tiles) + "</div>"


def _workload_chart(d: Dict[str, Any]) -> str:
    rows = d["workload"]
    if not rows:
        return '<p class="empty">No scores stored yet.</p>'
    days = sorted({r["simulated_day"] for r in rows})
    by_day = {day: [r for r in rows if r["simulated_day"] == day] for day in days}
    peak = max(sum(r["alerts"] for r in v) for v in by_day.values())
    top = max(peak, 100) * 1.1
    cols = []
    for day in days:
        segs = []
        for r in sorted(by_day[day], key=lambda r: PERIODS.index(r["evaluation_period"])):
            h = 100 * r["alerts"] / top
            part = "" if r["is_full_day"] else " partial"
            tip = (f"Day {day} · {PERIOD_LABEL[r['evaluation_period']]} · {r['alerts']:,} alerts from "
                   f"{r['transactions_scored']:,} transactions ({r['hours_covered']} hours"
                   f"{'' if r['is_full_day'] else ', partial day'})")
            segs.append(f'<div class="seg{part}" style="height:{h:.2f}%;background:var({PERIOD_VAR[r["evaluation_period"]]})" '
                        f'title="{_e(tip)}"></div>')
        cols.append(f'<div class="col" role="img" aria-label="Day {day}">' + "".join(segs) + "</div>")
    ref = 100 * 100 / top
    legend = "".join(f'<span><span class="sw" style="background:var({PERIOD_VAR[p]})"></span>{_e(PERIOD_LABEL[p])}</span>'
                     for p in PERIODS) + '<span>Faded bars: partial days</span>'
    axis = "".join(f"<span>{day if day % 5 == 0 or day == 1 else ''}</span>" for day in days)
    table = "".join(f"<tr><td>{r['simulated_day']}</td><td>{_e(PERIOD_LABEL[r['evaluation_period']])}</td>"
                    f"<td class='n'>{r['hours_covered']}</td><td class='n'>{r['transactions_scored']:,}</td>"
                    f"<td class='n'>{r['alerts']:,}</td><td class='n'>{r['first_time_recipient_alerts']:,}</td></tr>" for r in rows)
    return (f'<div class="legend">{legend}</div><div class="cols">{"".join(cols)}'
            f'<div class="ref" style="bottom:{ref:.2f}%"><span>100 alerts/day</span></div></div>'
            f'<div class="xaxis">{axis}</div><p class="note">Simulated day (1–31). Hover a bar for exact counts.</p>'
            f'<details><summary class="note">Table view</summary><div class="scroll"><table><thead><tr><th>Day</th><th>Period</th>'
            f'<th class="n">Hours</th><th class="n">Transactions</th><th class="n">Alerts</th><th class="n">First-time recipient</th>'
            f'</tr></thead><tbody>{table}</tbody></table></div></details>')


def _hbars(items: List[tuple], unit: str = "alerts") -> str:
    if not items:
        return '<p class="empty">No data yet.</p>'
    top = max(v for _, v in items) or 1
    return "".join(f'<div class="hbar" title="{_e(label)}: {v:,} {unit}"><span>{_e(label)}</span>'
                   f'<span class="t"><span class="f" style="display:block;width:{100 * v / top:.1f}%"></span></span>'
                   f'<span class="n">{v:,}</span></div>' for label, v in items)


def _mix_panels(d: Dict[str, Any]) -> str:
    mix = d["mix"]
    if not mix:
        return '<div class="card wide"><h2>Alert mix</h2><p class="empty">No scores stored yet.</p></div>'
    types = sorted({m["payment_type"] for m in mix})
    cell = {(t, h): sum(m["alerts"] for m in mix if m["payment_type"] == t and m["hour_band"] == h) for t in types for h in HOUR_BANDS}
    top = max(cell.values()) or 1
    rows = "".join("<tr><th>" + _e(t) + "</th>" + "".join(
        f'<td class="c" style="background:color-mix(in oklab, var(--seq-1) {8 + 70 * cell[(t, h)] / top:.0f}%, var(--surface-1))" '
        f'title="{_e(t)}, {h}: {cell[(t, h)]:,} alerts">{cell[(t, h)]:,}</td>' for h in HOUR_BANDS) + "</tr>" for t in types)
    heat = (f'<table class="heat"><thead><tr><th>Type</th>' + "".join(f'<th class="n">{h}</th>' for h in HOUR_BANDS)
            + f"</tr></thead><tbody>{rows}</tbody></table>")
    amounts = [(b, sum(m["alerts"] for m in mix if m["amount_band"] == b)) for b in AMOUNT_BANDS]
    per = "".join(f"<tr><td>{_e(PERIOD_LABEL[p])}</td><td class='n'>{sum(m['alerts'] for m in mix if m['evaluation_period'] == p):,}</td></tr>"
                  for p in PERIODS)
    return (f'<div class="card"><h2>Alerts by payment type and hour</h2><p class="note">All stored alerts; darker = more alerts.</p>'
            f'<div class="scroll">{heat}</div></div>'
            f'<div class="card"><h2>Alerts by amount</h2>{_hbars(amounts)}'
            f'<table style="margin-top:12px"><thead><tr><th>Period</th><th class="n">Alerts</th></tr></thead><tbody>{per}</tbody></table></div>')


def _reason_panels(d: Dict[str, Any]) -> str:
    rs = d["reasons"]
    model = {}
    rules = {}
    for r in rs:
        tgt = model if r["reason_source"] == "top_model_contributor" else rules
        tgt[r["reason"]] = tgt.get(r["reason"], 0) + int(r["alerts"])
    m_items = sorted(((FEATURE_TEXT.get(k, k), v) for k, v in model.items()), key=lambda x: -x[1])
    r_items = sorted(((RULE_TEXT.get(k, k), v) for k, v in rules.items()), key=lambda x: -x[1])
    return (f'<div class="card"><h2>Top model contributor on each alert</h2><p class="note">{_e(MODEL_NOTE)}</p>{_hbars(m_items)}</div>'
            f'<div class="card"><h2>Rule conditions met by alerts</h2><p class="note">{_e(RULE_NOTE)} '
            f'An alert can meet several rules, so these counts do not add up to the alert total.</p>{_hbars(r_items)}</div>')


def _benchmark(d: Dict[str, Any]) -> str:
    part_text = {"validation": "Validation hours 451–550", "development_holdout": "Development hours 551–650",
                 "sealed_final_test": "Sealed test 651–743 (recorded once, P1.5)"}
    bench = sorted(d["benchmark"], key=lambda r: (PARTITION_ORDER.index(r["evaluation_partition"])
                                                  if r["evaluation_partition"] in PARTITION_ORDER else 9, r["system_name"]))
    rows = "".join(
        f"<tr{' class=hl' if 'active champion' in r['system_name'] else ''}>"
        f"<td>{_e(part_text.get(r['evaluation_partition'], r['evaluation_partition']))}</td><td>{_e(r['system_name'])}</td>"
        f"<td class='n'>{r['alerts_per_day']:.1f}</td><td class='n'>{_pct(r['precision'])}</td><td class='n'>{_pct(r['recall'])}</td>"
        f"<td class='n'>{'–' if r['average_precision'] is None else f'{r['average_precision']:.3f}'}</td></tr>" for r in bench)
    return (f'<div class="card wide"><h2>Model evaluation</h2><p class="note">From <code>vw_risk_detection_benchmark</code>. '
            f'The active champion (E4) appears only in the sealed-test rows, its one held-out period. Precision and recall are '
            f'comparable only within a period, because the fraud rate differs between periods.</p>'
            f'<div class="scroll"><table><thead><tr><th>Period</th><th>System</th><th class="n">Alerts/day</th>'
            f'<th class="n">Precision</th><th class="n">Recall</th><th class="n">Avg precision</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></div>')


def _rails(d: Dict[str, Any]) -> str:
    rows = "".join(f"<tr><td>{_e(r['payment_rail'])}</td><td class='n'>{int(r['transaction_count']):,}</td>"
                   f"<td class='n'>{float(r['rail_volume_share_pct']):.1f}%</td><td class='n'>{int(r['labelled_fraud_count']):,}</td>"
                   f"<td class='n'>{float(r['labelled_fraud_rate_pct']):.3f}%</td></tr>" for r in d["rails"])
    return (f'<div class="card"><h2>Dataset by payment type</h2><p class="note">Descriptive counts of the simulated dataset.</p>'
            f'<div class="scroll"><table><thead><tr><th>Type</th><th class="n">Transactions</th><th class="n">Share</th>'
            f'<th class="n">Labelled fraud</th><th class="n">Fraud rate</th></tr></thead><tbody>{rows}</tbody></table></div></div>')


def _queue(d: Dict[str, Any]) -> str:
    q = d["queue"]
    if not q:
        body = '<p class="empty">No alerts stored yet.</p>'
    else:
        rows = "".join(
            f"<tr><td class='n'>{r['transaction_id']:,}</td><td>Day {r['simulated_day']}, {int(r['transaction_hour']):02d}:00</td>"
            f"<td>{_e(r['type'])}</td><td class='n'>${float(r['amount']):,.0f}</td><td class='n'>{float(r['risk_score']):.4f}</td>"
            f"<td>{_e(r['model_summary'])}</td><td>{' '.join(f'<span class=tag>{_e(x)}</span>' for x in r['rule_ids'])}</td></tr>"
            for r in q)
        body = (f'<div class="scroll"><table><thead><tr><th class="n">ID</th><th>When</th><th>Type</th><th class="n">Amount</th>'
                f'<th class="n">Score</th><th>Model explanation</th><th>Rule conditions met</th></tr></thead><tbody>{rows}</tbody></table></div>')
    return (f'<div class="card wide"><h2>Latest alerts</h2><p class="note">The {len(q)} most recent alerts with their reason codes. '
            f'{_e(MODEL_NOTE)} {_e(RULE_NOTE)}</p>{body}</div>')


def render(d: Dict[str, Any]) -> str:
    champ = d["champion"]
    src = d.get("source", {})
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinSight Dashboard</title><style>{CSS}</style></head>
<body><main>
<h1>FinSight dashboard</h1>
<p class="sub">Champion <code>{_e(champ['artifact'])}</code> · {_e(champ['model_name'])} · generated {_e(d.get('generated_utc', ''))}
 · source {_e(src.get('database', ''))}</p>
{_tiles(d)}
<div class="disclaimer">Scores are risk scores, not probabilities. Training hours (1–550) are hours the model learned from, so its alerts
there look better than they would on new data. Sealed-test figures are the one-time P1.5 result. Simulated (PaySim) data.</div>
<div class="grid">
<div class="card wide"><h2>Daily alert workload</h2><p class="note">Alerts the champion raised per simulated day at the operating threshold.</p>
{_workload_chart(d)}</div>
{_mix_panels(d)}
{_reason_panels(d)}
{_benchmark(d)}
{_rails(d)}
{_queue(d)}
</div>
<p class="note">Scoring runs: {_int(d['runs']['runs'])} · last run {_e(d['runs']['last_run'] or '–')}.
Generated by <code>src/finsight/dashboard.py</code> from views 01–08.</p>
</main></body></html>
"""


def main() -> None:
    from finsight.champion import read_registry
    from finsight.load_to_postgres import DatabaseConfig
    import psycopg2

    parser = argparse.ArgumentParser(description="Build the FinSight HTML/CSS dashboard (read-only)")
    parser.add_argument("--out", default="dashboard")
    parser.add_argument("--alerts", type=int, default=25)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = DatabaseConfig.from_env()
    conn = psycopg2.connect(host=cfg.host, port=cfg.port, dbname=cfg.name, user=cfg.user, password=cfg.password)
    try:
        data = collect(conn, read_registry()["active"], args.alerts)
    finally:
        conn.close()
    data["generated_utc"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    data["source"] = {"database": f"{cfg.host}:{cfg.port}/{cfg.name}"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    (out / "index.html").write_text(render(data), encoding="utf-8")
    logger.info("wrote %s/index.html from %s", out, data["source"]["database"])


if __name__ == "__main__":
    main()
