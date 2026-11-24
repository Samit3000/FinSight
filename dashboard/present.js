/* FinSight presentation — eight screens, one idea each. Arrow keys move between screens; N shows what to say.
   Every number comes from window.FINSIGHT_DATA (recorded results); text is inserted with textContent only. */
(function () {
  "use strict";
  const D = window.FINSIGHT_DATA, S = (D && D.study) || {};
  const $ = (s) => document.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v; else if (k === "style") n.setAttribute("style", v);
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
    }
    for (const c of kids.flat()) if (c != null && c !== false) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
    return n;
  }
  function sv(tag, attrs, ...kids) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) n.setAttribute(k, v);
    for (const c of kids.flat()) if (c != null) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
    return n;
  }
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const int = (v) => Math.round(+v).toLocaleString("en-US");
  const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
  const f2 = (v) => (+v).toFixed(2);

  // ---------------------------------------------------------------- facts (all from recorded results)
  const tx = D.daily.reduce((s, d) => s + d.transactions, 0), fraud = D.daily.reduce((s, d) => s + d.labelled_fraud, 0);
  const sealed = (S.sealed || {}).systems || {}, E4 = sealed.E4 || {}, E0 = sealed.E0 || {};
  const b4 = (E4.at_budget || {})[100] || {}, b0 = (E0.at_budget || {})[100] || {};
  const never = (sealed.dummy || {}).at_threshold || {};
  const dev = (S.development || {}).results || {}, ap = (m) => (dev[m] && dev[m].F3 ? dev[m].F3.average_precision : null);
  const f3m = (dev.E4 || {}).F3 || {};
  const split = { trainRows: f3m.train_rows, trainFraud: f3m.train_fraud, calRows: f3m.rows, calFraud: f3m.labelled_fraud, testRows: E4.rows, testFraud: E4.labelled_fraud };
  const threshold = D.registry.active.operating_point.threshold;
  const rails = Object.fromEntries(D.rails.map((r) => [r.payment_rail, r]));
  const shap = S.shap || [];
  const shareOf = (pred) => shap.filter((r) => pred(r.feature)).reduce((s, r) => s + r.share_pct, 0) / 100;
  const hourBands = (E4.breakdown_100 || {}).by_hour_band || [];
  const example = D.alerts.find((a) => a.id === 6168672) || D.alerts.filter((a) => a.period !== "training").sort((a, b) => b.score - a.score)[0];
  const dAP = (S.sealed || {}).paired_ap_difference_E4_minus_E0 || {};

  // ---------------------------------------------------------------- small visuals
  function needle() {   // 775 squares, one of them fraud
    const cols = 31, rows = 25, s = 12, g = 3, W = cols * (s + g), H = rows * (s + g);
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "775 squares, one of them highlighted: the share of transactions that are fraud" });
    for (let i = 0; i < cols * rows; i++) {
      const x = (i % cols) * (s + g), y = Math.floor(i / cols) * (s + g), hit = i === 397;
      svg.append(sv("rect", { x, y, width: s, height: s, rx: 2.5, fill: hit ? css("--m-E0") : css("--grid") }));
    }
    return el("div", { class: "svgbox" }, svg);
  }
  function hbar(rows, { fmt = int, max = null, colorFor = () => "--accent" } = {}) {
    const mx = max || Math.max(...rows.map((r) => r.value)) || 1;
    return el("div", { class: "hbars" }, rows.map((r) => el("div", { class: "hbar" }, el("span", {}, r.label),
      el("span", { class: "track", style: "height:14px" }, el("span", { class: "fill", style: `display:block;width:${(100 * r.value) / mx}%;background:var(${colorFor(r)})` })),
      el("span", { class: "v" }, fmt(r.value)))));
  }
  function stepChart() {   // AP on development fold F3 as the two ideas are added
    const steps = [["Starting model", ap("E0"), "--m-E0"], ["+ weight fraud ×50", ap("E2"), "--m-E2"], ["+ recipient history", ap("E1"), "--m-E1"], ["Both ideas (final model)", ap("E4"), "--m-E4"]];
    const W = 560, H = 260, m = { l: 40, r: 10, t: 24, b: 54 }, iw = W - m.l - m.r, ih = H - m.t - m.b, Y = (v) => ih - (v / 0.8) * ih, bw = iw / steps.length;
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Average precision for each model version" });
    const g = sv("g", { transform: `translate(${m.l},${m.t})` });
    for (const t of [0, 0.2, 0.4, 0.6, 0.8]) g.append(sv("line", { x1: 0, x2: iw, y1: Y(t), y2: Y(t), stroke: css("--grid") }), sv("text", { x: -8, y: Y(t) + 4, "text-anchor": "end", "font-size": 11, fill: css("--text-3") }, t.toFixed(1)));
    steps.forEach(([label, v, c], i) => {
      const x = i * bw + bw * 0.18, w = bw * 0.64;
      g.append(sv("rect", { x, y: Y(v), width: w, height: ih - Y(v), rx: 5, fill: `var(${c})` }),
        sv("text", { x: x + w / 2, y: Y(v) - 7, "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: css("--text") }, f2(v)));
      label.split(" (")[0].split(/(?<=model|history|×50)\s/).forEach((part, j) =>
        g.append(sv("text", { x: x + w / 2, y: ih + 18 + j * 15, "text-anchor": "middle", "font-size": 12, fill: css("--text-2") }, part)));
    });
    g.append(sv("line", { x1: 0, x2: iw, y1: ih, y2: ih, stroke: css("--border") }));
    svg.append(g);
    return el("div", { class: "svgbox" }, svg);
  }

  // ---------------------------------------------------------------- the eight screens
  const SLIDES = [
    {
      tab: "Start", title: "FinSight: finding fraud when a team can only check about 100 transactions a day",
      body: () => [
        el("p", { class: "lead" }, `A machine-learning model that ranks ${int(tx)} payments by how likely they are to be fraud, so that a small fraud team looks at the right ones first.`),
        el("div", { class: "big3" },
          el("div", { class: "stat hi" }, el("b", {}, pct(b4.precision, 0)), el("span", {}, "of the model's daily alerts are real fraud (final test)")),
          el("div", { class: "stat" }, el("b", {}, pct(b4.recall, 0)), el("span", {}, "of all fraud caught while checking only 100 a day")),
          el("div", { class: "stat" }, el("b", {}, `${f2(E4.average_precision)} vs ${f2(E0.average_precision)}`), el("span", {}, "ranking score: our model vs. the starting model"))),
        el("p", { class: "plain" }, el("b", {}, "The story in one line: "), "fraud is rare, so we built a model that puts it at the top of a short daily list, tested it fairly on data it had never seen, and it clearly beat the starting model."),
      ],
      notes: [
        "Good morning. My project is FinSight, a fraud-detection model.",
        `The setting: a payment company has millions of transactions, but its fraud team can only check about 100 a day. So the question is not just "is this fraud?" but "which 100 should we check first?"`,
        `I'll show the data, how I tested fairly, the model, the results and its limits. The headline: on the final test, about ${pct(b4.precision, 0)} of the model's daily alerts were real fraud.`,
      ],
    },
    {
      tab: "Problem", title: `Fraud is a needle in a haystack: about 1 in ${int(tx / fraud)} transactions`,
      body: () => [
        el("div", { class: "cols wide-left" },
          el("div", { class: "panel" }, el("h2", {}, `Each square is one transaction. Only one of these ${int(775)} is fraud.`), needle()),
          el("div", {},
            el("div", { class: "stat", style: "margin-bottom:12px" }, el("b", {}, int(tx)), el("span", {}, "simulated mobile-money transactions (PaySim), 31 days")),
            el("div", { class: "stat", style: "margin-bottom:12px" }, el("b", {}, int(fraud)), el("span", {}, `of them are fraud (${pct(fraud / tx, 2)})`)),
            el("div", { class: "stat hi" }, el("b", {}, "≈ 100 a day"), el("span", {}, "how many alerts the fraud team can review")))),
        el("p", { class: "plain" }, el("b", {}, "Why this is hard: "), "with so few frauds, a model can look very \"accurate\" while catching nothing. And the team's time is the real limit, so the model must rank, not just say yes or no."),
      ],
      notes: [
        `The dataset is PaySim: ${int(tx)} simulated mobile-money transactions over 31 days. Only ${int(fraud)} are fraud, about 1 in ${int(tx / fraud)}.`,
        "This is called class imbalance. It matters because a lazy model that always says 'not fraud' would still be right 99.9% of the time.",
        "The second constraint is people: a fraud team can only review about 100 alerts a day. So I judge the model on how many real frauds it puts into those 100.",
      ],
    },
    {
      tab: "Data", title: "Fraud only happens in two payment types, and plain accuracy cannot see it",
      body: () => [
        el("div", { class: "cols" },
          el("div", { class: "panel" }, el("h2", {}, "Frauds by payment type"),
            hbar(["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"].map((t) => ({ label: t, value: rails[t] ? rails[t].labelled_fraud_count : 0 })), { colorFor: (r) => (r.value ? "--m-E0" : "--grid") }),
            el("p", { class: "plain" }, "Typical pattern: money is ", el("b", {}, "transferred"), " to another account, then ", el("b", {}, "cashed out"), ".")),
          el("div", { class: "panel" }, el("h2", {}, "Why we don't use accuracy"),
            el("table", { class: "ptable" }, el("thead", {}, el("tr", {}, el("th", {}, "Final test"), el("th", { class: "r" }, "Accuracy"), el("th", { class: "r" }, "Frauds caught"))),
              el("tbody", {},
                el("tr", {}, el("td", {}, "A model that never says fraud"), el("td", { class: "r" }, pct(never.accuracy, 1)), el("td", { class: "r" }, "0")),
                el("tr", { class: "win" }, el("td", {}, "Our model (100 alerts/day)"), el("td", { class: "r" }, pct(b4.accuracy, 1)), el("td", { class: "r" }, `${int(b4.tp)} of ${int(split.testFraud)}`)))),
            el("p", { class: "plain" }, "Accuracy barely changes, yet one model is useless and the other catches hundreds of frauds. So we measure ", el("b", {}, "precision"), " (how many alerts are real) and ", el("b", {}, "recall"), " (how much fraud we catch).")),
        ),
      ],
      notes: [
        "Two things about the data. First, fraud only appears in TRANSFER and CASH_OUT: the fraudster moves money to another account and then cashes it out.",
        `Second, accuracy is misleading here. A model that never flags anything is ${pct(never.accuracy, 1)} accurate on the test data and catches zero frauds. Ours is ${pct(b4.accuracy, 1)} accurate, almost the same number, but catches ${int(b4.tp)} frauds.`,
        "So instead I use precision, meaning of the alerts we raise, how many are real fraud, and recall, meaning of all fraud, how much we catch.",
      ],
    },
    {
      tab: "Fair test", title: "We tested like predicting the future: learn on the past, check on later data, once",
      body: () => [
        el("div", { class: "tl" },
          el("div", { class: "t1" }, el("b", {}, "1 · Learn"), el("span", {}, `Hours 1–550 · ${int(split.trainRows)} transactions. The model learns patterns here.`)),
          el("div", { class: "t2" }, el("b", {}, "2 · Set the cut-off"), el("span", {}, "Hours 551–650. Pick the score that gives about 100 alerts a day.")),
          el("div", { class: "t3" }, el("b", {}, "3 · Final exam"), el("span", {}, `Hours 651–743 · ${int(split.testRows)} transactions. Opened once.`))),
        el("div", { class: "tl-labels" }, el("span", {}, "earlier"), el("span", {}), el("span", { style: "text-align:right" }, "later →")),
        el("div", { class: "cols", style: "margin-top:18px" },
          el("div", { class: "panel" }, el("h2", {}, "Why not shuffle the data randomly?"),
            el("p", { style: "margin:0;font-size:15px;color:var(--text-2)" }, "Shuffling lets the model learn from the future and be tested on the past, which looks better than real life. A bank always predicts tomorrow from yesterday, so we split by time.")),
          el("div", { class: "panel" }, el("h2", {}, "Why only once?"),
            el("p", { style: "margin:0;font-size:15px;color:var(--text-2)" }, "If you check the exam many times and keep tweaking, you end up fitting the exam. We wrote down the pass/fail rule first, then opened the final exam once. That is called pre-registration."))),
      ],
      notes: [
        "How I tested fairly. The data runs in time order, so I split it by time: the model learns on the first 550 hours.",
        "Hours 551 to 650 are used to choose the alert cut-off, the score that gives about 100 alerts a day, and to compare model versions.",
        "The last 93 hours are the final exam. I wrote down the pass/fail rule before looking, then ran the exam once. That stops me from tuning the model to the test, which would make the results look better than they really are.",
      ],
    },
    {
      tab: "Model", title: "The model: gradient boosting, plus two ideas that made it much better",
      body: () => [
        el("div", { class: "cols" },
          el("div", { class: "panel" }, el("h2", {}, "What the model sees for each transaction (8 inputs)"),
            el("div", { class: "chips" }, el("span", {}, "amount"), el("span", {}, "hour of day"), el("span", {}, "payment type (5 yes/no flags)"), el("span", { class: "key" }, "recipient history")),
            el("p", { class: "plain" }, el("b", {}, "Gradient boosting "), "builds 50 small decision trees one after another; each new tree fixes mistakes the earlier ones made. It handles mixed numbers well and trains in about a minute."),
            el("p", { class: "plain" }, el("b", {}, "Idea 1 · recipient history: "), "how many payments the receiving account got before. Brand-new receiving accounts are suspicious.", el("br"),
              el("b", {}, "Idea 2 · weight fraud ×50: "), "during training, missing a fraud counts 50 times more than a false alarm, so the model pays attention to the rare cases.")),
          el("div", { class: "panel" }, el("h2", {}, "How much each idea helped (ranking score, hours 551–650)"), stepChart(),
            el("p", { class: "note", style: "font-size:13px" }, "Ranking score = average precision: 1 is perfect, random guessing gets about 0.01. Recipient history gave the biggest jump."))),
      ],
      notes: [
        "The model is gradient boosting, from scikit-learn. It builds 50 small decision trees in sequence, each correcting the previous ones' mistakes.",
        "It sees 8 simple inputs: the amount, the hour, the payment type, and one feature I added: recipient history, meaning how many payments the receiving account had received before. New accounts receiving big transfers are a classic fraud sign.",
        `The second idea is class weighting: missing a fraud costs 50 times more than a false alarm during training. On the right you can see the effect: the starting model scores ${f2(ap("E0"))}, recipient history lifts it to ${f2(ap("E1"))}, and with both ideas it reaches ${f2(ap("E4"))}.`,
      ],
    },
    {
      tab: "Results", title: `On the final exam, ${pct(b4.precision, 0)} of alerts were real fraud, and it caught ${pct(b4.recall, 0)} of all fraud`,
      body: () => [
        el("div", { class: "cols" },
          el("div", { class: "panel" }, el("h2", {}, `Final exam (hours 651–743), checking 100 alerts a day`),
            el("div", { class: "cm2" },
              el("div", {}), el("div", { class: "h" }, "Model said: fraud"), el("div", { class: "h" }, "Model said: fine"),
              el("div", { class: "rh" }, "Really fraud"), el("div", { class: "c good" }, el("b", {}, int(b4.tp)), el("span", {}, "caught")), el("div", { class: "c bad" }, el("b", {}, int(b4.fn)), el("span", {}, "missed")),
              el("div", { class: "rh" }, "Really fine"), el("div", { class: "c bad" }, el("b", {}, int(b4.fp)), el("span", {}, "false alarms")), el("div", { class: "c good" }, el("b", {}, int(b4.tn)), el("span", {}, "correctly left alone"))),
            el("p", { class: "note", style: "font-size:13px" }, `This 2×2 table is the confusion matrix. Precision = ${int(b4.tp)} ÷ (${int(b4.tp)} + ${int(b4.fp)}) = ${pct(b4.precision)}. Recall = ${int(b4.tp)} ÷ (${int(b4.tp)} + ${int(b4.fn)}) = ${pct(b4.recall)}.`)),
          el("div", { class: "panel" }, el("h2", {}, "Our model vs. the starting model, same 100 alerts a day"),
            el("table", { class: "ptable" }, el("thead", {}, el("tr", {}, el("th", {}, ""), el("th", { class: "r" }, "Starting model"), el("th", { class: "r" }, "Our model"))),
              el("tbody", {},
                el("tr", { class: "win" }, el("td", {}, "Alerts that are real fraud (precision)"), el("td", { class: "r" }, pct(b0.precision)), el("td", { class: "r" }, pct(b4.precision))),
                el("tr", { class: "win" }, el("td", {}, "Share of fraud caught (recall)"), el("td", { class: "r" }, pct(b0.recall)), el("td", { class: "r" }, pct(b4.recall))),
                el("tr", { class: "win" }, el("td", {}, "False alarms in 3.9 days"), el("td", { class: "r" }, int(b0.fp)), el("td", { class: "r" }, int(b4.fp))),
                el("tr", { class: "win" }, el("td", {}, "Ranking score (average precision)"), el("td", { class: "r" }, f2(E0.average_precision)), el("td", { class: "r" }, f2(E4.average_precision))))),
            el("p", { class: "plain" }, el("b", {}, "Is the gain real? "), `The improvement is ${f2(dAP.point)} with a 95% confidence range of ${f2(dAP.ci95[0])} to ${f2(dAP.ci95[1])}, which never comes near zero. It passed the rule written before the exam.`))),
      ],
      notes: [
        `These are the final-exam results, with both models allowed exactly 100 alerts a day so the comparison is fair.`,
        `The confusion matrix: of the alerts, ${int(b4.tp)} were real fraud and only ${int(b4.fp)} were false alarms, so precision is ${pct(b4.precision)}. It caught ${int(b4.tp)} of ${int(split.testFraud)} frauds, so recall is ${pct(b4.recall)}. The other ${int(b4.fn)} were missed, because 100 checks a day is simply not enough to catch everything.`,
        `Compared with the starting model, precision went from ${pct(b0.precision)} to ${pct(b4.precision)} and recall from ${pct(b0.recall)} to ${pct(b4.recall)}. The ranking score improved by ${f2(dAP.point)}, and the 95% confidence range stays well above zero, so it is not luck.`,
      ],
    },
    {
      tab: "Why flagged", title: "Every alert comes with its reasons, so an analyst knows what to check",
      body: () => {
        const a = example, mx = Math.max(...a.top_contributors.map((c) => Math.abs(c.contribution)));
        const label = { recipient_prior_in_count: "Brand-new recipient", transaction_hour: `At ${String(a.hour).padStart(2, "0")}:00 at night`, type_PAYMENT: "Not an ordinary payment", amount: "Large amount", type_CASH_OUT: "Cash-out", type_CASH_IN: "Not a cash-in" };
        return [el("div", { class: "cols" },
          el("div", { class: "panel alert-card" }, el("h2", {}, "A real alert from hours the model never trained on"),
            el("dl", { class: "facts" }, ...[["Transaction", `#${a.id} · ${a.type}`], ["Amount", "$" + int(a.amount)], ["Time", `${String(a.hour).padStart(2, "0")}:00`], ["Recipient's earlier payments", a.recipient_prior === 0 ? "none" : int(a.recipient_prior)], ["Model score", `${a.score.toFixed(3)} (cut-off ${threshold.toFixed(3)})`]].flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])),
            el("h2", { style: "margin-top:6px" }, "Why the model flagged it"),
            a.top_contributors.map((c) => el("div", { class: "contrib" }, el("span", {}, label[c.feature] || c.description), el("span", { class: "track" }, el("span", { class: "bar", style: `display:block;width:${(100 * Math.abs(c.contribution)) / mx}%` })), el("span", { class: "v" }, "+" + c.contribution.toFixed(1)))),
            el("div", { class: "rules" }, a.rule_ids.map((r) => el("span", { class: "chip" }, D.rule_names[r] || r)))),
          el("div", { class: "panel" }, el("h2", {}, "What the model relies on overall"),
            hbar([{ label: "Payment type", value: shareOf((f) => f.startsWith("type_")) }, { label: "Recipient history", value: shareOf((f) => f === "recipient_prior_in_count") },
              { label: "Hour of day", value: shareOf((f) => f === "transaction_hour") }, { label: "Amount", value: shareOf((f) => f === "amount") }], { fmt: (v) => pct(v, 0), max: 1, colorFor: (r) => (r.label === "Recipient history" ? "--accent" : "--seq-2") }),
            el("p", { class: "plain" }, "These come from ", el("b", {}, "SHAP"), ", a standard method that splits each score into how much each input pushed it up or down. ", el("b", {}, "Important: "), "it shows what the model relied on, not proof that a transaction is fraud.")))];
      },
      notes: [
        `A model is only useful if analysts trust it, so every alert comes with reasons. This example is a ${"$" + int(example.amount)} transfer at ${String(example.hour).padStart(2, "0")}:00 to an account that had never received a payment before.`,
        "The bars come from SHAP, which breaks a prediction into contributions from each input. Here, 'brand-new recipient' and 'night-time' pushed the score up most. The chips are simple business rules it also meets, shown for context.",
        "On the right, overall, payment type matters most because it rules out three of the five types, and recipient history is second. I'm careful to say this explains the model; it doesn't prove fraud.",
      ],
    },
    {
      tab: "Limits", title: "What it can't do yet, and what I would do next",
      body: () => [
        el("div", { class: "cols" },
          el("div", { class: "panel" }, el("h2", {}, "Limitations"),
            el("ul", { class: "bul" },
              el("li", {}, el("b", {}, "Simulated data. "), "PaySim imitates real payments but fraud follows simple patterns. Real-world results would be lower."),
              el("li", {}, el("b", {}, "Small final exam. "), `${int(split.testFraud)} frauds over about 4 days, so numbers have some uncertainty (we report ranges).`),
              el("li", {}, el("b", {}, "Misses daytime fraud. "), "Most night-time fraud is caught, but busy daytime traffic hides it (chart)."),
              el("li", {}, el("b", {}, "Scores are not probabilities. "), "A score of 0.95 means 'rank high', not '95% chance of fraud'."))),
          el("div", { class: "panel" }, el("h2", {}, "Share of fraud caught, by time of day (final exam)"),
            hbar(hourBands.map((h) => ({ label: { "00-05": "Night 00–05", "06-08": "Morning 06–08", "09-17": "Daytime 09–17", "18-23": "Evening 18–23" }[h.segment] || h.segment, value: h.recall })), { fmt: (v) => pct(v, 0), max: 1, colorFor: (r) => (r.label.startsWith("Night") ? "--accent" : "--m-E0") }),
            el("h2", { style: "margin-top:16px" }, "Next steps"),
            el("ul", { class: "bul" },
              el("li", {}, "Test on real or differently simulated data."),
              el("li", {}, "Add sender-side and speed features (how fast money moves) to catch daytime fraud."),
              el("li", {}, "Choose the daily alert number from the cost of fraud vs. the cost of review.")))),
        el("p", { class: "plain" }, el("b", {}, "Thank you. "), "Happy to take questions. (The full technical report is under Details.)"),
      ],
      notes: [
        "Finally, the honest limits. The data is simulated, so real-world performance would be lower and would need testing on real data.",
        `The final exam is small, ${int(split.testFraud)} frauds in about 4 days, so I report confidence ranges rather than single numbers.`,
        `The model catches most night-time fraud, ${pct((hourBands[0] || {}).recall || 0, 0)}, but much less in the daytime, because legitimate traffic hides it. Next I'd add features about the sender and how fast money moves.`,
        "Thank you, I'm happy to take questions.",
      ],
    },
  ];

  // ---------------------------------------------------------------- shell
  let cur = 0, notesOn = false;
  function render() {
    const s = SLIDES[cur], stage = $("#stage");
    stage.replaceChildren(el("section", { class: "slide", "aria-label": s.tab }, el("div", { class: "step" }, `${cur + 1} of ${SLIDES.length} · ${s.tab}`), el("h1", {}, s.title), ...s.body()));
    $("#notes-body").replaceChildren(...s.notes.map((p) => el("p", {}, p)));
    $("#notes").hidden = !notesOn;
    $("#count").textContent = `${cur + 1} / ${SLIDES.length}`;
    $("#prev").disabled = cur === 0; $("#next").disabled = cur === SLIDES.length - 1;
    document.querySelectorAll("#dots button").forEach((b, i) => b.setAttribute("aria-current", i === cur ? "step" : "false"));
    if (location.hash !== "#" + (cur + 1)) history.replaceState(null, "", "#" + (cur + 1));
  }
  const go = (i) => { cur = Math.max(0, Math.min(SLIDES.length - 1, i)); render(); scrollTo(0, 0); };
  function init() {
    if (!D || !S.sealed) { $("#stage").replaceChildren(el("p", { class: "banner" }, "No data: run py src/finsight/dashboard.py --out dashboard.")); return; }
    $("#dots").replaceChildren(...SLIDES.map((s, i) => el("button", { onclick: () => go(i) }, `${i + 1} ${s.tab}`)));
    $("#prev").addEventListener("click", () => go(cur - 1));
    $("#next").addEventListener("click", () => go(cur + 1));
    $("#notes-toggle").addEventListener("click", () => { notesOn = !notesOn; $("#notes-toggle").setAttribute("aria-pressed", String(notesOn)); render(); });
    $("#theme-toggle").addEventListener("click", () => {
      const now = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      document.documentElement.dataset.theme = now === "dark" ? "light" : "dark";
      try { localStorage.setItem("finsight.theme", document.documentElement.dataset.theme); } catch (e) { /* ignore */ }
      render();
    });
    try { const th = localStorage.getItem("finsight.theme"); if (th) document.documentElement.dataset.theme = th; } catch (e) { /* ignore */ }
    document.addEventListener("keydown", (e) => {
      if (e.target.closest("input, textarea, select")) return;
      if (e.key === "ArrowRight" || e.key === "PageDown" || e.key === " ") { e.preventDefault(); go(cur + 1); }
      if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); go(cur - 1); }
      if (e.key === "Home") go(0); if (e.key === "End") go(SLIDES.length - 1);
      if (e.key.toLowerCase() === "n") $("#notes-toggle").click();
    });
    const h = parseInt((location.hash || "").slice(1), 10);
    go(h >= 1 && h <= SLIDES.length ? h - 1 : 0);
    document.body.dataset.ready = "true";
  }
  try { init(); } catch (err) { document.body.dataset.error = String(err && err.message || err); throw err; }
})();
