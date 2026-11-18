# P1.4.1 — Rolling daily alert-cap investigation

Config `hgb_baseline7_rcpt_cw50` (E4, leading candidate, not promoted) · development data only (highest step loaded:
650) · data sha256 `74cba5b396ce…` · numbers in `p141_rolling_cap.json`.

**Investigation only.** No alert policy has been implemented in the codebase. The simulation script ran outside the
repository (session scratchpad), and it reused the repo's P1.4 fold setup, threshold and tie-handling functions
without changing them.

## Disclosure

- **Fold reuse.** F1 and F2 were used to discover the recipient feature (P1.1), to screen candidates (P1.3) and to
  choose the policy (P1.4). F3 has now been used three times: P1.3 confirmation, P1.4 confirmation and this
  investigation. E4's feature was found by analysing champion errors on steps 451–650, which contains F3, and the
  champion it was compared against was chosen on 551–743. **No fold is independent.** These numbers are for comparing
  policies, not for estimating production performance.
- **Same models as P1.4.** E4 is trained on steps ≤ T−100. Thresholds come from the calibration window
  (T−99..T). Policies are applied to the test window without retraining.
- **Locked test.** Steps 651–743 were not read. The live database was not touched.
- **Time resolution.** PaySim records time only to the hour (`step`), and order within an hour is unknown. When a
  cap runs out partway through an hour, each eligible transaction in that hour is counted as reviewed with the same
  probability (expected values), which matches the repo's tie rule.
- **Day boundaries.** A simulated day runs from hour 00 to 23. Days cut off at the edge of a window get a pro-rated
  cap.
- **Uncertainty.** No confidence intervals are given, and each window has only 3–4 full days.

## Policies compared (K = 50, 100, 200 alerts per day)

| | Policy | Works in real time? | How it works |
|---|---|---|---|
| **B** | Fixed threshold (P1.4 choice) | Yes | Alert whenever score ≥ a threshold set on the calibration window; no cap |
| **D1** | End-of-day top-K | **No.** It needs the whole day's scores. | Each day, review that day's K highest scores. This is the best a daily cap can do. |
| **D2** | B's threshold + hard daily cap | Yes | B's alerts in arrival order until K are used, then stop until midnight |
| **D3** | Rolling 24-hour threshold + hard daily cap | Yes | Each hour, the threshold is the K-th highest score of the previous 24 hours (scores only, no labels), with D2's cap |

## Headline results at 100 alerts/day

| Fold | Policy | Alerts/day | Whole-day range | Day-to-day variation (CV) | Precision | Recall | Recall 06–23 | Recall 00–05 | Fraud lost to cap |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| F1 | B | 114.5 | 106–144 | 0.146 | 0.864 | **35.6%** | 19.4% | 77.0% | – |
| F1 | D1 | 100.1 | 100–100 | 0 | 0.907 | 32.7% | 16.3% | 74.5% | – |
| F1 | D2 | 99.3 | 100–100 | 0 | 0.894 | 32.0% | 14.3% | 77.0% | 42.1 |
| F1 | D3 | 95.7 | 87–100 | 0.057 | 0.893 | 30.8% | 14.1% | 73.3% | 15.0 |
| F2 | B | 64.1 | 62–69 | 0.047 | 0.936 | 24.8% | 11.8% | 63.4% | – |
| F2 | D1 | 100.1 | 100–100 | 0 | 0.771 | **31.9%** | 17.0% | 76.0% | – |
| F2 | D2 | 64.1 | 62–69 | 0.047 | 0.936 | 24.8% | 11.8% | 63.4% | 0 |
| F2 | D3 | 95.9 | 94–100 | 0.029 | 0.647 | 25.7% | **9.3%** | 74.3% | 52.4 |
| F3 | B | 97.9 | 81–131 | 0.205 | 0.865 | 31.4% | 16.9% | 69.4% | – |
| F3 | D1 | 99.8 | 100–100 | 0 | 0.909 | **33.6%** | 18.5% | 73.2% | – |
| F3 | D2 | 89.8 | 81–100 | 0.091 | 0.860 | 28.6% | 13.5% | 68.4% | 31.0 |
| F3 | D3 | 94.2 | 89–100 | 0.056 | 0.833 | 29.0% | 13.2% | 70.6% | 53.0 |

"Fraud lost to cap" counts frauds that the same rule without a cap would have flagged but that arrived after that
day's budget was used up. CV is the standard deviation of whole-day alert counts divided by their mean.

### All budgets (recall / precision / alerts per day)

| Fold | K | B | D1 | D2 | D3 |
|---|---:|---|---|---|---|
| F1 | 50 | 25.1% / 0.970 / 71.8 | 17.4% / 0.964 / 50.2 | 17.6% / 0.986 / 49.6 | 16.5% / 0.979 / 46.7 |
| F1 | 100 | 35.6% / 0.864 / 114.5 | 32.7% / 0.907 / 100.1 | 32.0% / 0.894 / 99.3 | 30.8% / 0.893 / 95.7 |
| F1 | 200 | 42.2% / 0.512 / 228.7 | 45.9% / 0.636 / 199.9 | 37.5% / 0.598 / 174.1 | 37.7% / 0.557 / 187.5 |
| F2 | 50 | 11.8% / 0.984 / 29.0 | 19.6% / 0.952 / 49.9 | 11.8% / 0.984 / 29.0 | 17.6% / 0.942 / 45.1 |
| F2 | 100 | 24.8% / 0.936 / 64.1 | 31.9% / 0.771 / 100.1 | 24.8% / 0.936 / 64.1 | 25.7% / 0.647 / 95.9 |
| F2 | 200 | 43.2% / 0.538 / 194.2 | 45.5% / 0.551 / 199.9 | 39.1% / 0.524 / 180.6 | 35.8% / 0.452 / 191.6 |
| F3 | 50 | 21.1% / 0.960 / 59.5 | 18.0% / 0.976 / 49.9 | 17.4% / 0.952 / 49.5 | 17.7% / 0.976 / 49.0 |
| F3 | 100 | 31.4% / 0.865 / 97.9 | 33.6% / 0.909 / 99.8 | 28.6% / 0.860 / 89.8 | 29.0% / 0.833 / 94.2 |
| F3 | 200 | 44.5% / 0.746 / 161.3 | 53.5% / 0.722 / 200.2 | 41.4% / 0.740 / 151.4 | 46.1% / 0.686 / 181.6 |

When B over-alerts (F1 at 50 and 100, F3 at 50), its higher recall comes from extra reviews, not better ranking. At
equal volume, D1 has the highest recall of D1–D3 in 8 of 9 fold–budget cells. The exception is F1 at 50/day, where
D2 is marginally ahead (17.6% vs 17.4%).

## High- and low-volume days (100/day)

| Fold · day | Transactions | Fraud | B alerts (TP) | D1 TP | D2 TP · budget used up at hour | D3 TP · used up at hour |
|---|---:|---:|---|---:|---|---|
| F1 · 17 (busiest) | 425,766 | 320 | 144 (120) | 96.4 | 92.0 · 16:00 | 97.0 · 21:00 |
| F1 · 18 (after the collapse) | 20,999 | 268 | 108 (106) | 98.0 | 98.0 · 19:00 | 85.0 · – |
| F2 · 22 | 53,437 | 256 | 69 (66) | 76.0 | 66.0 · – | **51.0 · 05:00** |
| F2 · 23 | 48,546 | 188 | 60 (53) | 56.0 | 53.0 · – | **40.6 · 05:00** |
| F3 · 25 (busiest) | 57,853 | 240 | 131 (86) | 82.0 | **57.7 · 05:00** | 56.6 · 05:00 |
| F3 · 26 | 13,885 | 272 | 81 (81) | 97.0 | 81.0 · – | 90.0 · – |

**High-volume days.** Under a fixed threshold, the number of alerts tracks legitimate volume: more legitimate
transactions means more of them score above the threshold. On F3 day 25, B raised 131 alerts, of which 45 were false
positives (precision 66%, against 92–100% on the other days). Adding a cap in arrival order (D2) doesn't fix this. It
uses up the budget on whatever arrives first. On day 25 the budget ran out at **05:00**, and about 28 frauds that B
would still have flagged later that day went unreviewed (86 → 57.7 caught).

**Low-volume days.** In F2, the thresholds were set on steps 351–450, which averaged 257k transactions a day, but the
test days had only 20–53k. Fraud stayed at about 190–270 a day, but with so few legitimate transactions, few scores
cleared B's threshold. B used only 60–69 of the 100 daily reviews, and the cap in D2 does nothing about that.
Filling the empty slots costs precision: D1 adds 71.5 frauds at 77% precision, while D3's rolling threshold filled
them mostly with **night-time legitimate transactions** (98.6% of its false positives were in 00–05). On days 22 and
23, D3's budget ran out at **05:00**. Daytime recall fell to 9.3%, below B's 11.8%.

**Why caps run out at night.** The day starts at midnight, and night scores are high: hour of day is still a strong
fraud signal in PaySim, where night legitimate volume is small. Under an arrival-order cap, night alerts reach the
budget first, so daytime fraud arriving after exhaustion is never reviewed. This is the main risk of a naive
real-time cap.

## Day vs night, false positives and missed fraud (100/day)

- **Recall in hours 06–23 is low under every policy:** 9–19%, against 63–77% in hours 00–05. Capping makes daytime
  recall worse: D2 and D3 never beat B or D1 on hours 06–23, because budget runs out before the day is over.
- **Where false positives fall.** In F2 and F3, 88–99% of false positives fall in hours 00–05 under every policy. In
  F1, at the peak-volume period, 52–68% fall in 18–23 and 19–32% in 09–17. So the false-positive mix also shifts
  with volume.
- **What is missed.** Total misses at 100/day are 744–804 (F1), 686–758 (F2) and 748–804 (F3) of 1,156 / 1,008 /
  1,126 frauds. Most missed fraud is low-scoring daytime fraud that no policy reaches at this budget. A cap adds
  15–53 extra misses per fold through budget exhaustion. The 200/day results show how much more fraud a larger review
  team would reach. On F3, D1 at 200/day catches 53.5%, with daytime recall of 39.1%.

## The trade-off

| At 100/day | Workload stability | Recall vs uncapped B | Main failure |
|---|---|---|---|
| B fixed threshold | Poor: daily load ranges 81–144 (CV up to 0.21); 36% under target on F2 | reference | Load follows legitimate volume: overloads busy days, wastes capacity on quiet days |
| D2 threshold + cap | Caps busy days only; can't fill quiet days | −3.6 / 0 / −2.8 pts | Budget runs out early; 31–42 frauds lost on busy windows |
| D3 rolling threshold + cap | Best of the real-time options (CV ≤ 0.057, 94–96/day) | −4.8 / +0.9 / −2.4 pts | Fills with night-time false positives; runs out at 05:00; 15–53 frauds lost |
| D1 end-of-day top-K | Perfect (exactly K) | −2.9 / **+7.1** / **+2.2** pts | Needs up to 24 hours of review delay; not a real-time decision |

**Cost of a stable workload.** On these folds, a stable daily load costs 0–5 recall points when the cap works
in arrival order (D2 or D3). Done by ranking within the day (D1), it is roughly neutral to positive (−2.9 to +7.1
points). The loss comes from **arrival order**, not from the cap itself. A first-come cap misses lower-ranked fraud
when budget runs out, but it also misses higher-ranked fraud that arrives late in the day. D1 shows that ranking
within the day recovers it.

## How a rolling cap would work in a real-time system

Today each transaction gets a risk score when it arrives. Nothing in the code decides alerts in real time yet.

1. **Scoring.** E4 needs the recipient count *before* the current hour. A live system keeps a recipient → count
   store and updates it when each hour closes. That matches the hourly approximation used in training and avoids
   counting same-hour transactions.
2. **Cap and pacing.** A daily counter starts at K. A plain midnight reset combined with arrival order spends the
   budget at night (seen above). Two ways to avoid that:
   - (a) a **rolling 24-hour window** rather than a calendar day;
   - (b) **hourly pacing**: each hour gets a share of K in proportion to expected review value, and unused share
     carries forward.
3. **Priority queue rather than first-come.** Alerts above a low floor threshold go into a queue sorted by score.
   Analysts always take the highest-scoring open item. Each item expires after a review deadline (for example 6 or
   24 hours). D1 is this design's upper bound when the deadline is 24 hours.
4. **When the budget is used up**, every option has a cost:
   - (i) **Drop alerts** (D2/D3 as simulated): the cheapest option, and the one that loses fraud.
   - (ii) **Queue overflow** past the deadline: the fraud may settle before review.
   - (iii) **Raise the threshold** for the rest of the day: protects capacity, but lowers recall for late-day fraud.
   - (iv) **Escalate the very highest scores** (above the 50/day threshold, 96–98% precision here): send them to
     an automatic hold or step-up authentication, so they never wait in the queue or get dropped. This is a business
     decision with customer-friction cost.

   Whichever option is used, the system should log every alert it suppresses, so missed fraud can be measured later.
5. **Monitoring.** Track daily alert counts, precision on reviewed items, threshold drift and how often the budget
   runs out. A fixed threshold drifted 15% over and 36% under target here, so it needs recalibration on a schedule,
   or the rolling window.

PaySim can't tell us how quickly a review has to happen to stop a fraud (there's no settlement timing). So the cost of
D1's delay can't be measured here.

## Recommendation (for approval, nothing implemented)

1. **Don't use a first-come hard cap (D2 or D3) as the production policy.** It stabilises the load but loses fraud
   whenever the budget runs out, and that happens at 05:00 on busy or quiet days.
2. **Next step (P1.4.2, if approved): simulate a priority queue with review deadlines.** Test deadlines of 6, 12 and
   24 hours, plus hourly pacing and high-score escalation. Choose on F1/F2 with a rule declared in advance, report
   F3 once more with that reuse disclosed, then implement the chosen policy with tests.
3. **Until then, keep B as the reference policy.** Report its daily load range next to every recall figure, so
   capacity drift stays visible.
4. **The locked test is unaffected.** The P1.5 single final run still needs separate approval and has not happened.
