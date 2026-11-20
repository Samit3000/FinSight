# Alert reason codes — sample

Champion `models/hgb_e4_recipient.joblib` · steps 577–600 (simulated day 25) · threshold 0.9463 · 125 alerts · generated 2026-11-20T11:39:35Z

> Model attribution: shows how each input moved this transaction's model score relative to the model's average (log-odds). It is not evidence that the transaction is fraudulent and does not establish a cause.
>
> Rule conditions met: deterministic checks from the rule engine, shown for context. They do not raise alerts in FinSight and are not evidence of fraud on their own.

Steps 551–650 are the champion's threshold-calibration window, so this sample illustrates the format; it is not a performance estimate. Labels were not read.

| Step | Hour | Type | Amount | Score | Top model contributors (log-odds) | Rule conditions met |
|---:|---:|---|---:|---:|---|---|
| 577 | 00 | TRANSFER | $681,526 | 0.9645 | first-time recipient (no earlier payments received) +2.94; hour 00:00 +2.36; type is not PAYMENT +1.58 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 577 | 00 | CASH_OUT | $681,526 | 0.9665 | hour 00:00 +2.95; type is CASH_OUT +1.80; amount $681,525.62 +1.80 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 577 | 00 | TRANSFER | $1,396,825 | 0.9748 | first-time recipient (no earlier payments received) +2.99; amount $1,396,825.00 +2.06; hour 00:00 +2.01 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 577 | 00 | CASH_OUT | $1,396,825 | 0.9805 | amount $1,396,825.00 +3.36; type is not PAYMENT +1.75; type is CASH_OUT +1.75 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 577 | 00 | TRANSFER | $763,171 | 0.9645 | first-time recipient (no earlier payments received) +2.96; hour 00:00 +2.16; type is not PAYMENT +1.59 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 577 | 00 | CASH_OUT | $763,171 | 0.9665 | hour 00:00 +2.29; amount $763,170.75 +2.23; type is not PAYMENT +1.72 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 577 | 00 | TRANSFER | $727,793 | 0.9645 | first-time recipient (no earlier payments received) +2.96; hour 00:00 +2.16; type is not PAYMENT +1.59 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 577 | 00 | CASH_OUT | $727,793 | 0.9729 | hour 00:00 +2.44; amount $727,793.00 +2.44; type is not PAYMENT +1.65 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 578 | 01 | TRANSFER | $978,103 | 0.9816 | first-time recipient (no earlier payments received) +2.61; hour 01:00 +2.51; amount $978,103.38 +1.92 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | CASH_OUT | $978,103 | 0.9807 | amount $978,103.38 +2.91; type is not PAYMENT +1.86; type is not CASH_IN +1.84 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | TRANSFER | $4,961,014 | 0.9860 | first-time recipient (no earlier payments received) +2.70; hour 01:00 +2.39; amount $4,961,014.00 +1.92 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | CASH_OUT | $4,961,014 | 0.9871 | amount $4,961,014.00 +2.93; type is CASH_OUT +2.42; type is not PAYMENT +1.87 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | TRANSFER | $190,816 | 0.9465 | first-time recipient (no earlier payments received) +2.91; hour 01:00 +2.10; type is TRANSFER +1.79 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | TRANSFER | $152,372 | 0.9465 | first-time recipient (no earlier payments received) +2.94; hour 01:00 +2.01; type is TRANSFER +1.82 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | TRANSFER | $736,677 | 0.9757 | hour 01:00 +2.65; first-time recipient (no earlier payments received) +2.56; type is not PAYMENT +1.72 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | CASH_OUT | $736,677 | 0.9669 | hour 01:00 +2.79; amount $736,677.38 +2.16; type is not PAYMENT +1.70 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | TRANSFER | $118,107 | 0.9465 | first-time recipient (no earlier payments received) +2.92; hour 01:00 +1.99; type is TRANSFER +1.83 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | TRANSFER | $6,274,753 | 0.9860 | first-time recipient (no earlier payments received) +2.70; hour 01:00 +2.39; amount $6,274,752.50 +1.92 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | CASH_OUT | $6,274,753 | 0.9872 | amount $6,274,752.50 +3.02; type is not PAYMENT +1.93; type is not CASH_IN +1.81 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 578 | 01 | TRANSFER | $195,563 | 0.9465 | first-time recipient (no earlier payments received) +2.91; hour 01:00 +2.10; type is TRANSFER +1.79 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 579 | 02 | TRANSFER | $3,085,337 | 0.9876 | first-time recipient (no earlier payments received) +2.74; hour 02:00 +2.67; type is not PAYMENT +1.84 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 579 | 02 | CASH_OUT | $3,085,337 | 0.9887 | amount $3,085,336.75 +2.62; hour 02:00 +2.01; type is not PAYMENT +1.98 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 579 | 02 | TRANSFER | $150,158 | 0.9703 | first-time recipient (no earlier payments received) +2.90; hour 02:00 +2.53; type is TRANSFER +1.74 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 579 | 02 | TRANSFER | $1,632,500 | 0.9838 | hour 02:00 +2.84; first-time recipient (no earlier payments received) +2.72; type is not PAYMENT +1.81 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 579 | 02 | CASH_OUT | $1,632,500 | 0.9839 | type is CASH_OUT +2.37; hour 02:00 +2.22; amount $1,632,500.25 +2.17 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 579 | 02 | TRANSFER | $32,373 | 0.9719 | first-time recipient (no earlier payments received) +2.89; hour 02:00 +2.47; type is TRANSFER +1.75 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 579 | 02 | CASH_OUT | $32,373 | 0.9489 | hour 02:00 +3.96; type is not PAYMENT +1.75; type is not CASH_IN +1.61 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 579 | 02 | TRANSFER | $1,540,625 | 0.9838 | hour 02:00 +2.84; first-time recipient (no earlier payments received) +2.72; type is not PAYMENT +1.81 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 579 | 02 | CASH_OUT | $1,540,625 | 0.9839 | type is CASH_OUT +2.38; hour 02:00 +2.23; amount $1,540,624.75 +2.21 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 579 | 02 | TRANSFER | $254,565 | 0.9703 | first-time recipient (no earlier payments received) +2.87; hour 02:00 +2.63; type is TRANSFER +1.71 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | CASH_OUT | $124,544 | 0.9504 | hour 03:00 +4.26; type is not PAYMENT +1.72; type is not CASH_IN +1.57 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | CASH_OUT | $105,602 | 0.9504 | hour 03:00 +4.21; type is not PAYMENT +1.72; type is not CASH_IN +1.58 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | CASH_OUT | $29,551 | 0.9466 | hour 03:00 +4.12; type is not PAYMENT +1.72; type is not CASH_IN +1.57 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | CASH_OUT | $56,778 | 0.9496 | hour 03:00 +4.27; type is not PAYMENT +1.72; type is not CASH_IN +1.57 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | TRANSFER | $112,902 | 0.9745 | first-time recipient (no earlier payments received) +2.90; hour 03:00 +2.62; type is TRANSFER +1.75 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | TRANSFER | $546,757 | 0.9786 | hour 03:00 +3.08; first-time recipient (no earlier payments received) +2.73; type is not PAYMENT +1.74 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | TRANSFER | $835,771 | 0.9832 | hour 03:00 +2.97; first-time recipient (no earlier payments received) +2.62; type is not PAYMENT +1.80 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | CASH_OUT | $835,771 | 0.9761 | hour 03:00 +2.93; type is CASH_OUT +1.88; amount $835,770.81 +1.84 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | TRANSFER | $105,861 | 0.9745 | first-time recipient (no earlier payments received) +2.90; hour 03:00 +2.62; type is TRANSFER +1.75 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | TRANSFER | $282,501 | 0.9745 | first-time recipient (no earlier payments received) +2.87; hour 03:00 +2.72; type is not PAYMENT +1.72 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | CASH_OUT | $282,501 | 0.9535 | hour 03:00 +4.08; type is not PAYMENT +1.76; type is not CASH_IN +1.62 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | TRANSFER | $2,925,112 | 0.9876 | first-time recipient (no earlier payments received) +2.74; hour 03:00 +2.71; type is not PAYMENT +1.85 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | CASH_OUT | $2,925,112 | 0.9887 | amount $2,925,111.75 +2.57; hour 03:00 +2.05; type is not PAYMENT +1.99 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | TRANSFER | $628,697 | 0.9799 | hour 03:00 +3.21; first-time recipient (no earlier payments received) +2.59; type is not PAYMENT +1.77 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 580 | 03 | CASH_OUT | $628,697 | 0.9801 | hour 03:00 +3.82; type is not PAYMENT +1.84; type is not CASH_IN +1.74 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 581 | 04 | TRANSFER | $121,143 | 0.9745 | first-time recipient (no earlier payments received) +2.90; hour 04:00 +2.63; type is TRANSFER +1.75 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 581 | 04 | TRANSFER | $155,003 | 0.9745 | first-time recipient (no earlier payments received) +2.92; hour 04:00 +2.64; type is TRANSFER +1.74 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 581 | 04 | TRANSFER | $414,153 | 0.9742 | first-time recipient (no earlier payments received) +2.86; hour 04:00 +2.71; type is not PAYMENT +1.72 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 581 | 04 | TRANSFER | $801,764 | 0.9813 | hour 04:00 +3.04; first-time recipient (no earlier payments received) +2.58; type is not PAYMENT +1.78 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 581 | 04 | CASH_OUT | $801,764 | 0.9826 | hour 04:00 +3.38; type is not PAYMENT +1.84; type is not CASH_IN +1.78 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 581 | 04 | TRANSFER | $2,562,460 | 0.9838 | hour 04:00 +2.84; first-time recipient (no earlier payments received) +2.71; type is not PAYMENT +1.81 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 581 | 04 | CASH_OUT | $2,562,460 | 0.9846 | type is CASH_OUT +2.41; hour 04:00 +2.20; amount $2,562,459.50 +2.12 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $161,347 | 0.9504 | hour 05:00 +4.26; type is not PAYMENT +1.72; type is not CASH_IN +1.57 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $218,717 | 0.9535 | hour 05:00 +4.11; type is not PAYMENT +1.76; type is not CASH_IN +1.61 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $781,989 | 0.9813 | hour 05:00 +3.04; first-time recipient (no earlier payments received) +2.58; type is not PAYMENT +1.78 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $781,989 | 0.9768 | hour 05:00 +3.37; type is not PAYMENT +1.81; type is not CASH_IN +1.72 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $27,501 | 0.9747 | first-time recipient (no earlier payments received) +2.89; hour 05:00 +2.55; type is TRANSFER +1.75 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $305,710 | 0.9745 | first-time recipient (no earlier payments received) +2.87; hour 05:00 +2.72; type is not PAYMENT +1.72 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $115,912 | 0.9504 | hour 05:00 +4.23; type is not PAYMENT +1.72; type is not CASH_IN +1.58 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $65,062 | 0.9496 | hour 05:00 +4.27; type is not PAYMENT +1.72; type is not CASH_IN +1.57 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $350,319 | 0.9562 | hour 05:00 +4.00; type is not PAYMENT +1.78; type is not CASH_IN +1.63 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $25,073 | 0.9514 | hour 05:00 +4.15; type is not PAYMENT +1.75; type is not CASH_IN +1.61 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $277,192 | 0.9504 | hour 05:00 +4.24; type is not PAYMENT +1.72; type is not CASH_IN +1.58 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $233,968 | 0.9504 | hour 05:00 +4.27; type is not PAYMENT +1.72; type is not CASH_IN +1.57 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $134,773 | 0.9504 | hour 05:00 +4.27; type is not PAYMENT +1.72; type is not CASH_IN +1.57 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $184,529 | 0.9745 | first-time recipient (no earlier payments received) +2.89; hour 05:00 +2.73; type is not PAYMENT +1.72 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $184,131 | 0.9745 | first-time recipient (no earlier payments received) +2.89; hour 05:00 +2.73; type is not PAYMENT +1.72 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $181,544 | 0.9745 | first-time recipient (no earlier payments received) +2.89; hour 05:00 +2.73; type is not PAYMENT +1.72 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $181,544 | 0.9535 | hour 05:00 +4.11; type is not PAYMENT +1.76; type is not CASH_IN +1.61 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $3,349 | 0.9747 | first-time recipient (no earlier payments received) +2.89; hour 05:00 +2.53; type is TRANSFER +1.75 | RULE_002_DRAINAGE_RAIL |
| 582 | 05 | TRANSFER | $6,106,150 | 0.9876 | first-time recipient (no earlier payments received) +2.74; hour 05:00 +2.71; type is not PAYMENT +1.85 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $6,106,150 | 0.9859 | type is CASH_OUT +2.48; amount $6,106,149.50 +2.36; hour 05:00 +2.08 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $1,158,373 | 0.9838 | hour 05:00 +2.80; first-time recipient (no earlier payments received) +2.62; type is not PAYMENT +1.81 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $1,158,373 | 0.9839 | amount $1,158,373.25 +2.50; hour 05:00 +2.24; type is CASH_OUT +1.98 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $232,711 | 0.9504 | hour 05:00 +4.27; type is not PAYMENT +1.72; type is not CASH_IN +1.57 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $29,103 | 0.9507 | hour 05:00 +4.14; type is not PAYMENT +1.73; type is not CASH_IN +1.59 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $92,582 | 0.9504 | hour 05:00 +4.23; type is not PAYMENT +1.72; type is not CASH_IN +1.58 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $5,179 | 0.9562 | hour 05:00 +3.96; type is not PAYMENT +1.78; type is not CASH_IN +1.63 | RULE_002_DRAINAGE_RAIL |
| 582 | 05 | TRANSFER | $1,270,566 | 0.9838 | hour 05:00 +2.80; first-time recipient (no earlier payments received) +2.62; type is not PAYMENT +1.81 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $1,270,566 | 0.9839 | amount $1,270,565.75 +2.52; hour 05:00 +2.18; type is CASH_OUT +2.01 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $227,640 | 0.9504 | hour 05:00 +4.26; type is not PAYMENT +1.72; type is not CASH_IN +1.57 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $1,070,032 | 0.9838 | hour 05:00 +2.80; first-time recipient (no earlier payments received) +2.62; type is not PAYMENT +1.81 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $1,070,032 | 0.9711 | amount $1,070,031.62 +2.49; hour 05:00 +2.13; type is CASH_OUT +2.06 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $123,069 | 0.9745 | first-time recipient (no earlier payments received) +2.90; hour 05:00 +2.63; type is TRANSFER +1.75 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $1,235 | 0.9532 | hour 05:00 +3.98; type is not PAYMENT +1.75; type is not CASH_IN +1.60 | RULE_002_DRAINAGE_RAIL |
| 582 | 05 | CASH_OUT | $67,753 | 0.9496 | hour 05:00 +4.32; type is not PAYMENT +1.71; type is not CASH_IN +1.57 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $22,684 | 0.9535 | hour 05:00 +4.16; type is not PAYMENT +1.76; type is not CASH_IN +1.61 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $279,220 | 0.9535 | hour 05:00 +4.08; type is not PAYMENT +1.76; type is not CASH_IN +1.62 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | CASH_OUT | $4,946 | 0.9562 | hour 05:00 +3.96; type is not PAYMENT +1.78; type is not CASH_IN +1.63 | RULE_002_DRAINAGE_RAIL |
| 582 | 05 | CASH_OUT | $180,604 | 0.9504 | hour 05:00 +4.26; type is not PAYMENT +1.72; type is not CASH_IN +1.57 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 582 | 05 | TRANSFER | $47,662 | 0.9741 | first-time recipient (no earlier payments received) +2.88; hour 05:00 +2.58; type is TRANSFER +1.75 | RULE_002_DRAINAGE_RAIL, RULE_003_OVERNIGHT_DRAINAGE |
| 583 | 06 | CASH_OUT | $1,764,389 | 0.9688 | amount $1,764,389.12 +2.35; type is CASH_OUT +2.17; type is not PAYMENT +1.76 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 583 | 06 | CASH_OUT | $2,149,658 | 0.9787 | amount $2,149,658.50 +2.50; type is CASH_OUT +2.26; type is not PAYMENT +1.81 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 585 | 08 | CASH_OUT | $8,286,343 | 0.9888 | amount $8,286,342.50 +4.13; type is CASH_OUT +3.79; type is not PAYMENT +1.36 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 585 | 08 | CASH_OUT | $1,298,921 | 0.9493 | amount $1,298,920.75 +4.43; type is CASH_OUT +2.34; type is not CASH_IN +1.46 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 586 | 09 | CASH_OUT | $10,000,000 | 0.9840 | amount $10,000,000.00 +4.60; type is CASH_OUT +2.17; type is not CASH_IN +1.36 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_004_LARGE_ROUND_AMOUNT |
| 586 | 09 | CASH_OUT | $10,000,000 | 0.9853 | amount $10,000,000.00 +4.20; type is CASH_OUT +3.76; type is not PAYMENT +1.33 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_004_LARGE_ROUND_AMOUNT |
| 586 | 09 | CASH_OUT | $10,000,000 | 0.9853 | amount $10,000,000.00 +4.20; type is CASH_OUT +3.76; type is not PAYMENT +1.33 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL, RULE_004_LARGE_ROUND_AMOUNT |
| 586 | 09 | CASH_OUT | $2,581,550 | 0.9775 | amount $2,581,550.00 +4.36; type is CASH_OUT +2.20; type is not CASH_IN +1.33 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 587 | 10 | CASH_OUT | $1,249,722 | 0.9645 | amount $1,249,721.75 +4.49; type is CASH_OUT +2.25; type is not CASH_IN +1.53 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 588 | 11 | CASH_OUT | $2,709,729 | 0.9791 | amount $2,709,729.25 +4.08; type is CASH_OUT +3.53; type is not PAYMENT +1.29 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 590 | 13 | CASH_OUT | $2,117,504 | 0.9806 | amount $2,117,503.50 +4.09; type is CASH_OUT +3.56; type is not PAYMENT +1.29 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 591 | 14 | CASH_OUT | $4,863,106 | 0.9839 | amount $4,863,106.50 +4.62; type is CASH_OUT +2.18; type is not CASH_IN +1.36 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 591 | 14 | CASH_OUT | $1,777,759 | 0.9658 | amount $1,777,758.62 +4.27; type is CASH_OUT +1.90; type is not CASH_IN +1.30 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 592 | 15 | CASH_OUT | $1,170,213 | 0.9542 | amount $1,170,213.12 +4.41; type is CASH_OUT +2.17; type is not CASH_IN +1.49 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 592 | 15 | CASH_OUT | $1,798,216 | 0.9724 | amount $1,798,215.75 +3.96; type is CASH_OUT +3.42; type is not PAYMENT +1.26 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 594 | 17 | CASH_OUT | $1,592,602 | 0.9668 | amount $1,592,602.25 +4.04; type is CASH_OUT +3.38; type is not PAYMENT +1.24 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 595 | 18 | CASH_OUT | $2,209,035 | 0.9789 | amount $2,209,035.25 +4.10; type is CASH_OUT +3.48; type is not PAYMENT +1.29 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 596 | 19 | TRANSFER | $3,002,224 | 0.9536 | first-time recipient (no earlier payments received) +4.16; amount $3,002,224.00 +2.36; type is not CASH_IN +1.16 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 596 | 19 | TRANSFER | $5,296,878 | 0.9536 | first-time recipient (no earlier payments received) +4.16; amount $5,296,878.00 +2.36; type is not CASH_IN +1.16 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 597 | 20 | CASH_OUT | $1,015,052 | 0.9669 | amount $1,015,051.69 +4.19; type is not CASH_IN +1.59; type is not PAYMENT +1.29 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 598 | 21 | TRANSFER | $5,566,368 | 0.9678 | first-time recipient (no earlier payments received) +4.28; amount $5,566,368.00 +2.17; type is not CASH_IN +1.20 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 598 | 21 | CASH_OUT | $5,566,368 | 0.9896 | amount $5,566,368.00 +3.89; type is CASH_OUT +3.51; type is not PAYMENT +1.39 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | TRANSFER | $3,455,440 | 0.9697 | first-time recipient (no earlier payments received) +4.33; amount $3,455,440.25 +2.03; type is not CASH_IN +1.22 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | CASH_OUT | $3,455,440 | 0.9900 | type is CASH_OUT +3.77; amount $3,455,440.25 +3.64; type is not PAYMENT +1.39 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | TRANSFER | $2,095,700 | 0.9718 | first-time recipient (no earlier payments received) +4.29; amount $2,095,699.88 +1.91; hour 23:00 +1.39 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | CASH_OUT | $2,095,700 | 0.9881 | amount $2,095,699.88 +3.73; type is CASH_OUT +1.81; type is not CASH_IN +1.43 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | TRANSFER | $1,158,071 | 0.9490 | first-time recipient (no earlier payments received) +3.95; amount $1,158,070.88 +2.18; type is not CASH_IN +1.29 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | CASH_OUT | $1,158,071 | 0.9792 | amount $1,158,070.88 +3.85; type is not CASH_IN +1.68; type is not PAYMENT +1.38 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | TRANSFER | $983,417 | 0.9477 | first-time recipient (no earlier payments received) +4.00; amount $983,417.50 +2.08; hour 23:00 +1.28 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | CASH_OUT | $983,417 | 0.9749 | amount $983,417.50 +3.62; type is not CASH_IN +1.64; type is not PAYMENT +1.34 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | TRANSFER | $7,363,709 | 0.9697 | first-time recipient (no earlier payments received) +4.33; amount $7,363,709.00 +2.03; type is not CASH_IN +1.22 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | CASH_OUT | $7,363,709 | 0.9900 | type is CASH_OUT +3.71; amount $7,363,709.00 +3.52; type is not PAYMENT +1.39 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | TRANSFER | $1,649,040 | 0.9647 | first-time recipient (no earlier payments received) +4.27; amount $1,649,039.62 +1.85; hour 23:00 +1.37 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |
| 600 | 23 | CASH_OUT | $1,649,040 | 0.9815 | type is CASH_OUT +3.55; amount $1,649,039.62 +3.45; type is not PAYMENT +1.33 | RULE_001_HIGH_VALUE, RULE_002_DRAINAGE_RAIL |

125 of 125 alerts meet at least one rule condition.
