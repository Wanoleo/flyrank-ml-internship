Capstone Report —

* Author: Wania Arif
* Lane: Refresh / Content Opportunity Scoring
* Repo: https://github.com/Wanoleo/flyrank-ml-internship
* Date: 2026-09-09

## 0. Abstract

Content editors managing thousands of pages need a way to know which ones to
review first as search performance shifts. Using page-level features from
FlyRank's 79M-row search warehouse — prior-period impressions, query
concentration, and 60-day position volatility — this project trains a
RandomForest to flag pages likely to decline, validated on clients it never
saw during training. The model reaches 0.705 ROC-AUC versus 0.623 for the
strongest single-signal baseline (position volatility alone), a +0.082
improvement that held up under a client-grouped split, not just a random
one. Raw accuracy (0.646) is actually below the task's 67.7% base rate — a
deliberate finding, not a flaw, since accuracy is misleading under this
class imbalance and AUC is the honest metric here. The output is a ranked,
reason-coded action list an editor can use to prioritize review today.

## 1. Problem framing

**Decision:** Which pages should a content editor prioritize for review
when search impressions start dropping.

**Unit of analysis:** One content item (page), within one client.

**Output:** A ranked list of pages by estimated decline risk, with
plain-language reason codes.

**Action taken:** An editor or SEO specialist reviews the highest-risk pages
first instead of working through pages in an arbitrary order.

**Cost of a wrong call:** A false positive wastes editorial time reviewing a
page that didn't need it. A false negative lets a genuinely declining page
go unreviewed and keep losing traffic.

**Why ML helps:** No single feature was strong enough alone (baseline AUC
0.623) — combining prior impressions, query-mix signals, and position
volatility produced a meaningfully better ranking (0.705) than any one
signal used by itself.

## 2. Data safety

**Data used:** `fact_content_daily_performance` (daily impressions, clicks,
avg. position — most recent 60 days) and `fact_content_query_90d`
(query-level concentration signals), from the FlyRank/internship-warehouse
release on Hugging Face.

**Deliberately excluded:** No pre-computed `trend_direction`/`trend_pct`
style label-derived fields were used as features — the label was built from
scratch off raw impression counts to avoid inheriting someone else's label
logic or leakage. `client_hash_id` and `content_hash_id` are pseudonymous
identifiers used only for joining tables and for the grouped train/test
split — never passed to the model as features.

**Leakage check:** Features are computed from the prior-30-day window;
the label is computed from the following last-30-day window. No feature is
derived from the same days used to define the label.

**Confirmed:** No client names, raw URLs, or query text appear anywhere in
`work/` — only hashed IDs and aggregated numeric features.

## 3. Baseline

**Baseline:** `pos_volatility_60d` (standard deviation of average search
position over 60 days) used directly as a ranking score, with no model.
Chosen as the baseline because feature-importance analysis showed it was
the single strongest raw signal available.

**Baseline ROC-AUC (held-out clients): 0.623**

This is a fair comparison because it's scored on the exact same
GroupShuffleSplit test set as the model, using the same metric.

## 4. Model / analysis

**Method:** RandomForestClassifier (200 trees), chosen for its ability to
combine several moderate signals non-linearly without heavy tuning.

**Features:** `imp_prev30`, `visible_queries`, `rare_share`, `anon_share`,
`top_query_share`, `pos_volatility_60d`.

**Left out on purpose:** any client- or content-identifying field, and any
field derived from the same window as the label (to prevent leakage).

**Target/proxy, one sentence:** `is_declining` = 1 if a page's impressions
fell more than 20% from the prior 30-day window to the last 30-day window,
else 0 — a proxy for "needs review," not a claim about cause.

## 5. Evaluation

**Split:** GroupShuffleSplit by `client_hash_id` (75/25), not a random row
split — this tests whether the model generalizes to clients it has never
seen, which is the real-world condition a production tool would face.

**Metrics, model vs. baseline, same split:**

| Metric | Baseline (volatility alone) | Model (6 features) |
|---|---|---|
| ROC-AUC | 0.623 | 0.705 |
| Accuracy | — | 0.646 |
| Base rate (majority class) | 0.677 | 0.677 |

**Error analysis:** Accuracy (0.646) sits *below* the 0.677 base rate,
because roughly 68% of held-out pages are already labeled "declining" —
a model that always guessed "declining" would score higher accuracy while
learning nothing. The model instead trades some accuracy for genuine class
separation: recall on the harder, minority (non-declining) class reached
0.668, up from 0.336 in an earlier version of the model that lacked the
volatility feature. AUC (0.705) is the metric that reflects this correctly;
accuracy does not.

## 6. Interpretation

**Feature importances:** `pos_volatility_60d` (0.242) is the strongest
signal, ahead of `rare_share` (0.180), `imp_prev30` (0.177), `anon_share`
(0.168), `top_query_share` (0.134), and `visible_queries` (0.099). Pages
with unstable rankings over the past 60 days are the clearest early
indicator of decline — more so than raw impression volume.

**Surprise / negative result worth naming:** A random (non-grouped) split
initially made the model look only marginally better than the base rate;
switching to a client-grouped split revealed the model actually performs
*below* the base rate on accuracy — an important, well-understood negative
result about what a naive evaluation would have hidden, not a flaw to bury.

**Pattern in top-ranked pages:** "unstable ranking position" and
"over-reliant on one query" co-occur most often among the highest-risk
pages in the held-out ranked list, suggesting these two signals together
are a stronger warning than either alone.

## 7. Recommendation

**Action list:** All 51,701 held-out pages scored 0-1 by decline risk,
ranked highest first, each with reason codes.

**Priority band:** Top decile (score ≥ 0.835, roughly 5,170 pages) is the
recommended first-review queue — sized to whatever review capacity a team
actually has.

**How an editor uses it tomorrow:** Pull the top decile from
`ranked_action_list.csv`. Read the reason code before opening the page —
"over-reliant on one query" suggests diversifying target queries/content
breadth; "unstable ranking position" suggests checking for recent technical
or content changes; "already low impressions" pages may be lower priority
than pages still losing meaningful traffic.

**Confidence and limits:** This is directional prioritization (ROC-AUC
0.705 — meaningfully better than chance, not perfect), not a guarantee any
individual page will decline. See Limitations below.

## 8. Reproducibility

**To re-run from a fresh clone:**
1. Open `work/notebooks/w03_working_with_the_full_release.ipynb` (or the
   equivalent cells now in `capstone.ipynb`) in Colab.
2. Add a Colab Secret named `HF_TOKEN` with a Hugging Face read token that
   has accepted access to `FlyRank/internship-warehouse`.
3. Runtime → Run all. The feature-build cell (60-day DuckDB aggregation)
   takes roughly 2–6 minutes.
4. This produces `capstone_features.csv`, which `work/notebooks/capstone.ipynb`
   loads directly (upload it via the `files.upload()` cell in Section 2).
5. Run all cells in `capstone.ipynb` in order — this reproduces the model,
   both AUC numbers, the ranked action list, and both charts.

**Random seed:** `random_state=42` throughout (train/test split and model).

**Sealed/holdout evaluation:** The GroupShuffleSplit test set (`X_te2`,
`y_te2`) is built once, in the cell before Section 4, and every reported
metric (AUC, accuracy, classification report, ranked action list) is scored
against that same held-out set — not re-split or re-sampled afterward. The
cell that builds it is committed in this notebook, and `ranked_action_list.csv`
is the metrics/output file it produced, so this is checkable from the repo,
not asserted on faith.

**Environment:** Google Colab default runtime; `duckdb`, `huggingface_hub`
installed via the first setup cell; `scikit-learn`, `pandas`, `numpy`,
`matplotlib` from Colab's preinstalled environment.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).
