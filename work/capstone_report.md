# Capstone Report — Lane 2: Refresh / Content Opportunity Scoring

- **Author:** Muhammad Mohid Shafiq
- **Lane:** 2: Refresh / Content Opportunity Scoring (confirmed in ML-07)
- **Repo:** https://github.com/WingManOO7/ML-INTER
- **Paper:** https://wingmanoo7.github.io/ML-INTER/ (`submission/paper_url.txt`)
- **Date:** October 2026

> Every number below is produced by an executed notebook and stored in a committed receipt under
> `work/outputs/`. The deployed paper is rendered from those same receipts by `work/scripts/build_paper.py`.

## 0. Abstract

Content teams can review only about fifty pages every two weeks, out of tens of thousands that earn
search impressions, so the useful question is which pages losing visibility deserve that attention
first. On the FlyRank internship warehouse (78.8M pseudonymized daily Search Console rows), I score
659,482 page-dates from 51 sites at ten biweekly decision dates (January–May 2026) and seal June 2026
for one final test. I compare the transparent ML-07 rule (rank by impressions lost last week) with
learned rankers under a rolling-origin, client-grouped design, scoring every 50-page queue on precision
*and* on the share of achievable lost visibility it covers. A learned probability alone reaches the
highest precision (0.81) but covers only 8% of the visibility at stake. Weighting the rule's lost
impressions by that probability raises precision@50 from 0.64 to 0.69 (+0.054, 95% CI +0.030 to +0.079)
at equal visibility (0.41 vs 0.42). In sealed June the precision lead held (0.85 vs 0.81), while
visibility came in slightly below the rule (0.54 vs 0.56). The output is a decision-support review queue
with action routes, reason codes, calibrated confidence and drift triggers. It makes no causal claim.

## 1. Problem framing

- **Decision:** the order of an editor's review queue, about 50 pages per two-week cycle (ML-03).
- **Unit:** one page at one decision date T. Features come from the 14 days before T; the outcome from
  the 14 days from T.
- **Output:** a ranked queue. Every row carries an action (`REFRESH_REVIEW`, `SITE_LEVEL_CHECK`,
  `TECHNICAL_CHECK`, `MONITOR`), a reason code and a confidence band.
- **Cost of a wrong call:**
  - a wasted review hour (the page was fine);
  - a missed loss (a decaying page never surfaced);
  - and the one ML-08 exposed: a page that does decline but is too small to matter. This is why the
    queue is scored on precision **and** visibility.
- **Why ML:** 42k–83k eligible pages compete for 50 slots each cycle. The question is whether a learned
  probability puts the same budget on more real, lasting losses than a transparent rule does.

## 2. Data safety

- **Data:** warehouse release v20260703. `fact_content_daily_performance` month partitions 2025-12 →
  2026-06, Search Console columns only, every row filtered `gsc_data_available IS TRUE`;
  `dim_content.content_created_date` only.
- **Excluded:**
  - `content_updated_date`, `last_optimized_date` and the publish/delete flags: export snapshots, after
    T for 76–100% of pages (ML-04, ML-09);
  - GA4, AI and scroll columns (patchy availability, out of scope);
  - keyword-tool fields (undated);
  - clicks.
- **IDs:** pseudonyms used for grouping and splitting only. ML-09 shows what an ID feature does under a
  careless split (+0.21 precision that disappears under the honest one).
- **Leakage:**
  - 21 unit tests on synthetic data;
  - an outcome-feature guard in the shared module;
  - a planted-leak test (AUC 0.614 → 0.994 with the guard bypassed) and a shuffled-label test (AUC 0.499);
  - product flags don't exist in the release.
- **Public safety:** nothing client-identifying appears in `work/` or the paper. The queue CSVs carry
  pseudonymous IDs and are gitignored; the receipts and paper hold aggregates only.

## 3. Baseline

The frozen **ML-07 rule**: flag a page when week-2 impressions are more than 20% below week 1, and rank
by impressions lost. If a page stays at its week-2 level, its two-week loss equals that number, so the
rule is a *persistence forecast* of the loss: a fair and strong comparison. It is recomputed in the
capstone run on the same 40 queues: precision@50 **0.640**, visibility@50 **0.412**.

## 4. Model / analysis

- **Policy:** `score = P(decline) × lost weekly impressions`, with P from logistic regression on the
  five ML-04 contract features: `log_imp_pre`, `pos_pre`, `active_days_pre`, `momentum_pre`,
  `age_days_at_T`.
- **Label:** `declined_robust` means the 14 days from T fall more than 20% below the 14 days before,
  after dropping each window's single biggest day. ML-07 found one-day spikes fooling the plain label;
  the contract label is kept as a robustness check.
- **Selection rule (declared before the backtest):** beat the rule on precision@50 without losing
  visibility@50, then pick the simplest. Gradient boosting and three extra features scored within noise,
  so the five-feature logistic version is the frozen policy.

## 5. Evaluation

- **Design:** rolling origin × client groups. At 8 test dates (Feb 2 → May 11), train on earlier dates
  and other clients only, then score the held-out clients: **40 queues**. 95% intervals come from a
  bootstrap that resamples whole dates.
- **Development:**
  - policy 0.694 vs rule 0.640 precision@50 (+0.054 [+0.030, +0.079]);
  - visibility 0.421 vs 0.412 (+0.009 [−0.031, +0.056]);
  - AUC +0.042 [+0.026, +0.059];
  - higher or equal precision in 34 of 40 queues and on all 8 dates.
- **Sealed June** (computed once; fingerprint `d9e8d9bfbd813d89`; 163,200 pages, 47 clients, 4 unseen;
  10 queues):
  - precision 0.854 vs 0.810 (9 of 10 queues);
  - visibility 0.544 vs 0.560 (2 of 10 queues): **a miss on the guardrail**;
  - June base rates 0.48 and 0.59, beyond the development range.
- **Errors:**
  - Probability-only rankers reach 0.81–0.83 precision but cover 8–9% of the visibility: they pick tiny
    collapsed pages.
  - Inside the policy's top 50, collapses (90% persist) and site-wide slides (80%) are right but need a
    different action, so they became routes.
  - Clean slides persist 63% of the time and carry 56% of the lost visibility.

## 6. Interpretation

- **The decline signal is the slide itself.** The week-over-week drop is CONFIRMED (decline rate 0.72
  for collapsed pages vs 0.20–0.23 for rising pages). Page age, the classic refresh-flag signal, is flat
  (0.32–0.38).
- **Persistence, not early warning.** On a stricter "fell further" label, the slide falls to chance:
  the system triages visible drops and doesn't foresee new ones.
- **A negative result for the paper I audited.** FlyRank's Finding #1, "older pages decline", does not
  replicate with a forward label inside clients. Its Finding #4 refresh multiplier rests on update dates
  that are mostly bulk system activity, and on pages that already had 3.4× the impressions before they
  were optimised.

## 7. Recommendation

- **How to use the queue:**
  1. Release it at each decision date unless a pre-release trigger fires (T1 feature drift vs the
     previous cycle, T2 site-level share of the top 50).
  2. Work `REFRESH_REVIEW` rows in score order.
  3. Send `SITE_LEVEL_CHECK` rows to site owners first and `TECHNICAL_CHECK` rows to indexing/redirect
     checks.
  4. Hold `MONITOR` rows for one cycle.
  5. Dedupe identical-series twins.
- **Confidence:** the bands are calibrated on the backtest (high 83%, medium 64%, low 46% of picks
  persisted). The 2026-06-22 queue: 30 refresh reviews, 12 site-level checks, 4 technical checks and
  4 monitors in the top 50.
- **Limits:**
  - a short and regime-shifting history;
  - high per-cycle variance;
  - the label partly measures the market;
  - Search Console only, above-floor pages only;
  - **no causal claim**.

## 8. Reproducibility

```bash
pip install -r requirements.txt pytest nbconvert ipykernel
export HF_TOKEN=...                         # a Hugging Face READ token, never in a cell
python -m pytest work/tests                 # 21 tests, no data needed
# execute in order: w03 -> w04 -> w05 -> w06 -> w07 -> capstone
jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
python work/scripts/build_paper.py          # re-renders docs/index.html from the receipts
```

- **Determinism:** seed 42 everywhere. Frames are sorted on build (DuckDB's GROUP BY order is not
  stable), and reruns reproduce the receipts byte for byte.
- **Environment:** Python 3.10, pandas 2.3.3, scikit-learn 1.7.2, DuckDB 1.5.6.
- **Sealed evaluation:** the cell that builds the sealed frame is `capstone.ipynb` §4b, and its metrics
  file is `work/outputs/sealed_june_receipts.json`. The cell re-asserts the fingerprint and the numbers
  on every rerun.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).
