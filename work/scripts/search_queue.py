"""search_queue — shared building blocks for my Lane 2 capstone (refresh-review queue).

The notebooks from ML-09 on (w06, w07, capstone) import this one file, so the frame, the labels,
the frozen ML-07 baseline and the metrics are defined once and tested once
(work/tests/test_search_queue.py). The ML-04 to ML-08 notebooks keep their own inline code on
purpose: they are the record of how each piece was first built.

Decision setup (ML-04 contract, extended to many decision dates):
- a decision date T is a Monday; the feature window is the 14 days before T (week 1 + week 2),
  plus a 14-day look-back before that for the longer trend;
- the outcome window is the 14 days from T and is used for labels only;
- a page is scored at T if it had >= 100 impressions in the feature window, was tracked from the
  window's first day and has a valid position.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = 'FlyRank/internship-warehouse'
KEYS = ['client_hash_id', 'content_hash_id']
WINDOW = 14
MIN_IMPRESSIONS = 100

# the five ML-04 contract features, plus three pre-T candidates added after the ML-08 error analysis
CONTRACT_FEATURES = ['log_imp_pre', 'pos_pre', 'active_days_pre', 'momentum_pre', 'age_days_at_T']
EXTRA_FEATURES = ['momentum_4w', 'peak_day_share_pre', 'site_momentum']
# every column computed from the outcome window: none of them may ever be a feature
OUTCOME_COLUMNS = {'imp_out', 'max_day_out', 'declined', 'declined_robust', 'gap', 'gap_robust', 'out_days'}


def decision_dates(first, last, step_days=WINDOW):
    """Mondays from `first` to `last`, every `step_days` days."""
    dates = pd.date_range(first, last, freq=f'{step_days}D')
    assert all(d.dayofweek == 0 for d in dates), 'decision dates must be Mondays'
    return list(dates)


def months_needed(dates):
    """Month partitions a set of decision dates touches (look-back through outcome window)."""
    months = set()
    for T in dates:
        for d in pd.date_range(T - pd.Timedelta(days=2 * WINDOW), T + pd.Timedelta(days=WINDOW - 1)):
            months.add(d.strftime('%Y-%m'))
    return sorted(months)


def open_warehouse(months, token, cache_dir=None):
    """DuckDB connection with a `daily` view over the given month partitions and `dim_content`.

    Files are fetched once into the Hugging Face cache; later runs read local copies.
    """
    import duckdb
    from huggingface_hub import hf_hub_download, list_repo_files

    def fetch(path):
        return Path(hf_hub_download(REPO, path, repo_type='dataset', token=token, cache_dir=cache_dir)).as_posix()

    files = list_repo_files(REPO, repo_type='dataset', token=token)
    daily = [fetch(f) for f in files
             if any(f.startswith(f'fact_content_daily_performance/month={m}/') for m in months)]
    assert daily, f'no partitions found for {months}'
    con = duckdb.connect()
    con.execute('SET enable_progress_bar = false')
    con.execute(f"CREATE OR REPLACE VIEW daily AS SELECT * FROM read_parquet({daily})")
    con.execute(f"CREATE OR REPLACE VIEW dim_content AS SELECT * FROM read_parquet('{fetch('dim_content.parquet')}')")
    return con


def frame_sql(T):
    """The SQL that aggregates one decision date. Features use only days before T."""
    T = pd.Timestamp(T)
    d = lambda x: f"DATE '{x.date()}'"
    back, f_start, wk2, o_end = (T - pd.Timedelta(days=2 * WINDOW), T - pd.Timedelta(days=WINDOW),
                                 T - pd.Timedelta(days=WINDOW // 2), T + pd.Timedelta(days=WINDOW - 1))
    ok = 'gsc_data_available IS TRUE'
    pos = f'{ok} AND gsc_avg_position > 0'
    pre = f'report_date >= {d(f_start)} AND report_date < {d(T)}'
    return f'''
        SELECT client_hash_id, content_hash_id,
               MIN(report_date) FILTER (WHERE {pre})                                         AS first_seen,
               SUM(gsc_impressions) FILTER (WHERE {ok} AND {pre})                            AS imp_pre,
               SUM(gsc_impressions) FILTER (WHERE {ok} AND report_date >= {d(f_start)} AND report_date < {d(wk2)}) AS imp_wk1,
               SUM(gsc_impressions) FILTER (WHERE {ok} AND report_date >= {d(wk2)} AND report_date < {d(T)})      AS imp_wk2,
               SUM(gsc_impressions) FILTER (WHERE {ok} AND report_date < {d(f_start)})       AS imp_back,
               MAX(gsc_impressions) FILTER (WHERE {ok} AND {pre})                            AS max_day_pre,
               SUM(gsc_sum_position) FILTER (WHERE {pos} AND {pre})
                 / NULLIF(SUM(gsc_impressions) FILTER (WHERE {pos} AND {pre}), 0)          AS pos_pre,
               COUNT(*) FILTER (WHERE {ok} AND {pre})                                        AS active_days_pre,
               COUNT(*) FILTER (WHERE report_date >= {d(T)})                                 AS out_days,
               SUM(gsc_impressions) FILTER (WHERE {ok} AND report_date >= {d(T)})            AS imp_out,
               MAX(gsc_impressions) FILTER (WHERE {ok} AND report_date >= {d(T)})            AS max_day_out
        FROM daily
        WHERE report_date >= {d(back)} AND report_date <= {d(o_end)}
        GROUP BY ALL
    '''


def finish_frame(raw, created, T):
    """Eligibility, features and labels from the aggregated rows (pure pandas, unit-tested)."""
    T = pd.Timestamp(T)
    f_start = T - pd.Timedelta(days=WINDOW)
    p = raw.copy()
    p = p[(p['imp_pre'].fillna(0) >= MIN_IMPRESSIONS) & (pd.to_datetime(p['first_seen']) == f_start)]
    p = p.dropna(subset=['pos_pre']).merge(created, on='content_hash_id', how='left')
    fill = ['imp_wk1', 'imp_wk2', 'imp_back', 'imp_out', 'max_day_out', 'out_days']
    p[fill] = p[fill].fillna(0)

    # features — everything here is known before T
    p['decision_date'] = T
    p['log_imp_pre'] = np.log1p(p['imp_pre'])
    p['momentum_pre'] = np.log((p['imp_wk2'] + 1) / (p['imp_wk1'] + 1))
    p['age_days_at_T'] = (T - pd.to_datetime(p['content_created_date'])).dt.days
    p['momentum_4w'] = np.log((p['imp_pre'] + 1) / (p['imp_back'] + 1))
    p['peak_day_share_pre'] = p['max_day_pre'] / p['imp_pre']
    site = p.groupby('client_hash_id')[['imp_wk1', 'imp_wk2']].sum()
    p['site_momentum'] = p['client_hash_id'].map(np.log((site['imp_wk2'] + 1) / (site['imp_wk1'] + 1)))
    p = p[p['age_days_at_T'] > 0]

    # labels — outcome window only
    p['declined'] = (p['imp_out'] < 0.8 * p['imp_pre']).astype(int)
    trimmed_pre, trimmed_out = p['imp_pre'] - p['max_day_pre'], p['imp_out'] - p['max_day_out']
    p['declined_robust'] = (trimmed_out < 0.8 * trimmed_pre).astype(int)
    p['gap'] = (p['imp_pre'] - p['imp_out']).clip(lower=0)
    p['gap_robust'] = (trimmed_pre - trimmed_out).clip(lower=0)
    # a label needs all 14 outcome days tracked; otherwise a tracking gap would look like a decline
    p['label_ok'] = p['out_days'] == WINDOW
    return normalise(p)


def normalise(p):
    """Plain numpy dtypes and a fixed row order.

    DuckDB hands back nullable Int64/Float64 and returns GROUP BY rows in a different order each
    run; plain dtypes and sorted rows keep cached and fresh frames identical and seeded models
    reproducible.
    """
    p = p.copy()
    for c in p.columns:
        if c in KEYS:
            continue
        if c in ('first_seen', 'content_created_date', 'decision_date'):
            p[c] = pd.to_datetime(p[c]).astype('datetime64[ns]')
        elif c == 'label_ok':
            p[c] = p[c].astype(bool)
        elif c in ('declined', 'declined_robust', 'active_days_pre', 'out_days', 'age_days_at_T'):
            p[c] = p[c].astype('int64')
        else:
            p[c] = p[c].astype('float64')
    return p.sort_values(['decision_date', *KEYS]).reset_index(drop=True)


def build_frame(con, T, created=None):
    if created is None:
        created = con.sql('SELECT content_hash_id, content_created_date FROM dim_content').df()
    return finish_frame(con.sql(frame_sql(T)).df(), created, T)


def build_frames(con, dates, cache_dir=None):
    """Frames for many decision dates, cached as parquet (gitignored) so reruns skip the scans."""
    import duckdb  # DuckDB reads and writes parquet itself, so no pyarrow is needed

    created = con.sql('SELECT content_hash_id, content_created_date FROM dim_content').df()
    out = []
    for T in dates:
        path = Path(cache_dir) / f'frame_{pd.Timestamp(T).date()}.parquet' if cache_dir else None
        if path is not None and path.exists():
            f = normalise(duckdb.read_parquet(path.as_posix()).df())
        else:
            f = build_frame(con, T, created)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                duckdb.from_df(f).write_parquet(path.as_posix())
        out.append(f)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- baseline and rankings
def rule_score(df):
    """The frozen ML-07 rule: weekly impressions lost when the slide exceeds 20%, else 0.

    imp_pre only breaks ties (always < 1, scores are whole impressions), as in the ML-07 queue.
    """
    flagged = (df['imp_wk2'] + 1) / (df['imp_wk1'] + 1) < 0.8
    tie = 0.5 * df['imp_pre'] / (df['imp_pre'].max() + 1)
    return np.where(flagged, df['imp_wk1'] - df['imp_wk2'], 0) + tie


def lost_impressions(df):
    """Weekly impressions lost from week 1 to week 2 (0 if the page did not drop)."""
    return (df['imp_wk1'] - df['imp_wk2']).clip(lower=0).to_numpy(dtype=float)


def queue_score(p_decline, df):
    """The frozen capstone policy: chance the drop persists x impressions it already lost.

    If a page simply stays at its week-2 level, its two-week loss equals imp_wk1 - imp_wk2 (the ML-07
    rule's score), so the rule is a 'persistence forecast' of the loss; the model adds the probability.
    """
    return np.asarray(p_decline, dtype=float) * lost_impressions(df)


def fit_policy_model(train, label='declined_robust', features=CONTRACT_FEATURES):
    """The frozen model: logistic regression on the five ML-04 contract features (seedless, deterministic)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    assert_no_outcome_features(features)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(train[features], train[label])


# routing: the ML-07/ML-08/backtest error analysis, promoted to policy (first matching rule wins)
COLLAPSE_SHARE, SITE_SLIDE, SPIKE_SHARE, NEW_PAGE_DAYS = 0.05, np.log(0.8), 0.5, 30
ROUTES = [
    ('TECHNICAL_CHECK', 'COLLAPSE_TO_NEAR_ZERO', lambda d: d['imp_wk2'] < COLLAPSE_SHARE * d['imp_wk1']),
    ('SITE_LEVEL_CHECK', 'SITE_WIDE_SLIDE', lambda d: d['site_momentum'] < SITE_SLIDE),
    ('MONITOR', 'SPIKE_FADE', lambda d: d['peak_day_share_pre'] > SPIKE_SHARE),
    ('MONITOR', 'NEW_PAGE_SETTLING', lambda d: d['age_days_at_T'] < NEW_PAGE_DAYS),
]


def route(df):
    """Action and reason code per page. Pages with no warning go to REFRESH_REVIEW."""
    action = np.full(len(df), 'REFRESH_REVIEW', dtype=object)
    reason = np.full(len(df), 'SLIDING_WITH_DEMAND', dtype=object)
    done = np.zeros(len(df), dtype=bool)
    for act, why, test in ROUTES:
        hit = np.asarray(test(df), dtype=bool) & ~done
        action[hit], reason[hit], done = act, why, done | hit
    no_drop = lost_impressions(df) == 0
    action[no_drop], reason[no_drop] = 'NO_ACTION', 'NO_DROP_BEFORE_T'
    return pd.DataFrame({'action': action, 'reason_code': reason}, index=df.index)


def assert_no_outcome_features(features):
    bad = OUTCOME_COLUMNS & set(features)
    assert not bad, f'outcome-window columns cannot be features: {sorted(bad)}'


# ---------------------------------------------------------------- metrics
def precision_at_k(y, score, k=50):
    order = np.argsort(-np.asarray(score, dtype=float), kind='stable')[:k]
    return float(np.asarray(y)[order].mean())


def visibility_at_k(gap, score, k=50):
    """Share of the best achievable impressions-lost that the top-k actually covers (0-1)."""
    gap = np.asarray(gap, dtype=float)
    got = gap[np.argsort(-np.asarray(score, dtype=float), kind='stable')[:k]].sum()
    best = np.sort(gap)[::-1][:k].sum()
    return float(got / best) if best > 0 else float('nan')


def queue_metrics(df, score, label='declined_robust', gap='gap_robust', k=50):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = df[label].to_numpy()
    out = {'n': len(df), 'base_rate': float(y.mean()),
           f'P@{k}': precision_at_k(y, score, k), f'visibility@{k}': visibility_at_k(df[gap], score, k)}
    if 0 < y.sum() < len(y):
        out.update({'AP': float(average_precision_score(y, score)), 'AUC': float(roc_auc_score(y, score))})
    return out


# ---------------------------------------------------------------- splits
def client_folds(clients, n_folds=5, seed=42):
    """Stable client -> fold map (same client, same fold, at every decision date)."""
    clients = np.array(sorted(set(clients)))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(clients))
    return {c: int(i % n_folds) for i, c in zip(range(len(clients)), clients[order])}


def assert_split_clean(train, test):
    shared = set(train['client_hash_id']) & set(test['client_hash_id'])
    assert not shared, f'{len(shared)} client(s) on both sides of the split'


def assert_time_forward(train, test):
    """Every training label must be observed before the earliest test decision date."""
    last_label_day = pd.to_datetime(train['decision_date']).max() + pd.Timedelta(days=WINDOW - 1)
    first_test = pd.to_datetime(test['decision_date']).min()
    assert last_label_day < first_test, f'training labels end {last_label_day.date()} >= test date {first_test.date()}'


def rolling_backtest(frames, models, label='declined_robust', n_folds=5, first_test=2, seed=42):
    """Rolling-origin x client-grouped backtest.

    For every decision date from index `first_test` on, and every client fold: train on EARLIER dates
    and OTHER clients only, score the held-out clients at that date. `models` maps a name to
    (features, make_fn); each fitted model's probability is stored as column `p_<name>`.
    """
    frames = frames[frames['label_ok']].copy()
    folds = client_folds(frames['client_hash_id'], n_folds, seed)
    frames['fold'] = frames['client_hash_id'].map(folds)
    dates = sorted(frames['decision_date'].unique())
    out = []
    for T in dates[first_test:]:
        for f in range(n_folds):
            train = frames[(frames['decision_date'] < T) & (frames['fold'] != f)]
            test = frames[(frames['decision_date'] == T) & (frames['fold'] == f)]
            if len(test) == 0 or test[label].nunique() < 2:
                continue
            assert_split_clean(train, test)
            assert_time_forward(train, test)
            scored = test.copy()
            for name, (features, make) in models.items():
                assert_no_outcome_features(features)
                scored[f'p_{name}'] = make().fit(train[features], train[label]).predict_proba(test[features])[:, 1]
            out.append(scored)
    return pd.concat(out, ignore_index=True)


def cell_metrics(pred, scores, label='declined_robust', gap='gap_robust', k=50):
    """queue_metrics for every (decision date, fold) cell and every named score column/function."""
    rows = []
    for (d, f), cell in pred.groupby(['decision_date', 'fold']):
        for name, score in scores.items():
            s = score(cell) if callable(score) else cell[score]
            m = queue_metrics(cell, np.asarray(s, dtype=float), label, gap, k)
            rows.append({'decision_date': d, 'fold': f, 'policy': name, **m})
    out = pd.DataFrame(rows)
    out['lift@50'] = out['P@50'] / out['base_rate']
    return out


def date_bootstrap(cells, policy, reference, metric='P@50', n=2000, seed=42):
    """Mean paired difference (policy - reference) with a 95% interval, resampling whole decision dates.

    Folds of the same date share a market week, so dates (not cells) are the resampling unit.
    """
    wide = cells.pivot_table(index=['decision_date', 'fold'], columns='policy', values=metric)
    diff = (wide[policy] - wide[reference]).rename('diff').reset_index()
    by_date = diff.groupby('decision_date')['diff'].apply(np.array)
    rng = np.random.RandomState(seed)
    dates = list(by_date.index)
    boots = [np.concatenate([by_date[dates[i]] for i in rng.randint(0, len(dates), len(dates))]).mean() for _ in range(n)]
    return float(diff['diff'].mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def psi(expected, actual, bins=10):
    """Population stability index of `actual` against `expected` (quantile bins of expected)."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
