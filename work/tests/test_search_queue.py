"""Tests for work/scripts/search_queue.py — the paths whose failure would invalidate the paper.

Run from the repo root:  python -m pytest work/tests -q
Everything here uses small synthetic tables, so no data access or token is needed.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import search_queue as sq  # noqa: E402

T = pd.Timestamp('2026-03-16')  # a Monday


def page_days(client, content, start, end, impressions, position=5.0, flag=True):
    """One row per day for one page; `impressions` is a number or a {date: value} dict."""
    rows = []
    for d in pd.date_range(start, end):
        imp = impressions.get(d, 0) if isinstance(impressions, dict) else impressions
        has = flag if flag is None else (flag and imp > 0)
        rows.append({'report_date': d.date(), 'client_hash_id': client, 'content_hash_id': content,
                     'gsc_data_available': has, 'gsc_impressions': imp if has else 0,
                     'gsc_avg_position': position if has else 0.0,
                     'gsc_sum_position': int(round(position * imp)) if has else 0})
    return rows


def frame_from(rows, created=None):
    con = duckdb.connect()
    daily = pd.DataFrame(rows)
    con.register('daily_df', daily)
    con.execute('CREATE VIEW daily AS SELECT * FROM daily_df')
    if created is None:
        created = pd.DataFrame({'content_hash_id': daily['content_hash_id'].unique(),
                                'content_created_date': pd.Timestamp('2025-06-01')})
    return sq.build_frame(con, T, created)


START, END = T - pd.Timedelta(days=28), T + pd.Timedelta(days=13)
PRE_START, PRE_END = T - pd.Timedelta(days=14), T - pd.Timedelta(days=1)


def steady_page(content='content_a', client='client_1', pre=100, out=100):
    imp = {d: (pre if d < T else out) for d in pd.date_range(START, END)}
    return page_days(client, content, START, END, imp)


# ------------------------------------------------------------------ windows and labels
def test_features_never_see_the_outcome_window():
    calm = frame_from(steady_page(out=100))
    crash = frame_from(steady_page(out=1))
    features = sq.CONTRACT_FEATURES + sq.EXTRA_FEATURES
    pd.testing.assert_frame_equal(calm[features], crash[features])
    assert calm.loc[0, 'declined'] == 0 and crash.loc[0, 'declined'] == 1


def test_label_threshold_is_twenty_percent():
    assert frame_from(steady_page(pre=100, out=79)).loc[0, 'declined'] == 1   # 21% drop
    assert frame_from(steady_page(pre=100, out=81)).loc[0, 'declined'] == 0   # 19% drop


def test_robust_label_ignores_a_single_spike_day():
    imp = {d: 100 for d in pd.date_range(START, END)}
    imp[PRE_START + pd.Timedelta(days=3)] = 20_000            # one burst inside the feature window
    f = frame_from(page_days('client_1', 'content_a', START, END, imp))
    assert f.loc[0, 'declined'] == 1            # the plain label calls a return-to-normal a decline
    assert f.loc[0, 'declined_robust'] == 0     # the spike-robust label does not
    assert f.loc[0, 'peak_day_share_pre'] > 0.9


def test_week_split_and_momentum():
    imp = {d: (200 if d < PRE_START + pd.Timedelta(days=7) else 50) for d in pd.date_range(START, END)}
    f = frame_from(page_days('client_1', 'content_a', START, END, imp))
    assert f.loc[0, 'imp_wk1'] == 7 * 200 and f.loc[0, 'imp_wk2'] == 7 * 50
    assert np.isclose(f.loc[0, 'momentum_pre'], np.log((350 + 1) / (1400 + 1)))


def test_null_or_false_flag_rows_never_count():
    rows = steady_page()
    for r in rows:
        if pd.Timestamp(r['report_date']) == PRE_START:
            r['gsc_data_available'] = None          # NULL flag: must be treated as "not available"
            r['gsc_impressions'] = 50_000
    f = frame_from(rows)
    assert f.loc[0, 'imp_pre'] == 13 * 100


def test_eligibility_rules():
    rows = steady_page('ok')
    rows += page_days('client_1', 'too_small', START, END, 7)                          # 98 < 100 impressions
    rows += page_days('client_1', 'late_start', PRE_START + pd.Timedelta(days=1), END, 100)
    rows += page_days('client_1', 'no_position', START, END, 100, position=0.0)
    f = frame_from(rows)
    assert list(f['content_hash_id']) == ['ok']


def test_untracked_outcome_days_make_the_label_unusable():
    rows = [r for r in steady_page() if pd.Timestamp(r['report_date']) < T + pd.Timedelta(days=10)]
    f = frame_from(rows)
    assert not f.loc[0, 'label_ok']


# ------------------------------------------------------------------ baseline and metrics
def test_rule_score_matches_the_ml07_order():
    df = pd.DataFrame({'imp_wk1': [1000, 1000, 500, 100], 'imp_wk2': [100, 900, 100, 100],
                       'imp_pre': [1100, 1900, 600, 200]})
    order = np.argsort(-sq.rule_score(df))
    assert list(order) == [0, 2, 1, 3]       # flagged by lost impressions; unflagged by volume


def test_precision_and_visibility_at_k():
    y = np.array([1, 0, 1, 1, 0])
    score = np.array([5, 4, 3, 2, 1])
    assert sq.precision_at_k(y, score, 2) == 0.5
    assert sq.precision_at_k(y, score, 4) == 0.75
    gap = np.array([10, 0, 30, 5, 0])
    assert sq.visibility_at_k(gap, score, 2) == pytest.approx(10 / 40)
    assert sq.visibility_at_k(gap, gap, 2) == 1.0          # ranking by the true loss is the best possible
    assert sq.visibility_at_k(gap, -gap, 2) == 0.0         # ranking the smallest losses first covers nothing


def test_precision_ties_keep_input_order():
    y = np.array([0, 1, 1])
    assert sq.precision_at_k(y, np.zeros(3), 1) == 0.0


# ------------------------------------------------------------------ splits and leak guards
def test_split_guards():
    a = pd.DataFrame({'client_hash_id': ['c1', 'c2'], 'decision_date': [T - pd.Timedelta(days=28)] * 2})
    b = pd.DataFrame({'client_hash_id': ['c3'], 'decision_date': [T]})
    sq.assert_split_clean(a, b)
    sq.assert_time_forward(a, b)
    with pytest.raises(AssertionError):
        sq.assert_split_clean(a, a)
    with pytest.raises(AssertionError):
        sq.assert_time_forward(a.assign(decision_date=T - pd.Timedelta(days=7)), b)


def test_outcome_columns_are_refused_as_features():
    sq.assert_no_outcome_features(sq.CONTRACT_FEATURES + sq.EXTRA_FEATURES)
    for leak in ['declined', 'imp_out', 'gap_robust']:
        with pytest.raises(AssertionError):
            sq.assert_no_outcome_features(sq.CONTRACT_FEATURES + [leak])


def test_client_folds_are_stable():
    clients = [f'c{i}' for i in range(20)]
    assert sq.client_folds(clients) == sq.client_folds(list(reversed(clients)))
    assert set(sq.client_folds(clients).values()) == set(range(5))


def test_decision_dates_and_months():
    dates = sq.decision_dates('2026-01-05', '2026-02-02')
    assert [d.day for d in dates] == [5, 19, 2]
    assert sq.months_needed([T]) == ['2026-02', '2026-03']


def test_queue_score_is_probability_times_lost_impressions():
    df = pd.DataFrame({'imp_wk1': [1000, 100, 500], 'imp_wk2': [400, 300, 500]})
    assert list(sq.lost_impressions(df)) == [600, 0, 0]
    assert list(sq.queue_score([0.5, 0.9, 0.9], df)) == [300, 0, 0]


def test_routes_follow_the_documented_precedence():
    base = {'imp_wk1': 1000.0, 'imp_wk2': 500.0, 'site_momentum': 0.0, 'peak_day_share_pre': 0.2, 'age_days_at_T': 200}
    cases = pd.DataFrame([
        base,                                                         # clean slide
        {**base, 'imp_wk2': 10.0, 'site_momentum': -1.0},             # collapse wins over site-wide
        {**base, 'site_momentum': -1.0, 'peak_day_share_pre': 0.9},   # site-wide wins over spike
        {**base, 'peak_day_share_pre': 0.9},
        {**base, 'age_days_at_T': 10},
        {**base, 'imp_wk2': 1200.0},                                  # no drop at all
    ])
    r = sq.route(cases)
    assert list(r['reason_code']) == ['SLIDING_WITH_DEMAND', 'COLLAPSE_TO_NEAR_ZERO', 'SITE_WIDE_SLIDE',
                                      'SPIKE_FADE', 'NEW_PAGE_SETTLING', 'NO_DROP_BEFORE_T']
    assert list(r['action']) == ['REFRESH_REVIEW', 'TECHNICAL_CHECK', 'SITE_LEVEL_CHECK', 'MONITOR', 'MONITOR', 'NO_ACTION']


def synthetic_panel(n_clients=10, pages=40, dates=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for d in range(dates):
        for c in range(n_clients):
            x = rng.normal(size=(pages, len(sq.CONTRACT_FEATURES)))
            y = (x[:, 3] + rng.normal(scale=0.5, size=pages) < 0).astype(int)
            for i in range(pages):
                rows.append({'client_hash_id': f'c{c}', 'content_hash_id': f'p{c}_{i}',
                             'decision_date': T + pd.Timedelta(days=14 * d), 'label_ok': True,
                             **dict(zip(sq.CONTRACT_FEATURES, x[i])), 'declined_robust': y[i]})
    return pd.DataFrame(rows)


def test_rolling_backtest_is_time_forward_and_client_grouped():
    from sklearn.linear_model import LogisticRegression
    panel = synthetic_panel()
    pred = sq.rolling_backtest(panel, {'lr': (sq.CONTRACT_FEATURES, lambda: LogisticRegression())}, first_test=1)
    assert pred['decision_date'].min() == T + pd.Timedelta(days=14)        # the first date is never a test date
    assert pred.groupby(['decision_date', 'content_hash_id']).size().max() == 1
    assert set(pred['fold']) == set(range(5))
    assert pred['p_lr'].between(0, 1).all()


def test_rolling_backtest_refuses_an_outcome_feature():
    panel = synthetic_panel().assign(declined=0)
    with pytest.raises(AssertionError):
        sq.rolling_backtest(panel, {'leaky': (['declined'], lambda: None)}, first_test=1)


def test_date_bootstrap_brackets_the_mean_and_is_seeded():
    cells = pd.DataFrame([{'decision_date': d, 'fold': f, 'policy': p, 'P@50': v}
                          for d in range(6) for f in range(3)
                          for p, v in [('a', 0.6 + 0.01 * d), ('b', 0.5)]])
    mean, lo, hi = sq.date_bootstrap(cells, 'a', 'b')
    assert lo <= mean <= hi and mean == pytest.approx(0.125)
    assert sq.date_bootstrap(cells, 'a', 'b') == (mean, lo, hi)


def test_psi_is_zero_for_same_distribution_and_grows_with_shift():
    rng = np.random.RandomState(0)
    x = rng.normal(size=5000)
    assert sq.psi(x, x) == pytest.approx(0, abs=1e-9)
    assert sq.psi(x, x + 1) > 0.25 > sq.psi(x, rng.normal(size=5000))


def test_cache_round_trip_is_identical(tmp_path):
    rows = steady_page('a') + steady_page('b', client='client_2', out=20)
    con = duckdb.connect()
    con.register('daily_df', pd.DataFrame(rows))
    con.execute('CREATE VIEW daily AS SELECT * FROM daily_df')
    con.execute("CREATE TABLE dim_content AS SELECT DISTINCT content_hash_id, DATE '2025-06-01' AS content_created_date FROM daily_df")
    fresh = sq.build_frames(con, [T], cache_dir=tmp_path)
    cached = sq.build_frames(con, [T], cache_dir=tmp_path)
    pd.testing.assert_frame_equal(fresh, cached)
