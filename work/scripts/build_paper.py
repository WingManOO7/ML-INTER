"""Render the deployed research paper, docs/index.html, from the committed metric receipts.

Run from the repo root (the capstone notebook runs it in its last code cell):
    python work/scripts/build_paper.py

Every number on the page is read from work/outputs/*.json, so the paper cannot drift from the
notebooks that produced them. Charts are inline SVG (no libraries, no network), themed for light and
dark, with a hover tooltip on every mark and a table view under every chart.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work' / 'outputs'
REPO = 'https://github.com/WingManOO7/ML-INTER'
BLOB = f'{REPO}/blob/main'

cap = json.loads((OUT / 'capstone_receipts.json').read_text(encoding='utf-8'))
w06 = json.loads((OUT / 'w06_validation_receipts.json').read_text(encoding='utf-8'))
w07 = json.loads((OUT / 'w07_playbook_receipts.json').read_text(encoding='utf-8'))
sealed_file = json.loads((OUT / 'sealed_june_receipts.json').read_text(encoding='utf-8'))

bt, sealed, live, data = cap['backtest'], cap['sealed'], cap['live'], cap['data']
R = bt['results']
RULE, POL, SORT, PROB = 'ML-07 rule (baseline)', 'LR x lost impressions (policy)', 'slide sort (no training)', 'LR probability alone'
boot = bt['bootstrap']
ST = sealed['table']
esc = html.escape


# numpy rounding, the same rule pandas uses for the notebook tables, so the page and the notebooks print
# identical digits (Python's format() would turn 0.6395 into 0.639, pandas into 0.640)
def f2(x):
    return f'{np.round(float(x), 2):.2f}'


def f3(x):
    return f'{np.round(float(x), 3):.3f}'


def pct(x, d=0):
    return f'{100 * float(x):.{d}f}%'


def signed(x, d=3):
    return f'{float(x):+.{d}f}'


def n(x):
    return f'{int(float(x)):,}'


# ------------------------------------------------------------------ chart primitives (inline SVG)
W, H = 560, 330
M = dict(l=56, r=20, t=16, b=46)


def scale(v, d0, d1, r0, r1):
    return r0 + (float(v) - d0) / (d1 - d0) * (r1 - r0)


def axis_y(ticks, y0, y1, fmt=lambda t: f'{t:.1f}', x0=M['l'], x1=W - M['r'], label=''):
    out = []
    for t in ticks:
        y = scale(t, y0, y1, H - M['b'], M['t'])
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt(t)}</text>')
    if label:
        out.append(f'<text class="axis-label" transform="translate(14 {(H - M["b"] + M["t"]) / 2:.0f}) rotate(-90)" text-anchor="middle">{esc(label)}</text>')
    return ''.join(out)


def legend(items):
    """items: (css class, label, kind) where kind is 'dot' | 'line' | 'bar'."""
    parts = []
    for cls, label, kind in items:
        if kind == 'line':
            key = f'<svg width="22" height="10" aria-hidden="true"><line class="{cls}" x1="1" x2="21" y1="5" y2="5"/></svg>'
        elif kind == 'bar':
            key = f'<svg width="12" height="12" aria-hidden="true"><rect class="{cls}" width="12" height="12" rx="3"/></svg>'
        else:
            key = f'<svg width="12" height="12" aria-hidden="true"><circle class="{cls}" cx="6" cy="6" r="5"/></svg>'
        parts.append(f'<span class="key">{key}{esc(label)}</span>')
    return f'<div class="legend">{"".join(parts)}</div>'


def table(headers, rows, caption=None, num_from=1):
    head = ''.join(f'<th{" class=num" if i >= num_from else ""}>{esc(h)}</th>' for i, h in enumerate(headers))
    body = ''.join('<tr>' + ''.join(f'<td{" class=num" if i >= num_from else ""}>{c}</td>' for i, c in enumerate(r)) + '</tr>'
                   for r in rows)
    # the caption sits outside the scroll container so it never scrolls away with a wide table
    cap_html = f'<p class="tcap">{caption}</p>' if caption else ''
    return f'{cap_html}<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def table_view(headers, rows):
    return f'<details class="tv"><summary>Table view</summary>{table(headers, rows)}</details>'


def bar_right(x0, x1, y, h, r=4):
    """Horizontal bar, square at the baseline (x0), 4px rounded data end (x1)."""
    r = min(r, (x1 - x0) / 2, h / 2)
    return (f'M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} V{y + h - r:.1f} '
            f'Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z')


def bar_up(x, y0, y1, w, r=4):
    """Vertical column from baseline y0 up to y1 with a 4px rounded cap."""
    r = min(r, (y0 - y1) / 2, w / 2)
    return (f'M{x:.1f},{y0:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + w - r:.1f} '
            f'Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f} V{y0:.1f} Z')


# ------------------------------------------------------------------ Figure 1: the precision-visibility frontier
def fig_frontier():
    x0, x1, y0, y1 = 0.0, 0.7, 0.3, 0.9
    style = {POL: 's1', RULE: 's2', SORT: 's3'}
    short = {POL: 'Policy: P × lost impressions', RULE: 'Frozen rule (ML-07)', SORT: 'Slide sort (no training)',
             PROB: 'LR probability alone', 'LR x impressions': 'LR × total impressions',
             'HGB (8 features) x lost impressions': 'HGB × lost', 'LR (8 features) x lost impressions': 'LR-8 × lost',
             'random order': 'Random order'}
    # label offsets hand-placed so labels never collide
    offset = {POL: (10, -10, 'start'), RULE: (10, 18, 'start'), SORT: (10, 4, 'start'), PROB: (10, 16, 'start'),
              'LR x impressions': (-10, -10, 'end'), 'HGB (8 features) x lost impressions': (-10, 20, 'end'),
              'LR (8 features) x lost impressions': (-10, -12, 'end'), 'random order': (10, 4, 'start')}
    body = [axis_y([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], y0, y1, label='precision@50')]
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        x = scale(t, x0, x1, M['l'], W - M['r'])
        body.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{M["t"]}" y2="{H - M["b"]}"/>'
                    f'<text class="tick" x="{x:.1f}" y="{H - M["b"] + 18}" text-anchor="middle">{t:.1f}</text>')
    body.append(f'<text class="axis-label" x="{(M["l"] + W - M["r"]) / 2:.0f}" y="{H - 6}" text-anchor="middle">'
                f'visibility@50 (share of the achievable lost impressions the 50 picks cover)</text>')
    # the HGB and 8-feature variants sit within 0.02 of the policy: plotted they only hide it, so they
    # live in the table view and Table 1 instead
    variants = {'HGB (8 features) x lost impressions', 'LR (8 features) x lost impressions'}
    order = [k for k in R if k not in style and k not in variants] + [SORT, RULE, POL]
    rows = [[esc(short[k]), f3(R[k]['P@50']), f3(R[k]['visibility@50']), f2(R[k]['lift']), f3(R[k]['AUC'])] for k in variants]
    for name in order:
        v = R[name]
        px, py = scale(v['visibility@50'], x0, x1, M['l'], W - M['r']), scale(max(v['P@50'], y0), y0, y1, H - M['b'], M['t'])
        cls = style.get(name, 'muted')
        tip = f"{short[name]}: precision@50 {f3(v['P@50'])} (SD {f2(v['P@50 sd'])}), visibility@50 {f3(v['visibility@50'])} (SD {f2(v['visibility sd'])})"
        dx, dy, anchor = offset[name]
        body.append(f'<g class="mark"><title>{esc(tip)}</title><circle class="hit" cx="{px:.1f}" cy="{py:.1f}" r="12"/>'
                    f'<circle class="dot {cls}" cx="{px:.1f}" cy="{py:.1f}" r="{7 if name == POL else 5.5}"/></g>'
                    f'<text class="dlabel{" strong" if name == POL else ""}" x="{px + dx:.1f}" y="{py + dy:.1f}" text-anchor="{anchor}">{esc(short[name])}</text>')
        rows.append([esc(short[name]), f3(v['P@50']), f3(v['visibility@50']), f2(v['lift']), f3(v['AUC'])])
    svg = f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Precision versus visibility for each ranking policy">{"".join(body)}</svg>'
    return (legend([('dot s1', 'Policy (P × lost impressions)', 'dot'), ('dot s2', 'Frozen rule', 'dot'),
                    ('dot s3', 'Slide sort', 'dot'), ('dot muted', 'Other rankings', 'dot')]) + svg +
            table_view(['ranking', 'precision@50', 'visibility@50', 'lift', 'AUC'], rows[::-1]))


# ------------------------------------------------------------------ Figure 2: precision by decision date
def fig_dates():
    dev = bt['per_date']
    sd = sealed['by_date_P@50']
    dates = [r['decision_date'] for r in dev] + list(sd)
    rule = [r['rule P@50'] for r in dev] + [sd[d][RULE] for d in sd]
    pol = [r['policy P@50'] for r in dev] + [sd[d][POL] for d in sd]
    base = [r['base rate'] for r in dev] + [sealed['base_rate_by_date'][d] for d in sd]
    y0, y1 = 0.0, 1.0
    step = (W - M['l'] - M['r']) / len(dates)
    xs = [M['l'] + step * (i + 0.5) for i in range(len(dates))]
    sx = xs[len(dev)] - step / 2
    body = [f'<rect class="sealed-band" x="{sx:.1f}" y="{M["t"]}" width="{W - M["r"] - sx:.1f}" height="{H - M["b"] - M["t"]}"/>',
            f'<text class="band-label" x="{(sx + W - M["r"]) / 2:.1f}" y="{M["t"] + 14}" text-anchor="middle">sealed June</text>',
            axis_y([0, 0.2, 0.4, 0.6, 0.8, 1.0], y0, y1, label='precision@50 (mean of 5 client-fold queues)')]
    for x, d in zip(xs, dates):
        body.append(f'<text class="tick" x="{x:.1f}" y="{H - M["b"] + 18}" text-anchor="middle">{d[5:7]}/{d[8:]}</text>')
    body.append(f'<text class="axis-label" x="{(M["l"] + W - M["r"]) / 2:.0f}" y="{H - 6}" text-anchor="middle">decision date (2026, month/day)</text>')
    rows = []
    for series, cls, label in [(base, 'base', 'base rate'), (rule, 's2', 'frozen rule'), (pol, 's1', 'policy')]:
        pts = [(x, scale(v, y0, y1, H - M['b'], M['t'])) for x, v in zip(xs, series)]
        body.append(f'<polyline class="line {cls}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        for (x, y), d, v in zip(pts, dates, series):
            body.append(f'<g class="mark"><title>{esc(f"{d} · {label}: {float(v):.3f}")}</title>'
                        f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="10"/><circle class="dot {cls}" cx="{x:.1f}" cy="{y:.1f}" r="{3.5 if cls == "base" else 4.5}"/></g>')
    for i, d in enumerate(dates):
        rows.append([d + (' (sealed)' if i >= len(dev) else ''), f3(base[i]), f3(rule[i]), f3(pol[i])])
    svg = f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Precision at 50 by decision date for the policy, the rule and the base rate">{"".join(body)}</svg>'
    return (legend([('line s1', 'Policy', 'line'), ('line s2', 'Frozen rule', 'line'), ('line base', 'Base rate', 'line')]) + svg +
            table_view(['decision date', 'base rate', 'rule P@50', 'policy P@50'], rows))


# ------------------------------------------------------------------ Figure 3: the boundary experiment
def fig_boundary():
    splits = ['random rows', 'client-grouped', 'rolling-origin + client-grouped']
    models = [('Week-5 model: LR, 5 features', 's1', 'Week-5 model (LR, 5 features)'),
              ('HGB + client code + date code', 's2', 'HGB + client code + date code')]
    h, gap, group_gap = 18, 6, 26
    top = M['t'] + 6
    x0, x1 = 0.0, 1.0
    left, right = 210, W - 56
    plot_bottom = top + len(splits) * (2 * h + gap) + (len(splits) - 1) * group_gap
    Hb = plot_bottom + 44                      # this chart is only as tall as its bars
    body, rows, y = [], [], top
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        x = scale(t, x0, x1, left, right)
        body.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" y2="{plot_bottom + 4}"/>'
                    f'<text class="tick" x="{x:.1f}" y="{plot_bottom + 20}" text-anchor="middle">{t:.2f}</text>')
    for s in splits:
        body.append(f'<text class="cat" x="{left - 10}" y="{y + h + gap / 2 + 4:.1f}" text-anchor="end">{esc(s)}</text>')
        for key, cls, label in models:
            v = w06['boundary'][f'{key} | {s}']
            xe = scale(v['P@50 mean'], x0, x1, left, right)
            tip = f"{label} · {s}: precision@50 {v['P@50 mean']:.3f} ± {v['P@50 std']:.3f}, AUC {v['AUC mean']:.3f}"
            body.append(f'<g class="mark"><title>{esc(tip)}</title><path class="bar {cls}" d="{bar_right(left, xe, y, h)}"/></g>'
                        f'<text class="value" x="{xe + 6:.1f}" y="{y + h - 4:.1f}">{v["P@50 mean"]:.2f}</text>')
            rows.append([esc(label), esc(s), f3(v['P@50 mean']), f3(v['P@50 std']), f3(v['AUC mean'])])
            y += h + gap
        y += group_gap - gap
    body.append(f'<text class="axis-label" x="{(left + right) / 2:.0f}" y="{Hb - 6}" text-anchor="middle">precision@50 (mean over test queues)</text>')
    svg = f'<svg class="chart" viewBox="0 0 {W} {Hb}" role="img" aria-label="Precision at 50 under three validation designs">{"".join(body)}</svg>'
    return (legend([('bar s1', 'Week-5 model (LR, 5 features)', 'bar'), ('bar s2', 'HGB + client code + date code', 'bar')]) + svg +
            table_view(['model', 'split', 'P@50 mean', 'P@50 SD', 'AUC'], rows))


# ------------------------------------------------------------------ Figure 4: the two signal checks
def fig_signals():
    panels = [('slide', 'momentum_pre', 'Week 2 ÷ week 1 before T (CONFIRMED)'),
              ('age', 'age_days_at_T', 'Page age at T, days (no trend)')]
    short = {'collapsed (<0.5x)': '<0.5×', 'down 20-50%': '0.5–0.8×', 'about flat': '≈1×', 'up 25-100%': '1.25–2×',
             'up >2x': '>2×', '<30 d': '<30', '30-89 d': '30–89', '90-179 d': '90–179', '180-364 d': '180–364', '365+ d': '365+'}
    pw, gapw = (W - 40) / 2, 40
    base = sum(r['rate'] * r['n'] for r in cap['signals']['slide']) / sum(r['n'] for r in cap['signals']['slide'])
    body, rows = [], []
    for p, (key, col, title) in enumerate(panels):
        ox = p * (pw + gapw)
        left, right, top, bottom = ox + 34, ox + pw, M['t'] + 22, H - M['b'] - 10
        body.append(f'<text class="panel-title" x="{ox + 34}" y="{M["t"] + 6}">{esc(title)}</text>')
        for t in [0, 0.2, 0.4, 0.6, 0.8]:
            yy = scale(t, 0, 0.8, bottom, top)
            body.append(f'<line class="grid" x1="{left}" x2="{right}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
                        f'<text class="tick" x="{left - 6}" y="{yy + 4:.1f}" text-anchor="end">{t:.1f}</text>')
        recs = cap['signals'][key]
        slot = (right - left) / len(recs)
        bw = min(24, slot * 0.55)
        for i, r in enumerate(recs):
            cx = left + slot * (i + 0.5)
            ytop = scale(r['rate'], 0, 0.8, bottom, top)
            tip = f"{r[col]}: decline rate {r['rate']:.3f} (n = {int(r['n']):,})"
            body.append(f'<g class="mark"><title>{esc(tip)}</title><path class="bar s1" d="{bar_up(cx - bw / 2, bottom, ytop, bw)}"/></g>'
                        f'<text class="value" x="{cx:.1f}" y="{ytop - 5:.1f}" text-anchor="middle">{r["rate"]:.2f}</text>'
                        f'<text class="tick small" x="{cx:.1f}" y="{bottom + 16}" text-anchor="middle">{esc(short[r[col]])}</text>')
            rows.append([esc(title.split(' (')[0]), esc(r[col]), n(r['n']), f3(r['rate'])])
        yb = scale(base, 0, 0.8, bottom, top)
        body.append(f'<line class="refline" x1="{left}" x2="{right}" y1="{yb:.1f}" y2="{yb:.1f}"/>')
    body.append(f'<text class="axis-label" x="{W / 2:.0f}" y="{H - 6}" text-anchor="middle">share of pages that declined over the next 14 days (hairline: overall rate {base:.2f})</text>')
    svg = f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Decline rate by slide bucket and by page age">{"".join(body)}</svg>'
    return svg + table_view(['signal', 'bucket', 'pages', 'decline rate'], rows)


# ------------------------------------------------------------------ text helpers bound to the receipts
d_p, lo_p, hi_p = boot['P@50']
d_v, lo_v, hi_v = boot['visibility@50']
d_a, lo_a, hi_a = boot['AUC']
base_rates = [r['base_rate_robust'] for r in data['per_date']]
dev_dates = cap['meta']['dev_dates']
bnd = w06['boundary']
wk5 = {s: bnd[f'Week-5 model: LR, 5 features | {s}'] for s in ['random rows', 'client-grouped', 'rolling-origin + client-grouped']}
ids = {s: bnd[f'HGB + client code + date code | {s}'] for s in ['random rows', 'client-grouped', 'rolling-origin + client-grouped']}
calib = {r['p_lr5'].split(' ')[0]: float(r['observed_precision']) for r in w07['confidence_calibration']}
routes = {r['reason_code']: r for r in bt['routes']}
sealed_base = sealed['base_rate_by_date']
sealed_pol, sealed_rule = ST[POL], ST[RULE]
cov = w07['coverage']
mix = {(m['action'], m['reason_code']): m for m in live['mix_top50']}
n_refresh = int(mix.get(('REFRESH_REVIEW', 'SLIDING_WITH_DEMAND'), {}).get('pages', 0))
n_site = int(mix.get(('SITE_LEVEL_CHECK', 'SITE_WIDE_SLIDE'), {}).get('pages', 0))
n_tech = int(mix.get(('TECHNICAL_CHECK', 'COLLAPSE_TO_NEAR_ZERO'), {}).get('pages', 0))
n_mon = sum(int(m['pages']) for k, m in mix.items() if k[0] == 'MONITOR')
replay = w07['trigger_replay']
fired = [r for r in replay if r['fires'] not in ('—', '-', '')]

route_label = {'REFRESH_REVIEW': 'Refresh review', 'SITE_LEVEL_CHECK': 'Site-level check', 'TECHNICAL_CHECK': 'Technical check',
               'MONITOR': 'Monitor', 'NO_ACTION': 'No action'}

main_rows = []
for name, label in [('random order', 'Random order'), (RULE, 'Frozen rule (ML-07 baseline)'), (SORT, 'Slide sort, no training'),
                    (PROB, 'Logistic regression probability alone'), ('LR x impressions', 'LR probability × total impressions'),
                    ('HGB (8 features) x lost impressions', 'Gradient boosting (8 features) × lost impressions'),
                    ('LR (8 features) x lost impressions', 'LR (8 features) × lost impressions'),
                    (POL, '<strong>Policy: LR (5 features) × lost impressions</strong>')]:
    v = R[name]
    main_rows.append([label, f"{f3(v['P@50'])} <span class=sd>± {f2(v['P@50 sd'])}</span>", f2(v['lift']),
                      f"{f3(v['visibility@50'])} <span class=sd>± {f2(v['visibility sd'])}</span>", f3(v['AUC']),
                      f"{int(v['P >= rule'])}/{bt['queues']}", f"{int(v['vis >= rule'])}/{bt['queues']}"])

sealed_rows = []
for name, label in [('random order', 'Random order'), (RULE, 'Frozen rule (ML-07 baseline)'), (SORT, 'Slide sort, no training'),
                    (PROB, 'LR probability alone'), (POL, '<strong>Policy: LR × lost impressions</strong>')]:
    v = ST[name]
    sealed_rows.append([label, f3(v['P@50']), f2(v['lift@50']), f3(v['visibility@50']), f3(v['AUC'])])

route_rows = []
for code, why in [('SLIDING_WITH_DEMAND', 'none of the warnings below'), ('SITE_WIDE_SLIDE', 'the whole site lost ≥ 20% from week 1 to week 2'),
                  ('COLLAPSE_TO_NEAR_ZERO', 'week 2 has < 5% of week-1 impressions'), ('NEW_PAGE_SETTLING', 'page younger than 30 days'),
                  ('SPIKE_FADE', 'one day held > 50% of the feature window')]:
    r = routes[code]
    route_rows.append([route_label[r['action']], f'<code>{code}</code>', esc(why), pct(r['share_of_picks']), f2(r['precision']), pct(r['drop_share'])])

live_rows = []
for r in live['top10']:
    live_rows.append([r['rank'], route_label[r['action']], f"<code>{r['reason_code']}</code>", r['confidence'], f2(r['p_decline']),
                      n(r['lost_weekly_impressions']), n(r['imp_pre']), f"{float(r['pos_pre']):.1f}", r['age_days_at_T']])

replay_rows = [[r['decision_date'], f2(r['T1 max PSI']), f2(r['T2 site share top50']), signed(r['T3 base-rate move'], 2),
                'yes' if r['T4 model < rule (2 in a row)'] == 'True' else 'no', r['fires'] if r['fires'] not in ('—', '-', '') else '—']
               for r in replay]

data_rows = [[r['decision_date'], n(r['pages']), r['clients'], f3(r['base_rate_robust']), f3(r['base_rate_contract'])] for r in data['per_date']]
v = cap['meta']['versions']

# ------------------------------------------------------------------ the page
CSS = r'''
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--band:rgba(42,120,214,.07);--link:#1c5cab;--code:#f0efec;--good:#006300}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--band:rgba(57,135,229,.12);--link:#86b6ef;--code:#262624;--good:#0ca30c}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--band:rgba(57,135,229,.12);--link:#86b6ef;--code:#262624;--good:#0ca30c}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font:17px/1.62 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:46rem;margin:0 auto;padding:2.5rem 16px 4rem}
header .eyebrow{font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin:0 0 .6rem}
h1{font-size:clamp(1.6rem,4.6vw,2.3rem);line-height:1.18;margin:0 0 .8rem;letter-spacing:-.01em}
.subtitle{font-size:1.08rem;color:var(--ink2);margin:0 0 1rem}
.byline{font-size:.92rem;color:var(--ink2);margin:0 0 1.6rem}
h2{font-size:1.35rem;margin:2.8rem 0 .7rem;padding-top:.6rem;border-top:1px solid var(--grid)}
h3{font-size:1.06rem;margin:1.8rem 0 .5rem}
p,li{color:var(--ink)}a{color:var(--link)}
code{font:.86em ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--code);padding:.08em .32em;border-radius:4px;overflow-wrap:anywhere}
pre{overflow-x:auto;max-width:100%;background:var(--code);padding:.8rem 1rem;border-radius:8px;font-size:.84rem;line-height:1.5}
pre code{background:none;padding:0;overflow-wrap:normal;white-space:pre}
td code{white-space:nowrap;overflow-wrap:normal}
p,li,figcaption{overflow-wrap:break-word}
.abstract{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.1rem 1.25rem;margin:0 0 1.4rem}
.abstract h2{border:0;margin:0 0 .4rem;padding:0;font-size:1rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(10.5rem,1fr));gap:.75rem;margin:0 0 1.6rem}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.85rem 1rem}
.tile .label{font-size:.82rem;color:var(--ink2)}.tile .value{font-size:1.7rem;font-weight:600;line-height:1.2;margin:.15rem 0}
.tile .note{font-size:.8rem;color:var(--muted)}
.claims li{margin:.35rem 0}
figure{margin:1.6rem 0;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.9rem .9rem .7rem}
figure svg.chart{display:block;width:100%;height:auto}
.legend svg{display:inline-block;flex:none}
figcaption{font-size:.9rem;color:var(--ink2);margin-top:.6rem}
figcaption b{color:var(--ink)}
.legend{display:flex;flex-wrap:wrap;gap:.35rem 1rem;font-size:.82rem;color:var(--ink2);margin:0 0 .4rem}
.key{display:inline-flex;align-items:center;gap:.35rem}
svg text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.tick{font-size:13px;fill:var(--muted);font-variant-numeric:tabular-nums}.tick.small{font-size:12px}
.axis-label{font-size:13px;fill:var(--ink2)}.cat{font-size:13.5px;fill:var(--ink2)}
.panel-title{font-size:13.5px;fill:var(--ink);font-weight:600}
.dlabel{font-size:13px;fill:var(--ink2)}.dlabel.strong{fill:var(--ink);font-weight:600}
.value{font-size:12.5px;fill:var(--ink2);font-variant-numeric:tabular-nums}
.band-label{font-size:12px;fill:var(--ink2)}
.value,.dlabel,.band-label{paint-order:stroke;stroke:var(--surface);stroke-width:3px;stroke-linejoin:round}
.grid{stroke:var(--grid);stroke-width:1}.refline{stroke:var(--muted);stroke-width:1.5}
.sealed-band{fill:var(--band)}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.s1{stroke:var(--s1)}.line.s2{stroke:var(--s2)}.line.base{stroke:var(--muted);stroke-width:1.5}
.dot{stroke:var(--surface);stroke-width:2}.dot.s1{fill:var(--s1)}.dot.s2{fill:var(--s2)}.dot.s3{fill:var(--s3)}.dot.muted,.dot.base{fill:var(--muted)}
.bar.s1,rect.bar.s1{fill:var(--s1)}.bar.s2,rect.bar.s2{fill:var(--s2)}
.legend .line{stroke-width:2.5}.legend circle{stroke:none}.legend rect.s1{fill:var(--s1)}.legend rect.s2{fill:var(--s2)}
.legend circle.s1{fill:var(--s1)}.legend circle.s2{fill:var(--s2)}.legend circle.s3{fill:var(--s3)}.legend circle.muted{fill:var(--muted)}
.hit{fill:transparent}.mark{cursor:default}.mark:hover .dot,.mark:hover .bar{opacity:.8}
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:.8rem 0}
table{border-collapse:collapse;width:100%;font-size:.88rem}
.tcap{color:var(--ink2);font-size:.86rem;margin:1.1rem 0 -.35rem}
th,td{padding:.42rem .55rem;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--ink2);white-space:nowrap}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.sd{color:var(--muted);font-size:.85em}
details.tv{margin-top:.5rem;font-size:.86rem}details.tv summary{cursor:pointer;color:var(--ink2)}
.callout{border-left:3px solid var(--s1);padding:.2rem 0 .2rem 1rem;margin:1.2rem 0;color:var(--ink)}
.small{font-size:.9rem;color:var(--ink2)}
footer{margin-top:3rem;font-size:.86rem;color:var(--ink2)}
ol.refs li{margin:.3rem 0;font-size:.92rem}
'''

TITLE = 'Which Pages Should an Editor Review First? Ranking Search-Visibility Drops by Risk and Stake'
DESC = ('A rolling-origin, client-grouped evaluation of a decision-support review queue on 79 million rows of '
        'pseudonymized Search Console data from the FlyRank ML Internship.')

abstract = (
    f'Content teams can review only about fifty pages every two weeks, out of tens of thousands that earn search impressions, '
    f'so the useful question is which pages losing visibility deserve that attention first. '
    f'On a pseudonymized warehouse of 78.8 million daily Search Console rows (FlyRank ML Internship release), we score '
    f'{n(data["rows"])} page-dates from {data["clients"]} sites at ten biweekly decision dates (January–May 2026) and seal June 2026 for one final test. '
    f'We compare a transparent rule (rank by impressions lost last week) with learned rankers under a rolling-origin, client-grouped design '
    f'(train only on earlier dates and other sites) and score every 50-page queue on precision and on the share of achievable lost visibility it covers. '
    f'A learned probability alone reaches the highest precision ({f2(R[PROB]["P@50"])}) but covers only {pct(R[PROB]["visibility@50"])} of the visibility at stake, '
    f'while weighting the rule\'s lost impressions by that probability raises precision@50 from {f2(R[RULE]["P@50"])} to {f2(R[POL]["P@50"])} '
    f'({signed(d_p)}, 95% interval {signed(lo_p)} to {signed(hi_p)}) at equal visibility ({f2(R[RULE]["visibility@50"])} vs {f2(R[POL]["visibility@50"])}); '
    f'in sealed June the precision lead held ({f2(sealed_pol["P@50"])} vs {f2(sealed_rule["P@50"])}, higher in {sealed["policy_P_ge_rule"]} of {sealed["fold_queues"]} queues) '
    f'while visibility came in slightly below the rule ({f2(sealed_pol["visibility@50"])} vs {f2(sealed_rule["visibility@50"])}). '
    f'The output is a decision-support review queue with action routes, reason codes, calibrated confidence and drift triggers: it ranks human attention and makes no causal claim about refresh impact.'
)

page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(TITLE)}</title>
<meta name="description" content="{esc(DESC)}">
<meta property="og:title" content="{esc(TITLE)}">
<meta property="og:description" content="{esc(DESC)}">
<style>{CSS}</style>
</head>
<body>
<main>
<header>
<p class="eyebrow">Research paper · FlyRank ML Internship capstone · Lane 2: refresh / content opportunity scoring</p>
<h1>{esc(TITLE)}</h1>
<p class="subtitle">A rolling-origin, client-grouped evaluation of a decision-support review queue, with one sealed month.</p>
<p class="byline">Muhammad Mohid Shafiq · October 2026 · <a href="{REPO}">Repository</a> · <a href="{BLOB}/work/notebooks/capstone.ipynb">Capstone notebook</a></p>
</header>

<section class="abstract" aria-labelledby="abstract">
<h2 id="abstract">Abstract</h2>
<p>{abstract}</p>
</section>

<div class="tiles">
<div class="tile"><div class="label">Precision@50 vs the frozen rule</div><div class="value">{signed(d_p)}</div><div class="note">95% interval {signed(lo_p)} to {signed(hi_p)}; 40 queues</div></div>
<div class="tile"><div class="label">Visibility@50 vs the rule</div><div class="value">{signed(d_v)}</div><div class="note">95% interval {signed(lo_v)} to {signed(hi_v)}: no measurable loss</div></div>
<div class="tile"><div class="label">Sealed June, policy vs rule</div><div class="value">{f2(sealed_pol["P@50"])} vs {f2(sealed_rule["P@50"])}</div><div class="note">precision@50; higher in {sealed["policy_P_ge_rule"]} of {sealed["fold_queues"]} queues</div></div>
</div>

<h2 id="intro">1 · Introduction and problem statement</h2>
<p>Every two weeks a content editor has time for about fifty page reviews. The portfolio behind this study offers {n(min(r["pages"] for r in data["per_date"]))} to {n(max(r["pages"] for r in data["per_date"]))} pages with real search exposure per cycle, so a review queue spends roughly one slot per thousand pages. <b>The decision this work supports is the order of that queue</b>: which pages, losing search visibility right now, deserve a human look first.</p>
<p>The obvious answer is a rule: review the pages that lost the most impressions last week. It is transparent and decent. The obvious upgrade is a model that predicts which pages will keep declining. Our first model did exactly that: <b>0.96 precision@50</b> on one held-out split. It was also nearly useless, because its fifty picks were tiny pages whose collapse no one would spend an hour on. That tension shapes this paper: a review queue has to be <b>right</b> (precision) and it has to <b>matter</b> (visibility at stake).</p>
<p><b>Claims.</b></p>
<ol class="claims">
<li><b>C1.</b> Weighting the rule's lost impressions by a learned probability that the drop persists beats the frozen rule on precision@50 by {signed(d_p)} (95% interval {signed(lo_p)} to {signed(hi_p)}) without a measurable loss of visibility, across 40 rolling-origin, client-grouped queues (§4.1–4.2).</li>
<li><b>C2.</b> Precision alone picks the wrong system. Probability-only rankers, and a no-training sort on the same signal, reach 0.81–0.83 precision but cover 8–9% of the achievable lost visibility, and extra model complexity buys nothing measurable (§4.1).</li>
<li><b>C3.</b> The evaluation boundary is load-bearing. The Week-5 model falls from 0.96 on one split to {f2(wk5["rolling-origin + client-grouped"]["P@50 mean"])} ± {f2(wk5["rolling-origin + client-grouped"]["P@50 std"])} under the honest design, and a model handed client and date codes gains {signed(ids["random rows"]["P@50 mean"] - ids["rolling-origin + client-grouped"]["P@50 mean"], 2)} precision from a careless split (§4.4).</li>
<li><b>C4.</b> In a sealed month the precision lead replicated, but the visibility guardrail slipped and the label regime moved beyond anything seen in development. Monitoring has to read lift over a moving base rate (§4.5, §6).</li>
</ol>

<h2 id="data">2 · Data</h2>
<p><b>Source.</b> The FlyRank ML Internship warehouse release (build v20260703): 78,835,655 pseudonymized daily rows of Google Search Console and Analytics data from 2025-01-27 to 2026-06-30, plus content and client dimensions. Clients, pages, URLs and queries appear only as salted hashes. We used:</p>
<ul>
<li><code>fact_content_daily_performance</code>, month partitions 2025-12 → 2026-06, Search Console columns only. Every row is filtered with <code>gsc_data_available IS TRUE</code>.</li>
<li><code>dim_content.content_created_date</code>, for page age, and nothing else from that table.</li>
</ul>
<p><b>Decision dates.</b> Every second Monday from <b>2026-01-05 to 2026-05-11</b> for development (ten dates, last outcome window ending 2026-05-24), then <b>2026-05-25 and 2026-06-08</b> as the sealed test, whose outcomes fall in June. A page is scored at a date if it earned ≥ 100 impressions in the 14 days before it, was tracked for all 14 days and has a valid position. That gives {n(data["rows"])} page-dates from {n(data["pages"])} distinct pages and {data["clients"]} clients. {n(data["dropped_incomplete"])} rows (0.8%) without a fully tracked outcome window carry no label.</p>
{table(["decision date", "eligible pages", "clients", "base rate (robust)", "base rate (contract)"], data_rows, "Table A. The development panel. The base rate moves from 0.14 to 0.57: every result is read against its own date.")}
<p><b>Excluded, and why.</b></p>
{table(["excluded", "why"], [
    ["<code>content_updated_date</code>, <code>last_optimized_date</code>, publish and delete flags", "export snapshots: the update date already lies after the decision date for 76–100% of pages"],
    ["GA4, AI-referral and scroll columns", "outside this search question; GA4 availability is FALSE or NULL on most rows"],
    ["keyword-tool fields (volume, CPC, competition)", "no date: cannot show they were known at the decision date"],
    ["clicks", "not needed for a visibility label (CTR is a different lane)"],
    ["every ID as a feature", "pseudonyms for grouping and splitting only; §4.4 shows what an ID feature does"],
    ["pages under 100 impressions", "too small for a 20% change to mean anything"],
], num_from=9)}
<p class="small"><b>Public safety.</b> This page reports aggregates only. No client names, domains, URLs, queries or titles appear anywhere in the release or here, and the queues with pseudonymous IDs stay out of the public repository.</p>

<h2 id="method">3 · Methodology</h2>
<h3>3.1 Unit, label and assumptions</h3>
<p>One observation is <b>one page at one decision date T</b>. Features use only the 14 days before T, split into week 1 and week 2; the label uses only the 14 days from T. The label <code>declined_robust</code> is true when impressions over the outcome window fall more than <b>20%</b> below the feature window, <b>after removing each window's single biggest day</b>. An earlier review found one-day spikes fooling both the rule and a plain sum label. The plain contract label is reported as a robustness check. We assume an editor capacity of 50 pages per cycle and judge pages against their own recent level.</p>
<h3>3.2 Features</h3>
{table(["feature", "definition", "knowable at T because"], [
    ["<code>log_imp_pre</code>", "log impressions, 14 days before T", "every day summed ends before T"],
    ["<code>pos_pre</code>", "impression-weighted position, position 0 skipped", "positions already reported"],
    ["<code>active_days_pre</code>", "days with impressions, of 14", "counted before T"],
    ["<code>momentum_pre</code>", "log(week-2 ÷ week-1 impressions)", "compares the past with the past"],
    ["<code>age_days_at_T</code>", "T minus the page's creation date", "the creation date precedes T and never moves"],
], num_from=9)}
<p>Three further pre-T candidates (four-week trend, peak-day share, site-wide slide) were tested and did not earn a place (§4.1).</p>
<h3>3.3 Baseline and policy</h3>
<p><b>Baseline: the frozen rule.</b> Flag a page when its week-2 impressions are more than 20% below week 1, and rank flagged pages by impressions lost (week 1 − week 2). It was frozen in Week 4, before any model existed. One property matters: if a page simply stays at its week-2 level, its two-week loss is exactly that number, so <b>the rule is a persistence forecast of the loss</b>.</p>
<p><b>Policy.</b> <code>score = P(decline) × lost weekly impressions</code>, where P comes from logistic regression on the five features. The rule supplies the stake and the model supplies the chance that the drop is real and lasting. A probability alone ranks tiny, fully collapsed pages first; the multiplication fixes that.</p>
<p><b>Selection rule, declared in advance.</b> Precision@50 is the primary metric, fixed in Week 2. Visibility@50 (the share of the 50 largest achievable losses the queue actually covers) is a guardrail. It was added after the Week-5 error analysis and before the sealed test. A candidate must beat the rule on precision without losing visibility, and among those that pass the simplest wins.</p>
<h3>3.4 Validation design</h3>
<p><b>Rolling origin × client groups.</b> Clients are split into five fixed folds. For each of the eight test dates from 2026-02-02 to 2026-05-11, every model is trained only on <i>earlier</i> dates and on clients in <i>other</i> folds, then builds one 50-page queue per held-out fold: 8 × 5 = <b>40 queues</b>, each scored against its own base rate. Paired differences carry 95% intervals from a bootstrap that resamples <b>whole dates</b>, because the five folds of one date share a market week. The <b>sealed month</b> was read exactly once, by one notebook cell, after everything was frozen. That cell fingerprints the frozen setup (<code>{esc(sealed["fingerprint"])}</code>) and fails if anything changes afterwards.</p>
<h3>3.5 Leakage controls</h3>
<ul>
<li>Windows are tested in code: 21 unit tests on synthetic data cover window boundaries, <code>IS TRUE</code> handling, both labels, eligibility, the rule's order, split guards and the metric maths.</li>
<li>The shared module refuses any outcome-window column as a feature. With that guard bypassed on purpose, a planted leak lifts AUC from {f3(w06["leakage"]["planted_leak_auc"][0])} to {f3(w06["leakage"]["planted_leak_auc"][1])}.</li>
<li>Shuffled labels fall to chance (AUC {f3(w06["leakage"]["shuffled_auc"])}).</li>
<li>No single feature is suspiciously strong (single-feature AUCs {f2(min(w06["leakage"]["single_feature_auc"].values()))}–{f2(max(w06["leakage"]["single_feature_auc"].values()))}).</li>
<li>Product decision flags do not exist in the release.</li>
</ul>
<h3>3.6 The two signals behind the rule</h3>
<figure>
{fig_signals()}
<figcaption><b>Figure 1. The slide predicts decline; page age does not.</b> Pages that lost more than half their impressions from week 1 to week 2 went on to decline {pct(cap["signals"]["slide"][0]["rate"])} of the time, against {pct(cap["signals"]["slide"][3]["rate"])} for pages that rose. Page age, the signal behind classic refresh flags, shows no trend ({pct(min(r["rate"] for r in cap["signals"]["age"]))}–{pct(max(r["rate"] for r in cap["signals"]["age"]))}), so it stays out of the rule. All ten development dates, {n(data["rows"] - data["dropped_incomplete"])} labelled page-dates.</figcaption>
</figure>

<h2 id="results">4 · Results</h2>
<h3>4.1 Main comparison: 40 rolling-origin, client-grouped queues</h3>
{table(["ranking", "precision@50", "lift", "visibility@50", "AUC", "P ≥ rule", "vis ≥ rule"], main_rows,
       "Table 1. Every policy scored on the same 40 queues (mean ± SD across queues). Lift = precision ÷ that queue's base rate. The last two columns count the queues where a policy matched or beat the rule.")}
<figure>
{fig_frontier()}
<figcaption><b>Figure 2. Precision and visibility trade off; the policy moves the rule up, not sideways.</b> Probability-only rankers sit top-left: often right, rarely about pages that matter. Ranking by probability × total impressions sits bottom-right. The policy keeps the rule's visibility and adds precision. Its gradient-boosting and 8-feature variants sit within 0.02 of it on both axes (Table 1), so they are left off the plot. Across queues the SD is about 0.15 on both axes, so read the positions, not the third decimal.</figcaption>
</figure>
<h3>4.2 Uncertainty</h3>
<p>Policy minus rule, paired by queue, with 95% intervals from the whole-date bootstrap:</p>
<ul>
<li>precision@50: <b>{signed(d_p)}</b> [{signed(lo_p)}, {signed(hi_p)}]</li>
<li>visibility@50: <b>{signed(d_v)}</b> [{signed(lo_v)}, {signed(hi_v)}]</li>
<li>AUC: <b>{signed(d_a)}</b> [{signed(lo_a)}, {signed(hi_a)}]</li>
</ul>
<p>The precision interval excludes zero. The visibility interval straddles it: <b>no measurable loss</b>, which is exactly what the guardrail asks. The policy matches or beats the rule on precision in {int(R[POL]["P >= rule"])} of 40 queues and on all {len(bt["per_date"])} test dates, but on visibility in only {int(R[POL]["vis >= rule"])} of 40. On visibility it is a tie, not a win. On the contract label the picture holds: precision {f3(bt["contract_label"][POL]["P@50"])} vs {f3(bt["contract_label"][RULE]["P@50"])}, visibility {f3(bt["contract_label"][POL]["visibility@50"])} vs {f3(bt["contract_label"][RULE]["visibility@50"])}.</p>
<h3>4.3 Over time</h3>
<figure>
{fig_dates()}
<figcaption><b>Figure 3. The policy stays above the rule on every development date, and both ride a moving base rate.</b> The base rate runs from {f2(min(base_rates))} to {f2(max(base_rates))} in development and reaches {f2(max(sealed_base.values()))} in sealed June, so raw precision has to be read against it. The shaded band is the sealed month.</figcaption>
</figure>
<h3>4.4 The boundary experiment</h3>
<figure>
{fig_boundary()}
<figcaption><b>Figure 4. A careless split rewards memorising sites and weeks.</b> A gradient-boosting model given a client code and a date code reaches {f2(ids["random rows"]["P@50 mean"])} under random rows, {f2(ids["client-grouped"]["P@50 mean"])} with clients grouped and {f2(ids["rolling-origin + client-grouped"]["P@50 mean"])} under the honest design. The Week-5 model, which has no identity features, loses less ({f2(wk5["random rows"]["P@50 mean"])} → {f2(wk5["rolling-origin + client-grouped"]["P@50 mean"])}), but its earlier single-split figure of 0.96 still overstated it.</figcaption>
</figure>
<h3>4.5 The sealed month: June 2026, computed once</h3>
<p>The frozen model, trained on all development data, scored the two sealed dates: {n(sealed["pages"])} pages from {sealed["clients"]} clients, {sealed["new_clients"]} of them never seen in development. The same five client folds give 10 queues.</p>
{table(["ranking", "precision@50", "lift", "visibility@50", "AUC"], sealed_rows,
       f"Table 2. Sealed June (dates 2026-05-25 and 2026-06-08; base rates {f2(sealed_base['2026-05-25'])} and {f2(sealed_base['2026-06-08'])}). First computed {esc(sealed_file['first_computed'])}; every later run re-asserts these numbers.")}
<p><b>The precision lead replicated:</b> {f3(sealed_pol["P@50"])} vs {f3(sealed_rule["P@50"])}, higher or equal in {sealed["policy_P_ge_rule"]} of {sealed["fold_queues"]} queues. A single portfolio-wide queue per date agrees ({f2(sealed["portfolio_queue_P@50"]["2026-05-25"][POL])} vs {f2(sealed["portfolio_queue_P@50"]["2026-05-25"][RULE])} and {f2(sealed["portfolio_queue_P@50"]["2026-06-08"][POL])} vs {f2(sealed["portfolio_queue_P@50"]["2026-06-08"][RULE])}).</p>
<p><b>The visibility guardrail did not hold queue by queue:</b> {f3(sealed_pol["visibility@50"])} vs {f3(sealed_rule["visibility@50"])}, higher or equal in only {sealed["policy_vis_ge_rule"]} of {sealed["fold_queues"]}. The gap is small next to the development spread, but it points toward the rule, so we report it as a miss rather than a tie.</p>
<p>June is also a different regime. A base rate of {f2(sealed_base["2026-06-08"])} sits above the whole development range. When most pages are falling, every ranking looks better in raw precision, so lift is the fairer reading (policy {f2(sealed_pol["lift@50"])}×, rule {f2(sealed_rule["lift@50"])}×).</p>
<h3>4.6 Error analysis, promoted to policy</h3>
<p>Reading the policy's top-50 picks by warning sign showed something unexpected. The "suspicious" picks are not where the model is wrong. Collapses and site-wide slides persist <i>more</i> often than clean slides. What changes is the right <i>action</i>. So the warnings became routes that change the action, not the rank:</p>
{table(["action", "reason code", "when", "share of picks", "precision", "share of lost visibility"], route_rows,
       "Table 3. The policy's top-50 picks across the 40 development queues, by route.", num_from=3)}

<h2 id="limits">5 · Limitations and honest framing</h2>
<ul>
<li><b>A short history.</b> Eight development dates and two sealed dates, with the base rate moving from 0.14 to 0.59. These results describe this portfolio in this period. A longer, multi-season backtest is the most valuable missing experiment.</li>
<li><b>High variance.</b> The policy's precision@50 ranges widely across queues (SD {f2(R[POL]["P@50 sd"])}). The mean advantage is real; any single cycle can go either way.</li>
<li><b>The label partly measures the market.</b> A fixed 20% threshold counts site-wide and portfolio-wide swings as page declines. On 2026-04-27 the whole portfolio lost 18% of its impressions. Routes and triggers mitigate this; a site-relative label is the natural next step.</li>
<li><b>Persistence, not early warning.</b> The strongest signal is a drop already visible at T. On pages asked to fall <i>further</i> below their week-2 level, the slide signal falls to chance. The queue triages visible drops; it does not foresee new ones.</li>
<li><b>The visibility guardrail slipped in June</b> ({f3(sealed_pol["visibility@50"])} vs {f3(sealed_rule["visibility@50"])}).</li>
<li><b>Coverage.</b> Search Console only, and pages above the floor only: about {pct(cov["share of tracked pages that are eligible"])} of tracked pages, holding {pct(cov["share of all impressions held by eligible pages"])} of impressions. Absence from the queue is not evidence of health. There are no click, engagement or conversion outcomes.</li>
<li><b>Data quality.</b> About 1% of sizeable queue rows share an identical Search Console series with another record. A small share of rows report an impossible position below 1. The content dates are export snapshots.</li>
<li><b>Observational, not causal.</b> Every number here is <b>observed</b> or <b>measured</b> on one pseudonymized portfolio. Nothing shows that reviewing or refreshing a flagged page changes its outcome; that would need an experiment, such as randomising which queued pages are reviewed first. The system is <b>decision-support</b>: it orders human attention.</li>
</ul>

<h2 id="recommendations">6 · Ranked recommendations: the action playbook</h2>
<p>Applied to the latest date the release supports, <b>2026-06-22</b> (outcome not yet in the data), the frozen system scored {n(live["pages"])} pages; the {n(live["queue"])} that lost impressions before that date entered the queue. Both pre-release triggers were quiet (feature drift PSI {f2(live["psi_max"])}; site-level share of the top 50 {f2(live["site_share_top50"])}), so the queue would be released. Its top 50 hold {n_refresh} refresh reviews, {n_site} site-level checks, {n_tech} technical checks and {n_mon} pages to monitor.</p>
<ol>
<li><b>Work the refresh reviews in score order.</b> Each is a large page whose loss is likely to persist. The confidence band tells you how likely: in the backtest, high-band picks persisted {pct(calib["high"])} of the time, medium {pct(calib["medium"])}, low {pct(calib["low"])}.</li>
<li><b>Send site-level rows to whoever owns the site, before touching pages.</b> When a whole site slides, a page rewrite is the wrong first move.</li>
<li><b>Run a technical check on collapses</b> (indexing, redirects, canonicals). A page that falls to near zero in a week usually has a visibility problem, not a content problem.</li>
<li><b>Hold spike and new-page rows for one cycle.</b> The largest page in this queue is routed to monitoring: one day carried its whole window.</li>
<li><b>Dedupe before review, and never automate an edit.</b> Every row is a human decision. Nothing here deletes, redirects or rewrites anything on its own.</li>
</ol>
{table(["rank", "action", "reason code", "confidence", "P(decline)", "lost / week", "impressions (14 d)", "position", "age (d)"], live_rows,
       "Table 4. The top 10 of the 2026-06-22 queue (no identifiers). The score is P(decline) × lost weekly impressions; the route sets the action.", num_from=4)}
<h3>Monitoring and retraining</h3>
<p>Refit at every decision date on all labelled history, exactly as the backtest does. Two checks run before release: T1, feature drift against the previous cycle (PSI &gt; 0.25), and T2, more than half the top 50 routed to site-level checks. Two run after labels arrive: T3, a base-rate move &gt; 0.10, which only annotates the cycle; and T4, the model losing to the rule two cycles in a row, which means falling back to the rule. Replayed over development, the stop-level checks fired only at the portfolio event of 2026-04-27 and the cycle after it, and they caught it from week-2 data <i>before</i> any label existed.</p>
{table(["date", "T1 max PSI", "T2 site share", "T3 base move", "T4", "fires"], replay_rows, "Table 5. Trigger replay over the development history.")}

<h2 id="repro">7 · Reproducibility</h2>
<p>Everything regenerates from a fresh clone with one Hugging Face read token in the environment (<code>HF_TOKEN</code>). The first run downloads about 1 GB of month partitions into the Hugging Face cache and builds per-date frames into a gitignored cache. Reruns take minutes.</p>
<pre><code>pip install -r requirements.txt pytest nbconvert ipykernel
python -m pytest work/tests                # 21 tests, no data needed
# execute in order: w03 → w04 → w05 → w06 → w07 → capstone
jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
python work/scripts/build_paper.py          # re-renders this page from the receipts</code></pre>
<p class="small">Environment: Python {v["python"]}, pandas {v["pandas"]}, scikit-learn {v["scikit-learn"]}, DuckDB {v["duckdb"]}; seed 42 for every stochastic step; data: FlyRank internship warehouse release v20260703.</p>
{table(["result", "regenerated by", "receipt"], [
    ["Table 1, Figures 2–3, §4.2", f'<a href="{BLOB}/work/notebooks/capstone.ipynb">capstone.ipynb</a> §4a', f'<a href="{BLOB}/work/outputs/capstone_receipts.json">capstone_receipts.json</a>'],
    ["Table 2 (sealed June)", f'<a href="{BLOB}/work/notebooks/capstone.ipynb">capstone.ipynb</a> §4b, one cell', f'<a href="{BLOB}/work/outputs/sealed_june_receipts.json">sealed_june_receipts.json</a>'],
    ["Figure 4, leakage audit, paper audit", f'<a href="{BLOB}/work/notebooks/w06_validation_audit.ipynb">w06_validation_audit.ipynb</a>', f'<a href="{BLOB}/work/outputs/w06_validation_receipts.json">w06_validation_receipts.json</a>'],
    ["Playbook, calibration, Table 5", f'<a href="{BLOB}/work/notebooks/w07_action_playbook.ipynb">w07_action_playbook.ipynb</a>', f'<a href="{BLOB}/work/outputs/w07_playbook_receipts.json">w07_playbook_receipts.json</a>'],
    ["Data contract, baseline, first model", f'<a href="{BLOB}/work/notebooks/w03_data_contract.ipynb">w03</a> · <a href="{BLOB}/work/notebooks/w04_baseline_score.ipynb">w04</a> · <a href="{BLOB}/work/notebooks/w05_model.ipynb">w05</a>', "w03/w04/w05 receipts in work/outputs"],
    ["Frames, labels, rule, metrics, splits", f'<a href="{BLOB}/work/scripts/search_queue.py">search_queue.py</a>', f'<a href="{BLOB}/work/tests/test_search_queue.py">21 unit tests</a>'],
], num_from=9)}

<h2 id="credit">8 · Acknowledgments and data credit</h2>
<p><b>Built on the FlyRank ML Internship dataset — <a href="https://flyrank.ai">flyrank.ai</a>.</b> The pseudonymized warehouse release made it possible to study real search performance without exposing any client. Thanks to the FlyRank ML Internship team for the data, the lane framing and the evaluation standards this work tries to meet.</p>
<h3>References</h3>
<ol class="refs">
<li>FlyRank (2026). <i>The State of AI-Driven SEO in Numbers</i>, March 2026 data report. Audited in <a href="{BLOB}/work/notebooks/w06_validation_audit.ipynb">w06</a>.</li>
<li>Zinkevich, M. <i>Rules of Machine Learning: Best Practices for ML Engineering.</i> Google. <a href="https://developers.google.com/machine-learning/guides/rules-of-ml">developers.google.com</a></li>
<li>Hyndman, R. J. &amp; Athanasopoulos, G. <i>Forecasting: Principles and Practice</i>, 3rd ed., §5.10 Time series cross-validation. <a href="https://otexts.com/fpp3/tscv.html">otexts.com/fpp3</a></li>
<li>Kaufman, S., Rosset, S., Perlich, C. &amp; Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. <i>ACM TKDD</i> 6(4). <a href="https://doi.org/10.1145/2382577.2382579">doi:10.1145/2382577.2382579</a></li>
<li>Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR</i> 12. Cross-validation with groups: <a href="https://scikit-learn.org/stable/modules/cross_validation.html">scikit-learn.org</a></li>
</ol>
<footer>Every number on this page is generated by <code>work/scripts/build_paper.py</code> from the committed metric receipts. Observed, measured, directional, decision-support.</footer>
</main>
</body>
</html>
'''

out = ROOT / 'docs' / 'index.html'
out.write_text(page, encoding='utf-8')
(ROOT / 'docs' / '.nojekyll').write_text('', encoding='utf-8')
print(f'wrote {out.relative_to(ROOT)} ({len(page):,} bytes)')
