"""Collect loss logs from runs/ and produce the figures / tables used in the report."""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = os.path.join('..', 'results')
os.makedirs(FIG, exist_ok=True)
LRS = ['2.5e-4', '5e-4', '1e-3', '2e-3', '3e-3', '4e-3', '5e-3']
VARIANTS = ['baseline', 'rmsnorm', 'swiglu', 'nope', 'rope', 'gqa']
LABEL = {'baseline': 'Baseline (LayerNorm, GELU, learned PE, MHA)', 'rmsnorm': 'RMSNorm', 'swiglu': 'SwiGLU',
         'nope': 'NoPE', 'rope': 'RoPE', 'gqa': 'GQA (6 q-heads, 3 kv-heads)'}
COLOR = {'baseline': '#222222', 'rmsnorm': '#1f5fa8', 'swiglu': '#c0392b', 'nope': '#8e44ad', 'rope': '#27ae60', 'gqa': '#d35400'}
NAMES = {'baseline': 'Baseline', 'rmsnorm': 'RMSNorm', 'swiglu': 'SwiGLU', 'nope': 'NoPE', 'rope': 'RoPE', 'gqa': 'GQA'}

def load(name):
    p = os.path.join('runs', name, 'loss_log.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    if d['log'][-1]['iter'] < 5000:  # unfinished run
        return None
    it = np.array([r['iter'] for r in d['log']]); tr = np.array([r['train'] for r in d['log']]); va = np.array([r['val'] for r in d['log']])
    d.update(it=it, tr=tr, va=va, best=va.min(), best_it=int(it[va.argmin()]), final=va[-1], tr_at_best=tr[va.argmin()],
             va1000=va[list(it).index(1000)])
    return d

# ---- LR sweep table ----
sweep = {v: {lr: load(f'{v}_lr{lr}') for lr in LRS} for v in VARIANTS}
best_lr = {}
print('min val loss (iter of min) per learning rate')
for v in VARIANTS:
    row = []
    for lr in LRS:
        d = sweep[v][lr]
        row.append(f"{d['best']:.4f} ({d['best_it']})" if d else '   --   ')
    done = {lr: d['best'] for lr, d in sweep[v].items() if d}
    if done:
        best_lr[v] = min(done, key=done.get)
    print(f"{v:9s}", ' | '.join(row), ' best lr', best_lr.get(v))

# ---- seeds (best lr, seeds 1337/1338/1339) ----
seed_stats = {}
for v, lr in best_lr.items():
    runs = [load(f'{v}_lr{lr}')] + [load(f'{v}_lr{lr}_s{s}') for s in (1338, 1339)]
    runs = [r for r in runs if r]
    b = np.array([r['best'] for r in runs]); fnl = np.array([r['final'] for r in runs])
    seed_stats[v] = dict(lr=lr, n=len(runs), best_all=b, best_mean=b.mean(), best_std=b.std(ddof=1) if len(b) > 1 else 0.0,
                         final_mean=fnl.mean(), best_it=np.mean([r['best_it'] for r in runs]),
                         tr_at_best=np.mean([r['tr_at_best'] for r in runs]), va1000=np.mean([r['va1000'] for r in runs]),
                         n_params=runs[0]['n_params'], runs=runs)
print('\nbest-lr results over seeds: mean +- std of min val loss')
for v, s in seed_stats.items():
    print(f"{v:9s} lr={s['lr']} n={s['n']} best={s['best_mean']:.4f}+-{s['best_std']:.4f} "
          f"final={s['final_mean']:.4f} best_it={s['best_it']:.0f} train@best={s['tr_at_best']:.4f} params={s['n_params']/1e6:.3f}M")
json.dump({v: {k: val for k, val in s.items() if k not in ('runs', 'best_all')} for v, s in seed_stats.items()},
          open(os.path.join(FIG, 'summary.json'), 'w'), indent=1, default=float)

# ---- Figure: baseline train / val ----
d = sweep['baseline']['1e-3']
if d:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    tc = np.array(d['train_curve'])
    ax.plot(tc[:, 0], tc[:, 1], color='#9bb7dd', lw=0.6, label='training loss (current mini-batch, dropout on, every 10 iters)')
    ax.plot(d['it'], d['tr'], 'o-', color='#1f5fa8', ms=3.5, label='training loss (eval mode, 200 batches)')
    ax.plot(d['it'], d['va'], 's-', color='#c0392b', ms=3.5, label='validation loss (eval mode, 200 batches)')
    ax.set_ylim(0.5, 4.4); ax.set_xlabel('iteration'); ax.set_ylabel('cross-entropy loss (nats / char)')
    ax.set_title('Baseline nanoGPT, shakespeare_char (default config, lr = 1e-3)'); ax.grid(alpha=.3); ax.legend(fontsize=8.5)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'baseline_loss.png'), dpi=160); plt.close()

# ---- Figures: one per comparison (left: train/val of seed 1337; right: val mean over 3 seeds, band = min..max) ----
panels = {'rmsnorm': ('RMSNorm vs. LayerNorm', ['rmsnorm']), 'swiglu': ('SwiGLU vs. GELU MLP', ['swiglu']),
          'pos': ('Positional encoding: learned vs. NoPE vs. RoPE', ['nope', 'rope']), 'gqa': ('GQA (group size 2) vs. MHA', ['gqa'])}
for key, (title, vs) in panels.items():
    fig, (ax, az) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for v in ['baseline'] + vs:
        if v not in seed_stats:
            continue
        st = seed_stats[v]; r = st['runs'][0]; lab = f"{NAMES[v]} (lr {st['lr']})"
        ax.plot(r['it'], r['va'], '-', color=COLOR[v], lw=1.6, label=lab + ', val')
        ax.plot(r['it'], r['tr'], '--', color=COLOR[v], lw=1.0, label=lab + ', train')
        V = np.array([x['va'] for x in st['runs']])
        az.plot(r['it'], V.mean(0), '-o', ms=2.5, color=COLOR[v], lw=1.5, label=f"{lab}: min {st['best_mean']:.4f}")
        az.fill_between(r['it'], V.min(0), V.max(0), color=COLOR[v], alpha=0.15, lw=0)
    ax.set_ylim(0.55, 2.3); ax.set_xlim(0, 5000); ax.set_xlabel('iteration'); ax.set_ylabel('loss'); ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc='lower left'); ax.set_title(title + ' (seed 1337)', fontsize=9.5)
    az.set_ylim(1.43, 1.68); az.set_xlim(500, 5000); az.set_xlabel('iteration'); az.set_ylabel('validation loss'); az.grid(alpha=.3)
    az.legend(fontsize=7, loc='upper right'); az.set_title('validation loss, mean of 3 seeds (band: min-max)', fontsize=9.5)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f'cmp_{key}.png'), dpi=150); plt.close()

# ---- Figure: all variants at the same (default) learning rate ----
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for v in VARIANTS:
    d = sweep[v]['1e-3']
    if d:
        ax.plot(d['it'], d['va'], '-o', ms=2.5, color=COLOR[v], lw=1.4, label=f"{NAMES[v]}: min {d['best']:.4f} @ {d['best_it']}")
ax.set_ylim(1.44, 1.9); ax.set_xlim(250, 5000); ax.set_xlabel('iteration'); ax.set_ylabel('validation loss'); ax.grid(alpha=.3)
ax.legend(fontsize=7.5, ncol=2); ax.set_title('All variants at the default learning rate 1e-3 (seed 1337)', fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'same_lr.png'), dpi=160); plt.close()

# ---- Figure: LR sweep ----
fig, ax = plt.subplots(figsize=(6.5, 3.8))
x = np.array([float(lr) for lr in LRS])
for v in VARIANTS:
    y = [sweep[v][lr]['best'] if sweep[v][lr] else np.nan for lr in LRS]
    ax.plot(x, y, 'o-', color=COLOR[v], label=v)
ax.set_xscale('log'); ax.set_xticks(x); ax.set_xticklabels(LRS); ax.set_xlabel('peak learning rate')
ax.set_ylabel('min validation loss'); ax.set_title('Learning-rate sweep (seed 1337)'); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'lr_sweep.png'), dpi=160); plt.close()

# ---- HTML tables for the report ----
rows = []
for v in VARIANTS:
    cells = []
    for lr in LRS:
        d = sweep[v][lr]
        if d is None:
            cells.append('<td>&ndash;</td>')
        else:
            cells.append(f"<td>{d['best']:.4f} ({d['best_it']})</td>")
    rows.append(f"<tr><td>{NAMES[v]}</td>{''.join(cells)}</tr>")
lr_html = ('<table class="tbl num"><tr><th>Variant</th>' + ''.join(f'<th>lr = {lr}</th>' for lr in LRS) + '</tr>'
           + ''.join(rows) + '</table>')
open(os.path.join(FIG, 'lr_table.html'), 'w').write(lr_html)

bench = json.load(open(os.path.join(FIG, 'bench.json'))) if os.path.exists(os.path.join(FIG, 'bench.json')) else {}
base = seed_stats.get('baseline')
rows = []
for v in VARIANTS:
    if v not in seed_stats:
        continue
    s = seed_stats[v]; b = bench.get(v, {})
    if v == 'baseline' or base is None:
        delta, tstat = '', ''
    else:
        se = np.sqrt(s['best_all'].var(ddof=1) / len(s['best_all']) + base['best_all'].var(ddof=1) / len(base['best_all']))
        delta, tstat = f"{s['best_mean'] - base['best_mean']:+.4f}", f"{(s['best_mean'] - base['best_mean']) / se:+.1f}"
    rows.append(f"<tr><td>{NAMES[v]}</td><td>{s['lr']}</td><td>{b.get('params', float('nan'))/1e6:.2f}M</td>"
                f"<td>{s['best_mean']:.4f} &plusmn; {s['best_std']:.4f}</td><td>{delta}</td><td>{tstat}</td>"
                f"<td>{sweep[v]['1e-3']['va1000']:.3f}</td><td>{s['best_it']:.0f}</td><td>{s['tr_at_best']:.3f}</td><td>{s['final_mean']:.3f}</td>"
                f"<td>{b.get('ms_per_iter', float('nan')):.1f}</td></tr>")
sum_html = ('<table class="tbl num small"><tr><th>Variant</th><th>best lr</th><th>params</th><th>min val loss<br>(mean &plusmn; std, 3 seeds)</th>'
            '<th>&Delta; vs. baseline</th><th>Welch <i>t</i></th><th>val loss @ iter 1000<br>(all at lr 1e-3)</th><th>iter of min</th><th>train loss at min</th><th>final val loss<br>(iter 5000)</th><th>ms / iter</th></tr>'
            + ''.join(rows) + '</table>')
open(os.path.join(FIG, 'summary_table.html'), 'w').write(sum_html)

# ---- Figure: generalization (val loss vs. train loss, eval mode) at each variant's best LR ----
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for v in VARIANTS:
    if v not in seed_stats:
        continue
    r = seed_stats[v]['runs'][0]
    m = r['it'] >= 250
    ax.plot(r['tr'][m], r['va'][m], 'o-', ms=2.5, lw=1.2, color=COLOR[v], label=f"{NAMES[v]} (lr {seed_stats[v]['lr']})")
ax.invert_xaxis(); ax.set_xlim(2.0, 0.5); ax.set_ylim(1.44, 1.9)
ax.set_xlabel('training loss (eval mode)  →  training progresses'); ax.set_ylabel('validation loss')
ax.set_title('Generalization: validation vs. training loss (seed 1337)', fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'generalization.png'), dpi=160); plt.close()

# ---- Figure: NoPE learning-rate sensitivity ----
fig, ax = plt.subplots(figsize=(6.4, 3.8))
cmap = {'2.5e-4': '#e3d0f0', '5e-4': '#c7a4e0', '1e-3': '#a569d0', '2e-3': '#7d3cb5', '3e-3': '#4a1a7a', '4e-3': '#220a40', '5e-3': '#000000'}
for lr in LRS:
    d = sweep['nope'][lr]
    if d:
        ax.plot(d['it'], d['va'], '-', color=cmap[lr], lw=1.4, label=f'NoPE, lr {lr}')
for v in ['baseline', 'rope']:
    if v in seed_stats:
        r = seed_stats[v]['runs'][0]
        ax.plot(r['it'], r['va'], '--', color=COLOR[v], lw=1.2, label=f"{NAMES[v]}, lr {seed_stats[v]['lr']}")
ax.axhline(2.4875, color='k', lw=0.9, ls=':', label='count-based bigram model (2.49)')
ax.set_ylim(1.4, 2.8); ax.set_xlabel('iteration'); ax.set_ylabel('validation loss'); ax.grid(alpha=.3); ax.legend(fontsize=7.5, ncol=2)
ax.set_title('NoPE: validation loss for different learning rates', fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'nope_lr.png'), dpi=160); plt.close()

# ---- dropout follow-up ----
rows = []
for v in ['baseline', 'swiglu']:
    if v not in seed_stats:
        continue
    lr = seed_stats[v]['lr']
    d3 = load(f'{v}_lr{lr}_do0.3'); d2 = seed_stats[v]['runs'][0]
    if d3:
        rows.append(f"<tr><td>{NAMES[v]}</td><td>{lr}</td><td>{d2['best']:.4f} ({d2['best_it']})</td><td>{d2['final']:.3f}</td>"
                    f"<td>{d3['best']:.4f} ({d3['best_it']})</td><td>{d3['final']:.3f}</td></tr>")
        print(f"dropout follow-up {v}: d0.2 {d2['best']:.4f}@{d2['best_it']} final {d2['final']:.3f} | d0.3 {d3['best']:.4f}@{d3['best_it']} final {d3['final']:.3f}")
if rows:
    open(os.path.join(FIG, 'dropout_table.html'), 'w').write(
        '<table class="tbl num"><tr><th>Variant</th><th>lr</th><th>min val loss, dropout 0.2 (iter)</th><th>final val, dropout 0.2</th>'
        '<th>min val loss, dropout 0.3 (iter)</th><th>final val, dropout 0.3</th></tr>' + ''.join(rows) + '</table>')
