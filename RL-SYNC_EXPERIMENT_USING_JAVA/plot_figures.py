"""
Generate all figures for ML-Sync paper.
All figures match the actual experimental data exactly.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import csv, os, math

DATA  = "./mlsync/data"
FIGS  = "./mlsync/figures"
os.makedirs(FIGS, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.labelsize':    10,
    'axes.titlesize':    11,
    'legend.fontsize':   8.5,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.35,
    'grid.linewidth':    0.6,
    'figure.dpi':        180,
})

COLORS = {
    'RL-Sync':    '#2166ac',
    'FairMutex':  '#d6604d',
    'CLH':        '#4dac26',
    'ExpBackoff': '#8073ac',
}
MARKERS = {'RL-Sync': 'o', 'FairMutex': 's', 'CLH': '^', 'ExpBackoff': 'D'}

# ── Load sweep_results.csv ────────────────────────────────────────────────────
sweep = {}  # method -> {threads: {metric: value}}
with open(f"{DATA}/sweep_results.csv") as f:
    for row in csv.DictReader(f):
        m = row['method']; tc = int(row['threads'])
        if m not in sweep: sweep[m] = {}
        sweep[m][tc] = {k: float(v) for k,v in row.items() if k not in ('method','threads')}

methods = ['RL-Sync', 'FairMutex', 'CLH', 'ExpBackoff']
thread_counts = sorted({tc for m in sweep for tc in sweep[m]})

# ── Load convergence.csv ──────────────────────────────────────────────────────
conv_ep, conv_rl, conv_mx = [], [], []
try:
    with open(f"{DATA}/convergence.csv") as f:
        for row in csv.DictReader(f):
            conv_ep.append(int(row['episode']))
            conv_rl.append(float(row['rl_mean_wait_ms']))
            conv_mx.append(float(row['mutex_baseline_ms']))
except:
    pass

# ── Load stats_rl_vs_mutex.csv ────────────────────────────────────────────────
stats_mx = {}  # threads -> {metric: {mean_A, mean_B, delta_pct, welch_t, cohens_d}}
try:
    with open(f"{DATA}/stats_rl_vs_mutex.csv") as f:
        for row in csv.DictReader(f):
            tc = int(row['threads']); mt = row['metric']
            if tc not in stats_mx: stats_mx[tc] = {}
            stats_mx[tc][mt] = {k: float(v) for k,v in row.items() if k not in ('threads','metric')}
except:
    pass

# ── Load Q-table ──────────────────────────────────────────────────────────────
qt_data = []
try:
    with open(f"{DATA}/qtable.csv") as f:
        for row in csv.DictReader(f):
            qt_data.append({k: (int(v) if k in ('nWaiting','csOccupied','loadBucket')
                               else (v if k=='policy' else float(v)))
                            for k,v in row.items()})
except:
    pass

# =============================================================================
# Figure 1: Mean Wait Time vs Thread Count (scalability sweep)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

# Left: all 4 methods (exclude CLH to avoid huge y-axis domination)
ax = axes[0]
show_methods = ['RL-Sync', 'FairMutex', 'ExpBackoff']
for m in show_methods:
    xs = sorted(sweep[m].keys())
    ys = [sweep[m][tc]['meanWait_ms'] for tc in xs]
    es = [sweep[m][tc]['ci95_ms'] for tc in xs]
    ax.errorbar(xs, ys, yerr=es, label=m, color=COLORS[m],
                marker=MARKERS[m], markersize=5, linewidth=1.5,
                capsize=3, capthick=1.0)
ax.set_xlabel('Number of threads')
ax.set_ylabel('Mean wait time (ms)')
ax.set_title('(a) Wait Time vs Concurrency')
ax.set_xticks(thread_counts)
ax.legend(loc='upper left', framealpha=0.7)

# Right: ZDR
ax2 = axes[1]
for m in show_methods:
    xs = sorted(sweep[m].keys())
    ys = [sweep[m][tc]['zdr_pct'] for tc in xs]
    ax2.plot(xs, ys, label=m, color=COLORS[m],
             marker=MARKERS[m], markersize=5, linewidth=1.5)
ax2.axhline(y=95, color='gray', linestyle='--', linewidth=0.8, alpha=0.7,
            label='95% ZDR target')
ax2.set_xlabel('Number of threads')
ax2.set_ylabel('Zero-delay rate (%)')
ax2.set_title('(b) ZDR vs Concurrency')
ax2.set_xticks(thread_counts)
ax2.set_ylim(-5, 110)
ax2.legend(loc='upper right', framealpha=0.7)

plt.tight_layout(pad=0.8)
plt.savefig(f"{FIGS}/fig1_scalability.pdf", bbox_inches='tight')
plt.savefig(f"{FIGS}/fig1_scalability.png", bbox_inches='tight')
plt.close()
print("Fig 1 done")

# =============================================================================
# Figure 2: RL Training Convergence (6-thread configuration)
# =============================================================================
if conv_ep:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    # Smooth with EMA for display
    def ema(x, a=0.15):
        s = [x[0]]; [s.append(a*v+(1-a)*s[-1]) for v in x[1:]]; return s
    rl_s  = ema(conv_rl, 0.12)
    ax.plot(conv_ep, conv_rl, color=COLORS['RL-Sync'], alpha=0.25, linewidth=0.7)
    ax.plot(conv_ep, rl_s, color=COLORS['RL-Sync'], linewidth=1.8, label='RL-Sync (smoothed)')
    if conv_mx:
        ax.axhline(y=conv_mx[0], color=COLORS['FairMutex'], linewidth=1.4,
                   linestyle='--', label='FairMutex (eval)')
    ax.set_xlabel('Training episode')
    ax.set_ylabel('Mean wait time (ms)')
    ax.set_title('Learning Curve (6 threads)')
    ax.legend(framealpha=0.7)
    plt.tight_layout(pad=0.6)
    plt.savefig(f"{FIGS}/fig2_convergence.pdf", bbox_inches='tight')
    plt.savefig(f"{FIGS}/fig2_convergence.png", bbox_inches='tight')
    plt.close()
    print("Fig 2 done")

# =============================================================================
# Figure 3: Q-Table heatmap (nWaiting x csOccupied, loadBucket=1 slice)
# =============================================================================
if qt_data:
    lb_slice = 1  # moderate load
    nw_vals = sorted(set(r['nWaiting'] for r in qt_data))
    cs_vals = [0, 1]
    # Q(admit) - Q(backoff) = advantage of admitting
    adv = np.zeros((len(nw_vals), len(cs_vals)))
    for r in qt_data:
        if r['loadBucket'] == lb_slice:
            ni = nw_vals.index(r['nWaiting'])
            ci = cs_vals.index(r['csOccupied'])
            adv[ni, ci] = r['Q_admit'] - r['Q_backoff']

    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    vmax = max(abs(adv.max()), abs(adv.min()), 0.01)
    im = ax.imshow(adv, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax,
                   origin='lower')
    ax.set_xticks([0,1]); ax.set_xticklabels(['CS free','CS occupied'])
    ax.set_yticks(range(len(nw_vals))); ax.set_yticklabels(nw_vals)
    ax.set_xlabel('CS status'); ax.set_ylabel('Waiting processes (nWait)')
    ax.set_title('Q-Advantage: Admit − Back-off\n(load bucket 1)')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Q(admit)−Q(back-off)', fontsize=8)
    # Annotate
    for i in range(len(nw_vals)):
        for j in range(len(cs_vals)):
            pol = 'A' if adv[i,j] > 0 else 'B'
            ax.text(j, i, pol, ha='center', va='center',
                    color='black', fontsize=7, fontweight='bold')
    plt.tight_layout(pad=0.6)
    plt.savefig(f"{FIGS}/fig3_qtable_heatmap.pdf", bbox_inches='tight')
    plt.savefig(f"{FIGS}/fig3_qtable_heatmap.png", bbox_inches='tight')
    plt.close()
    print("Fig 3 done")

# =============================================================================
# Figure 4: P99 wait time (tail latency) — important for real-time claim
# =============================================================================
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for m in show_methods:
    xs = sorted(sweep[m].keys())
    ys = [sweep[m][tc]['p99_ms'] for tc in xs]
    ax.plot(xs, ys, label=m, color=COLORS[m],
            marker=MARKERS[m], markersize=5, linewidth=1.5)
ax.set_xlabel('Number of threads')
ax.set_ylabel('P99 wait time (ms)')
ax.set_title('Tail Latency (99th percentile)')
ax.set_xticks(thread_counts)
ax.legend(framealpha=0.7)
plt.tight_layout(pad=0.6)
plt.savefig(f"{FIGS}/fig4_p99.pdf", bbox_inches='tight')
plt.savefig(f"{FIGS}/fig4_p99.png", bbox_inches='tight')
plt.close()
print("Fig 4 done")

# =============================================================================
# Figure 5: Contention rate vs thread count
# =============================================================================
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for m in ['RL-Sync', 'FairMutex', 'ExpBackoff']:
    xs = sorted(sweep[m].keys())
    ys = [sweep[m][tc]['contention_pct'] for tc in xs]
    ax.plot(xs, ys, label=m, color=COLORS[m],
            marker=MARKERS[m], markersize=5, linewidth=1.5)
ax.set_xlabel('Number of threads')
ax.set_ylabel('Contention rate (%)')
ax.set_title('CS Contention Rate')
ax.set_xticks(thread_counts)
ax.legend(framealpha=0.7)
plt.tight_layout(pad=0.6)
plt.savefig(f"{FIGS}/fig5_contention.pdf", bbox_inches='tight')
plt.savefig(f"{FIGS}/fig5_contention.png", bbox_inches='tight')
plt.close()
print("Fig 5 done")

print(f"\nAll figures written to {FIGS}")

# Print data summary for paper writing
print("\n=== Data for paper tables ===")
for tc in thread_counts:
    print(f"\nThreads={tc}:")
    for m in methods:
        if tc in sweep.get(m,{}):
            d = sweep[m][tc]
            print(f"  {m:12s} wait={d['meanWait_ms']:.3f}±{d['ci95_ms']:.3f}  "
                  f"p99={d['p99_ms']:.3f}  zdr={d['zdr_pct']:.1f}%  cont={d['contention_pct']:.1f}%")
