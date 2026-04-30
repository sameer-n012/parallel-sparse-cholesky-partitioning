"""
plot_results.py  —  Sparse Cholesky ordering benchmark visualizations
Usage:  python plot_results.py [results_dir] [output_dir]
Defaults: results_dir=results, output_dir=figures
Produces publication-ready figures as PDFs + PNGs.
"""

import glob
import json
import math
import os
import sys
import warnings
from collections import defaultdict

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import LogNorm

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Color palette — distinct, colorblind-friendly
COLORS = {
    "natural": "#555555",
    "amd": "#2196F3",
    "metis": "#FF9800",
    "nesdis": "#4CAF50",
}
ORDER_LABELS = {
    "natural": "Natural",
    "amd": "AMD",
    "metis": "METIS",
    "nesdis": "NESDIS",
}
ORDERINGS = ["natural", "amd", "metis", "nesdis"]

FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK = 10
FONT_LEGEND = 10

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=180, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


# ── load data ─────────────────────────────────────────────────────────────────

# Load both `results_*.json` and (if present) `results.json`. This avoids silently
# ignoring multithread runs if you have both formats sitting in `results/`.
files = sorted(glob.glob(os.path.join(RESULTS_DIR, "results_*.json")))
single = os.path.join(RESULTS_DIR, "results.json")
if os.path.exists(single):
    files.append(single)
files = sorted(set(files))
assert files, f"No results_*.json or results.json found in '{RESULTS_DIR}'"

records_full, error_records = [], []
for f in files:
    payload = json.load(open(f))
    if isinstance(payload, dict):
        payload = [payload]
    for rec in payload:
        if rec.get("error"):
            error_records.append(rec)
            continue
        # Keep all records here, including different `nthreads`. Any de-dup for
        # the original figures is applied later when building `all_records`.
        records_full.append(rec)

# For the existing figures (which compare orderings per matrix), select one
# record per (group, matrix_name, ordering) as the baseline. Prefer the default
# thread count (typically 4 for this project), then fall back to 1, then to the
# smallest available thread count.
PREFERRED_BASELINE_THREADS = 4
baseline_by_key = {}
for rec in records_full:
    key = (rec.get("group"), rec.get("matrix_name"), rec.get("ordering"))
    nt = int(rec.get("nthreads", 1) or 1)
    prev = baseline_by_key.get(key)
    if prev is None:
        baseline_by_key[key] = rec
        continue
    prev_nt = int(prev.get("nthreads", 1) or 1)

    # Prefer the project default threads (4), then 1, then the smallest.
    def _rank(t: int) -> tuple[int, int]:
        if t == PREFERRED_BASELINE_THREADS:
            return (0, t)
        if t == 1:
            return (1, t)
        return (2, t)

    if _rank(nt) < _rank(prev_nt):
        baseline_by_key[key] = rec

all_records = list(baseline_by_key.values())

# Build lookup: (group, name) -> {ordering -> record}
matrix_index = defaultdict(dict)
for r in all_records:
    matrix_index[(r["group"], r["matrix_name"])][r["ordering"]] = r

full_keys = [k for k, v in matrix_index.items() if set(v.keys()) == set(ORDERINGS)]


def col(field, ordering=None, keys=None):
    """Collect a field from all records (or full_keys subset)."""
    if keys is None:
        src = [r for r in all_records if r["ordering"] == ordering]
    else:
        src = [matrix_index[k][ordering] for k in keys]
    return [r[field] for r in src]


def mcol(metric, ordering, keys=None):
    """Collect a metrics sub-field."""
    if keys is None:
        src = [r for r in all_records if r["ordering"] == ordering]
    else:
        src = [matrix_index[k][ordering] for k in keys]
    return [r["metrics"][metric] for r in src]


print(f"Loaded {len(all_records)} records, {len(full_keys)} fully-covered matrices.")
print(f"Generating figures in '{OUT_DIR}/'...\n")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Fill-in ratio vs natural ordering  (box plots, all matrices)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 1: fill-in ratio box plots")

fig, ax = plt.subplots(figsize=(7, 4.5))

nat_fill = {k: matrix_index[k]["natural"]["metrics"]["fill_in"] for k in full_keys}
data = []
labels_plot = []
orderings_wo_nat = [o for o in ORDERINGS if o != "natural"]
for o in orderings_wo_nat:
    if o == "natural":
        continue
    ratios = [
        matrix_index[k][o]["metrics"]["fill_in"] / max(nat_fill[k], 1)
        for k in full_keys
        if nat_fill[k] > 0
    ]
    data.append(ratios)
    labels_plot.append(ORDER_LABELS[o])

bp = ax.boxplot(
    data,
    patch_artist=True,
    notch=False,
    medianprops=dict(color="white", linewidth=2),
    flierprops=dict(marker="o", markersize=2, alpha=0.4),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)
for patch, o in zip(bp["boxes"], orderings_wo_nat):
    patch.set_facecolor(COLORS[o])
    patch.set_alpha(0.85)
for flier, o in zip(bp["fliers"], orderings_wo_nat):
    flier.set(markerfacecolor=COLORS[o], markeredgecolor=COLORS[o])

ax.axhline(
    1.0, color="gray", linestyle="--", linewidth=1, label="Natural ordering baseline"
)
ax.set_yscale("log")
ax.set_xticklabels(labels_plot, fontsize=FONT_TICK)
ax.set_ylabel("Fill-in ratio relative to natural", fontsize=FONT_LABEL)
ax.set_title(
    "Fill-in Reduction Across Orderings",
    fontsize=FONT_TITLE,
)
ax.legend(fontsize=FONT_LEGEND)
fig.tight_layout()
save(fig, "01_fillin_ratio_boxplot")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Factorization time box plots (log scale)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 2: factorization time box plots")

fig, ax = plt.subplots(figsize=(7, 4.5))

data = []
for o in ORDERINGS:
    vals = [
        r["time_factor_s_median"]
        for r in all_records
        if r["ordering"] == o and r["time_factor_s_median"] > 0
    ]
    data.append(vals)

bp = ax.boxplot(
    data,
    patch_artist=True,
    notch=False,
    medianprops=dict(color="white", linewidth=2),
    flierprops=dict(marker="o", markersize=2, alpha=0.4),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)
for patch, o in zip(bp["boxes"], ORDERINGS):
    patch.set_facecolor(COLORS[o])
    patch.set_alpha(0.85)
for flier, o in zip(bp["fliers"], ORDERINGS):
    flier.set(markerfacecolor=COLORS[o], markeredgecolor=COLORS[o])

ax.set_yscale("log")
ax.set_xticklabels([ORDER_LABELS[o] for o in ORDERINGS], fontsize=FONT_TICK)
ax.set_ylabel("Median time (s)", fontsize=FONT_LABEL)
ax.set_title("Total Analysis + Factorization Time", fontsize=FONT_TITLE)
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
fig.tight_layout()
save(fig, "02_factor_time_boxplot")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Elimination tree height box plots (log scale)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3: etree height box plots")

fig, ax = plt.subplots(figsize=(7, 4.5))

data = []
for o in ORDERINGS:
    vals = [
        r["metrics"]["etree_height"]
        for r in all_records
        if r["ordering"] == o and r["metrics"]["etree_height"] > 0
    ]
    data.append(vals)

bp = ax.boxplot(
    data,
    patch_artist=True,
    notch=False,
    medianprops=dict(color="white", linewidth=2),
    flierprops=dict(marker="o", markersize=2, alpha=0.4),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)
for patch, o in zip(bp["boxes"], ORDERINGS):
    patch.set_facecolor(COLORS[o])
    patch.set_alpha(0.85)
for flier, o in zip(bp["fliers"], ORDERINGS):
    flier.set(markerfacecolor=COLORS[o], markeredgecolor=COLORS[o])

ax.set_yscale("log")
ax.set_xticklabels([ORDER_LABELS[o] for o in ORDERINGS], fontsize=FONT_TICK)
ax.set_ylabel("Elimination tree height", fontsize=FONT_LABEL)
ax.set_title(
    "Elimination Tree Height",
    fontsize=FONT_TITLE,
)
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
fig.tight_layout()
save(fig, "03_etree_height_boxplot")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Fill-in ratio by size bin (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4: fill-in ratio by size bin")

SIZE_BINS = ["tiny\n(<500)", "small\n(500–5k)", "medium\n(5k–50k)", "large\n(>50k)"]
SIZE_BREAKS = [0, 500, 5000, 50000, float("inf")]


def size_idx(n):
    for i, (lo, hi) in enumerate(zip(SIZE_BREAKS, SIZE_BREAKS[1:])):
        if lo <= n < hi:
            return i
    return len(SIZE_BINS) - 1


# Collect fill ratio (nnz_L / nnz_A_lower) per bin per ordering
bin_data = {o: [[] for _ in SIZE_BINS] for o in ORDERINGS}
for r in all_records:
    m = r["metrics"]
    if m["nnz_A_lower"] > 0:
        ratio = m["nnz_L"] / m["nnz_A_lower"]
        bin_data[r["ordering"]][size_idx(m["n"])].append(ratio)

means = {
    o: [np.mean(bin_data[o][i]) if bin_data[o][i] else 0 for i in range(len(SIZE_BINS))]
    for o in ORDERINGS
}

x = np.arange(len(SIZE_BINS))
w = 0.18
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, o in enumerate(ORDERINGS):
    ax.bar(
        x + (i - 1.5) * w,
        means[o],
        w,
        color=COLORS[o],
        label=ORDER_LABELS[o],
        alpha=0.88,
    )

ax.set_xticks(x)
ax.set_xticklabels(SIZE_BINS, fontsize=FONT_TICK)
ax.set_ylabel("Mean fill-in ratio  (nnz(L) / nnz(A_lower))", fontsize=FONT_LABEL)
ax.set_title("Fill-in Ratio Scaling with Matrix Size", fontsize=FONT_TITLE)
ax.legend(fontsize=FONT_LEGEND)
fig.tight_layout()
save(fig, "04_fillin_by_size")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Fill-in vs factorization time scatter (log-log, colored by ordering)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 5: fill-in vs factor time scatter")

fig, ax = plt.subplots(figsize=(7, 5))
for o in ORDERINGS:
    recs = [
        r
        for r in all_records
        if r["ordering"] == o
        and r["metrics"]["fill_in"] > 0
        and r["time_factor_s_median"] > 0
    ]
    xs = [r["metrics"]["fill_in"] for r in recs]
    ys = [r["time_factor_s_median"] for r in recs]
    ax.scatter(xs, ys, s=12, alpha=0.5, color=COLORS[o], label=ORDER_LABELS[o])

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Fill-in  (nnz(L) − nnz(A_lower))", fontsize=FONT_LABEL)
ax.set_ylabel("Factorization time (s, median)", fontsize=FONT_LABEL)
ax.set_title(
    "Fill-in vs. Factorization Time\n(log–log, r ≈ 0.96 across orderings)",
    fontsize=FONT_TITLE,
)
ax.legend(fontsize=FONT_LEGEND, markerscale=2)
fig.tight_layout()
save(fig, "05_fillin_vs_time_scatter")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Speedup vs natural, heatmap by matrix kind
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 6: speedup heatmap by matrix kind")

# Only non-natural orderings; only full-coverage matrices
NON_NAT = ["amd", "metis", "nesdis"]

kind_speedups = defaultdict(lambda: defaultdict(list))
for k in full_keys:
    nat_t = matrix_index[k]["natural"]["time_factor_s_median"]
    if nat_t == 0:
        continue
    kind = matrix_index[k]["natural"].get("matrix_kind", "unknown")
    for o in NON_NAT:
        bt = matrix_index[k][o]["time_factor_s_median"]
        if bt > 0:
            kind_speedups[kind][o].append(nat_t / bt)

# Filter kinds with enough data
min_count = 3
kinds_filtered = sorted(
    [
        kd
        for kd, od in kind_speedups.items()
        if all(len(od[o]) >= min_count for o in NON_NAT)
    ]
)

kinds_filtered = [
    k
    for k in kinds_filtered
    if "subsequent" not in k.lower() and "duplicate" not in k.lower()
]

heat = np.array(
    [[np.median(kind_speedups[kd][o]) for o in NON_NAT] for kd in kinds_filtered]
)

fig, ax = plt.subplots(figsize=(6, max(4, 0.42 * len(kinds_filtered))))
im = ax.imshow(
    heat,
    aspect="auto",
    cmap="YlOrRd",
    norm=LogNorm(vmin=max(heat.min(), 0.1), vmax=heat.max()),
)
plt.colorbar(im, ax=ax, label="Median speedup vs. natural (log scale)")
ax.set_xticks(range(len(NON_NAT)))
ax.set_xticklabels([ORDER_LABELS[o] for o in NON_NAT], fontsize=FONT_TICK)
ax.set_yticks(range(len(kinds_filtered)))
ax.set_yticklabels(kinds_filtered, fontsize=FONT_TICK - 1)
ax.set_title(
    "Factorization Speedup vs. Natural\nby Matrix Kind (median)", fontsize=FONT_TITLE
)
# Annotate cells
for i in range(len(kinds_filtered)):
    for j in range(len(NON_NAT)):
        v = heat[i, j]
        ax.text(
            j,
            i,
            f"{v:.1f}×",
            ha="center",
            va="center",
            fontsize=7,
            color="black" if v < heat.max() * 0.5 else "white",
        )
fig.tight_layout()
save(fig, "06_speedup_heatmap_by_kind")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Analyze vs factor time grouped bars (overhead analysis)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 7: analyze vs factor overhead")

fig, ax = plt.subplots(figsize=(7, 4.5))

medians_analyze = []
medians_factor = []
for o in ORDERINGS:
    recs = [r for r in all_records if r["ordering"] == o]
    medians_analyze.append(np.median([r["time_analyze_s_median"] for r in recs]))
    medians_factor.append(np.median([r["time_factor_s_median"] for r in recs]))

x = np.arange(len(ORDERINGS))
w = 0.32
bars_a = ax.bar(
    x - w / 2,
    medians_analyze,
    w,
    label="Symbolic analyze",
    color=[COLORS[o] for o in ORDERINGS],
    alpha=0.6,
    hatch="//",
)
bars_f = ax.bar(
    x + w / 2,
    medians_factor,
    w,
    label="Numeric factor",
    color=[COLORS[o] for o in ORDERINGS],
    alpha=0.95,
)

ax.set_xticks(x)
ax.set_xticklabels([ORDER_LABELS[o] for o in ORDERINGS], fontsize=FONT_TICK)
ax.set_yscale("log")
ax.set_ylabel("Median time (s)", fontsize=FONT_LABEL)
ax.set_title(
    "Symbolic Analyze vs. Numerical Factorization Time",
    fontsize=FONT_TITLE,
)
ax.legend(fontsize=FONT_LEGEND)
# Add ratio annotations
for i, o in enumerate(ORDERINGS):
    ratio = medians_analyze[i] / max(medians_factor[i], 1e-15)
    ax.text(
        i,
        max(medians_analyze[i], medians_factor[i]) * 1.6,
        f"{ratio:.1f}×",
        ha="center",
        fontsize=8,
        color="#333",
    )

fig.tight_layout()
save(fig, "07_analyze_vs_factor_overhead")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — METIS vs AMD fill-in scatter (log-log)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 8: METIS vs AMD fill-in scatter")

fig, ax = plt.subplots(figsize=(6, 6))

amd_fill = []
metis_fill = []
for k in full_keys:
    af = matrix_index[k]["amd"]["metrics"]["fill_in"]
    mf = matrix_index[k]["metis"]["metrics"]["fill_in"]
    if af > 0 and mf > 0:
        amd_fill.append(af)
        metis_fill.append(mf)

amd_arr = np.array(amd_fill)
metis_arr = np.array(metis_fill)

# Color by who wins
colors_scatter = np.where(metis_arr <= amd_arr, COLORS["metis"], COLORS["amd"])
ax.scatter(amd_arr, metis_arr, c=colors_scatter, s=18, alpha=0.7, linewidths=0)

# y=x line
lo = min(amd_arr.min(), metis_arr.min())
hi = max(amd_arr.max(), metis_arr.max())
ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="METIS = AMD")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("AMD fill-in", fontsize=FONT_LABEL)
ax.set_ylabel("METIS fill-in", fontsize=FONT_LABEL)
ax.set_title("METIS vs. AMD Fill-in", fontsize=FONT_TITLE)

n_amd_better = int((amd_arr < metis_arr).sum())
n_metis_better = int((metis_arr <= amd_arr).sum())
patches = [
    mpatches.Patch(color=COLORS["metis"], label=f"METIS Better ({n_metis_better})"),
    mpatches.Patch(color=COLORS["amd"], label=f"AMD Better ({n_amd_better})"),
    # mpatches.Patch(color="black"),
]
ax.legend(handles=patches, fontsize=FONT_LEGEND)
ax.set_aspect("equal", "box")
fig.tight_layout()
save(fig, "08_metis_vs_amd_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9/10 — Thread scaling scatters (structural problem only)
#   x = elimination tree metric, y = factor time, color = nthreads
# ─────────────────────────────────────────────────────────────────────────────

THREADS_OF_INTEREST = [1, 4, 8, 16]
THREAD_COLORS = {
    1: "#1f77b4",  # tab:blue
    4: "#ff7f0e",  # tab:orange
    8: "#2ca02c",  # tab:green
    16: "#d62728",  # tab:red
}


def _thread_scaling_points(*, ordering: str, metric: str):
    xs_by_t = {t: [] for t in THREADS_OF_INTEREST}
    ys_by_t = {t: [] for t in THREADS_OF_INTEREST}
    for r in records_full:
        if r.get("matrix_kind") != "structural problem":
            continue
        if r.get("ordering") != ordering:
            continue
        nt = int(r.get("nthreads", 1) or 1)
        if nt not in xs_by_t:
            continue
        m = r.get("metrics") or {}
        if not isinstance(m, dict) or metric not in m:
            continue
        y = r.get("time_factor_s_median")
        if y is None:
            continue
        xs_by_t[nt].append(m[metric])
        ys_by_t[nt].append(y)
    return xs_by_t, ys_by_t


def thread_scaling_scatter(metric: str, *, x_label: str, xscale: str, name: str):
    print(f"{name}: thread scaling scatter ({metric})")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharey=True)
    axes = axes.ravel()

    any_points = {t: False for t in THREADS_OF_INTEREST}
    for i, o in enumerate(ORDERINGS):
        ax = axes[i]
        xs_by_t, ys_by_t = _thread_scaling_points(ordering=o, metric=metric)
        for t in THREADS_OF_INTEREST:
            xs, ys = xs_by_t[t], ys_by_t[t]
            if not xs:
                continue
            any_points[t] = True
            ax.scatter(
                xs,
                ys,
                s=18,
                alpha=0.7,
                linewidths=0,
                c=THREAD_COLORS.get(t, "#333333"),
            )

        ax.set_title(ORDER_LABELS[o], fontsize=FONT_TICK + 1)
        ax.set_yscale("log")
        if xscale == "log":
            ax.set_xscale("log")
        ax.set_xlabel(x_label, fontsize=FONT_TICK)
        if i % 2 == 0:
            ax.set_ylabel("Factor time (s, median)", fontsize=FONT_TICK)

    handles = [
        mpatches.Patch(color=THREAD_COLORS[t], label=f"{t} threads")
        for t in THREADS_OF_INTEREST
        if any_points.get(t)
    ]
    fig.suptitle(
        f"Thread Scaling (Structural Problem Matrices)\n{metric} vs factor time",
        fontsize=FONT_TITLE,
        y=1.02,
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(handles),
            fontsize=FONT_LEGEND,
            frameon=False,
        )
    fig.tight_layout()
    save(fig, name)


thread_scaling_scatter(
    "etree_height",
    x_label="Elimination tree height",
    xscale="linear",
    name="09_thread_scaling_scatter_etree_height",
)
thread_scaling_scatter(
    "max_col_nnz_L",
    x_label="max nnz per column in L",
    xscale="log",
    name="10_thread_scaling_scatter_max_col_nnz_L",
)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\nDone. All figures saved to '{OUT_DIR}/'.")
