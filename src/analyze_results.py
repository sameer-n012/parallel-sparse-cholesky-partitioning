"""
analyze_results.py
Analyzes sparse Cholesky benchmark results from results/results_*.json
"""

import json
import glob
import sys
import os
from collections import defaultdict
import math
import numpy as np
import scipy.stats as st

INCLUDE_DUPLICATES = False
INCLUDE_SUBSEQUENTS = True

def mean(xs):
    return np.mean(xs) if xs else float('nan')

def median(xs):
    return np.median(xs) if xs else float('nan')

def stdev(xs):
    return np.std(xs) if len(xs) >= 2 else float('nan')

def sem(xs):
    return st.sem(xs) if len(xs) >= 2 else float('nan')

def fmt(x, decimals=3):
    if isinstance(x, float):
        if math.isnan(x): return "N/A"
        if abs(x) < 1e-3 or abs(x) > 1e6:
            return f"{x:.{decimals}e}"
    return f"{x:.{decimals}f}" if isinstance(x, float) else str(x)

def section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def subsection(title):
    print(f"\n--- {title} ---")

def table(rows, headers):
    """Print a simple aligned text table."""
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(str(cell)))
    fmt_row = lambda r: "  ".join(str(c).ljust(col_w[i]) for i, c in enumerate(r))
    print(fmt_row(headers))
    print("  ".join("-" * w for w in col_w))
    for row in rows:
        print(fmt_row(row))


def main():

    results_dir = "results"
    files = glob.glob(os.path.join(results_dir, "results_*.json"))

    if not files:
        print(f"No results_*.json files found in '{results_dir}'. Pass the directory as an argument.")
        sys.exit(1)

    all_records = []
    error_records = []
    dup_records = []

    for f in sorted(files):
        with open(f) as fh:
            data = json.load(fh)
        for rec in data:
            if "error" in rec:
                error_records.append(rec)
            elif "duplicate" in rec["matrix_kind"] and not INCLUDE_DUPLICATES:
                dup_records.append(rec)
            elif "subsequent" in rec["matrix_kind"] and not INCLUDE_SUBSEQUENTS:
                dup_records.append(rec)
            else:
                all_records.append(rec)

    section("DATASET OVERVIEW")
    print(f"  Total result records (success): {len(all_records)}")
    print(f"  Error records skipped:          {len(error_records)}")
    print(f"  Duplicate records skipped:      {len(dup_records)}")
    print(f"  JSON files loaded:              {len(files)}")

    ORDERINGS = sorted({r["ordering"] for r in all_records})
    MATRIX_KINDS = sorted({r.get("matrix_kind", "unknown") for r in all_records})

    unique_matrices = {(r["group"], r["matrix_name"]) for r in all_records}
    print(f"  Unique matrices:                {len(unique_matrices)}")
    print(f"  Orderings present:              {', '.join(ORDERINGS)}")
    print(f"  Matrix kinds:                   {', '.join(MATRIX_KINDS)}")

    # Count matrices per ordering
    subsection("Record counts per ordering")
    for o in ORDERINGS:
        n = sum(1 for r in all_records if r["ordering"] == o)
        print(f"  {o:<20} {n}")

    # Error summary
    if error_records:
        subsection("Error summary")
        err_by_ordering = defaultdict(list)
        for r in error_records:
            err_by_ordering[r.get("ordering","?")].append(
                f"{r['group']}/{r['matrix_name']}: {r['error']}"
            )
        for o, errs in sorted(err_by_ordering.items()):
            print(f"  {o}: {len(errs)} errors")
            for e in errs[:5]:
                print(f"    {e}")
            if len(errs) > 5:
                print(f"    ... and {len(errs)-5} more")

    # Duplicate summary
    if dup_records:
        subsection("Duplicate summary")
        dup_by_kind = defaultdict(list)
        for r in dup_records:
            dup_by_kind[r.get("matrix_kind","?")].append(
                f"{r['group']}/{r['matrix_name']}"
            )
        for k, dups in sorted(dup_by_kind.items()):
            print(f"  {k}: {len(dups)} duplicates")
            print(f"    {','.join(dups)}")

    # ── per-matrix comparison ─────────────────────────────────────────────────────

    # Build index: (group, name) -> {ordering -> record}
    matrix_index = defaultdict(dict)
    for r in all_records:
        key = (r["group"], r["matrix_name"])
        matrix_index[key][r["ordering"]] = r

    # Only consider matrices where ALL orderings are available
    full_keys = [k for k, v in matrix_index.items() if set(v.keys()) == set(ORDERINGS)]
    partial_keys = [k for k, v in matrix_index.items() if set(v.keys()) != set(ORDERINGS)]

    section("FILL-IN COMPARISON ACROSS ORDERINGS")
    print(f"  Matrices with all orderings present: {len(full_keys)}")
    print(f"  Matrices with partial coverage:      {len(partial_keys)}")

    if full_keys:
        subsection("Fill-in ratio vs natural ordering (mean over all matrices)")
        nat_fillins = []
        ratio_sums = defaultdict(list)
        for key in full_keys:
            nat = matrix_index[key]["natural"]["metrics"]["fill_in"]
            if nat == 0:
                continue
            nat_fillins.append(nat)
            for o in ORDERINGS:
                ratio_sums[o].append(matrix_index[key][o]["metrics"]["fill_in"] / nat)
        for o in ORDERINGS:
            rs = ratio_sums[o]
            print(f"  {o:<20} mean ratio={fmt(mean(rs))}  median={fmt(median(rs))}  min={fmt(min(rs))}  max={fmt(max(rs))}")

        subsection("nnz(L) absolute values — mean over matrices, by ordering")
        for o in ORDERINGS:
            vals = [matrix_index[k][o]["metrics"]["nnz_L"] for k in full_keys]
            print(f"  {o:<20} mean={fmt(mean(vals),0)}  median={fmt(median(vals),0)}")

        subsection("Matrices where natural ordering beats AMD on fill-in (sanity check)")
        natural_wins = [(k, matrix_index[k]["natural"]["metrics"]["fill_in"],
                            matrix_index[k]["amd"]["metrics"]["fill_in"])
                        for k in full_keys
                        if "amd" in matrix_index[k]
                        and matrix_index[k]["natural"]["metrics"]["fill_in"]
                            <= matrix_index[k]["amd"]["metrics"]["fill_in"]]
        if natural_wins:
            for k, nf, af in natural_wins[:10]:
                print(f"  {k[0]}/{k[1]}: natural={nf}, amd={af}")
        else:
            print("  None — AMD always reduces fill-in (expected).")

    # ── etree structure ────────────────────────────────────────────────────────────

    section("ELIMINATION TREE STRUCTURE")

    subsection("Tree height by ordering — mean/median/max over all matrices")
    rows = []
    for o in ORDERINGS:
        recs = [r for r in all_records if r["ordering"] == o]
        heights = [r["metrics"]["etree_height"] for r in recs]
        rows.append([o, fmt(mean(heights),1), fmt(median(heights),1), str(max(heights)), str(min(heights))])
    table(rows, ["ordering", "mean_height", "median_height", "max_height", "min_height"])

    subsection("Height reduction ratio vs natural ordering (full-coverage matrices only)")
    if full_keys and "natural" in ORDERINGS:
        for o in [x for x in ORDERINGS if x != "natural"]:
            ratios = [matrix_index[k][o]["metrics"]["etree_height"] /
                    matrix_index[k]["natural"]["metrics"]["etree_height"]
                    for k in full_keys
                    if matrix_index[k]["natural"]["metrics"]["etree_height"] > 0]
            print(f"  {o:<20} mean height ratio vs natural: {fmt(mean(ratios))}  "
                f"(min={fmt(min(ratios))}, max={fmt(max(ratios))})")

    subsection("Max level width (parallelism proxy) by ordering")
    rows = []
    for o in ORDERINGS:
        recs = [r for r in all_records if r["ordering"] == o]
        max_widths = [max(r["metrics"]["etree_widths"]) for r in recs]
        rows.append([o, fmt(mean(max_widths),1), fmt(median(max_widths),1),
                    str(max(max_widths)), str(min(max_widths))])
    table(rows, ["ordering", "mean_max_width", "median_max_width", "max", "min"])

    subsection("Sum of level widths > 1 (levels with any parallelism) by ordering")
    for o in ORDERINGS:
        recs = [r for r in all_records if r["ordering"] == o]
        parallel_levels = [sum(1 for w in r["metrics"]["etree_widths"] if w > 1) for r in recs]
        print(f"  {o:<20} mean parallel levels: {fmt(mean(parallel_levels),1)}  "
            f"median: {fmt(median(parallel_levels),1)}")

    # ── timing analysis ────────────────────────────────────────────────────────────

    section("TIMING ANALYSIS")

    subsection("Mean factorization time (seconds) by ordering")
    rows = []
    for o in ORDERINGS:
        recs = [r for r in all_records if r["ordering"] == o]
        times = [r["time_factor_s_mean"] for r in recs]
        rows.append([o, fmt(mean(times)), fmt(median(times)), fmt(min(times)), fmt(max(times))])
    table(rows, ["ordering", "mean_time", "median_time", "min_time", "max_time"])

    subsection("Mean analyze (symbolic) time by ordering")
    rows = []
    for o in ORDERINGS:
        recs = [r for r in all_records if r["ordering"] == o]
        times = [r["time_analyze_s_mean"] for r in recs]
        rows.append([o, fmt(mean(times)), fmt(median(times))])
    table(rows, ["ordering", "mean_analyze_s", "median_analyze_s"])

    if full_keys:
        subsection("Factorization speedup vs natural ordering (median times, full-coverage matrices)")
        for o in [x for x in ORDERINGS if x != "natural"]:
            speedups = [matrix_index[k]["natural"]["time_factor_s_median"] /
                        matrix_index[k][o]["time_factor_s_median"]
                        for k in full_keys
                        if matrix_index[k][o]["time_factor_s_median"] > 0]
            wins = sum(1 for s in speedups if s > 1)
            print(f"  {o:<20} mean speedup={fmt(mean(speedups))}  "
                f"median={fmt(median(speedups))}  "
                f"matrices where faster: {wins}/{len(speedups)}")

        subsection("Analyze overhead: ratio analyze/factor time by ordering")
        for o in ORDERINGS:
            recs = [r for r in all_records if r["ordering"] == o
                    and r["time_factor_s_mean"] > 0]
            ratios = [r["time_analyze_s_mean"] / r["time_factor_s_mean"] for r in recs]
            print(f"  {o:<20} mean analyze/factor ratio: {fmt(mean(ratios))}")

    # ── matrix size scaling ────────────────────────────────────────────────────────

    section("SCALING WITH MATRIX SIZE")

    # Bin matrices by n
    def size_bin(n):
        if n < 500:    return "tiny   (<500)"
        if n < 5000:   return "small  (500-5k)"
        if n < 50000:  return "medium (5k-50k)"
        return              "large  (>50k)"

    subsection("Fill-in ratio (nnz_L / nnz_A_lower) by size bin and ordering")
    bins = defaultdict(lambda: defaultdict(list))
    for r in all_records:
        m = r["metrics"]
        if m["nnz_A_lower"] > 0:
            ratio = m["nnz_L"] / m["nnz_A_lower"]
            bins[size_bin(m["n"])][r["ordering"]].append(ratio)

    size_order = ["tiny   (<500)", "small  (500-5k)", "medium (5k-50k)", "large  (>50k)"]
    for sz in size_order:
        if sz not in bins: continue
        print(f"\n  {sz}")
        for o in ORDERINGS:
            vals = bins[sz][o]
            if vals:
                print(f"    {o:<20} mean fill ratio={fmt(mean(vals))}  n={len(vals)}")

    subsection("Max column nnz in L (proxy for supernode/frontal size) by ordering")
    rows = []
    for o in ORDERINGS:
        recs = [r for r in all_records if r["ordering"] == o]
        vals = [r["metrics"]["max_col_nnz_L"] for r in recs]
        rows.append([o, fmt(mean(vals),1), fmt(median(vals),1), str(max(vals))])
    table(rows, ["ordering", "mean_max_col_nnz", "median_max_col_nnz", "max"])

    # ── top/bottom performers ──────────────────────────────────────────────────────

    section("NOTABLE MATRICES")

    if full_keys:
        subsection("Top 10 matrices by fill-in reduction (natural vs best ordering)")
        improvements = []
        for key in full_keys:
            nat = matrix_index[key]["natural"]["metrics"]["fill_in"]
            if nat == 0: continue
            best_o = min([o for o in ORDERINGS if o != "natural"],
                        key=lambda o: matrix_index[key][o]["metrics"]["fill_in"])
            best_fill = matrix_index[key][best_o]["metrics"]["fill_in"]
            improvements.append((key, nat, best_fill, best_o, nat / max(best_fill, 1)))
        improvements.sort(key=lambda x: -x[4])
        rows = [(f"{k[0]}/{k[1]}", str(nf), str(bf), bo, fmt(ratio))
                for k, nf, bf, bo, ratio in improvements[:10]]
        table(rows, ["matrix", "fill_natural", "fill_best", "best_ord", "reduction_ratio"])

        subsection("Top 10 matrices by factorization speedup (vs natural)")
        speedups = []
        for key in full_keys:
            nat_t = matrix_index[key]["natural"]["time_factor_s_median"]
            if nat_t == 0: continue
            best_o = min([o for o in ORDERINGS if o != "natural"],
                        key=lambda o: matrix_index[key][o]["time_factor_s_median"])
            best_t = matrix_index[key][best_o]["time_factor_s_median"]
            speedups.append((key, nat_t, best_t, best_o, nat_t / max(best_t, 1e-15)))
        speedups.sort(key=lambda x: -x[4])
        rows = [(f"{k[0]}/{k[1]}", fmt(nt), fmt(bt), bo, fmt(sp))
                for k, nt, bt, bo, sp in speedups[:10]]
        table(rows, ["matrix", "time_natural_s", "time_best_s", "best_ord", "speedup"])

        subsection("Matrices where METIS does NOT beat AMD on fill-in (if both present)")
        if "metis" in ORDERINGS and "amd" in ORDERINGS:
            metis_worse = [(k,
                            matrix_index[k]["amd"]["metrics"]["fill_in"],
                            matrix_index[k]["metis"]["metrics"]["fill_in"])
                        for k in full_keys
                        if matrix_index[k]["metis"]["metrics"]["fill_in"]
                            > matrix_index[k]["amd"]["metrics"]["fill_in"]]
            print(f"  Count: {len(metis_worse)} / {len(full_keys)}")
            for k, af, mf in sorted(metis_worse, key=lambda x: x[2]/max(x[1],1), reverse=True)[:10]:
                print(f"    {k[0]}/{k[1]}: amd={af}, metis={mf}  (metis/amd ratio={fmt(mf/max(af,1))})")

    # ── correlation hints ──────────────────────────────────────────────────────────

    section("CORRELATION HINTS (for scatter plots)")

    subsection("Pearson r: etree_height vs time_factor_s_mean, by ordering")

    def pearson(xs, ys):
        n = len(xs)
        if n < 2: return float('nan')
        mx, my = mean(xs), mean(ys)
        num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
        return num/den if den > 0 else float('nan')

    for o in ORDERINGS:
        recs = [r for r in all_records if r["ordering"] == o]
        heights = [r["metrics"]["etree_height"] for r in recs]
        times   = [r["time_factor_s_mean"] for r in recs]
        print(f"  {o:<20} r = {fmt(pearson(heights, times))}")

    subsection("Pearson r: fill_in vs time_factor_s_mean, by ordering")
    for o in ORDERINGS:
        recs = [r for r in all_records if r["ordering"] == o]
        fills = [r["metrics"]["fill_in"] for r in recs]
        times = [r["time_factor_s_mean"] for r in recs]
        print(f"  {o:<20} r = {fmt(pearson(fills, times))}")

    subsection("Pearson r: n vs time_factor_s_mean, by ordering")
    for o in ORDERINGS:
        recs = [r for r in all_records if r["ordering"] == o]
        ns    = [r["metrics"]["n"] for r in recs]
        times = [r["time_factor_s_mean"] for r in recs]
        print(f"  {o:<20} r = {fmt(pearson(ns, times))}")

    subsection("Pearson r: etree_height vs fill_in, by ordering")
    for o in ORDERINGS:
        recs = [r for r in all_records if r["ordering"] == o]
        heights = [r["metrics"]["etree_height"] for r in recs]
        fills   = [r["metrics"]["fill_in"] for r in recs]
        print(f"  {o:<20} r = {fmt(pearson(heights, fills))}")

    # ── matrix kind breakdown ──────────────────────────────────────────────────────

    section("BREAKDOWN BY MATRIX KIND")

    kind_index = defaultdict(lambda: defaultdict(list))
    for r in all_records:
        kind_index[r.get("matrix_kind","unknown")][r["ordering"]].append(r)

    subsection("Mean fill-in ratio (nnz_L / nnz_A_lower) by matrix kind and ordering")
    for kind in sorted(kind_index.keys()):
        print(f"\n  {kind}")
        for o in ORDERINGS:
            recs = kind_index[kind][o]
            if not recs: continue
            ratios = [r["metrics"]["nnz_L"] / max(r["metrics"]["nnz_A_lower"], 1) for r in recs]
            print(f"    {o:<20} mean={fmt(mean(ratios))}  n={len(recs)}")

    subsection("Mean etree height by matrix kind and ordering")
    for kind in sorted(kind_index.keys()):
        print(f"\n  {kind}")
        for o in ORDERINGS:
            recs = kind_index[kind][o]
            if not recs: continue
            heights = [r["metrics"]["etree_height"] for r in recs]
            print(f"    {o:<20} mean={fmt(mean(heights),1)}")

    print()
    print("=" * 60)
    print("\tDONE")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()