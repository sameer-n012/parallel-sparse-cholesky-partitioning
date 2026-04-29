import os
import time
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
import scipy.stats as st
from sksparse.cholmod import analyze

from .metrics import factor_metrics


def _set_thread_env(nthreads: int):
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)


def run_cholesky_once(
    A_csc,
    ordering: str,
    diag_shift: float = 0.0,
) -> dict[str, Any]:
    """
    Analyze, factorize with CHOLMOD and return timings/metrics
    """

    if diag_shift and diag_shift > 0:
        A_csc = (A_csc + diag_shift * sp.eye(A_csc.shape[0], format="csc")).tocsc()

    try:
        t0 = time.perf_counter()
        Fsym = analyze(A_csc, ordering_method=ordering)
        t1 = time.perf_counter()
        F = Fsym.factorize(A_csc)
        t2 = time.perf_counter()

        L = F.L()
        m = factor_metrics(A_csc=A_csc, L_csc=L)

        return {
            "ordering": ordering,
            "error": None,
            "time_analyze_s": t1 - t0,
            "time_factor_s": t2 - t1,
            "metrics": m,
        }

    except Exception as e:
        return {
            "ordering": ordering,
            "error": f"Error in run_cholesky_once: {e}",
            "time_analyze_s": None,
            "time_factor_s": None,
            "metrics": None,
        }


def run_cholesky_retry(
    A_csc,
    ordering: str,
    diag_shift_schedule: list[float] = [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1],
) -> Optional[dict]:
    """
    Run CHOLMOD, retrying with an increasing diagonal shift if error
    """

    last = None
    for s in diag_shift_schedule:
        last = run_cholesky_once(A_csc, ordering=ordering, diag_shift=s)
        if not last["error"] and not last["metrics"]:
            last["metrics"]["diag_shift_used"] = float(s)
            return last
    return last


def run_bench(
    A_csc,
    ordering: str,
    nthreads: int,
    repeats: int,
) -> dict[str, Any]:
    _set_thread_env(nthreads)

    runs = []
    for _ in range(repeats):
        runs.append(run_cholesky_retry(A_csc, ordering=ordering))

    ok_runs = [r for r in runs if not r["error"]]
    if not ok_runs:
        return {
            "ordering": ordering,
            "nthreads": nthreads,
            "repeats": repeats,
            "error": runs[-1].error,
        }

    times_an = np.array([r["time_analyze_s"] for r in ok_runs])
    times_fac = np.array([r["time_factor_s"] for r in ok_runs])
    metrics = ok_runs[-1]["metrics"] or {}

    return {
        "ordering": ordering,
        "nthreads": nthreads,
        "repeats": repeats,
        "time_analyze_s_median": np.median(times_an),
        "time_factor_s_median": np.median(times_fac),
        "time_analyze_s_mean": np.mean(times_an),
        "time_factor_s_mean": np.mean(times_fac),
        "time_analyze_s_se": st.sem(times_an),
        "time_factor_s_se": st.sem(times_fac),
        "metrics": metrics,
    }
