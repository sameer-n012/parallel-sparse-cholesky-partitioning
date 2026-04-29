"""Small smoke test for SuiteSparse download + CHOLMOD factorization.

Run this on the target machine where dependencies exist.
"""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.matrices import list_matrix_refs, materialize_matrix
    from src.cholesky_bench import run_bench

    ref = list_matrix_refs(kind="curated_spd", limit=1)[0]
    A = materialize_matrix(ref, kind="curated_spd", data_dir="data")
    out = run_bench(A, ordering="amd", nthreads=1, repeats=1)
    print(ref.name, out["ok"], out.get("time_factor_s_median"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
