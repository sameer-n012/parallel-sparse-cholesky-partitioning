import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.cholesky_bench import run_bench
from src.matrices import download_matrix, find_matrices, get_matrix, load_mat_csc


def set_env_threads(nthreads: int):
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)


def run_one(mat_id: int, args: argparse.Namespace) -> dict:
    """
    Runs the benchmark for one matrix.
    """

    m = get_matrix(mat_id)
    path = download_matrix(m, data_dir=args.data_dir)
    A = load_mat_csc(
        path,
        make_laplacian=args.matrix_kind == "laplacian",
    )
    res = run_bench(
        A, ordering=args.ordering, nthreads=args.nthreads, repeats=args.repeats
    )
    res["matrix_id"] = m.id
    res["matrix_name"] = m.name
    res["group"] = m.group
    res["matrix_kind"] = m.kind
    res["pattern_sym"] = m.psym
    res["numeric_sym"] = m.nsym
    res["nrows"] = m.rows
    res["ncols"] = m.cols
    res["nnz"] = m.nnz
    res["dtype"] = m.dtype
    res["2d3d"] = m.is2d3d
    res["spd"] = m.isspd
    return res


def run(args: argparse.Namespace):
    """
    Runs the benchmark
    """
    set_env_threads(args.nthreads)

    mat_ids = find_matrices(
        n=args.nmats,
        kind=args.matrix_kind,
        data_dir=args.data_dir,
    )

    results = []
    for mat_id in mat_ids:
        res = run_one(mat_id, args)
        results.append(res)

    out = Path(os.path.join(args.out_dir, f"results_{time.time()}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix-kind",
        required=True,
    )
    ap.add_argument("--matrix-name", required=True)
    ap.add_argument(
        "--ordering", required=True, choices=["natural", "amd", "metis", "nesdis"]
    )
    ap.add_argument("--nthreads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=10)  # num runs per matrix
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--nmats", type=int, default=100)  # num matrices
    args = ap.parse_args()

    # set in run_bench
    # set_env_threads(args.nthreads)

    run(args)
