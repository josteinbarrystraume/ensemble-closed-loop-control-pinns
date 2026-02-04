"""Example script for CL-CPINN.

These examples are intentionally explicit and verbose so that a user can
copy/paste and modify them without needing to remember CLI flags.

All examples call the main entry point:

    python ensemble_cl_cpinn_control.py <subcommand> [flags...]

Run from the repository root (where ensemble_cl_cpinn_control.py lives), e.g.:

    python examples/run_benchmarks_nonlinear2d.py

"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ensemble_cl_cpinn_control.py"


def run(cmd: list[str]) -> None:
    print("\n=== Running command ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)



def main() -> None:
    outdir = "outputs_benchmarks_nonlinear2d"

    # 1) TFC baseline
    run(
        [
            sys.executable,
            str(SCRIPT),
            "benchmarks",
            "--problem",
            "nonlinear2d",
            "--which",
            "tfc",
            "--outdir",
            outdir,
            "--alpha",
            "0.1",
            "--epochs_warm",
            "2000",
            "--epochs_hjb",
            "1200",
            "--make_plots",
            "1",
            "--save_history",
            "1",
            "--save_csv",
            "1",
        ]
    )

    # 2) Bellman/actor-critic baseline
    run(
        [
            sys.executable,
            str(SCRIPT),
            "benchmarks",
            "--problem",
            "nonlinear2d",
            "--which",
            "bellman",
            "--outdir",
            outdir,
            "--make_plots",
            "0",
            "--save_history",
            "0",
            "--save_csv",
            "1",
        ]
    )

    # 3) Simple grid-based VI baseline (2D only)
    run(
        [
            sys.executable,
            str(SCRIPT),
            "benchmarks",
            "--problem",
            "nonlinear2d",
            "--which",
            "vi",
            "--outdir",
            outdir,
            "--vi_grid",
            "61",
            "--vi_iters",
            "120",
            "--make_plots",
            "1",
            "--save_csv",
            "1",
        ]
    )


if __name__ == "__main__":
    main()
