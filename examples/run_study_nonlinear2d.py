"""Example script for CL-CPINN.

These examples are intentionally explicit and verbose so that a user can
copy/paste and modify them without needing to remember CLI flags.

All examples call the main entry point:

    python ensemble_cl_cpinn_control.py <subcommand> [flags...]

Run from the repository root (where ensemble_cl_cpinn_control.py lives), e.g.:

    python examples/run_study_nonlinear2d.py

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
    outdir = "outputs_study_nonlinear2d"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "study",
        "--problem",
        "nonlinear2d",
        "--outdir",
        outdir,
        "--device",
        "auto",
        # Sweep grid (paper-style)
        "--alphas",
        "0.01",
        "0.1",
        "1",
        "10",
        "--alpha_schedules",
        "constant",
        "linear@0.1",
        "cosine@0.1",
        "exp@0.1",
        "step@0.1",
        "--rank_by",
        "J_mse",
        "--ensemble",
        "5",
        "--seeds",
        "0",
        "1",
        "2",
        # Training budget
        "--epochs_warm",
        "2000",
        "--epochs_hjb",
        "1200",
        "--batch_interior",
        "512",
        "--batch_boundary",
        "512",
        # Baselines
        "--do_tfc",
        "1",
        "--do_bellman",
        "1",
        "--do_vi",
        "1",
        "--vi_grid",
        "61",
        "--vi_iters",
        "120",
        # RL baseline is irrelevant here
        "--do_rl",
        "0",
        # Optional diagnostics
        "--make_plots",
        "1",
        "--save_history",
        "1",
        "--save_csv",
        "1",
    ]

    run(cmd)


if __name__ == "__main__":
    main()
