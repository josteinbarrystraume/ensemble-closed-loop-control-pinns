"""Example script for CL-CPINN.

These examples are intentionally explicit and verbose so that a user can
copy/paste and modify them without needing to remember CLI flags.

All examples call the main entry point:

    python ensemble_cl_cpinn_control.py <subcommand> [flags...]

Run from the repository root (where ensemble_cl_cpinn_control.py lives), e.g.:

    python examples/run_ablation_nonlinear2d.py

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
    outdir = "outputs_ablation_nonlinear2d"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "ablation",
        "--problem",
        "nonlinear2d",
        "--outdir",
        outdir,
        "--device",
        "auto",
        "--ensemble",
        "5",
        "--seeds",
        "0",
        "1",
        "2",
        "--alpha",
        "0.1",
        "--alpha_schedule",
        "constant",
        "--epochs_warm",
        "2000",
        "--epochs_hjb",
        "1200",
        "--make_plots",
        "1",
        "--plot_runs",
        "all",
        "--save_history",
        "1",
        "--save_csv",
        "1",
    ]

    run(cmd)


if __name__ == "__main__":
    main()
