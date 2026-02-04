"""Example script for CL-CPINN.

These examples are intentionally explicit and verbose so that a user can
copy/paste and modify them without needing to remember CLI flags.

All examples call the main entry point:

    python ensemble_cl_cpinn_control.py <subcommand> [flags...]

Run from the repository root (where ensemble_cl_cpinn_control.py lives), e.g.:

    python examples/run_train_quick_smoke_test.py

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
    # This is intentionally SMALL (fast) — it is not meant to reproduce paper results.
    outdir = "outputs_smoke_test"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "train_cpinn",
        "--problem",
        "nonlinear2d",
        "--outdir",
        outdir,
        "--device",
        "auto",
        "--ensemble",
        "2",
        "--alpha",
        "0.1",
        "--alpha_schedule",
        "constant",
        "--epochs_warm",
        "50",
        "--epochs_hjb",
        "50",
        "--batch_interior",
        "128",
        "--batch_boundary",
        "128",
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
