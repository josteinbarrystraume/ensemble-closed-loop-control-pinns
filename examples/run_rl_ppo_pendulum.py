"""Example script for CL-CPINN.

These examples are intentionally explicit and verbose so that a user can
copy/paste and modify them without needing to remember CLI flags.

All examples call the main entry point:

    python ensemble_cl_cpinn_control.py <subcommand> [flags...]

Run from the repository root (where ensemble_cl_cpinn_control.py lives), e.g.:

    python examples/run_rl_ppo_pendulum.py

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
    # Requires: pip install gymnasium
    outdir = "outputs_rl_ppo_pendulum"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "rl_ppo",
        "--env",
        "Pendulum-v1",
        "--outdir",
        outdir,
        "--device",
        "auto",
        "--steps",
        "200000",
        "--seed",
        "0",
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
