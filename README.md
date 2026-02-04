# Closed-Loop Ensemble CPINN (CL-CPINN) — Reference Code

This folder contains a **single-file** research implementation:

- `ensemble_cl_cpinn_control.py` — the main script (CPINN training + baselines + sweeps + optional PPO)

The code is organized as one file on purpose (for portability when sharing with users/reviewers),
but it is heavily commented and split into clearly labeled sections.

---

## 1) What this code does

This code trains a neural approximation of the **infinite-horizon cost-to-go** \(J(x)\) for continuous-time
optimal control problems by minimizing the **Hamilton–Jacobi–Bellman (HJB) residual**.

Key features implemented here:

- **Closed-loop policy recovery**: the feedback law is recovered from the value gradient
  \(\nabla J(x)\) using a closed-form minimizer (quadratic control cost + control-affine dynamics).
- **Warm-starting** from a reference \(J^*(x)\) (analytic CARE for LQR-type problems or numerical value iteration for nonlinear pendulum).
- **Boundary supervision** (or anchor constraints) for identifiability and improved stability.
- **Alpha sweeps + alpha scheduling** to characterize the trade-off between value learning and policy recovery.
- **Ensembles** of CPINNs (with optional outlier-robust mean control).
- Baselines:
  - TFC-style anchor embedding
  - Bellman/actor–critic neural baseline
  - Simple 2D grid-based value iteration (baseline)
  - Optional PPO (Pendulum) if `gymnasium` is installed

---

## 2) Requirements

Required Python packages:

- `numpy`
- `scipy`
- `torch`
- `matplotlib`

Optional (only needed for PPO baselines):

- `gymnasium`

If you want a quick install (example):

```bash
pip install numpy scipy matplotlib torch
pip install gymnasium  # optional
```

> Tip: the CPINN training uses float64 by default (see `TORCH_DTYPE`), which is slower but typically more stable.

---

## 3) Quickstart: run a single training job

Train a 5-member ensemble on the cartpole LQR benchmark:

```bash
python ensemble_cl_cpinn_control.py train_cpinn \
  --problem cartpole_lqr \
  --ensemble 5 \
  --seed 0 \
  --alpha 0.1 \
  --alpha_schedule constant \
  --use_warm_start 1 \
  --use_boundary_loss 1 \
  --epochs_warm 2000 \
  --epochs_hjb 1200 \
  --outdir outputs_train_cartpole_lqr
```

This creates run artifacts under `outputs_train_cartpole_lqr/`.

---

## 4) Recommended “paper-style” study runs (tune → best CPINN → baselines)

The `study` subcommand is the most “end-to-end” workflow:

1) sweeps alphas (and optional ablation toggles/schedules),
2) selects the best configuration by a metric (e.g., `J_mse`),
3) retrains that best CPINN config, and
4) runs baselines (TFC/Bellman/VI) (+ optional PPO for pendulum problems).

### Recommended sweep grid (matches the manuscript tuning philosophy)

- `alphas`: `0.01 0.1 1 10`
- `alpha_schedules`: `constant linear@0.1 cosine@0.1 exp@0.1 step@0.1`
  - the `@0.1` means: `alpha_start = alpha_end * 0.1` (relative start factor)
- `ensemble`: `5`
- `seeds`: `0 1 2`
- `epochs_warm`: `2000`
- `epochs_hjb`: `1200`
- batch sizes: `512`

### Example: nonlinear2d

```bash
python ensemble_cl_cpinn_control.py study \
  --problem nonlinear2d \
  --outdir outputs_study_nonlinear2d \
  --device auto \
  --alphas 0.01 0.1 1 10 \
  --alpha_schedules constant linear@0.1 cosine@0.1 exp@0.1 step@0.1 \
  --rank_by J_mse \
  --ensemble 5 \
  --seeds 0 1 2 \
  --epochs_warm 2000 \
  --epochs_hjb 1200 \
  --vi_grid 61 \
  --vi_iters 120 \
  --do_rl 0
```

### Example: pendulum (nonlinear, uses a cached VI reference)

```bash
python ensemble_cl_cpinn_control.py study \
  --problem pendulum \
  --outdir outputs_study_pendulum \
  --device auto \
  --alphas 0.01 0.1 1 10 \
  --alpha_schedules constant linear@0.1 cosine@0.1 exp@0.1 step@0.1 \
  --rank_by J_mse \
  --ensemble 5 \
  --seeds 0 1 2 \
  --epochs_warm 2000 \
  --epochs_hjb 1200 \
  --pendulum_vi_grid 61 --pendulum_vi_iters 200 --pendulum_vi_dt 0.02 --pendulum_vi_u_points 41 \
  --do_rl 0
```

Notes for pendulum:
- The VI reference is cached as a `.npz` file in `--outdir`. Re-running the same configuration will reuse it.
- To force recomputation, add `--pendulum_vi_force 1`.

---

## 5) Example scripts (recommended)

See the `examples/` folder (created alongside this README). These scripts are meant to be copy/pasted and modified.

- `examples/run_study_nonlinear2d.py`
- `examples/run_study_pendulum.py`
- `examples/run_study_cartpole_lqr.py`
- `examples/run_train_quick_smoke_test.py`
- `examples/run_benchmarks_cartpole_lqr.py`

Each script uses `subprocess.run([...])` with recommended hyperparameters.

---

## 6) Output files and where to look

Under `--outdir`, each run writes:

- `*_summary.json` / `*_summary.csv`  
  Key metrics such as `J_mse`, `u_mse`, residual diagnostics, and runtime.

- `*_train_history.csv`  
  Per-epoch diagnostics (warm-start MSE, boundary loss, residual loss, and `alpha_t`).

- `*.png` (if `--make_plots 1`)  
  Loss curves and (for 2D) heatmaps.

---

## 7) Tips for users

- Start by reading the commented sections in `ensemble_cl_cpinn_control.py`:
  - **Control problems** (how systems + costs are defined)
  - **HJB residuals / losses** (the CPINN objective)
  - **train_cpinn_ensemble** (how warm-start + ensembling works)
  - **main()** (CLI wiring)

- If you add a new benchmark problem, implement a new subclass of `ControlAffineProblem`
  and register it in `build_problem()`.

---

## 8) Troubleshooting

- **“gymnasium not installed”**: only affects `rl_ppo` / `rl_pinn_hybrid` and pendulum RL baselines.
- **CUDA out of memory**: reduce `--batch_interior`, `--batch_boundary`, or use `--device cpu`.
- **Slow VI**: reduce `--vi_grid` or `--vi_iters` (2D baseline is intentionally simple, not optimized).

---

If you have questions about how a specific block maps to the manuscript equations, search within the
code for “HJB”, “residual”, or “optimal_u_from_costate”.
