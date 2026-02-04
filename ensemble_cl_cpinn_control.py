#!/usr/bin/env python3
"""
Closed-Loop Ensemble Control via CPINNs (CL-CPINN) — Reference Research Code

This script is a **single-file** implementation used to generate the experiments in the
CL-CPINN manuscript (submitted to CNSNS). It trains physics-informed neural networks for
**infinite-horizon**, **continuous-time** optimal control by minimizing the residual of the
stationary Hamilton–Jacobi–Bellman (HJB) equation.

The core idea is:

  1) Learn a differentiable approximation of the cost-to-go J(x) with a neural network J_θ(x)
  2) Recover the feedback policy from the value gradient (costate):
         u*(x) = argmin_u { L(x,u) + ∇J(x)·f(x,u) }
     For quadratic control penalties and control-affine dynamics, this argmin has a closed form:
         u*(x) = -1/2 R^{-1} f₂(x)^T ∇J(x)
  3) Optionally train an **ensemble** of networks and aggregate their implied controls to
     improve robustness (including an outlier-robust variant).

Why a single file?
------------------
Keeping the implementation self-contained makes it easy for users (and reviewers)
to run the code without navigating a complex repository. The trade-off is file length, so
this version includes extensive commentary and clear section boundaries.

Glossary (variable naming in this file)
---------------------------------------
- x                : state, shape (B, n)
- u                : control, shape (B, m)
- Jnet(x)          : learned cost-to-go, shape (B, 1)
- dJ = ∇J(x)        : value gradient/costate, shape (B, n)
- residual r(x)    : HJB residual, scalar per sample, shape (B,)
- warm-start       : supervised regression of Jnet(x) to a reference J*(x)
- boundary loss    : optional supervision of J(x) on boundary faces (or anchor constraint)
- alpha            : weight for the HJB residual term (can be scheduled over epochs)
- CPINN            : "closed-loop PINN" training (value network + closed-form policy recovery)

What problems are included?
---------------------------
- nonlinear2d       : 2-state nonlinear analytic benchmark (closed-form J*, u*)
- cubic_nd          : n-state nonlinear analytic benchmark with closed-form J*, u*
- lqr               : n-state linear–quadratic regulator with CARE reference
- cartpole_lqr      : linearized continuous-time cart-pole with CARE reference
- pendulum_lqr      : linearized pendulum with CARE reference
- pendulum          : nonlinear pendulum with a cached numerical VI reference for J*, u*

Main command-line entry points
------------------------------
- train_cpinn    : train a single CPINN or an ensemble for one configuration
- tune           : sweep alpha × {warm-start,boundary} × schedule options
- study          : tune → retrain best CPINN → run baselines (TFC, BellmanNN, VI) (+ optional PPO)
- benchmarks     : run one baseline (TFC, BellmanNN, or VI)
- rl_ppo         : PPO baseline (optional; requires gymnasium)
- rl_pinn_hybrid : PPO + an optional physics/HJB regularizer on the critic (optional)

Outputs
-------
All artifacts are written under --outdir in timestamped run folders:
- *_summary.json / *_summary.csv : scalar metrics
- *_train_history.csv            : per-epoch losses (if enabled)
- *.png                          : plots (if enabled)

See README.md (generated alongside this file) for end-to-end example scripts and
recommended hyperparameters matching the manuscript experiments.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import math
import time
import random
import sys
import subprocess
import os
import csv
import json

import numpy as np
import scipy.linalg
from scipy.interpolate import RegularGridInterpolator

import torch
import torch.nn as nn
import torch.optim as optim

# Optional plotting
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except Exception:
    gym = None


# -----------------------------
# Global settings
# -----------------------------
# NOTE:
# - This project uses float64 (TORCH_DTYPE) by default. In our experience,
#   float32 can be unstable for stiff HJB residuals and for second-order
#   autodiff operations.
# - `set_seed` sets Python/NumPy/Torch RNGs for reproducibility.
# - `get_device` centralizes CPU/GPU selection (use --device auto by default).
TORCH_DTYPE = torch.float64


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    This sets the RNG seeds for:
      - Python's built-in `random` module
      - NumPy's global RNG
      - PyTorch (CPU RNG; and CUDA RNG when CUDA is available)

    Args:
        seed: Integer seed used across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device: str = "auto") -> torch.device:
    """Resolve a user-supplied device string into a :class:`torch.device`.

    Args:
        device: One of:
          - 'auto'  : choose CUDA if available, else CPU
          - 'cpu'   : force CPU
          - 'cuda' / 'cuda:0' / ... : select a specific CUDA device

    Returns:
        A torch.device instance.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# -----------------------------
# Neural nets
# -----------------------------
# The primary learnable object is a value network J_θ(x) implemented as an MLP.
# We also define small helper networks used by baselines:
#   - Actor networks u_φ(x) for the Bellman/actor–critic baseline
#   - PPO policies/value nets for the optional RL baselines
#
# Shape conventions throughout this file:
#   x : (B, n)  batch of states
#   u : (B, m)  batch of controls
#   J : (B, 1)  scalar value per state
class MLP(nn.Module):
    """Simple fully-connected multi-layer perceptron (MLP).

    This is the workhorse network used for the value function J(x) and (in some
    baselines) for the actor u(x) and PPO policy/value heads.

    Design notes:
      - Xavier/Glorot initialization is used for stable early training.
      - Activation can be tanh/relu/silu.
      - The output layer uses a linear activation.

    Args:
        in_dim: Input dimension (state dimension).
        out_dim: Output dimension (1 for value nets; control_dim for actors).
        hidden: Tuple of hidden layer widths.
        act: Activation name.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Tuple[int, ...] = (64, 64),
        act: str = "tanh",
    ) -> None:
        super().__init__()
        if act == "tanh":
            activation: nn.Module = nn.Tanh()
            gain_name = "tanh"
        elif act == "relu":
            activation = nn.ReLU()
            gain_name = "relu"
        elif act == "silu":
            activation = nn.SiLU()
            gain_name = "relu"
        else:
            raise ValueError(f"Unknown activation: {act}")

        layers: List[nn.Module] = []
        dims = (in_dim,) + hidden
        for a, b in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(a, b)
            nn.init.xavier_uniform_(lin.weight, gain=nn.init.calculate_gain(gain_name))
            nn.init.zeros_(lin.bias)
            layers += [lin, activation]
        out = nn.Linear(dims[-1], out_dim)
        nn.init.xavier_uniform_(out.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing to [-1,1] then scaled to action bounds."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (64, 64), act: str = "tanh") -> None:
        super().__init__()
        self.mu = MLP(obs_dim, act_dim, hidden=hidden, act=act)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=TORCH_DTYPE))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(obs)
        log_std = self.log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self.forward(obs)
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        action = torch.tanh(pre_tanh)
        # log prob with tanh correction
        # log N(mu,std) - sum log(1 - tanh(a)^2)
        var = std**2
        logp = -0.5 * (((pre_tanh - mu) ** 2) / (var + 1e-8) + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi))
        logp = logp.sum(dim=-1, keepdim=True)
        logp -= torch.log(1 - action**2 + 1e-8).sum(dim=-1, keepdim=True)
        return action, logp
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-probability of a *squashed* action in [-1,1] under the policy.

        This matches the correction used in :meth:`sample`:
            log π(a|s) = log N(pre_tanh; μ, σ) - Σ log(1 - tanh(pre_tanh)^2)

        Args:
            obs: (B, obs_dim)
            action: (B, act_dim) in [-1,1] (tanh-squashed)

        Returns:
            logp: (B, 1)
        """
        mu, std = self.forward(obs)
        # Numerical safety near |a|=1
        a = torch.clamp(action, -0.999999, 0.999999)
        pre_tanh = _atanh(a)
        var = std**2
        logp = -0.5 * (((pre_tanh - mu) ** 2) / (var + 1e-8) + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi))
        logp = logp.sum(dim=-1, keepdim=True)
        logp -= torch.log(1 - a**2 + 1e-8).sum(dim=-1, keepdim=True)
        return logp

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Approximate entropy (pre-tanh Normal entropy).

        The exact entropy of a tanh-squashed Normal is more involved; for PPO-style
        entropy bonuses, the Normal entropy is a common, simple surrogate.
        """
        _, std = self.forward(obs)
        # Normal entropy per dim: 0.5*log(2πeσ^2) = 0.5 + 0.5*log(2π) + log σ
        ent = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std + 1e-8)
        return ent.sum(dim=-1, keepdim=True)



# -----------------------------
# Control problems (continuous-time, control-affine)
# -----------------------------
# All benchmarks implement a continuous-time, control-affine system:
#     xdot = f1(x) + f2(x) u
# with running cost (typically quadratic):
#     L(x,u) = x^T Q x + u^T R u
#
# IMPORTANT: For quadratic control costs and control-affine dynamics, the HJB
# minimization over u has a closed form, which is what enables *closed-loop*
# policy recovery from ∇J(x) without training a separate actor network.
#
# Some problems expose a reference solution J*(x), u*(x) (analytic CARE,
# closed-form nonlinear, or numerical value iteration). We use these for:
#   (i) warm-start initialization (supervised regression), and
#   (ii) quantitative evaluation (MSE of value and control).
@dataclass
class Domain:
    """Axis-aligned box domain used for sampling training/evaluation states.

    The CPINN training loop requires sampling:
      - interior points (uniform)
      - boundary points (uniform, then projected to a random face)

    Attributes:
        low:  Lower bounds per dimension, shape (n,)
        high: Upper bounds per dimension, shape (n,)
    """
    low: np.ndarray  # (n,)
    high: np.ndarray  # (n,)

    def sample_uniform(self, n: int) -> np.ndarray:
        u = np.random.rand(n, self.low.size)
        return self.low + (self.high - self.low) * u

    def sample_boundary(self, n: int) -> np.ndarray:
        """Uniform samples, then project one coordinate to a random boundary face."""
        x = self.sample_uniform(n)
        d = self.low.size
        which_dim = np.random.randint(0, d, size=n)
        which_side = np.random.randint(0, 2, size=n)  # 0 -> low, 1 -> high
        for i in range(n):
            dim = which_dim[i]
            x[i, dim] = self.low[dim] if which_side[i] == 0 else self.high[dim]
        return x


class ControlAffineProblem:
    """
    Continuous-time control-affine system: xdot = f1(x) + f2(x) u
    Quadratic running cost: x^T Q x + u^T R u
    """
    name: str = "base"
    state_dim: int
    control_dim: int
    domain: Domain

    def f1(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def f2(self, x: torch.Tensor) -> torch.Tensor:
        """Return shape (batch, state_dim, control_dim)."""
        raise NotImplementedError

    def Q(self) -> torch.Tensor:
        return torch.eye(self.state_dim, dtype=TORCH_DTYPE)

    def R(self) -> torch.Tensor:
        return torch.eye(self.control_dim, dtype=TORCH_DTYPE)

    def running_cost(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # x: (B,n), u:(B,m)
        Q = self.Q().to(x.device)
        R = self.R().to(x.device)
        x_cost = torch.einsum("bi,ij,bj->b", x, Q, x)
        u_cost = torch.einsum("bi,ij,bj->b", u, R, u)
        return x_cost + u_cost  # (B,)

    def optimal_u_from_costate(self, x: torch.Tensor, costate: torch.Tensor) -> torch.Tensor:
        """
        Implements u*(x) = -1/2 R^{-1} f2(x)^T costate, matching Eq.(25) in the manuscript.
        costate = ∂J/∂x (B,n)
        """
        R = self.R().to(x.device)
        Rinv = torch.linalg.inv(R)
        f2 = self.f2(x)  # (B,n,m)
        # f2^T costate => (B,m)
        tmp = torch.bmm(f2.transpose(1, 2), costate.unsqueeze(-1)).squeeze(-1)
        u = -0.5 * (tmp @ Rinv.T)
        return u

    # Optional ground truth references
    def has_analytic(self) -> bool:
        return False

    def J_star(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def u_star(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_states(self, n: int, boundary: bool = False) -> np.ndarray:
        return self.domain.sample_boundary(n) if boundary else self.domain.sample_uniform(n)

    def anchor_states(self) -> np.ndarray:
        """Default anchor at the origin if it lies in the domain."""
        return np.zeros((1, self.state_dim), dtype=np.float64)


class Nonlinear2DAnalytic(ControlAffineProblem):
    """
    The 2-state nonlinear benchmark used in the manuscript (Eq.(39)).
    Analytic reference:
        J*(x) = 0.5 x1^2 + x2^2
        λ*(x) = [x1, 2 x2]
        u*(x) = -b(x1) x2,  b(x1)=cos(2x1)+2
    """
    name = "nonlinear2d"

    def __init__(self, low=-10.0, high=10.0) -> None:
        self.state_dim = 2
        self.control_dim = 1
        self.domain = Domain(low=np.array([low, low], dtype=np.float64), high=np.array([high, high], dtype=np.float64))

    def f1(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        b = torch.cos(2 * x1) + 2.0
        f1_1 = -x1 + x2
        f1_2 = -0.5 * x1 - 0.5 * x2 * (1 - b**2)
        return torch.stack([f1_1, f1_2], dim=1)

    def f2(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        b = torch.cos(2 * x1) + 2.0
        # shape (B,2,1)
        z = torch.zeros_like(b)
        return torch.stack([z, b], dim=1).unsqueeze(-1)

    def Q(self) -> torch.Tensor:
        return torch.eye(2, dtype=TORCH_DTYPE)

    def R(self) -> torch.Tensor:
        return torch.ones((1, 1), dtype=TORCH_DTYPE)

    def has_analytic(self) -> bool:
        return True

    def J_star(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        return 0.5 * x1**2 + x2**2

    def u_star(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        b = np.cos(2 * x1) + 2.0
        return (-b * x2).reshape(-1, 1)




class CubicNDAnalytic(ControlAffineProblem):
    """Nonlinear *analytic* benchmark without linearization.

    This is a fully actuated, decoupled polynomial system (each coordinate is independent):

        xdot_i = x_i^3 + u_i

    with infinite-horizon running cost:

        L(x,u) = sum_i (x_i^2 + u_i^2)

    The HJB reduces to a 1D nonlinear ODE per coordinate, which admits a closed-form solution.
    The optimal value function and feedback are:

        J*(x) = sum_i J1(x_i)
        u*_i(x) = -x_i (x_i^2 + sqrt(x_i^4 + 1))

    where

        J1(s) = 0.5 s^4 + 0.5 s^2 sqrt(s^4 + 1) + 0.5 asinh(s^2)

    Notes:
    - This is *nonlinear* and has a true closed-form J* (no linearization).
    - It scales to higher dimensions cleanly (n can be 2, 4, 8, ...).
    """

    name = "cubic_nd"

    def __init__(self, n: int = 4, low: float = -2.0, high: float = 2.0) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.state_dim = int(n)
        self.control_dim = int(n)  # fully actuated
        self.domain = Domain(
            low=np.full(self.state_dim, float(low), dtype=np.float64),
            high=np.full(self.state_dim, float(high), dtype=np.float64),
        )

    def f1(self, x: torch.Tensor) -> torch.Tensor:
        return x**3

    def f2(self, x: torch.Tensor) -> torch.Tensor:
        # Identity input matrix, expanded across batch
        I = torch.eye(self.state_dim, dtype=TORCH_DTYPE, device=x.device)
        return I.unsqueeze(0).expand(x.shape[0], -1, -1)

    def Q(self) -> torch.Tensor:
        return torch.eye(self.state_dim, dtype=TORCH_DTYPE)

    def R(self) -> torch.Tensor:
        return torch.eye(self.control_dim, dtype=TORCH_DTYPE)

    def has_analytic(self) -> bool:
        return True

    @staticmethod
    def _J1(s: np.ndarray) -> np.ndarray:
        # s: (N,)
        s2 = s**2
        s4 = s2**2
        return 0.5 * s4 + 0.5 * s2 * np.sqrt(s4 + 1.0) + 0.5 * np.arcsinh(s2)

    def J_star(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        # sum over coordinates
        return np.sum(self._J1(x), axis=1)

    def u_star(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return -(x * (x**2 + np.sqrt(x**4 + 1.0)))

class LQRND(ControlAffineProblem):
    """
    Continuous-time LQR benchmark: xdot = A x + B u, cost = x^T Q x + u^T R u
    Analytic reference from continuous algebraic Riccati equation (CARE).
    """
    name = "lqr"

    def __init__(self, n: int = 4, m: int = 2, low=-3.0, high=3.0, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.state_dim = n
        self.control_dim = m
        self.domain = Domain(low=np.full(n, low, dtype=np.float64), high=np.full(n, high, dtype=np.float64))
        # Make a stable-ish A by shifting eigenvalues left.
        A = rng.normal(size=(n, n))
        A = A - 0.5 * np.eye(n)
        B = rng.normal(size=(n, m))
        Q = np.eye(n)
        R = np.eye(m)
        self.A_np = A
        self.B_np = B
        self.Q_np = Q
        self.R_np = R
        # CARE solution
        self.P_np = scipy.linalg.solve_continuous_are(A, B, Q, R)

    def f1(self, x: torch.Tensor) -> torch.Tensor:
        A = torch.tensor(self.A_np, dtype=TORCH_DTYPE, device=x.device)
        return x @ A.T

    def f2(self, x: torch.Tensor) -> torch.Tensor:
        B = torch.tensor(self.B_np, dtype=TORCH_DTYPE, device=x.device)
        B_expand = B.unsqueeze(0).expand(x.shape[0], -1, -1)
        return B_expand

    def Q(self) -> torch.Tensor:
        return torch.tensor(self.Q_np, dtype=TORCH_DTYPE)

    def R(self) -> torch.Tensor:
        return torch.tensor(self.R_np, dtype=TORCH_DTYPE)

    def has_analytic(self) -> bool:
        return True

    def J_star(self, x: np.ndarray) -> np.ndarray:
        # V(x)=x^T P x
        return np.einsum("bi,ij,bj->b", x, self.P_np, x)

    def u_star(self, x: np.ndarray) -> np.ndarray:
        # Optimal u = -R^{-1} B^T P x  (standard LQR)
        Rinv = np.linalg.inv(self.R_np)
        K = Rinv @ self.B_np.T @ self.P_np
        return (-(K @ x.T).T).reshape(-1, self.control_dim)




class CartPoleLQR(ControlAffineProblem):
    """Continuous-time *linearized* cart-pole around the upright equilibrium.

    This implements the standard inverted-pendulum-on-a-cart linearization with continuous force input.

    State: x = [cart_pos, cart_vel, pole_angle, pole_ang_vel]

    Linearized dynamics:
        xdot = A x + B u

    Running cost:
        L(x,u) = x^T Q x + u^T R u

    Optimal (analytic) solution from CARE:
        J*(x) = x^T P x
        u*(x) = -K x

    References for the (A,B) structure match standard control literature (e.g. CTMS).
    """

    name = "cartpole_lqr"

    def __init__(
        self,
        *,
        # Domain ranges
        x_low: float = -2.4,
        x_high: float = 2.4,
        xdot_low: float = -3.0,
        xdot_high: float = 3.0,
        theta_low: float = -0.5,
        theta_high: float = 0.5,
        thetadot_low: float = -3.0,
        thetadot_high: float = 3.0,
        # Physical params
        M: float = 1.0,
        m: float = 0.1,
        l: float = 0.5,
        g: float = 9.8,
        b: float = 0.0,
        I: float | None = None,
        # Cost weights
        Q_diag: Tuple[float, float, float, float] = (1.0, 0.1, 10.0, 0.1),
        R_scalar: float = 0.1,
    ) -> None:
        self.state_dim = 4
        self.control_dim = 1
        self.domain = Domain(
            low=np.array([x_low, xdot_low, theta_low, thetadot_low], dtype=np.float64),
            high=np.array([x_high, xdot_high, theta_high, thetadot_high], dtype=np.float64),
        )

        M = float(M)
        m = float(m)
        l = float(l)
        g = float(g)
        b = float(b)
        if I is None:
            # Uniform rod of length 2l, inertia about COM: I = (1/12) m L^2
            L = 2.0 * l
            I = (1.0 / 12.0) * m * (L**2)
        I = float(I)

        # Continuous-time linearized cart-pole (inverted pendulum on a cart)
        # Following the standard CTMS form with parameter:
        #   p = I*(M+m) + M*m*l^2
        p = I * (M + m) + M * m * (l**2)
        if abs(p) < 1e-12:
            raise ValueError('Invalid parameters: p is near zero')

        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -((I + m * l**2) * b) / p, (m**2 * g * l**2) / p, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, -(m * l * b) / p, (m * g * l * (M + m)) / p, 0.0],
            ],
            dtype=np.float64,
        )
        B = np.array([[0.0], [(I + m * l**2) / p], [0.0], [m * l / p]], dtype=np.float64)

        Q = np.diag(np.array(Q_diag, dtype=np.float64))
        R = np.array([[float(R_scalar)]], dtype=np.float64)

        self.A_np = A
        self.B_np = B
        self.Q_np = Q
        self.R_np = R
        self.P_np = scipy.linalg.solve_continuous_are(A, B, Q, R)

    def f1(self, x: torch.Tensor) -> torch.Tensor:
        A = torch.tensor(self.A_np, dtype=TORCH_DTYPE, device=x.device)
        return x @ A.T

    def f2(self, x: torch.Tensor) -> torch.Tensor:
        B = torch.tensor(self.B_np, dtype=TORCH_DTYPE, device=x.device)
        return B.unsqueeze(0).expand(x.shape[0], -1, -1)

    def Q(self) -> torch.Tensor:
        return torch.tensor(self.Q_np, dtype=TORCH_DTYPE)

    def R(self) -> torch.Tensor:
        return torch.tensor(self.R_np, dtype=TORCH_DTYPE)

    def has_analytic(self) -> bool:
        return True

    def J_star(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.einsum('bi,ij,bj->b', x, self.P_np, x)

    def u_star(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        Rinv = np.linalg.inv(self.R_np)
        K = Rinv @ self.B_np.T @ self.P_np
        return (-(K @ x.T).T).reshape(-1, 1)

class PendulumLQR(ControlAffineProblem):
    """Continuous-time *linearized* pendulum about the upright equilibrium (theta=0).

    This is an analytic-control benchmark that mirrors the Gymnasium Pendulum cost structure,
    but uses a linearization of the dynamics:
        theta_dot = omega
        omega_dot ≈ (3g/(2l)) * theta + (3/(m l^2)) * u

    Running cost (matches Pendulum-v1 reward up to sign):
        L = theta^2 + 0.1*omega^2 + 0.001*u^2

    The optimal cost-to-go is quadratic:
        J*(x) = x^T P x
    where P solves the continuous-time algebraic Riccati equation (CARE).

    Notes:
    - This provides a TRUE closed-form reference for J*(x) and u*(x) for the linearized system.
    - It is only a good approximation of the nonlinear swing-up dynamics near theta=0.
    """

    name = "pendulum_lqr"

    def __init__(
        self,
        theta_low: float = -math.pi,
        theta_high: float = math.pi,
        omega_low: float = -8.0,
        omega_high: float = 8.0,
        *,
        g: float = 10.0,
        m: float = 1.0,
        l: float = 1.0,
    ) -> None:
        self.state_dim = 2
        self.control_dim = 1
        self.domain = Domain(
            low=np.array([theta_low, omega_low], dtype=np.float64),
            high=np.array([theta_high, omega_high], dtype=np.float64),
        )
        self.g = float(g)
        self.m = float(m)
        self.l = float(l)

        A = np.array([[0.0, 1.0], [3.0 * self.g / (2.0 * self.l), 0.0]], dtype=np.float64)
        B = np.array([[0.0], [3.0 / (self.m * self.l * self.l)]], dtype=np.float64)
        Q = np.diag([1.0, 0.1]).astype(np.float64)
        R = np.array([[0.001]], dtype=np.float64)

        self.A_np = A
        self.B_np = B
        self.Q_np = Q
        self.R_np = R
        self.P_np = scipy.linalg.solve_continuous_are(A, B, Q, R)

    def f1(self, x: torch.Tensor) -> torch.Tensor:
        A = torch.tensor(self.A_np, dtype=TORCH_DTYPE, device=x.device)
        return x @ A.T

    def f2(self, x: torch.Tensor) -> torch.Tensor:
        B = torch.tensor(self.B_np, dtype=TORCH_DTYPE, device=x.device)
        return B.unsqueeze(0).expand(x.shape[0], -1, -1)

    def Q(self) -> torch.Tensor:
        return torch.tensor(self.Q_np, dtype=TORCH_DTYPE)

    def R(self) -> torch.Tensor:
        return torch.tensor(self.R_np, dtype=TORCH_DTYPE)

    def has_analytic(self) -> bool:
        return True

    def J_star(self, x: np.ndarray) -> np.ndarray:
        return np.einsum("bi,ij,bj->b", x, self.P_np, x)

    def u_star(self, x: np.ndarray) -> np.ndarray:
        Rinv = np.linalg.inv(self.R_np)
        K = Rinv @ self.B_np.T @ self.P_np
        return (-(K @ x.T).T).reshape(-1, self.control_dim)


def _angle_normalize_np(theta: np.ndarray) -> np.ndarray:
    """Map angles to [-pi, pi)."""
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


def _angle_normalize_torch(theta: torch.Tensor) -> torch.Tensor:
    """Torch version: map angles to [-pi, pi)."""
    two_pi = 2.0 * math.pi
    return torch.remainder(theta + math.pi, two_pi) - math.pi


class PendulumVI(ControlAffineProblem):
    """Nonlinear pendulum (Gymnasium-like) with a cached *numerical* HJB reference.

    There is no known closed-form J*(theta, omega) for the full nonlinear swing-up problem
    with torque bounds. To enable warm-start + boundary supervision and MSE metrics, this class
    computes (or loads) a reference cost-to-go V(theta, omega) and an approximate optimal policy
    u*(theta, omega) using a simple grid-based value iteration solver.

    IMPORTANT:
      - This reference is NUMERICAL, not analytic.
      - The CPINN training still uses the continuous-time HJB residual with a clipped closed-form
        minimizer (box constraint), which is consistent with torque bounds.

    Dynamics (same form as gym/gymnasium Pendulum):
        theta_dot = omega
        omega_dot = -3g/(2l) * sin(theta + pi) + 3/(m l^2) * u

    Cost:
        L = angle_normalize(theta)^2 + 0.1*omega^2 + 0.001*u^2

    Reference solver (discrete-time approximation):
        V(x) = min_u { L(x,u) dt + V(x + dt f(x,u)) }
    with periodic wrapping in theta.
    """

    name = "pendulum"

    def __init__(
        self,
        theta_low: float = -math.pi,
        theta_high: float = math.pi,
        omega_low: float = -8.0,
        omega_high: float = 8.0,
        *,
        u_max: float = 2.0,
        g: float = 10.0,
        m: float = 1.0,
        l: float = 1.0,
        vi_grid: int = 61,
        vi_iters: int = 200,
        vi_dt: float = 0.02,
        vi_u_points: int = 41,
        cache_dir: str = "outputs",
        cache_tag: str = "",
        force_recompute: bool = False,
    ) -> None:
        self.state_dim = 2
        self.control_dim = 1
        self.domain = Domain(
            low=np.array([theta_low, omega_low], dtype=np.float64),
            high=np.array([theta_high, omega_high], dtype=np.float64),
        )
        self.u_max = float(u_max)
        self.g = float(g)
        self.m = float(m)
        self.l = float(l)

        # Cost matrices used for the closed-form minimizer.
        # State cost isn't globally quadratic due to angle wrapping, but the control term is.
        self.Q_np = np.diag([1.0, 0.1]).astype(np.float64)
        self.R_np = np.array([[0.001]], dtype=np.float64)

        self.vi_grid = int(vi_grid)
        self.vi_iters = int(vi_iters)
        self.vi_dt = float(vi_dt)
        self.vi_u_points = int(vi_u_points)

        os.makedirs(cache_dir, exist_ok=True)
        tag = ("_" + cache_tag) if cache_tag else ""
        cache_name = f"pendulum_vi_ref_grid{self.vi_grid}_dt{self.vi_dt:g}_iters{self.vi_iters}_u{self.vi_u_points}{tag}.npz"
        self.cache_path = os.path.join(cache_dir, cache_name)

        self._ref_loaded = False
        self._theta_grid: Optional[np.ndarray] = None
        self._omega_grid: Optional[np.ndarray] = None
        self._V_grid: Optional[np.ndarray] = None
        self._U_grid: Optional[np.ndarray] = None
        self._V_interp: Optional[RegularGridInterpolator] = None
        self._U_interp: Optional[RegularGridInterpolator] = None

        if (not force_recompute) and os.path.isfile(self.cache_path):
            self._load_reference(self.cache_path)
        else:
            self._compute_reference()
            self._save_reference(self.cache_path)

    # --- Control-affine pieces ---
    def f1(self, x: torch.Tensor) -> torch.Tensor:
        theta = x[:, 0]
        omega = x[:, 1]
        theta_dot = omega
        omega_dot = (-3.0 * self.g / (2.0 * self.l)) * torch.sin(theta + math.pi)
        return torch.stack([theta_dot, omega_dot], dim=1)

    def f2(self, x: torch.Tensor) -> torch.Tensor:
        # constant input matrix
        B = torch.tensor([[0.0], [3.0 / (self.m * self.l * self.l)]], dtype=TORCH_DTYPE, device=x.device)
        return B.unsqueeze(0).expand(x.shape[0], -1, -1)

    def Q(self) -> torch.Tensor:
        return torch.tensor(self.Q_np, dtype=TORCH_DTYPE)

    def R(self) -> torch.Tensor:
        return torch.tensor(self.R_np, dtype=TORCH_DTYPE)

    def running_cost(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        theta = x[:, 0]
        omega = x[:, 1]
        th = _angle_normalize_torch(theta)
        # Ensure u has shape (B,1)
        if u.ndim == 1:
            u1 = u
        else:
            u1 = u[:, 0]
        return th**2 + 0.1 * omega**2 + 0.001 * (u1**2)

    def optimal_u_from_costate(self, x: torch.Tensor, costate: torch.Tensor) -> torch.Tensor:
        # unconstrained minimizer for quadratic Hamiltonian
        u = super().optimal_u_from_costate(x, costate)
        # enforce box torque constraint
        return torch.clamp(u, -self.u_max, self.u_max)

    # --- Reference interface ---
    def has_analytic(self) -> bool:
        # The reference is numerical, but we expose it through the same interface.
        return True

    def J_star(self, x: np.ndarray) -> np.ndarray:
        self._ensure_reference_ready()
        th = _angle_normalize_np(x[:, 0])
        om = np.clip(x[:, 1], self.domain.low[1], self.domain.high[1])
        pts = np.stack([th, om], axis=1)
        V = self._V_interp(pts)
        return np.asarray(V, dtype=np.float64).reshape(-1)

    def u_star(self, x: np.ndarray) -> np.ndarray:
        self._ensure_reference_ready()
        th = _angle_normalize_np(x[:, 0])
        om = np.clip(x[:, 1], self.domain.low[1], self.domain.high[1])
        pts = np.stack([th, om], axis=1)
        U = self._U_interp(pts)
        U = np.asarray(U, dtype=np.float64).reshape(-1, 1)
        return np.clip(U, -self.u_max, self.u_max)

    def anchor_states(self) -> np.ndarray:
        # Upright equilibrium
        return np.array([[0.0, 0.0]], dtype=np.float64)

    # --- Reference internals ---
    def _ensure_reference_ready(self) -> None:
        if not self._ref_loaded or self._V_interp is None or self._U_interp is None:
            raise RuntimeError("PendulumVI reference not initialized.")

    def _load_reference(self, path: str) -> None:
        data = np.load(path)
        self._theta_grid = data["theta_grid"].astype(np.float64)
        self._omega_grid = data["omega_grid"].astype(np.float64)
        self._V_grid = data["V"].astype(np.float64)
        self._U_grid = data["U"].astype(np.float64)

        self._V_interp = RegularGridInterpolator((self._theta_grid, self._omega_grid), self._V_grid, bounds_error=False, fill_value=None)
        self._U_interp = RegularGridInterpolator((self._theta_grid, self._omega_grid), self._U_grid, bounds_error=False, fill_value=None)
        self._ref_loaded = True
        print(f"[pendulum_vi] loaded reference from {path} (grid={self._V_grid.shape}, iters={self.vi_iters}, dt={self.vi_dt:g})")

    def _save_reference(self, path: str) -> None:
        assert self._theta_grid is not None and self._omega_grid is not None and self._V_grid is not None and self._U_grid is not None
        np.savez_compressed(
            path,
            theta_grid=self._theta_grid,
            omega_grid=self._omega_grid,
            V=self._V_grid,
            U=self._U_grid,
            meta=np.array([self.vi_grid, self.vi_iters, self.vi_dt, self.vi_u_points, self.u_max, self.g, self.m, self.l], dtype=np.float64),
        )
        print(f"[pendulum_vi] saved reference to {path}")

    def _compute_reference(self) -> None:
        # Grid: theta periodic, so use endpoint=False in [-pi,pi)
        theta_low, theta_high = float(self.domain.low[0]), float(self.domain.high[0])
        omega_low, omega_high = float(self.domain.low[1]), float(self.domain.high[1])

        # If user passes [-pi,pi], enforce periodic grid
        # We assume theta_high - theta_low ≈ 2*pi for best results.
        theta_grid = np.linspace(theta_low, theta_high, self.vi_grid, endpoint=False, dtype=np.float64)
        omega_grid = np.linspace(omega_low, omega_high, self.vi_grid, endpoint=True, dtype=np.float64)

        u_grid = np.linspace(-self.u_max, self.u_max, self.vi_u_points, dtype=np.float64)

        V = np.zeros((theta_grid.size, omega_grid.size), dtype=np.float64)
        U = np.zeros((theta_grid.size, omega_grid.size), dtype=np.float64)

        interp = RegularGridInterpolator((theta_grid, omega_grid), V, bounds_error=False, fill_value=None)

        dt = float(self.vi_dt)
        k = -1
        for k in range(self.vi_iters):
            V_new = np.empty_like(V)
            U_new = np.empty_like(U)
            interp.values = V

            # brute-force VI
            for i, th in enumerate(theta_grid):
                for j, om in enumerate(omega_grid):
                    best = float('inf')
                    best_u = 0.0
                    for u in u_grid:
                        # continuous-time dynamics Euler step
                        th_dot = om
                        om_dot = (-3.0 * self.g / (2.0 * self.l)) * math.sin(th + math.pi) + (3.0 / (self.m * self.l * self.l)) * u
                        th_next = th + dt * th_dot
                        om_next = om + dt * om_dot

                        th_next = float(_angle_normalize_np(np.array([th_next]))[0])
                        om_next = float(np.clip(om_next, omega_low, omega_high))

                        stage = (float(_angle_normalize_np(np.array([th]))[0]) ** 2) + 0.1 * (om ** 2) + 0.001 * (u ** 2)
                        v_next = float(interp((th_next, om_next)))
                        c = stage * dt + v_next
                        if c < best:
                            best = c
                            best_u = u
                    V_new[i, j] = best
                    U_new[i, j] = best_u

            diff = float(np.max(np.abs(V_new - V)))
            V, U = V_new, U_new
            if (k + 1) % 10 == 0 or k == 0:
                print(f"[pendulum_vi] iter={k+1:4d} max|dV|={diff:.3e}")
            if diff < 1e-5:
                break

        self._theta_grid = theta_grid
        self._omega_grid = omega_grid
        self._V_grid = V
        self._U_grid = U
        self._V_interp = RegularGridInterpolator((theta_grid, omega_grid), V, bounds_error=False, fill_value=None)
        self._U_interp = RegularGridInterpolator((theta_grid, omega_grid), U, bounds_error=False, fill_value=None)
        self._ref_loaded = True
        print(f"[pendulum_vi] VI complete: grid={V.shape}, iters={k+1}, dt={dt:g}, u_points={u_grid.size}")


# -----------------------------
# HJB residuals / losses
# -----------------------------
# The stationary infinite-horizon HJB equation can be written as:
#     0 = min_u { L(x,u) + ∇J(x) · f(x,u) }
# where f(x,u)=f1(x)+f2(x)u for control-affine systems.
#
# We train J_θ by sampling x in the state domain and minimizing the squared
# residual r(x)^2, where r(x) is the left-hand side evaluated at the closed-form
# minimizer u*(x; θ) derived from ∇J_θ(x).
#
# The training loop has two phases:
#   1) warm-start: regress J_θ(x) to a reference J*(x) if available
#   2) HJB training: minimize boundary/anchor loss + α_t * residual loss
#
# α_t can be scheduled (constant/linear/cosine/exp/step) to balance the
# relative importance of value fitting vs residual minimization over time.
def grad_wrt_x(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return dy/dx for scalar y per batch: y shape (B,1) or (B,), x shape (B,n)."""
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return g


def hjb_residual_closed_form_u(problem: ControlAffineProblem, Jnet: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Residual r(x) = x^T Q x + u^T R u + ∇J·(f1(x)+f2(x)u), with u from closed-form minimizer.
    """
    x = x.requires_grad_(True)
    J = Jnet(x)  # (B,1)
    dJ = grad_wrt_x(J, x)  # (B,n)
    u = problem.optimal_u_from_costate(x, dJ)  # (B,m)
    f = problem.f1(x) + torch.bmm(problem.f2(x), u.unsqueeze(-1)).squeeze(-1)  # (B,n)
    stage = problem.running_cost(x, u)  # (B,)
    inner = (dJ * f).sum(dim=1)  # (B,)
    return stage + inner  # (B,)


def hjb_residual_actor(problem: ControlAffineProblem, Jnet: nn.Module, Unet: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same residual, but with u from an actor network instead of closed-form minimization.
    Returns (residual, u).
    """
    x = x.requires_grad_(True)
    J = Jnet(x)
    dJ = grad_wrt_x(J, x)
    u = Unet(x)
    if u.ndim == 2:
        pass
    else:
        u = u.view(x.shape[0], -1)
    f = problem.f1(x) + torch.bmm(problem.f2(x), u.unsqueeze(-1)).squeeze(-1)
    stage = problem.running_cost(x, u)
    inner = (dJ * f).sum(dim=1)
    return stage + inner, u


@dataclass
class TrainConfig:
    """Hyperparameters controlling CPINN training.

    The training objective is a weighted sum of:
      - boundary/anchor loss  (enforces identifiability / boundary conditions)
      - HJB residual loss     (physics / optimality constraint)

    total_loss = boundary_weight * loss_boundary + alpha_t * loss_hjb

    where alpha_t can follow a schedule over HJB epochs.

    Notes:
      - `use_warm_start`: if a reference J*(x) exists, run supervised pre-training.
      - `use_boundary_loss`: if reference exists, enforce boundary supervision; otherwise
         fall back to an anchor constraint J(0)=0 (or problem.anchor_states()).
    """
    seed: int = 0
    device: str = "auto"
    hidden: Tuple[int, ...] = (64, 64)
    act: str = "tanh"
    lr_warm: float = 1e-3
    lr_hjb: float = 1e-4
    epochs_warm: int = 2000
    epochs_hjb: int = 2000
    batch_interior: int = 512
    batch_boundary: int = 512

    # Loss weighting: total_loss = boundary_weight * loss_b + alpha_t * loss_hjb
    # alpha is interpreted as the *final/end* alpha (alpha_end).
    alpha: float = 0.1

    # Alpha scheduling (applied during HJB training).
    # - alpha_schedule: constant|linear|cosine|exp|step
    # - alpha_start_factor: used when alpha_start is None (alpha_start = alpha * alpha_start_factor)
    # - alpha_hold_frac: fraction of epochs_hjb to hold at alpha_start before ramping
    # - alpha_ramp_frac: fraction of epochs_hjb used to ramp from alpha_start -> alpha (alpha_end)
    alpha_schedule: str = "constant"
    alpha_start: Optional[float] = None
    alpha_start_factor: float = 1.0
    alpha_hold_frac: float = 0.0
    alpha_ramp_frac: float = 1.0

    boundary_weight: float = 1.0
    use_boundary_loss: bool = True
    use_warm_start: bool = True
    print_every: int = 200
    record_every: int = 1
    weight_decay: float = 0.0


def _normalize_alpha_schedule(name: str) -> str:
    """Normalize schedule names and allow a few common aliases."""
    n = (name or "constant").strip().lower()
    aliases = {
        "const": "constant",
        "fixed": "constant",
        "none": "constant",
        "lin": "linear",
        "linear": "linear",
        "cos": "cosine",
        "cosine": "cosine",
        "exp": "exp",
        "exponential": "exp",
        "step": "step",
    }
    return aliases.get(n, n)


def alpha_schedule_value(cfg: "TrainConfig", ep: int) -> float:
    """Return scheduled alpha_t for HJB epoch ep (1-indexed)."""
    sched = _normalize_alpha_schedule(getattr(cfg, "alpha_schedule", "constant"))
    if sched not in {"constant", "linear", "cosine", "exp", "step"}:
        raise ValueError(f"Unknown alpha_schedule: {sched}")

    a_end = float(getattr(cfg, "alpha", 0.0))

    # Determine start value
    if getattr(cfg, "alpha_start", None) is not None:
        a_start = float(cfg.alpha_start)  # type: ignore[arg-type]
    else:
        a_start = float(a_end * float(getattr(cfg, "alpha_start_factor", 1.0)))

    if not np.isfinite(a_start):
        a_start = a_end

    if sched == "constant":
        return a_end

    T = max(1, int(getattr(cfg, "epochs_hjb", 1)))
    hold = int(round(float(getattr(cfg, "alpha_hold_frac", 0.0)) * T))
    hold = max(0, min(hold, T))
    ramp = int(round(float(getattr(cfg, "alpha_ramp_frac", 1.0)) * T))
    ramp = max(1, min(ramp, T))

    if sched == "step":
        return a_start if ep <= hold else a_end

    # Ramp schedules
    if ep <= hold:
        return a_start

    t = (ep - hold) / max(1, ramp)
    if t <= 0.0:
        return a_start
    if t >= 1.0:
        return a_end

    if sched == "linear":
        return a_start + t * (a_end - a_start)
    if sched == "cosine":
        return a_start + 0.5 * (1.0 - math.cos(math.pi * t)) * (a_end - a_start)
    if sched == "exp":
        # Geometric interpolation; fall back to linear if non-positive.
        if a_start > 0.0 and a_end > 0.0:
            return float(a_start * (a_end / a_start) ** t)
        return a_start + t * (a_end - a_start)

    # Should be unreachable
    return a_end


def alpha_schedule_desc(cfg: "TrainConfig") -> str:
    """Human-readable description of the alpha schedule for logging/plots."""
    sched = _normalize_alpha_schedule(getattr(cfg, "alpha_schedule", "constant"))
    a_end = float(getattr(cfg, "alpha", 0.0))

    if getattr(cfg, "alpha_start", None) is not None:
        a_start = float(cfg.alpha_start)  # type: ignore[arg-type]
        start_part = f"start={a_start:g}"
    else:
        sf = float(getattr(cfg, "alpha_start_factor", 1.0))
        start_part = f"start_factor={sf:g}"

    hold = float(getattr(cfg, "alpha_hold_frac", 0.0))
    ramp = float(getattr(cfg, "alpha_ramp_frac", 1.0))

    if sched == "constant":
        return f"constant (alpha={a_end:g})"

    return f"{sched} (end={a_end:g}, {start_part}, hold_frac={hold:g}, ramp_frac={ramp:g})"


def build_value_net(problem: ControlAffineProblem, cfg: TrainConfig) -> nn.Module:
    """Construct the value network J_θ(x) for a given problem.

    Returns:
        A torch.nn.Module mapping x -> J(x) with output shape (B, 1).
    """
    return MLP(problem.state_dim, 1, hidden=cfg.hidden, act=cfg.act).to(dtype=TORCH_DTYPE)


def build_actor_net(problem: ControlAffineProblem, cfg: TrainConfig) -> nn.Module:
    """Construct an actor network u_φ(x) for a given problem.

    This is used only by certain baselines (e.g., the Bellman/actor-critic).

    Returns:
        A torch.nn.Module mapping x -> u(x) with output shape (B, control_dim).
    """
    return MLP(problem.state_dim, problem.control_dim, hidden=cfg.hidden, act=cfg.act).to(dtype=TORCH_DTYPE)


def warm_start_train(
    problem: ControlAffineProblem,
    Jnet: nn.Module,
    cfg: TrainConfig,
    *,
    history: Optional[List[Dict[str, object]]] = None,
    run_id: str = "",
    member: int = -1,
) -> Dict[str, float]:
    """
    Warm-start on available analytic J*(x) when present. If not available, this is a no-op.
    """
    if not problem.has_analytic():
        return {"warm_mse": float("nan")}

    dev = get_device(cfg.device)
    Jnet.to(dev)
    Jnet.train()

    opt = optim.AdamW(Jnet.parameters(), lr=cfg.lr_warm, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    last_mse = float("nan")

    for ep in range(1, cfg.epochs_warm + 1):
        x_np = problem.sample_states(cfg.batch_interior, boundary=False)
        y_np = problem.J_star(x_np).reshape(-1, 1)

        x = torch.tensor(x_np, dtype=TORCH_DTYPE, device=dev)
        y = torch.tensor(y_np, dtype=TORCH_DTYPE, device=dev)

        pred = Jnet(x)
        loss = mse(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        last_mse = float(loss.item())

        if history is not None and (ep % cfg.record_every == 0 or ep in {1, cfg.epochs_warm}):
            history.append(
                {
                    "run_id": run_id,
                    "phase": "warm",
                    "epoch": int(ep),
                    "member": int(member),
                    "loss_total": float(loss.item()),
                    "loss_b": float("nan"),
                    "loss_hjb": float("nan"),
                    "alpha_t": float("nan"),
                }
            )

        if ep % cfg.print_every == 0 or ep == 1:
            print(f"[warm] ep={ep:5d} mse={loss.item():.4e}")

    return {"warm_mse": float(last_mse)}


def hjb_train(
    problem: ControlAffineProblem,
    Jnet: nn.Module,
    cfg: TrainConfig,
    *,
    history: Optional[List[Dict[str, object]]] = None,
    run_id: str = "",
    member: int = 0,
) -> Dict[str, float]:
    """Main CPINN training loop (HJB residual minimization).

    Each epoch:
      1) sample interior points x
      2) compute closed-loop control u*(x) from ∇J(x)
      3) compute the HJB residual r(x)
      4) compute residual MSE loss_hjb = E[r(x)^2]
      5) compute boundary/anchor loss loss_b
      6) update parameters with: loss = boundary_weight*loss_b + alpha_t*loss_hjb

    Args:
        problem: ControlAffineProblem defining f1,f2,Q,R and sampling.
        Jnet: Value network to train.
        cfg: Training configuration.
        history: Optional list to append per-epoch diagnostics.
        run_id: Optional run identifier (propagated into history rows).
        member: Ensemble member index (for logging only).

    Returns:
        Dict of final losses for quick summary tables.
    """
    dev = get_device(cfg.device)
    Jnet.to(dev)
    Jnet.train()

    opt = optim.AdamW(Jnet.parameters(), lr=cfg.lr_hjb, weight_decay=cfg.weight_decay)

    last_loss = float("nan")
    last_b = float("nan")
    last_hjb = float("nan")

    for ep in range(1, cfg.epochs_hjb + 1):
        x_int_np = problem.sample_states(cfg.batch_interior, boundary=False)
        x_int = torch.tensor(x_int_np, dtype=TORCH_DTYPE, device=dev)

        res = hjb_residual_closed_form_u(problem, Jnet, x_int)
        loss_hjb = (res**2).mean()

        if cfg.use_boundary_loss and problem.has_analytic():
            x_b_np = problem.sample_states(cfg.batch_boundary, boundary=True)
            y_b_np = problem.J_star(x_b_np).reshape(-1, 1)
            x_b = torch.tensor(x_b_np, dtype=TORCH_DTYPE, device=dev)
            y_b = torch.tensor(y_b_np, dtype=TORCH_DTYPE, device=dev)
            loss_b = ((Jnet(x_b) - y_b) ** 2).mean()
        else:
            # Anchor constraint at origin if no boundary info
            x0_np = problem.anchor_states()
            x0 = torch.tensor(x0_np, dtype=TORCH_DTYPE, device=dev)
            loss_b = (Jnet(x0) ** 2).mean()

        alpha_t = alpha_schedule_value(cfg, ep)
        loss = cfg.boundary_weight * loss_b + alpha_t * loss_hjb

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        last_loss = float(loss.item())
        last_b = float(loss_b.item())
        last_hjb = float(loss_hjb.item())

        if history is not None and (ep % cfg.record_every == 0 or ep in {1, cfg.epochs_hjb}):
            history.append(
                {
                    "run_id": run_id,
                    "phase": "hjb",
                    "epoch": int(ep),
                    "member": int(member),
                    "loss_total": float(loss.item()),
                    "loss_b": float(loss_b.item()),
                    "loss_hjb": float(loss_hjb.item()),
                    "alpha_t": float(alpha_t),
                }
            )

        if ep % cfg.print_every == 0 or ep == 1:
            print(f"[hjb]  ep={ep:5d} loss={loss.item():.4e}  b={loss_b.item():.3e}  hjb={loss_hjb.item():.3e}  alpha_t={alpha_t:.3g}")

    return {"final_loss": float(last_loss), "final_b": float(last_b), "final_hjb": float(last_hjb)}


# -----------------------------
# PINN-TFC style constrained value net
# -----------------------------
# Baseline: Theory of Functional Connections (TFC)-style constraint embedding.
#
# Instead of enforcing a boundary/anchor constraint with an explicit loss term,
# we build a value network that satisfies the constraint *by construction*:
#     J_hat(x) = J(anchor) + g(x) * N(x),    with g(anchor)=0
# where N(x) is a free neural network and g(x) is a simple function that is
# zero at the anchor (e.g., squared distance to anchor).
#
# This approach often improves stability on LQR-like problems, since it removes
# the need to balance boundary and residual losses.
class TFCCostToGo(nn.Module):
    """
    A lightweight TFC-like embedding to enforce an anchor constraint exactly.

    For an anchor constraint J(x_a)=J_a (default x_a=0, J_a=0),
    define:
        J_hat(x) = J_a + g(x) * N(x)
    where g(x_a)=0, e.g., g(x)=||x-x_a||^2.

    This matches the 'spirit' of PINN-TFC constraint embedding (removing explicit boundary loss),
    but is intentionally lightweight and general across dimensions.
    """
    def __init__(self, base_net: nn.Module, anchor_x: np.ndarray, anchor_val: float = 0.0) -> None:
        super().__init__()
        self.net = base_net
        self.register_buffer("anchor_x", torch.tensor(anchor_x.reshape(1, -1), dtype=TORCH_DTYPE))
        self.anchor_val = float(anchor_val)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        d = x - self.anchor_x.to(x.device)
        return (d**2).sum(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.anchor_val + self.g(x) * self.net(x)


def train_tfc(problem: ControlAffineProblem, cfg: TrainConfig) -> Tuple[nn.Module, Dict[str, float]]:
    """Train the lightweight TFC-style baseline.

    The TFC baseline embeds the anchor constraint exactly by construction and
    then trains using the same HJB residual objective (no boundary loss term).

    Returns:
        (trained_model, stats_dict)
    """
    base = build_value_net(problem, cfg)
    tfc = TFCCostToGo(base, anchor_x=problem.anchor_states()[0], anchor_val=0.0)
    # In TFC variant we do NOT include boundary loss; the constraint is embedded.
    cfg2 = dataclasses.replace(cfg, use_boundary_loss=False, boundary_weight=0.0)
    if cfg2.use_warm_start and problem.has_analytic():
        warm_start_train(problem, tfc, cfg2)
    stats = hjb_train(problem, tfc, cfg2)
    return tfc, stats


# -----------------------------
# Bellman neural network (actor-critic continuous-time HJB)
# -----------------------------
# Baseline: Bellman/actor–critic neural solver.
#
# Here we learn both:
#   - critic J_θ(x) (value function), and
#   - actor u_φ(x) (policy),
# by minimizing the HJB residual with u provided by the actor.
#
# The critic is trained to reduce residual + satisfy boundary/anchor; the actor
# is trained to minimize the Hamiltonian using gradients of the critic that are
# detached (so actor updates do not backprop through critic parameters).
@dataclass
class BellmanConfig:
    """Hyperparameters for the Bellman/actor--critic baseline.

    This baseline trains both a critic J(x) and an actor u(x) by minimizing the
    HJB residual evaluated with the actor's control.
    """
    seed: int = 0
    device: str = "auto"
    hidden: Tuple[int, ...] = (64, 64)
    act: str = "tanh"
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    epochs: int = 3000
    batch: int = 512
    boundary_weight: float = 1.0
    use_boundary: bool = True
    print_every: int = 200


def train_bellman_nn(problem: ControlAffineProblem, cfg: BellmanConfig) -> Tuple[nn.Module, nn.Module, Dict[str, float]]:
    """Train the Bellman/actor--critic neural baseline.

    Training alternates:
      - critic update: minimize boundary/anchor + HJB residual MSE
      - actor update : minimize the Hamiltonian using detached critic gradients

    Returns:
        (critic_net, actor_net, stats_dict)
    """
    set_seed(cfg.seed)
    dev = get_device(cfg.device)

    critic = build_value_net(problem, TrainConfig(hidden=cfg.hidden, act=cfg.act)).to(dev)
    actor = build_actor_net(problem, TrainConfig(hidden=cfg.hidden, act=cfg.act)).to(dev)

    opt_c = optim.AdamW(critic.parameters(), lr=cfg.lr_critic)
    opt_a = optim.AdamW(actor.parameters(), lr=cfg.lr_actor)

    for ep in range(1, cfg.epochs + 1):
        x_np = problem.sample_states(cfg.batch, boundary=False)
        x = torch.tensor(x_np, dtype=TORCH_DTYPE, device=dev)

        res, u = hjb_residual_actor(problem, critic, actor, x)
        loss_hjb = (res**2).mean()

        # boundary/anchor constraint for critic
        if cfg.use_boundary and problem.has_analytic():
            xb_np = problem.sample_states(cfg.batch // 2, boundary=True)
            yb_np = problem.J_star(xb_np).reshape(-1, 1)
            xb = torch.tensor(xb_np, dtype=TORCH_DTYPE, device=dev)
            yb = torch.tensor(yb_np, dtype=TORCH_DTYPE, device=dev)
            loss_b = ((critic(xb) - yb) ** 2).mean()
        else:
            x0 = torch.tensor(problem.anchor_states(), dtype=TORCH_DTYPE, device=dev)
            loss_b = (critic(x0) ** 2).mean()

        # Critic update: make residual small + respect boundary/anchor
        loss_c = cfg.boundary_weight * loss_b + loss_hjb
        opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        opt_c.step()

        # Actor update: minimize Hamiltonian (proxy: mean stage+gradJ·f)
        # We need ∇_x J(x) but we do NOT backprop through the critic when updating the actor.
        # Compute dJ in grad mode and immediately detach it so critic params are unaffected.
        x_ag = x.detach().requires_grad_(True)
        J = critic(x_ag)
        dJ = grad_wrt_x(J, x_ag).detach()
        u = actor(x.detach())
        f = problem.f1(x.detach()) + torch.bmm(problem.f2(x.detach()), u.unsqueeze(-1)).squeeze(-1)
        stage = problem.running_cost(x.detach(), u)
        ham = stage + (dJ * f).sum(dim=1)
        loss_a = ham.mean()

        opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        opt_a.step()

        if ep % cfg.print_every == 0 or ep == 1:
            print(f"[bellman] ep={ep:5d} critic={loss_c.item():.3e} (b={loss_b.item():.2e}, hjb={loss_hjb.item():.2e}) actor={loss_a.item():.3e}")

    return critic, actor, {"final_critic": float(loss_c.item()), "final_actor": float(loss_a.item())}


# -----------------------------
# Ensemble control simulation (continuous-time Euler)
# -----------------------------
# After training, we can simulate closed-loop trajectories to test whether the
# learned policy stabilizes the system and how robust it is to noise.
#
# We simulate one perturbed system per ensemble member (N parallel trajectories).
# Policies supported:
#   - individual          : each member controls its own trajectory
#   - mean                : average control across ensemble members is applied to all
#   - outlier_exclusion   : compute the mean over inlier states; outliers fallback to
#                           individual control (useful under observation noise).
@dataclass
class SimConfig:
    """Configuration for closed-loop trajectory simulation.

    Attributes:
        dt: Euler integration step size.
        steps: Number of integration steps.
        obs_noise_std: Std-dev of additive observation noise on x.
        init_perturb_std: Std-dev of initial-condition perturbations per ensemble member.
    """
    dt: float = 0.01
    steps: int = 2000
    obs_noise_std: float = 0.0
    init_perturb_std: float = 0.0


def chauvenet_outliers(x: np.ndarray) -> np.ndarray:
    """
    Multivariate Chauvenet-like criterion on ensemble state vectors.

    Returns: boolean mask of inliers (True=inlier).
    """
    n, d = x.shape
    mu = x.mean(axis=0)
    Sigma = np.cov(x.T) + 1e-9 * np.eye(d)
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    # multivariate normal density
    diff = x - mu
    quad = np.einsum("bi,ij,bj->b", diff, inv, diff)
    norm = 1.0 / (math.pow(2 * math.pi, d / 2) * math.sqrt(det))
    p = norm * np.exp(-0.5 * quad)
    C = 1.0 / (2.0 * n)
    return p >= C


def simulate_ensemble_closed_loop(
    problem: ControlAffineProblem,
    nets: List[nn.Module],
    policy: str,
    x0: np.ndarray,
    sim: SimConfig,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Simulate N parallel perturbed systems (one per network).

    policy:
      - "individual" : each net controls its own trajectory
      - "mean"       : mean control across nets applied to all
      - "outlier_exclusion" : exclude outlier states from mean; outliers get individual control
    """
    set_seed(seed)
    dev = next(nets[0].parameters()).device

    N = len(nets)
    x = np.tile(x0.reshape(1, -1), (N, 1)).astype(np.float64)
    x += sim.init_perturb_std * np.random.randn(*x.shape)

    traj = np.zeros((sim.steps + 1, N, problem.state_dim), dtype=np.float64)
    u_traj = np.zeros((sim.steps, N, problem.control_dim), dtype=np.float64)
    traj[0] = x

    for t in range(sim.steps):
        # noisy observation
        x_obs = x + sim.obs_noise_std * np.random.randn(*x.shape)

        # compute controls
        u_each = []
        for j, net in enumerate(nets):
            net.eval()
            xt = torch.tensor(x_obs[j:j+1], dtype=TORCH_DTYPE, device=dev)
            xt.requires_grad_(True)
            J = net(xt)
            dJ = grad_wrt_x(J, xt)
            uj = problem.optimal_u_from_costate(xt, dJ).detach().cpu().numpy()
            u_each.append(uj)
        u_each = np.vstack(u_each)  # (N,m)

        if policy == "individual":
            u = u_each
        elif policy == "mean":
            u_mean = u_each.mean(axis=0, keepdims=True)
            u = np.repeat(u_mean, N, axis=0)
        elif policy == "outlier_exclusion":
            inlier = chauvenet_outliers(x_obs)
            if inlier.any():
                u_mean = u_each[inlier].mean(axis=0, keepdims=True)
                u = u_each.copy()
                u[inlier] = np.repeat(u_mean, inlier.sum(), axis=0)
            else:
                u = u_each
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Euler step
        xt = torch.tensor(x, dtype=TORCH_DTYPE, device=dev)
        ut = torch.tensor(u, dtype=TORCH_DTYPE, device=dev)
        f = problem.f1(xt) + torch.bmm(problem.f2(xt), ut.unsqueeze(-1)).squeeze(-1)
        x = x + sim.dt * f.detach().cpu().numpy()

        traj[t + 1] = x
        u_traj[t] = u

    return {"x": traj, "u": u_traj}


# -----------------------------
# Metrics and plots
# -----------------------------
# Evaluation utilities used by sweeps/studies:
#   - For 2D problems: evaluate on a dense uniform grid to create heatmaps.
#   - For nD problems: evaluate on random Monte-Carlo samples.
# Metrics include:
#   - J_mse : mean squared error vs reference value function (if available)
#   - u_mse : mean squared error vs reference policy (if available)
#   - res_* : diagnostics of the HJB residual (mean/RMS/max)
def evaluate_on_grid_2d(
    problem: ControlAffineProblem,
    Jnet: nn.Module,
    grid_n: int = 201,
    device: str = "auto",
) -> Dict[str, np.ndarray]:
    """
    Only for 2D problems: compute value error, control error, and HJB residual over a grid.
    """
    assert problem.state_dim == 2, "Grid evaluation helper is for 2D problems."

    dev = get_device(device)
    Jnet.to(dev)
    Jnet.eval()

    x1 = np.linspace(problem.domain.low[0], problem.domain.high[0], grid_n)
    x2 = np.linspace(problem.domain.low[1], problem.domain.high[1], grid_n)
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")
    pts = np.stack([X1.ravel(), X2.ravel()], axis=1)

    with torch.no_grad():
        x = torch.tensor(pts, dtype=TORCH_DTYPE, device=dev)
        x.requires_grad_(True)

    # Need gradients, so not under no_grad
    x = torch.tensor(pts, dtype=TORCH_DTYPE, device=dev, requires_grad=True)
    J = Jnet(x)
    dJ = grad_wrt_x(J, x)
    u = problem.optimal_u_from_costate(x, dJ)
    res = (problem.running_cost(x, u) + (dJ * (problem.f1(x) + torch.bmm(problem.f2(x), u.unsqueeze(-1)).squeeze(-1))).sum(dim=1))

    out: Dict[str, np.ndarray] = {
        "x1": x1,
        "x2": x2,
        "J_pred": J.detach().cpu().numpy().reshape(grid_n, grid_n),
        "u_pred": u.detach().cpu().numpy().reshape(grid_n, grid_n, problem.control_dim),
        "residual": res.detach().cpu().numpy().reshape(grid_n, grid_n),
    }

    if problem.has_analytic():
        out["J_true"] = problem.J_star(pts).reshape(grid_n, grid_n)
        out["u_true"] = problem.u_star(pts).reshape(grid_n, grid_n, problem.control_dim)
        out["J_mse"] = np.mean((out["J_pred"] - out["J_true"]) ** 2)
        out["u_mse"] = np.mean((out["u_pred"] - out["u_true"]) ** 2)

    out["res_mean"] = float(np.mean(out["residual"]))
    out["res_rms"] = float(np.sqrt(np.mean(out["residual"] ** 2)))
    out["res_max"] = float(np.max(np.abs(out["residual"])))
    return out


def evaluate_on_samples(
    problem: ControlAffineProblem,
    Jnet: nn.Module,
    n: int = 5000,
    device: str = "auto",
) -> Dict[str, float]:
    """Dimension-agnostic evaluation on randomly sampled states.

    Useful for nD problems where grid evaluation is impractical.
    Returns scalar metrics only.
    """

    dev = get_device(device)
    Jnet.to(dev)
    Jnet.eval()

    pts = problem.sample_states(n, boundary=False)
    x = torch.tensor(pts, dtype=TORCH_DTYPE, device=dev, requires_grad=True)
    J = Jnet(x)
    dJ = grad_wrt_x(J, x)
    u = problem.optimal_u_from_costate(x, dJ)
    f = problem.f1(x) + torch.bmm(problem.f2(x), u.unsqueeze(-1)).squeeze(-1)
    res = problem.running_cost(x, u) + (dJ * f).sum(dim=1)

    out: Dict[str, float] = {
        "res_mean": float(res.detach().cpu().mean().item()),
        "res_rms": float(torch.sqrt((res**2).mean()).detach().cpu().item()),
        "res_max": float(res.detach().cpu().abs().max().item()),
    }

    if problem.has_analytic():
        J_true = problem.J_star(pts).reshape(-1, 1)
        u_true = problem.u_star(pts).reshape(-1, problem.control_dim)
        J_pred = J.detach().cpu().numpy()
        u_pred = u.detach().cpu().numpy()
        out["J_mse"] = float(np.mean((J_pred - J_true) ** 2))
        out["u_mse"] = float(np.mean((u_pred - u_true) ** 2))
    else:
        out["J_mse"] = float("nan")
        out["u_mse"] = float("nan")

    return out


def plot_heatmap(x1: np.ndarray, x2: np.ndarray, Z: np.ndarray, title: str, outpath: Optional[str] = None) -> None:
    """Utility: save a simple heatmap plot (for 2D grids).

    Args:
        x1, x2: 1D grid axes.
        Z: 2D array of values on the grid.
        title: Plot title.
        outpath: If provided, save to this path; otherwise show interactively.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(
        Z,
        origin="lower",
        extent=(x1.min(), x1.max(), x2.min(), x2.max()),
        aspect="auto",
    )
    plt.colorbar()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    else:
        plt.show()
    plt.close()


# -----------------------------
# Traditional numerical HJB baseline (2D value iteration)
# -----------------------------
# Baseline: simple 2D value iteration (VI) solver.
#
# This is NOT intended to be an optimized HJB solver. It exists to provide a
# transparent grid-based baseline for small 2D problems. Complexity grows quickly
# with grid resolution and action discretization.
def value_iteration_hjb_2d(
    problem: ControlAffineProblem,
    grid_n: int = 81,
    u_grid: Optional[np.ndarray] = None,
    dt: float = 0.02,
    iters: int = 200,
) -> Dict[str, np.ndarray]:
    """
    A simple discrete-time approximation:
        V(x) = min_u { L(x,u) dt + V(x + dt f(x,u)) }
    solved by value iteration on a uniform grid (2D only).

    This is intended as a *baseline* for the 2D benchmark.
    """
    assert problem.state_dim == 2 and problem.control_dim == 1, "Baseline implemented for 2D, 1D-control."

    if u_grid is None:
        u_grid = np.linspace(-30.0, 30.0, 121).reshape(-1, 1)

    x1 = np.linspace(problem.domain.low[0], problem.domain.high[0], grid_n)
    x2 = np.linspace(problem.domain.low[1], problem.domain.high[1], grid_n)
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")
    pts = np.stack([X1.ravel(), X2.ravel()], axis=1)

    V = np.zeros((grid_n, grid_n), dtype=np.float64)
    # Greedy control (argmin u) for each grid cell.
    U = np.zeros((grid_n, grid_n, 1), dtype=np.float64)
    interp = RegularGridInterpolator((x1, x2), V, bounds_error=False, fill_value=None)

    dev = torch.device("cpu")

    for k in range(iters):
        V_new = np.zeros_like(V)
        U_new = np.zeros_like(U)
        interp.values = V  # update

        for i in range(grid_n):
            for j in range(grid_n):
                x = np.array([[x1[i], x2[j]]], dtype=np.float64)
                best_cost = float("inf")
                best_u = 0.0
                # evaluate min over u_grid
                for u in u_grid:
                    xt = torch.tensor(x, dtype=TORCH_DTYPE, device=dev)
                    ut = torch.tensor(u.reshape(1, 1), dtype=TORCH_DTYPE, device=dev)
                    f = problem.f1(xt) + torch.bmm(problem.f2(xt), ut.unsqueeze(-1)).squeeze(-1)
                    x_next = (xt + dt * f).detach().cpu().numpy().reshape(2,)
                    # RegularGridInterpolator returns an array; make it a python float robustly.
                    v_next = float(np.asarray(interp(x_next)).reshape(-1)[0])
                    stage = float(problem.running_cost(xt, ut).detach().cpu().numpy().reshape(-1)[0])
                    cost = stage * dt + v_next
                    if cost < best_cost:
                        best_cost = cost
                        best_u = float(np.asarray(u).reshape(-1)[0])

                V_new[j, i] = best_cost  # note meshgrid indexing: row=y (x2), col=x (x1)
                U_new[j, i, 0] = best_u

        diff = np.max(np.abs(V_new - V))
        V = V_new
        U = U_new
        if (k + 1) % 10 == 0 or k == 0:
            print(f"[VI] iter={k+1:4d} max|dV|={diff:.3e}")
        if diff < 1e-5:
            break

    return {"x1": x1, "x2": x2, "V": V, "U": U}

@dataclass
class PPOConfig:
    """Hyperparameters for the lightweight PPO baseline implementation.

    This PPO code is intentionally minimal (no external RL libraries) to keep
    the repository lightweight and reproducible.
    """
    seed: int = 0
    device: str = "auto"
    steps: int = 200_000
    rollout_len: int = 2048
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs_per_rollout: int = 10
    batch_size: int = 256
    hidden: Tuple[int, ...] = (64, 64)
    act: str = "tanh"
    value_coef: float = 0.5
    entropy_coef: float = 0.0


def pendulum_dynamics_continuous(x: torch.Tensor, u: torch.Tensor, g: float = 10.0, m: float = 1.0, l: float = 1.0) -> torch.Tensor:
    """
    Continuous-time pendulum dynamics in (theta, theta_dot) coordinates.
    theta_dot = omega
    omega_dot = -3g/(2l) * sin(theta + pi) + 3/(m l^2) * u

    This matches the classic gym/gymnasium pendulum formulation (up to discretization).
    """
    theta = x[:, 0]
    omega = x[:, 1]
    u = u[:, 0]
    theta_dot = omega
    omega_dot = (-3.0 * g / (2.0 * l)) * torch.sin(theta + math.pi) + (3.0 / (m * l * l)) * u
    return torch.stack([theta_dot, omega_dot], dim=1)


def pendulum_running_cost(theta: torch.Tensor, omega: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Gymnasium pendulum reward is negative cost:
      cost = angle_normalize(theta)^2 + 0.1*omega^2 + 0.001*u^2
    Here we implement the cost (positive).
    """
    def angle_normalize(x):
        return ((x + math.pi) % (2 * math.pi)) - math.pi

    th = angle_normalize(theta)
    return th**2 + 0.1 * omega**2 + 0.001 * (u**2)



class CategoricalPolicy(nn.Module):
    """Categorical policy for discrete action spaces."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (64, 64), act: str = "tanh") -> None:
        super().__init__()
        self.logits = MLP(obs_dim, act_dim, hidden=hidden, act=act)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.logits(obs)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a).unsqueeze(-1)
        return a, logp


class PPOAgentContinuous(nn.Module):
    """PPO agent for continuous-action environments.

    Uses a tanh-squashed Gaussian policy (for bounded actions) and a scalar
    value function head.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), act="tanh") -> None:
        super().__init__()
        self.pi = GaussianPolicy(obs_dim, act_dim, hidden=hidden, act=act)
        self.v = MLP(obs_dim, 1, hidden=hidden, act=act)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, logp = self.pi.sample(obs)  # a in [-1,1]
        v = self.v(obs)
        return a, logp, v


class PPOAgentDiscrete(nn.Module):
    """PPO agent for discrete-action environments.

    Uses a categorical policy and a scalar value function head.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), act="tanh") -> None:
        super().__init__()
        self.pi = CategoricalPolicy(obs_dim, act_dim, hidden=hidden, act=act)
        self.v = MLP(obs_dim, 1, hidden=hidden, act=act)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, logp = self.pi.sample(obs)  # integer actions
        v = self.v(obs)
        return a, logp, v


def gae(rews: np.ndarray, vals: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation (GAE-λ).

    Args:
        rews: Rewards r_t, length T.
        vals: Value estimates V_t, length T+1 (includes bootstrap value).
        dones: Done flags, length T.
        gamma: Discount factor.
        lam: GAE parameter λ.

    Returns:
        (advantages, returns) both length T.
    """
    T = len(rews)
    adv = np.zeros(T, dtype=np.float64)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - dones[t]
        nextvalue = vals[t + 1]
        delta = rews[t] + gamma * nextvalue * nextnonterminal - vals[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + vals[:-1]
    return adv, ret


def _atanh(x: torch.Tensor) -> torch.Tensor:
    # numerically stable atanh for |x|<1
    """Numerically stable inverse tanh for |x|<1.

    Used to compute log-probabilities for tanh-squashed Gaussian policies.
    """
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def run_ppo(
    env_id: str,
    cfg: PPOConfig,
    physics_weight: float = 0.0,
    *,
    record_history: bool = True,
) -> Dict[str, object]:
    """Train PPO on a Gymnasium environment.

    This is intentionally lightweight (no external RL libs) and is used as a baseline.
    If Gymnasium is not installed, non-RL subcommands still work; only the RL subcommands
    will raise.

    When `physics_weight > 0` and `env_id` starts with "Pendulum", an additional
    physics/HJB residual penalty is applied to the critic in observation space
    (obs = [cos θ, sin θ, ω]). The actions used in the penalty are the *buffer* actions,
    so the regularizer targets the critic (and does not backprop through the actor).
    """
    if gym is None:
        raise ImportError(
            "gymnasium is required for rl_ppo / rl_pinn_hybrid. "
            "Install via: pip install gymnasium"
        )

    set_seed(cfg.seed)
    dev = get_device(cfg.device)

    env = gym.make(env_id)

    # Flatten helpers (some Gym envs already return flat arrays, but keep it robust)
    def _flat_obs(o: np.ndarray) -> np.ndarray:
        return np.asarray(o, dtype=np.float64).reshape(-1)

    obs_space = env.observation_space
    act_space = env.action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"Only Box observation spaces are supported (got {type(obs_space)}).")

    obs_dim = int(np.prod(obs_space.shape))

    discrete = isinstance(act_space, gym.spaces.Discrete)

    act_scale_np: Optional[np.ndarray] = None
    act_bias_np: Optional[np.ndarray] = None
    act_scale_t: Optional[torch.Tensor] = None
    act_bias_t: Optional[torch.Tensor] = None

    if discrete:
        agent: nn.Module = PPOAgentDiscrete(obs_dim, int(act_space.n), hidden=cfg.hidden, act=cfg.act).to(dev)
    else:
        if not isinstance(act_space, gym.spaces.Box):
            raise TypeError(f"Only Discrete/Box action spaces are supported (got {type(act_space)}).")
        act_dim = int(np.prod(act_space.shape))
        agent = PPOAgentContinuous(obs_dim, act_dim, hidden=cfg.hidden, act=cfg.act).to(dev)

        # Map tanh actions a∈[-1,1] to env bounds: u = bias + scale * a
        act_low = np.asarray(act_space.low, dtype=np.float64).reshape(-1)
        act_high = np.asarray(act_space.high, dtype=np.float64).reshape(-1)
        act_scale_np = (act_high - act_low) / 2.0
        act_bias_np = (act_high + act_low) / 2.0
        act_scale_t = torch.tensor(act_scale_np.reshape(1, -1), dtype=TORCH_DTYPE, device=dev)
        act_bias_t = torch.tensor(act_bias_np.reshape(1, -1), dtype=TORCH_DTYPE, device=dev)

    opt = optim.Adam(agent.parameters(), lr=float(cfg.lr))

    total_steps = 0
    ep_returns: List[float] = []
    episode_history: List[Dict[str, object]] = []
    train_history: List[Dict[str, object]] = []

    t0 = time.time()

    obs, _ = env.reset(seed=int(cfg.seed))
    obs = _flat_obs(obs)

    ep_ret = 0.0
    ep_len = 0
    ep_idx = 0
    rollout_idx = 0

    while total_steps < int(cfg.steps):
        rollout_idx += 1

        # Rollout buffers
        obs_buf: List[np.ndarray] = []
        act_buf: List[object] = []
        logp_buf: List[float] = []
        rew_buf: List[float] = []
        done_buf: List[bool] = []
        val_buf: List[float] = []

        # Collect rollout
        for _ in range(int(cfg.rollout_len)):
            obs_t = torch.tensor(obs.reshape(1, -1), dtype=TORCH_DTYPE, device=dev)

            with torch.no_grad():
                a_t, logp_t, v_t = agent.act(obs_t)  # type: ignore[attr-defined]

            if discrete:
                act_env = int(a_t.cpu().item())
                act_store = act_env
            else:
                a_np = a_t.cpu().numpy()[0]  # tanh action in [-1,1]
                assert act_scale_np is not None and act_bias_np is not None
                act_env = (act_bias_np + act_scale_np * a_np).reshape(act_space.shape)  # type: ignore[union-attr]
                act_store = a_np  # store tanh action for PPO update

            next_obs, rew, terminated, truncated, _ = env.step(act_env)  # type: ignore[arg-type]
            done = bool(terminated or truncated)

            obs_buf.append(obs.copy())
            act_buf.append(act_store)
            logp_buf.append(float(logp_t.cpu().item()))
            rew_buf.append(float(rew))
            done_buf.append(done)
            val_buf.append(float(v_t.cpu().item()))

            ep_ret += float(rew)
            ep_len += 1
            total_steps += 1

            obs = _flat_obs(next_obs)

            if done:
                ep_returns.append(ep_ret)
                if record_history:
                    episode_history.append({"episode_idx": ep_idx, "steps": ep_len, "return": ep_ret})
                ep_idx += 1
                ep_ret = 0.0
                ep_len = 0

                obs, _ = env.reset()
                obs = _flat_obs(obs)

            if total_steps >= int(cfg.steps):
                break

        # Bootstrap value for GAE
        obs_t = torch.tensor(obs.reshape(1, -1), dtype=TORCH_DTYPE, device=dev)
        with torch.no_grad():
            _, _, v_last = agent.act(obs_t)  # type: ignore[attr-defined]
        v_last_f = float(v_last.cpu().item())

        # GAE(λ)
        T = len(rew_buf)
        adv = np.zeros(T, dtype=np.float64)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 0.0 if done_buf[t] else 1.0
            next_value = v_last_f if t == T - 1 else float(val_buf[t + 1])
            delta = float(rew_buf[t]) + float(cfg.gamma) * next_value * next_nonterminal - float(val_buf[t])
            lastgaelam = delta + float(cfg.gamma) * float(cfg.lam) * next_nonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + np.asarray(val_buf, dtype=np.float64)

        # Normalize advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Convert buffers to tensors
        obs_t = torch.tensor(np.asarray(obs_buf), dtype=TORCH_DTYPE, device=dev)
        ret_t = torch.tensor(returns.reshape(-1, 1), dtype=TORCH_DTYPE, device=dev)
        adv_t = torch.tensor(adv.reshape(-1, 1), dtype=TORCH_DTYPE, device=dev)
        old_logp_t = torch.tensor(np.asarray(logp_buf).reshape(-1, 1), dtype=TORCH_DTYPE, device=dev)

        if discrete:
            act_t = torch.tensor(np.asarray(act_buf, dtype=np.int64).reshape(-1), dtype=torch.long, device=dev)
        else:
            act_t = torch.tensor(np.asarray(act_buf, dtype=np.float64), dtype=TORCH_DTYPE, device=dev).reshape(T, -1)

        # PPO update
        agent.train()
        n = int(obs_t.shape[0])
        idxs = np.arange(n)

        for _ in range(int(cfg.epochs_per_rollout)):
            np.random.shuffle(idxs)
            for start in range(0, n, int(cfg.batch_size)):
                mb = idxs[start : start + int(cfg.batch_size)]
                if mb.size == 0:
                    continue

                obs_mb = obs_t[mb]
                ret_mb = ret_t[mb]
                adv_mb = adv_t[mb]
                old_logp_mb = old_logp_t[mb]

                if discrete:
                    act_mb = act_t[mb]
                    logits = agent.pi(obs_mb)  # type: ignore[attr-defined]
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(act_mb).unsqueeze(-1)
                    entropy = dist.entropy().unsqueeze(-1)
                    v = agent.v(obs_mb)  # type: ignore[attr-defined]
                else:
                    act_mb = act_t[mb]
                    logp = agent.pi.log_prob(obs_mb, act_mb)  # type: ignore[attr-defined]
                    entropy = agent.pi.entropy(obs_mb)  # type: ignore[attr-defined]
                    v = agent.v(obs_mb)  # type: ignore[attr-defined]

                ratio = torch.exp(logp - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - float(cfg.clip), 1.0 + float(cfg.clip)) * adv_mb
                loss_pi = -torch.mean(torch.min(surr1, surr2))

                loss_v = 0.5 * torch.mean((v - ret_mb) ** 2)
                loss_ent = -float(cfg.entropy_coef) * torch.mean(entropy)

                loss = loss_pi + float(cfg.value_coef) * loss_v + loss_ent

                # Optional physics residual regularization (Pendulum-v1 only)
                if (
                    float(physics_weight) > 0.0
                    and (not discrete)
                    and env_id.lower().startswith("pendulum")
                    and obs_mb.shape[1] == 3
                    and act_mb.shape[1] == 1
                    and act_scale_t is not None
                    and act_bias_t is not None
                ):
                    obs_req = obs_mb.detach().requires_grad_(True)
                    J = agent.v(obs_req)  # type: ignore[attr-defined]
                    dJ = grad_wrt_x(J, obs_req)

                    theta = torch.atan2(obs_req[:, 1], obs_req[:, 0])
                    omega = obs_req[:, 2]

                    # Buffer actions are in [-1,1]; map to env units for the residual.
                    u_env = act_bias_t + act_scale_t * act_mb  # (B,1)

                    x_state = torch.stack([theta, omega], dim=1)
                    f_state = pendulum_dynamics_continuous(x_state, u_env)
                    theta_dot = f_state[:, 0]
                    omega_dot = f_state[:, 1]

                    dcos = -torch.sin(theta) * theta_dot
                    dsin = torch.cos(theta) * theta_dot
                    domega = omega_dot
                    f_obs = torch.stack([dcos, dsin, domega], dim=1)

                    cost = pendulum_running_cost(theta, omega, u_env[:, 0])
                    res = cost + (dJ * f_obs).sum(dim=1)
                    loss_phys = torch.mean(res**2)
                    loss = loss + float(physics_weight) * loss_phys

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

        agent.eval()

        if record_history:
            avg_last20 = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
            train_history.append(
                {
                    "rollout": int(rollout_idx),
                    "total_steps": int(total_steps),
                    "avg_return_last20": avg_last20,
                    "time_sec": float(time.time() - t0),
                }
            )

    env.close()

    avg_return_last20 = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
    out: Dict[str, object] = {
        "avg_return_last20": avg_return_last20,
        "n_episodes": int(len(ep_returns)),
        "time_sec": float(time.time() - t0),
        "ep_returns": ep_returns,
    }
    if record_history:
        out["episode_history"] = episode_history
        out["train_history"] = train_history
    return out
# -----------------------------
# Orchestration helpers
# -----------------------------
# Helper functions that wire together training, evaluation, sweeps, and baselines.
# These are the functions you would most likely modify if you:
#   - add a new benchmark problem
#   - change how metrics are computed
#   - change how the 'best' tuned configuration is selected
def build_problem(args: argparse.Namespace) -> ControlAffineProblem:
    """Factory for continuous-time control problems used by CPINN/benchmarks."""
    if args.problem == "nonlinear2d":
        return Nonlinear2DAnalytic(low=args.low, high=args.high)

    if args.problem == "lqr":
        n = int(getattr(args, "state_dim", 4))
        m = int(getattr(args, "control_dim", 2))
        return LQRND(n=n, m=m, low=args.low, high=args.high, seed=args.seed)

    if args.problem == "cubic_nd":
        n = int(getattr(args, "state_dim", 4))
        low = float(getattr(args, "cubic_low", -2.0))
        high = float(getattr(args, "cubic_high", 2.0))
        return CubicNDAnalytic(n=n, low=low, high=high)

    if args.problem == "cartpole_lqr":
        Qdiag = getattr(args, "cartpole_Q", [1.0, 0.1, 10.0, 0.1])
        if Qdiag is None or len(Qdiag) != 4:
            Qdiag = [1.0, 0.1, 10.0, 0.1]
        return CartPoleLQR(
            x_low=float(getattr(args, "cartpole_x_low", -2.4)),
            x_high=float(getattr(args, "cartpole_x_high", 2.4)),
            xdot_low=float(getattr(args, "cartpole_xdot_low", -3.0)),
            xdot_high=float(getattr(args, "cartpole_xdot_high", 3.0)),
            theta_low=float(getattr(args, "cartpole_theta_low", -0.5)),
            theta_high=float(getattr(args, "cartpole_theta_high", 0.5)),
            thetadot_low=float(getattr(args, "cartpole_thetadot_low", -3.0)),
            thetadot_high=float(getattr(args, "cartpole_thetadot_high", 3.0)),
            M=float(getattr(args, "cartpole_M", 1.0)),
            m=float(getattr(args, "cartpole_m", 0.1)),
            l=float(getattr(args, "cartpole_l", 0.5)),
            g=float(getattr(args, "cartpole_g", 9.8)),
            b=float(getattr(args, "cartpole_b", 0.0)),
            I=getattr(args, "cartpole_I", None),
            Q_diag=(float(Qdiag[0]), float(Qdiag[1]), float(Qdiag[2]), float(Qdiag[3])),
            R_scalar=float(getattr(args, "cartpole_R", 0.1)),
        )


    if args.problem == "pendulum":
        return PendulumVI(
            theta_low=float(getattr(args, "pendulum_theta_low", -math.pi)),
            theta_high=float(getattr(args, "pendulum_theta_high", math.pi)),
            omega_low=float(getattr(args, "pendulum_omega_low", -8.0)),
            omega_high=float(getattr(args, "pendulum_omega_high", 8.0)),
            u_max=float(getattr(args, "pendulum_u_max", 2.0)),
            g=float(getattr(args, "pendulum_g", 10.0)),
            m=float(getattr(args, "pendulum_m", 1.0)),
            l=float(getattr(args, "pendulum_l", 1.0)),
            vi_grid=int(getattr(args, "pendulum_vi_grid", 61)),
            vi_iters=int(getattr(args, "pendulum_vi_iters", 200)),
            vi_dt=float(getattr(args, "pendulum_vi_dt", 0.02)),
            vi_u_points=int(getattr(args, "pendulum_vi_u_points", 41)),
            cache_dir=str(getattr(args, "outdir", "outputs")),
            cache_tag=str(getattr(args, "pendulum_cache_tag", "")),
            force_recompute=bool(int(getattr(args, "pendulum_vi_force", 0))),
        )

    if args.problem == "pendulum_lqr":
        return PendulumLQR(
            theta_low=float(getattr(args, "pendulum_theta_low", -math.pi)),
            theta_high=float(getattr(args, "pendulum_theta_high", math.pi)),
            omega_low=float(getattr(args, "pendulum_omega_low", -8.0)),
            omega_high=float(getattr(args, "pendulum_omega_high", 8.0)),
            g=float(getattr(args, "pendulum_g", 10.0)),
            m=float(getattr(args, "pendulum_m", 1.0)),
            l=float(getattr(args, "pendulum_l", 1.0)),
        )

    raise ValueError(f"Unknown problem: {args.problem}")


def train_cpinn_ensemble(
    problem: ControlAffineProblem,
    cfg: TrainConfig,
    ensemble: int,
    *,
    run_id: str = "",
    history: Optional[List[Dict[str, object]]] = None,
) -> Tuple[List[nn.Module], Dict[str, float], List[Dict[str, float]]]:
    """Train an ensemble of CPINN value networks.

    Implementation detail (important for reproducibility):
      - We warm-start *one* base model (if enabled) and then clone its weights to all
        ensemble members. Each member then receives a tiny random perturbation so the
        ensemble does not collapse to identical solutions.

    Args:
        problem: Benchmark problem instance.
        cfg: Training configuration.
        ensemble: Number of ensemble members.
        run_id: Optional run identifier for logging.
        history: Optional per-epoch history list.

    Returns:
        (nets, warm_stats, member_train_stats)
    """
    set_seed(cfg.seed)
    dev = get_device(cfg.device)
    nets: List[nn.Module] = []
    member_train_stats: List[Dict[str, float]] = []

    # Warm-start a single model and clone weights for consistency (matches manuscript style).
    base = build_value_net(problem, cfg).to(dev)
    warm_stats: Dict[str, float] = {"warm_mse": float("nan")}
    if cfg.use_warm_start:
        warm_stats = warm_start_train(problem, base, cfg, history=history, run_id=run_id, member=-1)

    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    for j in range(ensemble):
        net = build_value_net(problem, cfg).to(dev)
        net.load_state_dict(base_state, strict=True)
        # small random perturbation to diversify ensemble
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1e-3 * torch.randn_like(p))
        print(f"\n=== Training ensemble member {j+1}/{ensemble} ===")
        st = hjb_train(problem, net, cfg, history=history, run_id=run_id, member=j)
        member_train_stats.append(st)
        nets.append(net)

    return nets, warm_stats, member_train_stats


# -----------------------------
# Sweep utilities: run IDs, CSV I/O, evaluation, and plotting
# -----------------------------
# This section contains small I/O and bookkeeping helpers:
#   - deterministic run_id strings for filesystem organization
#   - CSV/JSON writers for results and histories
#   - evaluation aggregations across ensemble members and seeds
#   - plotting helpers used to generate sweep/tune diagnostics
def make_run_id(
    prefix: str,
    problem_name: str,
    alpha: float,
    use_warm: bool,
    use_boundary: bool,
    seed: int,
    *,
    alpha_schedule: str = "constant",
    alpha_start: Optional[float] = None,
    alpha_start_factor: float = 1.0,
    alpha_hold_frac: float = 0.0,
    alpha_ramp_frac: float = 1.0,
) -> str:
    """Create a human- and filesystem-friendly run identifier string.

    The run_id encodes the most important hyperparameters (alpha, schedule, warm-start,
    boundary conditioning, seed) plus a timestamp. It is used to name output files.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    # Keep run_id filesystem-friendly
    a_str = f"{alpha:g}".replace("-", "m").replace(".", "p")

    sched = _normalize_alpha_schedule(alpha_schedule)
    if sched == "constant":
        sched_tag = "const"
    else:
        if alpha_start is not None:
            # Encode start relative to alpha end when possible
            sf = (float(alpha_start) / float(alpha)) if float(alpha) != 0.0 else float(alpha_start)
        else:
            sf = float(alpha_start_factor)

        sf_str = f"{sf:g}".replace("-", "m").replace(".", "p")
        sched_tag = f"{sched[:3]}sf{sf_str}"

        if abs(float(alpha_hold_frac)) > 1e-12:
            h_str = f"{float(alpha_hold_frac):g}".replace("-", "m").replace(".", "p")
            sched_tag += f"h{h_str}"
        if abs(float(alpha_ramp_frac) - 1.0) > 1e-12:
            r_str = f"{float(alpha_ramp_frac):g}".replace("-", "m").replace(".", "p")
            sched_tag += f"r{r_str}"

    return f"{prefix}_{problem_name}_a{a_str}_asch{sched_tag}_warm{int(use_warm)}_b{int(use_boundary)}_seed{seed}_{stamp}"


def _safe_makedirs(path: str) -> None:
    """Create a directory if it does not already exist (like `mkdir -p`)."""
    os.makedirs(path, exist_ok=True)


def write_rows_csv(path: str, rows: List[Dict[str, object]]) -> None:
    """Write a list of dictionaries to a CSV file.

    The CSV column set is the union of keys across all rows.
    """
    if not rows:
        return
    _safe_makedirs(os.path.dirname(path) or ".")
    # Union of keys, stable order
    fieldnames: List[str] = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: str, obj: object) -> None:
    """Write an object to JSON (pretty-printed, stable key order)."""
    _safe_makedirs(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def group_mean_std(
    rows: List[Dict[str, object]],
    *,
    key_fields: List[str],
    metric_fields: List[str],
) -> List[Dict[str, object]]:
    """Group rows by key_fields and compute mean/std over metric_fields.

    Assumes metrics are numeric (float/int) or NaN-like.
    """

    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for r in rows:
        key = tuple(r.get(k) for k in key_fields)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for key, grp in groups.items():
        row: Dict[str, object] = {k: v for k, v in zip(key_fields, key)}
        row["n"] = len(grp)
        for m in metric_fields:
            vals: List[float] = []
            for g in grp:
                v = g.get(m)
                if v is None:
                    continue
                try:
                    fv = float(v)  # type: ignore[arg-type]
                except Exception:
                    continue
                if not np.isnan(fv):
                    vals.append(fv)
            row[m + "_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[m + "_std"] = float(np.std(vals)) if vals else float("nan")
        out.append(row)

    # Deterministic ordering: sort by keys (best effort)
    def _keyfun(r: Dict[str, object]) -> Tuple:
        return tuple(r.get(k) for k in key_fields)

    out.sort(key=_keyfun)
    return out


def evaluate_one(
    problem: ControlAffineProblem,
    net: nn.Module,
    *,
    device: str,
    grid_n_2d: int = 121,
    n_eval: int = 5000,
) -> Dict[str, float]:
    """Return scalar evaluation metrics for a single network.

    - For 2D problems: uses a uniform grid (grid_n_2d x grid_n_2d)
    - For nD problems: uses Monte-Carlo samples (n_eval)
    """
    if problem.state_dim == 2:
        grid = evaluate_on_grid_2d(problem, net, grid_n=grid_n_2d, device=device)
        return {
            "J_mse": float(grid.get("J_mse", float("nan"))),
            "u_mse": float(grid.get("u_mse", float("nan"))),
            "res_rms": float(grid["res_rms"]),
            "res_max": float(grid["res_max"]),
            "res_mean": float(grid["res_mean"]),
        }
    # Generic nD fallback
    out = evaluate_on_samples(problem, net, n=n_eval, device=device)
    return {
        "J_mse": float(out.get("J_mse", float("nan"))),
        "u_mse": float(out.get("u_mse", float("nan"))),
        "res_rms": float(out["res_rms"]),
        "res_max": float(out["res_max"]),
        "res_mean": float(out["res_mean"]),
    }


def evaluate_ensemble(
    problem: ControlAffineProblem,
    nets: List[nn.Module],
    *,
    device: str,
    grid_n_2d: int = 121,
    n_eval: int = 5000,
    eval_all_members: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Evaluate each ensemble member (optionally) and return aggregate statistics.

    Returns:
        member_rows: per-member metrics
        summary: mean/std across members for the key metrics
    """
    idxs = list(range(len(nets))) if eval_all_members else [0]
    member_rows: List[Dict[str, object]] = []
    for mi in idxs:
        m = evaluate_one(problem, nets[mi], device=device, grid_n_2d=grid_n_2d, n_eval=n_eval)
        member_rows.append({"member": mi, **m})

    # aggregate
    def _col(name: str) -> List[float]:
        xs: List[float] = []
        for r in member_rows:
            v = float(r.get(name, float("nan")))
            xs.append(v)
        return xs

    summary: Dict[str, float] = {}
    for k in ["J_mse", "u_mse", "res_rms", "res_max", "res_mean"]:
        xs = np.array(_col(k), dtype=float)
        summary[k + "_mean"] = float(np.nanmean(xs))
        summary[k + "_std"] = float(np.nanstd(xs))
    return member_rows, summary


def evaluate_actor_critic(
    problem: ControlAffineProblem,
    critic: nn.Module,
    actor: nn.Module,
    *,
    device: str,
    grid_n_2d: int = 121,
    n_eval: int = 5000,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Evaluate an actor-critic pair.

    - J metrics computed on critic(x)
    - u metrics computed on actor(x)
    - residual computed using actor in the Hamiltonian

    Returns a dict with keys: J_mse, u_mse, res_rms, res_max, res_mean
    """
    dev = get_device(device)
    critic = critic.to(dev)
    actor = actor.to(dev)
    critic.eval()
    actor.eval()

    # Build evaluation points
    if problem.state_dim == 2:
        low = np.asarray(problem.domain.low, dtype=np.float64)
        high = np.asarray(problem.domain.high, dtype=np.float64)
        x1 = np.linspace(low[0], high[0], int(grid_n_2d))
        x2 = np.linspace(low[1], high[1], int(grid_n_2d))
        X1, X2 = np.meshgrid(x1, x2, indexing="ij")
        pts_np = np.stack([X1.reshape(-1), X2.reshape(-1)], axis=1).astype(np.float64)
    else:
        pts_np = problem.sample_states(int(n_eval), boundary=False).astype(np.float64)

    # Compute predictions + residuals in batches
    J_preds: List[np.ndarray] = []
    u_preds: List[np.ndarray] = []
    res_vals: List[np.ndarray] = []

    n_pts = int(pts_np.shape[0])
    bs = int(batch_size)
    for i in range(0, n_pts, bs):
        xb = torch.tensor(pts_np[i : i + bs], dtype=TORCH_DTYPE, device=dev)

        # predictions (no grad)
        with torch.no_grad():
            Jb = critic(xb).squeeze(-1)
            ub = actor(xb)

        # residual needs ∇J, so recompute with grad enabled
        xb_req = xb.detach().requires_grad_(True)
        res_b, _ = hjb_residual_actor(problem, critic, actor, xb_req)

        J_preds.append(Jb.detach().cpu().numpy())
        u_preds.append(ub.detach().cpu().numpy())
        res_vals.append(res_b.detach().cpu().numpy())

    J_pred = np.concatenate(J_preds, axis=0)
    u_pred = np.concatenate(u_preds, axis=0)
    res = np.concatenate(res_vals, axis=0)

    out: Dict[str, float] = {}
    out["res_rms"] = float(np.sqrt(np.mean(res**2)))
    out["res_max"] = float(np.max(np.abs(res)))
    out["res_mean"] = float(np.mean(np.abs(res)))

    if problem.has_analytic():
        J_true = problem.J_star(pts_np).reshape(-1)
        u_true = problem.u_star(pts_np)
        out["J_mse"] = float(np.mean((J_pred.reshape(-1) - J_true) ** 2))
        out["u_mse"] = float(np.mean((u_pred - u_true) ** 2))
    else:
        out["J_mse"] = float("nan")
        out["u_mse"] = float("nan")

    return out


def plot_loss_curves_for_run(
    history_rows: List[Dict[str, object]],
    *,
    run_id: str,
    outdir: str,
    title_prefix: str = "",
    alpha: Optional[float] = None,
    boundary_weight: Optional[float] = None,
    yscale: str = "log",
) -> None:
    """Create loss curve plots from history rows for a single run_id."""
    rows = [r for r in history_rows if r.get("run_id") == run_id]
    if not rows:
        return

    _safe_makedirs(outdir)

    def _plot_phase(phase: str, ykey: str, fname: str, members: Optional[List[int]] = None, yscale_override: Optional[str] = None) -> None:
        rs = [r for r in rows if r.get("phase") == phase]
        if members is not None:
            rs = [r for r in rs if int(r.get("member", -999)) in members]
        if not rs:
            return

        # group by member
        mems = sorted({int(r.get("member", -1)) for r in rs})
        plt.figure(figsize=(6, 4))
        for m in mems:
            rsm = [r for r in rs if int(r.get("member", -1)) == m]
            rsm.sort(key=lambda r: int(r.get("epoch", 0)))
            xs = [int(r.get("epoch", 0)) for r in rsm]
            ys = [float(r.get(ykey, float("nan"))) for r in rsm]
            plt.plot(xs, ys, label=f"m{m}")

        # mean/std across members (if multiple)
        if len(mems) > 1:
            # Assume all members share the same epoch list (true when record_every is fixed)
            by_epoch: Dict[int, List[float]] = {}
            for r in rs:
                ep = int(r.get("epoch", 0))
                by_epoch.setdefault(ep, []).append(float(r.get(ykey, float("nan"))))
            eps = sorted(by_epoch.keys())
            mu = [float(np.nanmean(by_epoch[e])) for e in eps]
            sig = [float(np.nanstd(by_epoch[e])) for e in eps]
            plt.plot(eps, mu, linewidth=2.5, label="mean")
            plt.fill_between(eps, np.maximum(1e-12, np.array(mu) - np.array(sig)), np.array(mu) + np.array(sig), alpha=0.2)

        ttl = f"{title_prefix}{phase} {ykey}"
        if alpha is not None:
            ttl += f" (alpha={alpha:g})"
        plt.title(ttl)
        plt.xlabel("epoch")
        plt.ylabel(ykey)
        ysc = yscale_override if yscale_override is not None else yscale
        if ysc:
            plt.yscale(ysc)
        if len(mems) <= 10:
            plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close()

    # Warm-start curve (member=-1)
    _plot_phase("warm", "loss_total", f"{run_id}_warm_mse.png", members=[-1])

    # HJB curves (members >= 0)
    hjb_members = sorted({int(r.get("member", -1)) for r in rows if r.get("phase") == "hjb" and int(r.get("member", -1)) >= 0})
    if hjb_members:
        _plot_phase("hjb", "loss_total", f"{run_id}_hjb_loss_total.png", members=hjb_members)
        _plot_phase("hjb", "loss_b", f"{run_id}_hjb_loss_b.png", members=hjb_members)
        _plot_phase("hjb", "loss_hjb", f"{run_id}_hjb_loss_hjb.png", members=hjb_members)
        # Alpha schedule (if present in history)
        if any(r.get("phase") == "hjb" and r.get("alpha_t") is not None and not (isinstance(r.get("alpha_t"), float) and np.isnan(r.get("alpha_t"))) for r in rows):
            _plot_phase("hjb", "alpha_t", f"{run_id}_alpha_t.png", members=hjb_members, yscale_override="linear")

        # Plot the weighted loss terms to visualize alpha vs boundary term dominance
        if alpha is not None and boundary_weight is not None:
            rs = [r for r in rows if r.get("phase") == "hjb" and int(r.get("member", -1)) in hjb_members]
            by_epoch_b: Dict[int, List[float]] = {}
            by_epoch_h: Dict[int, List[float]] = {}
            for r in rs:
                ep = int(r.get("epoch", 0))
                by_epoch_b.setdefault(ep, []).append(boundary_weight * float(r.get("loss_b", float("nan"))))
                a_t = float(r.get("alpha_t", alpha if alpha is not None else float("nan")))
                by_epoch_h.setdefault(ep, []).append(a_t * float(r.get("loss_hjb", float("nan"))))
            eps = sorted(set(by_epoch_b.keys()) & set(by_epoch_h.keys()))
            if eps:
                b_mu = [float(np.nanmean(by_epoch_b[e])) for e in eps]
                h_mu = [float(np.nanmean(by_epoch_h[e])) for e in eps]
                plt.figure(figsize=(6, 4))
                plt.plot(eps, b_mu, label="boundary_weight * loss_b")
                plt.plot(eps, h_mu, label="alpha * loss_hjb")
                plt.title(f"{title_prefix}HJB loss terms (alpha={alpha:g})")
                plt.xlabel("epoch")
                plt.ylabel("weighted term")
                if yscale:
                    plt.yscale(yscale)
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{run_id}_hjb_loss_terms.png"), dpi=200)
                plt.close()



def plot_rl_curves(stats: Dict[str, object], *, run_id: str, outdir: str, title_prefix: str = "") -> None:
    """Save basic PPO training curves (episode returns and update losses)."""
    _safe_makedirs(outdir)

    episode_history = stats.get("episode_history", [])
    train_history = stats.get("train_history", [])

    # Episode returns
    if isinstance(episode_history, list) and episode_history:
        xs = [int(r.get("episode", i + 1)) for i, r in enumerate(episode_history)]
        ys = [float(r.get("return", float("nan"))) for r in episode_history]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys)
        plt.title(f"{title_prefix}Episode return")
        plt.xlabel("episode")
        plt.ylabel("return")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{run_id}_episode_returns.png"), dpi=200)
        plt.close()

    # Avg return vs steps and losses vs steps
    if isinstance(train_history, list) and train_history:
        steps = [int(r.get("steps", 0)) for r in train_history]
        avg_ret = [float(r.get("avg_return_last20", float("nan"))) for r in train_history]

        plt.figure(figsize=(6, 4))
        plt.plot(steps, avg_ret)
        plt.title(f"{title_prefix}Avg return (last20) vs steps")
        plt.xlabel("environment steps")
        plt.ylabel("avg_return_last20")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{run_id}_avg_return_last20.png"), dpi=200)
        plt.close()

        # Losses (pi, v, total, phys if present)
        loss_pi = [float(r.get("loss_pi", float("nan"))) for r in train_history]
        loss_v = [float(r.get("loss_v", float("nan"))) for r in train_history]
        loss_total = [float(r.get("loss_total", float("nan"))) for r in train_history]
        loss_phys = [float(r.get("loss_phys", float("nan"))) for r in train_history]

        plt.figure(figsize=(6, 4))
        plt.plot(steps, loss_total, label="loss_total")
        plt.plot(steps, loss_pi, label="loss_pi")
        plt.plot(steps, loss_v, label="loss_v")
        # Only plot phys if at least one finite value exists
        if any(np.isfinite(loss_phys)):
            plt.plot(steps, loss_phys, label="loss_phys")
        plt.title(f"{title_prefix}PPO losses vs steps")
        plt.xlabel("environment steps")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{run_id}_losses.png"), dpi=200)
        plt.close()



def main() -> None:
    """CLI entry point.

    This function wires together the argparse interface and dispatches to the
    appropriate subcommand implementation.
    """
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(s):
        s.add_argument("--problem", choices=["nonlinear2d", "cubic_nd", "lqr", "cartpole_lqr", "pendulum", "pendulum_lqr"], default="nonlinear2d")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--device", type=str, default="auto")
        s.add_argument("--low", type=float, default=-10.0)
        s.add_argument("--high", type=float, default=10.0)

        # Dimensions (used by --problem lqr and cubic_nd)
        s.add_argument("--state_dim", type=int, default=4, help="State dimension for LQR/cubic_nd")
        s.add_argument("--control_dim", type=int, default=2, help="Control dimension for LQR (ignored for cubic_nd/cartpole_lqr)")

        # CubicNDAnalytic domain (overrides --low/--high for --problem cubic_nd)
        s.add_argument("--cubic_low", type=float, default=-2.0)
        s.add_argument("--cubic_high", type=float, default=2.0)

        # CartPole LQR (linearized) options (only used for --problem cartpole_lqr)
        s.add_argument("--cartpole_x_low", type=float, default=-2.4)
        s.add_argument("--cartpole_x_high", type=float, default=2.4)
        s.add_argument("--cartpole_xdot_low", type=float, default=-3.0)
        s.add_argument("--cartpole_xdot_high", type=float, default=3.0)
        s.add_argument("--cartpole_theta_low", type=float, default=-0.5)
        s.add_argument("--cartpole_theta_high", type=float, default=0.5)
        s.add_argument("--cartpole_thetadot_low", type=float, default=-3.0)
        s.add_argument("--cartpole_thetadot_high", type=float, default=3.0)
        s.add_argument("--cartpole_M", type=float, default=1.0, help="Cart mass")
        s.add_argument("--cartpole_m", type=float, default=0.1, help="Pole mass")
        s.add_argument("--cartpole_l", type=float, default=0.5, help="Pole COM distance (matches Gymnasium 'length')")
        s.add_argument("--cartpole_g", type=float, default=9.8)
        s.add_argument("--cartpole_b", type=float, default=0.0, help="Cart friction coefficient")
        s.add_argument("--cartpole_I", type=float, default=None, help="Pole inertia about COM (default: uniform rod)")
        s.add_argument("--cartpole_Q", type=float, nargs=4, default=[1.0, 0.1, 10.0, 0.1], help="Diagonal of Q (4 values)")
        s.add_argument("--cartpole_R", type=float, default=0.1, help="Scalar R")

        # Pendulum-specific knobs (only used when --problem pendulum or pendulum_lqr)
        s.add_argument("--pendulum_theta_low", type=float, default=-math.pi)
        s.add_argument("--pendulum_theta_high", type=float, default=math.pi)
        s.add_argument("--pendulum_omega_low", type=float, default=-8.0)
        s.add_argument("--pendulum_omega_high", type=float, default=8.0)
        s.add_argument("--pendulum_u_max", type=float, default=2.0)
        s.add_argument("--pendulum_g", type=float, default=10.0)
        s.add_argument("--pendulum_m", type=float, default=1.0)
        s.add_argument("--pendulum_l", type=float, default=1.0)
        s.add_argument("--pendulum_vi_grid", type=int, default=61)
        s.add_argument("--pendulum_vi_iters", type=int, default=200)
        s.add_argument("--pendulum_vi_dt", type=float, default=0.02)
        s.add_argument("--pendulum_vi_u_points", type=int, default=41)
        s.add_argument("--pendulum_vi_force", type=int, default=0, help="Force recompute of pendulum VI reference (ignores cache)")
        s.add_argument("--pendulum_cache_tag", type=str, default="", help="Extra tag to distinguish cached pendulum VI references")

        s.add_argument("--alpha", type=float, default=0.1)

        # Alpha scheduling (applies during HJB training)
        s.add_argument(
            "--alpha_schedule",
            type=str,
            default="constant",
            choices=["constant", "const", "linear", "lin", "cosine", "cos", "exp", "exponential", "step"],
            help="Alpha schedule during HJB training; --alpha sets alpha_end",
        )
        s.add_argument(
            "--alpha_start",
            type=float,
            default=None,
            help="Optional absolute alpha_start (overrides --alpha_start_factor)",
        )
        s.add_argument(
            "--alpha_start_factor",
            type=float,
            default=1.0,
            help="When --alpha_start is not set: alpha_start = alpha_end * alpha_start_factor",
        )
        s.add_argument(
            "--alpha_hold_frac",
            type=float,
            default=0.0,
            help="Fraction of HJB epochs to hold at alpha_start before ramp",
        )
        s.add_argument(
            "--alpha_ramp_frac",
            type=float,
            default=1.0,
            help="Fraction of HJB epochs used for ramp alpha_start -> alpha_end",
        )

        # Support both flag names; users often type --warm_epochs
        s.add_argument("--epochs_warm", "--warm_epochs", type=int, default=2000)
        s.add_argument("--epochs_hjb", type=int, default=2000)
        s.add_argument("--batch_interior", type=int, default=512)
        s.add_argument("--batch_boundary", type=int, default=512)
        s.add_argument("--use_warm_start", type=int, default=1)
        s.add_argument("--use_boundary_loss", type=int, default=1)
        s.add_argument("--boundary_weight", type=float, default=1.0)
        s.add_argument("--lr_warm", type=float, default=1e-3)
        s.add_argument("--lr_hjb", type=float, default=1e-4)
        s.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
        s.add_argument("--act", type=str, default="tanh")
        s.add_argument("--print_every", type=int, default=200)
        s.add_argument("--record_every", type=int, default=1, help="Record losses every N epochs for CSV/plots")
        s.add_argument("--make_plots", type=int, default=0)
        s.add_argument(
            "--plot_runs",
            type=str,
            choices=["none", "best", "all"],
            default="best",
            help="For sweeps: which runs to plot loss curves for (requires --make_plots 1)",
        )
        s.add_argument("--save_csv", type=int, default=1, help="Save results/summary tables to CSV")
        s.add_argument("--save_history", type=int, default=1, help="Save per-epoch loss history to CSV")
        s.add_argument("--outdir", type=str, default="outputs")

    # train_cpinn
    s_train = sub.add_parser("train_cpinn", help="Train CPINN (single or ensemble) on a continuous-time problem")
    add_common(s_train)
    s_train.add_argument("--ensemble", type=int, default=1)
    s_train.add_argument("--grid_n_eval", type=int, default=201, help="2D eval grid size (per axis)")
    s_train.add_argument("--n_eval", type=int, default=5000, help="nD eval samples (fallback when state_dim != 2)")
    s_train.add_argument("--eval_all_members", type=int, default=1, help="Evaluate all ensemble members (1) or only first (0)")

    # ablation
    s_ab = sub.add_parser("ablation", help="Run core stability ablations")
    add_common(s_ab)
    s_ab.add_argument("--ensemble", type=int, default=5)
    s_ab.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional list of seeds; overrides --seed")
    s_ab.add_argument("--grid_n_eval", type=int, default=121, help="2D eval grid size (per axis)")
    s_ab.add_argument("--n_eval", type=int, default=5000, help="nD eval samples (fallback when state_dim != 2)")
    s_ab.add_argument("--eval_all_members", type=int, default=1, help="Evaluate all ensemble members (1) or only first (0)")
    # --epochs_hjb is already defined in add_common(); override the *default* for this subcommand.
    s_ab.set_defaults(epochs_hjb=1200)
    s_ab.set_defaults(plot_runs="all")

    # alpha_sweep
    s_al = sub.add_parser("alpha_sweep", help="Sensitivity sweep over alpha")
    add_common(s_al)
    s_al.add_argument("--alphas", type=float, nargs="+", required=True)
    s_al.add_argument("--ensemble", type=int, default=1)
    s_al.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional list of seeds; overrides --seed")
    s_al.add_argument("--grid_n_eval", type=int, default=121, help="2D eval grid size (per axis)")
    s_al.add_argument("--n_eval", type=int, default=5000, help="nD eval samples (fallback when state_dim != 2)")
    s_al.add_argument("--eval_all_members", type=int, default=1, help="Evaluate all ensemble members (1) or only first (0)")
    # --epochs_hjb is already defined in add_common(); override the *default* for this subcommand.
    s_al.set_defaults(epochs_hjb=1200)

    # tune (alpha x ablation)
    s_tune = sub.add_parser("tune", help="Joint sweep over alpha and ablation toggles (warm start, boundary loss)")
    add_common(s_tune)
    s_tune.add_argument("--alphas", type=float, nargs="+", required=True)
    s_tune.add_argument("--alpha_schedules", type=str, nargs="+", default=None, help="Optional list of alpha schedule specs to sweep (e.g., constant linear@0.1 cosine@0.01 exp@0.1 step@0.1). Use @ to set alpha_start_factor relative to each alpha.")
    s_tune.add_argument("--ensemble", type=int, default=3)
    s_tune.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional list of seeds; overrides --seed")
    s_tune.add_argument("--grid_n_eval", type=int, default=121, help="2D eval grid size (per axis)")
    s_tune.add_argument("--n_eval", type=int, default=5000, help="nD eval samples (fallback when state_dim != 2)")
    s_tune.add_argument("--eval_all_members", type=int, default=1, help="Evaluate all ensemble members (1) or only first (0)")
    s_tune.add_argument(
        "--rank_by",
        type=str,
        default="auto",
        choices=["auto", "J_mse", "u_mse", "res_rms", "res_max"],
        help="Metric used to select the best configuration",
    )

    # study (tune + baselines + optional RL)
    s_study = sub.add_parser(
        "study",
        help="Run CPINN tune (alpha x ablation x sched) then baselines (TFC/Bellman/VI) and optional RL, and write a comparison CSV.",
    )
    add_common(s_study)
    s_study.add_argument("--alphas", type=float, nargs="+", required=True)
    s_study.add_argument(
        "--alpha_schedules",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of alpha schedule specs to sweep during tune (same as `tune`).",
    )
    s_study.add_argument("--ensemble", type=int, default=3)
    s_study.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional list of seeds; overrides --seed")
    s_study.add_argument("--grid_n_eval", type=int, default=121, help="2D eval grid size (per axis)")
    s_study.add_argument("--n_eval", type=int, default=5000, help="nD eval samples (fallback when state_dim != 2)")
    s_study.add_argument("--eval_all_members", type=int, default=1)
    s_study.add_argument(
        "--rank_by",
        type=str,
        default="auto",
        choices=["auto", "J_mse", "u_mse", "res_rms", "res_max"],
        help="Metric used to select the best CPINN configuration",
    )
    # VI baseline params (only used when --do_vi 1 and problem.state_dim == 2)
    s_study.add_argument("--vi_grid", type=int, default=61, help="VI grid size per axis (2D only)")
    s_study.add_argument("--vi_iters", type=int, default=120, help="VI iterations (2D only)")

    # Which baselines to run
    s_study.add_argument("--do_tfc", type=int, default=1)
    s_study.add_argument("--do_bellman", type=int, default=1)
    s_study.add_argument("--do_vi", type=int, default=1)
    s_study.add_argument("--do_rl", type=int, default=1, help="Only meaningful for pendulum-style problems")
    # RL params (if enabled)
    s_study.add_argument("--env", type=str, default="Pendulum-v1")
    s_study.add_argument("--rl_steps", type=int, default=200000)
    s_study.add_argument("--physics_weight", type=float, default=0.1)


    # baselines
    s_base = sub.add_parser("benchmarks", help="Run baseline methods (TFC, BellmanNN, numeric HJB where available)")
    add_common(s_base)
    s_base.add_argument("--which", choices=["tfc", "bellman", "vi"], required=True)
    s_base.add_argument("--vi_grid", type=int, default=61)
    s_base.add_argument("--vi_iters", type=int, default=120)
    s_base.add_argument("--grid_n_eval", type=int, default=121, help="2D eval grid size (per axis)")
    s_base.add_argument("--n_eval", type=int, default=5000, help="nD eval samples (fallback when state_dim != 2)")
    s_base.add_argument("--eval_all_members", type=int, default=0, help="Compatibility flag; ignored for baselines")


    # RL
    s_rl = sub.add_parser("rl_ppo", help="Train PPO on a Gymnasium environment (e.g., Pendulum-v1)")
    s_rl.add_argument("--env", type=str, default="Pendulum-v1")
    s_rl.add_argument("--steps", type=int, default=200000)
    s_rl.add_argument("--seed", type=int, default=0)
    s_rl.add_argument("--device", type=str, default="auto")
    s_rl.add_argument("--outdir", type=str, default="outputs")
    s_rl.add_argument("--save_csv", type=int, default=1)
    s_rl.add_argument("--save_history", type=int, default=1)
    s_rl.add_argument("--make_plots", type=int, default=0)

    s_hyb = sub.add_parser("rl_pinn_hybrid", help="Train PPO + physics/HJB residual critic regularizer")
    s_hyb.add_argument("--env", type=str, default="Pendulum-v1")
    s_hyb.add_argument("--steps", type=int, default=200000)
    s_hyb.add_argument("--physics_weight", type=float, default=0.1)
    s_hyb.add_argument("--seed", type=int, default=0)
    s_hyb.add_argument("--device", type=str, default="auto")
    s_hyb.add_argument("--outdir", type=str, default="outputs")
    s_hyb.add_argument("--save_csv", type=int, default=1)
    s_hyb.add_argument("--save_history", type=int, default=1)
    s_hyb.add_argument("--make_plots", type=int, default=0)

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.cmd in {"train_cpinn", "ablation", "alpha_sweep", "tune", "benchmarks"}:
        if args.problem == "lqr":
            args.state_dim = getattr(args, "state_dim", 4)
            args.control_dim = getattr(args, "control_dim", 2)
        problem = build_problem(args)

        cfg = TrainConfig(
            seed=args.seed,
            device=args.device,
            hidden=tuple(args.hidden),
            act=args.act,
            lr_warm=args.lr_warm,
            lr_hjb=args.lr_hjb,
            epochs_warm=args.epochs_warm,
            epochs_hjb=args.epochs_hjb,
            batch_interior=args.batch_interior,
            batch_boundary=args.batch_boundary,
            alpha=args.alpha,
            alpha_schedule=args.alpha_schedule,
            alpha_start=args.alpha_start,
            alpha_start_factor=args.alpha_start_factor,
            alpha_hold_frac=args.alpha_hold_frac,
            alpha_ramp_frac=args.alpha_ramp_frac,
            boundary_weight=args.boundary_weight,
            use_boundary_loss=bool(args.use_boundary_loss),
            use_warm_start=bool(args.use_warm_start),
            print_every=args.print_every,
            record_every=int(getattr(args, "record_every", 1)),
        )

    if args.cmd == "train_cpinn":
        run_id = make_run_id("train", args.problem, cfg.alpha, cfg.use_warm_start, cfg.use_boundary_loss, cfg.seed, alpha_schedule=cfg.alpha_schedule, alpha_start=cfg.alpha_start, alpha_start_factor=cfg.alpha_start_factor, alpha_hold_frac=cfg.alpha_hold_frac, alpha_ramp_frac=cfg.alpha_ramp_frac)
        history_rows: List[Dict[str, object]] = []
        history = history_rows if (args.save_history or args.make_plots) else None

        t0 = time.time()
        nets, warm_stats, member_train_stats = train_cpinn_ensemble(
            problem,
            cfg,
            ensemble=args.ensemble,
            run_id=run_id,
            history=history,
        )
        train_time = time.time() - t0

        # Evaluation metrics (by default: evaluate all ensemble members)
        member_eval_rows, eval_summary = evaluate_ensemble(
            problem,
            nets,
            device=cfg.device,
            grid_n_2d=int(args.grid_n_eval),
            n_eval=int(args.n_eval),
            eval_all_members=bool(args.eval_all_members),
        )

        run_row: Dict[str, object] = {
            "run_id": run_id,
            "cmd": "train_cpinn",
            "problem": args.problem,
            "seed": cfg.seed,
            "ensemble": args.ensemble,
            "alpha": cfg.alpha,
            "alpha_schedule": str(cfg.alpha_schedule),
            "alpha_start": float(cfg.alpha_start) if cfg.alpha_start is not None else float("nan"),
            "alpha_start_factor": float(cfg.alpha_start_factor),
            "alpha_hold_frac": float(cfg.alpha_hold_frac),
            "alpha_ramp_frac": float(cfg.alpha_ramp_frac),
            "use_warm_start": int(cfg.use_warm_start),
            "use_boundary_loss": int(cfg.use_boundary_loss),
            "boundary_weight": cfg.boundary_weight,
            "epochs_warm": cfg.epochs_warm,
            "epochs_hjb": cfg.epochs_hjb,
            "train_time_sec": float(train_time),
            "warm_mse": float(warm_stats.get("warm_mse", float("nan"))),
            # eval metrics are ensemble means
            "J_mse": float(eval_summary.get("J_mse_mean", float("nan"))),
            "u_mse": float(eval_summary.get("u_mse_mean", float("nan"))),
            "res_rms": float(eval_summary.get("res_rms_mean", float("nan"))),
            "res_max": float(eval_summary.get("res_max_mean", float("nan"))),
        }

        # Training-objective finals (across ensemble members)
        if member_train_stats:
            finals = np.array([float(s.get("final_loss", float("nan"))) for s in member_train_stats], dtype=float)
            run_row["train_final_loss"] = float(np.nanmean(finals))
            run_row["train_final_loss_std"] = float(np.nanstd(finals))

        # Save CSV artifacts
        if args.save_csv:
            write_rows_csv(os.path.join(args.outdir, f"{run_id}_run_summary.csv"), [run_row])
            write_rows_csv(
                os.path.join(args.outdir, f"{run_id}_member_eval.csv"),
                [{"run_id": run_id, **r} for r in member_eval_rows],
            )
            write_json(os.path.join(args.outdir, f"{run_id}_train_config.json"), dataclasses.asdict(cfg))

        if args.save_history and history is not None:
            write_rows_csv(os.path.join(args.outdir, f"{run_id}_loss_history.csv"), history_rows)

        if args.make_plots:
            # Loss curves
            if history is not None:
                plot_loss_curves_for_run(
                    history_rows,
                    run_id=run_id,
                    outdir=args.outdir,
                    title_prefix="train ",
                    alpha=cfg.alpha,
                    boundary_weight=cfg.boundary_weight,
                )

            # Grid heatmaps (first member)
            if problem.state_dim == 2:
                grid = evaluate_on_grid_2d(problem, nets[0], grid_n=int(args.grid_n_eval), device=cfg.device)
                plot_heatmap(
                    grid["x1"],
                    grid["x2"],
                    grid["residual"],
                    "Global HJB residual",
                    outpath=f"{args.outdir}/{run_id}_hjb_residual.png",
                )
                if "J_true" in grid:
                    plot_heatmap(
                        grid["x1"],
                        grid["x2"],
                        (grid["J_pred"] - grid["J_true"]) ** 2,
                        "Value MSE (J)",
                        outpath=f"{args.outdir}/{run_id}_J_mse.png",
                    )
                    plot_heatmap(
                        grid["x1"],
                        grid["x2"],
                        np.mean((grid["u_pred"] - grid["u_true"]) ** 2, axis=2),
                        "Control MSE (u)",
                        outpath=f"{args.outdir}/{run_id}_u_mse.png",
                    )

            # Quick boundary-initial-condition simulation for reviewer request
            if problem.state_dim == 2:
                sim = SimConfig(dt=0.01, steps=2000, obs_noise_std=0.01, init_perturb_std=0.01)
                x0s = [
                    np.array([problem.domain.low[0], problem.domain.low[1]]),
                    np.array([problem.domain.low[0], problem.domain.high[1]]),
                    np.array([problem.domain.high[0], problem.domain.low[1]]),
                    np.array([problem.domain.high[0], problem.domain.high[1]]),
                ]
                for k, x0 in enumerate(x0s):
                    out = simulate_ensemble_closed_loop(
                        problem, nets, policy="outlier_exclusion", x0=x0, sim=sim, seed=cfg.seed
                    )
                    xm = out["x"].mean(axis=1)
                    plt.figure(figsize=(6, 4))
                    plt.plot(xm[:, 0], label="$x_1$ (mean)")
                    plt.plot(xm[:, 1], label="$x_2$ (mean)")
                    plt.title(f"Boundary IC test {k+1}: x0={x0.tolist()}")
                    plt.xlabel("step")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"{args.outdir}/{run_id}_boundary_ic_{k+1}.png", dpi=200)
                    plt.close()
    elif args.cmd == "ablation":
        ablations = [
            ("warm+boundary", True, True),
            ("warm_only", True, False),
            ("boundary_only", False, True),
            ("neither", False, False),
        ]

        seeds = args.seeds if args.seeds is not None else [cfg.seed]
        run_rows: List[Dict[str, object]] = []
        member_eval_all: List[Dict[str, object]] = []
        history_all: List[Dict[str, object]] = []

        for name, use_warm, use_b in ablations:
            for seed in seeds:
                print(f"\n\n=== Ablation: {name} (warm={use_warm}, boundary={use_b}, seed={seed}) ===")
                cfg2 = dataclasses.replace(cfg, seed=int(seed), use_warm_start=use_warm, use_boundary_loss=use_b)
                run_id = make_run_id(f"ablation_{name}", args.problem, cfg2.alpha, cfg2.use_warm_start, cfg2.use_boundary_loss, cfg2.seed, alpha_schedule=cfg2.alpha_schedule, alpha_start=cfg2.alpha_start, alpha_start_factor=cfg2.alpha_start_factor, alpha_hold_frac=cfg2.alpha_hold_frac, alpha_ramp_frac=cfg2.alpha_ramp_frac)

                history_rows: List[Dict[str, object]] = []
                history = history_rows if (args.save_history or args.make_plots) else None

                t0 = time.time()
                nets, warm_stats, member_train_stats = train_cpinn_ensemble(
                    problem,
                    cfg2,
                    ensemble=args.ensemble,
                    run_id=run_id,
                    history=history,
                )
                train_time = time.time() - t0

                member_eval_rows, eval_summary = evaluate_ensemble(
                    problem,
                    nets,
                    device=cfg2.device,
                    grid_n_2d=int(args.grid_n_eval),
                    n_eval=int(args.n_eval),
                    eval_all_members=bool(args.eval_all_members),
                )

                run_row: Dict[str, object] = {
                    "run_id": run_id,
                    "cmd": "ablation",
                    "problem": args.problem,
                    "ablation": name,
                    "seed": cfg2.seed,
                    "ensemble": args.ensemble,
                    "alpha": cfg2.alpha,
                "alpha_schedule": str(cfg2.alpha_schedule),
                "alpha_start": float(cfg2.alpha_start) if cfg2.alpha_start is not None else float("nan"),
                "alpha_start_factor": float(cfg2.alpha_start_factor),
                "alpha_hold_frac": float(cfg2.alpha_hold_frac),
                "alpha_ramp_frac": float(cfg2.alpha_ramp_frac),
                "alpha_schedule": str(cfg2.alpha_schedule),
                "alpha_start": float(cfg2.alpha_start) if cfg2.alpha_start is not None else float("nan"),
                "alpha_start_factor": float(cfg2.alpha_start_factor),
                "alpha_hold_frac": float(cfg2.alpha_hold_frac),
                "alpha_ramp_frac": float(cfg2.alpha_ramp_frac),
                    "use_warm_start": int(cfg2.use_warm_start),
                    "use_boundary_loss": int(cfg2.use_boundary_loss),
                    "boundary_weight": cfg2.boundary_weight,
                    "epochs_warm": cfg2.epochs_warm,
                    "epochs_hjb": cfg2.epochs_hjb,
                    "train_time_sec": float(train_time),
                    "warm_mse": float(warm_stats.get("warm_mse", float("nan"))),
                    "J_mse": float(eval_summary.get("J_mse_mean", float("nan"))),
                    "J_mse_ens_std": float(eval_summary.get("J_mse_std", float("nan"))),
                    "u_mse": float(eval_summary.get("u_mse_mean", float("nan"))),
                    "u_mse_ens_std": float(eval_summary.get("u_mse_std", float("nan"))),
                    "res_rms": float(eval_summary.get("res_rms_mean", float("nan"))),
                    "res_rms_ens_std": float(eval_summary.get("res_rms_std", float("nan"))),
                    "res_max": float(eval_summary.get("res_max_mean", float("nan"))),
                    "res_max_ens_std": float(eval_summary.get("res_max_std", float("nan"))),
                }

                if member_train_stats:
                    finals = np.array([float(s.get("final_loss", float("nan"))) for s in member_train_stats], dtype=float)
                    run_row["train_final_loss"] = float(np.nanmean(finals))
                    run_row["train_final_loss_std"] = float(np.nanstd(finals))

                run_rows.append(run_row)
                member_eval_all.extend([{"run_id": run_id, "ablation": name, "seed": cfg2.seed, **r} for r in member_eval_rows])
                history_all.extend(history_rows)

        # Console summary aggregated over seeds (if multiple)
        metric_fields = ["J_mse", "u_mse", "res_rms", "res_max"]
        summary_rows = group_mean_std(run_rows, key_fields=["ablation"], metric_fields=metric_fields)
        if summary_rows:
            print("\nAblation summary (mean ± std over seeds; metrics are ensemble means):")
            print("ablation\tJ_MSE\tu_MSE\tres_RMS\tres_max\tn")
            for r in summary_rows:
                jm = r.get("J_mse_mean", float("nan"))
                js = r.get("J_mse_std", float("nan"))
                um = r.get("u_mse_mean", float("nan"))
                us = r.get("u_mse_std", float("nan"))
                rr = r.get("res_rms_mean", float("nan"))
                rs = r.get("res_rms_std", float("nan"))
                rx = r.get("res_max_mean", float("nan"))
                rxs = r.get("res_max_std", float("nan"))
                print(f"{r['ablation']}\t{float(jm):.3e}±{float(js):.1e}\t{float(um):.3e}±{float(us):.1e}\t{float(rr):.3e}±{float(rs):.1e}\t{float(rx):.3e}±{float(rxs):.1e}\t{r.get('n', '')}")

        # Save CSV artifacts
        if args.save_csv:
            write_rows_csv(os.path.join(args.outdir, "ablation_runs.csv"), run_rows)
            write_rows_csv(os.path.join(args.outdir, "ablation_members.csv"), member_eval_all)
            write_rows_csv(os.path.join(args.outdir, "ablation_summary.csv"), summary_rows)
            write_json(os.path.join(args.outdir, "ablation_train_config.json"), dataclasses.asdict(cfg))

        if args.save_history:
            write_rows_csv(os.path.join(args.outdir, "ablation_loss_history.csv"), history_all)

        # Plots
        if args.make_plots:
            plot_dir = os.path.join(args.outdir, "ablation_plots")
            _safe_makedirs(plot_dir)

            # Loss curves
            if args.plot_runs != "none":
                plot_ids: List[str] = []
                if args.plot_runs == "all":
                    plot_ids = [r["run_id"] for r in run_rows]
                else:
                    # best run (prefer J_mse if available)
                    def _score(rr: Dict[str, object]) -> float:
                        jm = float(rr.get("J_mse", float("nan")))
                        if not np.isnan(jm):
                            return jm
                        return float(rr.get("res_rms", float("inf")))

                    best = min(run_rows, key=_score) if run_rows else None
                    if best is not None:
                        plot_ids = [str(best["run_id"])]

                for rid in plot_ids:
                    # Look up alpha/boundary_weight for titles
                    rr = next((r for r in run_rows if r.get("run_id") == rid), None)
                    plot_loss_curves_for_run(
                        history_all,
                        run_id=rid,
                        outdir=plot_dir,
                        title_prefix=f"ablation {rr.get('ablation','')} ",
                        alpha=float(rr.get("alpha", float("nan"))) if rr else None,
                        boundary_weight=float(rr.get("boundary_weight", float("nan"))) if rr else None,
                    )

            # Summary bar chart (J_mse if analytic)
            if summary_rows and not all(np.isnan(float(r.get("J_mse_mean", float("nan")))) for r in summary_rows):
                labels = [str(r.get("ablation")) for r in summary_rows]
                vals = [float(r.get("J_mse_mean", float("nan"))) for r in summary_rows]
                errs = [float(r.get("J_mse_std", 0.0)) for r in summary_rows]
                plt.figure(figsize=(7, 4))
                plt.bar(labels, vals, yerr=errs, capsize=4)
                plt.ylabel("J_MSE")
                plt.title("Ablation summary (J_MSE)")
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "ablation_Jmse_bar.png"), dpi=200)
                plt.close()

    elif args.cmd == "alpha_sweep":
        seeds = args.seeds if args.seeds is not None else [cfg.seed]
        run_rows: List[Dict[str, object]] = []
        member_eval_all: List[Dict[str, object]] = []
        history_all: List[Dict[str, object]] = []

        for a in args.alphas:
            for seed in seeds:
                print(f"\n\n=== Alpha sweep: alpha={a} (seed={seed}) ===")
                cfg2 = dataclasses.replace(cfg, seed=int(seed), alpha=float(a))
                run_id = make_run_id("alpha", args.problem, cfg2.alpha, cfg2.use_warm_start, cfg2.use_boundary_loss, cfg2.seed, alpha_schedule=cfg2.alpha_schedule, alpha_start=cfg2.alpha_start, alpha_start_factor=cfg2.alpha_start_factor, alpha_hold_frac=cfg2.alpha_hold_frac, alpha_ramp_frac=cfg2.alpha_ramp_frac)

                history_rows: List[Dict[str, object]] = []
                history = history_rows if (args.save_history or args.make_plots) else None

                t0 = time.time()
                nets, warm_stats, member_train_stats = train_cpinn_ensemble(
                    problem,
                    cfg2,
                    ensemble=args.ensemble,
                    run_id=run_id,
                    history=history,
                )
                train_time = time.time() - t0

                member_eval_rows, eval_summary = evaluate_ensemble(
                    problem,
                    nets,
                    device=cfg2.device,
                    grid_n_2d=int(args.grid_n_eval),
                    n_eval=int(args.n_eval),
                    eval_all_members=bool(args.eval_all_members),
                )

                run_row: Dict[str, object] = {
                    "run_id": run_id,
                    "cmd": "alpha_sweep",
                    "problem": args.problem,
                    "alpha": cfg2.alpha,
                    "seed": cfg2.seed,
                    "ensemble": args.ensemble,
                    "use_warm_start": int(cfg2.use_warm_start),
                    "use_boundary_loss": int(cfg2.use_boundary_loss),
                    "boundary_weight": cfg2.boundary_weight,
                    "epochs_warm": cfg2.epochs_warm,
                    "epochs_hjb": cfg2.epochs_hjb,
                    "train_time_sec": float(train_time),
                    "warm_mse": float(warm_stats.get("warm_mse", float("nan"))),
                    "J_mse": float(eval_summary.get("J_mse_mean", float("nan"))),
                    "J_mse_ens_std": float(eval_summary.get("J_mse_std", float("nan"))),
                    "u_mse": float(eval_summary.get("u_mse_mean", float("nan"))),
                    "u_mse_ens_std": float(eval_summary.get("u_mse_std", float("nan"))),
                    "res_rms": float(eval_summary.get("res_rms_mean", float("nan"))),
                    "res_rms_ens_std": float(eval_summary.get("res_rms_std", float("nan"))),
                    "res_max": float(eval_summary.get("res_max_mean", float("nan"))),
                    "res_max_ens_std": float(eval_summary.get("res_max_std", float("nan"))),
                }

                if member_train_stats:
                    finals = np.array([float(s.get("final_loss", float("nan"))) for s in member_train_stats], dtype=float)
                    run_row["train_final_loss"] = float(np.nanmean(finals))
                    run_row["train_final_loss_std"] = float(np.nanstd(finals))

                run_rows.append(run_row)
                member_eval_all.extend([{"run_id": run_id, "seed": cfg2.seed, "alpha": cfg2.alpha, **r} for r in member_eval_rows])
                history_all.extend(history_rows)

        metric_fields = ["J_mse", "u_mse", "res_rms", "res_max"]
        summary_rows = group_mean_std(run_rows, key_fields=["alpha"], metric_fields=metric_fields)

        if summary_rows:
            print("\nAlpha sweep summary (mean ± std over seeds; metrics are ensemble means):")
            print("alpha\tJ_MSE\tu_MSE\tres_RMS\tres_max\tn")
            for r in summary_rows:
                a = float(r.get("alpha", float("nan")))
                jm = float(r.get("J_mse_mean", float("nan")))
                js = float(r.get("J_mse_std", float("nan")))
                um = float(r.get("u_mse_mean", float("nan")))
                us = float(r.get("u_mse_std", float("nan")))
                rr = float(r.get("res_rms_mean", float("nan")))
                rs = float(r.get("res_rms_std", float("nan")))
                rx = float(r.get("res_max_mean", float("nan")))
                rxs = float(r.get("res_max_std", float("nan")))
                print(f"{a:.3g}\t{jm:.3e}±{js:.1e}\t{um:.3e}±{us:.1e}\t{rr:.3e}±{rs:.1e}\t{rx:.3e}±{rxs:.1e}\t{r.get('n','')}")

        if args.save_csv:
            write_rows_csv(os.path.join(args.outdir, "alpha_sweep_runs.csv"), run_rows)
            write_rows_csv(os.path.join(args.outdir, "alpha_sweep_members.csv"), member_eval_all)
            write_rows_csv(os.path.join(args.outdir, "alpha_sweep_summary.csv"), summary_rows)
            write_json(os.path.join(args.outdir, "alpha_sweep_train_config.json"), dataclasses.asdict(cfg))

        if args.save_history:
            write_rows_csv(os.path.join(args.outdir, "alpha_sweep_loss_history.csv"), history_all)

        if args.make_plots:
            plot_dir = os.path.join(args.outdir, "alpha_sweep_plots")
            _safe_makedirs(plot_dir)

            # Metric vs alpha plots
            # Prefer J_mse if available (analytic problems), otherwise plot residual
            have_j = summary_rows and not all(np.isnan(float(r.get("J_mse_mean", float("nan"))) ) for r in summary_rows)
            x = [float(r.get("alpha", float("nan"))) for r in summary_rows]
            if have_j:
                y = [float(r.get("J_mse_mean", float("nan"))) for r in summary_rows]
                yerr = [float(r.get("J_mse_std", 0.0)) for r in summary_rows]
                plt.figure(figsize=(6, 4))
                plt.errorbar(x, y, yerr=yerr, fmt="o-")
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("alpha")
                plt.ylabel("J_MSE")
                plt.title("Alpha sweep: J_MSE")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "alpha_sweep_Jmse.png"), dpi=200)
                plt.close()

            # Residual RMS
            y = [float(r.get("res_rms_mean", float("nan"))) for r in summary_rows]
            yerr = [float(r.get("res_rms_std", 0.0)) for r in summary_rows]
            plt.figure(figsize=(6, 4))
            plt.errorbar(x, y, yerr=yerr, fmt="o-")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("alpha")
            plt.ylabel("Residual RMS")
            plt.title("Alpha sweep: residual RMS")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "alpha_sweep_resRMS.png"), dpi=200)
            plt.close()

            # Loss curves
            if args.plot_runs != "none" and run_rows:
                plot_ids: List[str] = []
                if args.plot_runs == "all":
                    plot_ids = [str(r["run_id"]) for r in run_rows]
                else:
                    def _score(rr: Dict[str, object]) -> float:
                        jm = float(rr.get("J_mse", float("nan")))
                        if not np.isnan(jm):
                            return jm
                        return float(rr.get("res_rms", float("inf")))

                    best = min(run_rows, key=_score)
                    plot_ids = [str(best["run_id"]) ]

                for rid in plot_ids:
                    rr = next((r for r in run_rows if r.get("run_id") == rid), None)
                    plot_loss_curves_for_run(
                        history_all,
                        run_id=rid,
                        outdir=plot_dir,
                        title_prefix="alpha_sweep ",
                        alpha=float(rr.get("alpha", float("nan"))) if rr else None,
                        boundary_weight=float(rr.get("boundary_weight", float("nan"))) if rr else None,
                    )

    elif args.cmd == "tune":
        # Joint alpha x ablation x alpha-schedule grid search
        ablations = [
            ("warm+boundary", True, True),
            ("warm_only", True, False),
            ("boundary_only", False, True),
            ("neither", False, False),
        ]

        seeds = args.seeds if args.seeds is not None else [cfg.seed]
        alphas = [float(a) for a in args.alphas]

        # Parse schedule specs (e.g., constant, linear@0.1)
        # If not provided, fall back to the single --alpha_schedule from add_common.
        sched_specs = args.alpha_schedules if args.alpha_schedules is not None else [str(cfg.alpha_schedule)]

        def _parse_sched_spec(spec: str) -> Tuple[str, Optional[float]]:
            s = str(spec).strip()
            if "@" in s:
                name, sf = s.split("@", 1)
                try:
                    sfv = float(sf)
                except Exception:
                    raise ValueError(f"Bad alpha schedule spec (start_factor): {spec}")
                return _normalize_alpha_schedule(name), sfv
            return _normalize_alpha_schedule(s), None

        schedule_grid: List[Tuple[str, Optional[float]]] = [_parse_sched_spec(s) for s in sched_specs]

        run_rows: List[Dict[str, object]] = []
        member_eval_all: List[Dict[str, object]] = []
        history_all: List[Dict[str, object]] = []

        for a in alphas:
            for sched_name, sched_sf in schedule_grid:
                for name, use_warm, use_b in ablations:
                    for seed in seeds:
                        # Construct per-run config
                        # Precedence for alpha_start:
                        #   1) schedule spec "...@sf" -> alpha_start_factor = sf
                        #   2) --alpha_start (absolute)
                        #   3) --alpha_start_factor (relative)
                        if sched_sf is not None:
                            a_start = None
                            a_sf = float(sched_sf)
                        else:
                            a_start = cfg.alpha_start
                            a_sf = float(cfg.alpha_start_factor)

                        cfg2 = dataclasses.replace(
                            cfg,
                            seed=int(seed),
                            alpha=float(a),
                            alpha_schedule=str(sched_name),
                            alpha_start=a_start,
                            alpha_start_factor=a_sf,
                            use_warm_start=use_warm,
                            use_boundary_loss=use_b,
                        )

                        sched_desc = alpha_schedule_desc(cfg2)
                        print(f"\n\n=== Tune: alpha={a} | sched={sched_desc} | {name} (warm={use_warm}, boundary={use_b}, seed={seed}) ===")

                        run_id = make_run_id(
                            f"tune_{name}",
                            args.problem,
                            cfg2.alpha,
                            cfg2.use_warm_start,
                            cfg2.use_boundary_loss,
                            cfg2.seed,
                            alpha_schedule=cfg2.alpha_schedule,
                            alpha_start=cfg2.alpha_start,
                            alpha_start_factor=cfg2.alpha_start_factor,
                            alpha_hold_frac=cfg2.alpha_hold_frac,
                            alpha_ramp_frac=cfg2.alpha_ramp_frac,
                        )

                        history_rows: List[Dict[str, object]] = []
                        history = history_rows if (args.save_history or args.make_plots) else None

                        t0 = time.time()
                        nets, warm_stats, member_train_stats = train_cpinn_ensemble(
                            problem,
                            cfg2,
                            ensemble=args.ensemble,
                            run_id=run_id,
                            history=history,
                        )
                        train_time = time.time() - t0

                        member_eval_rows, eval_summary = evaluate_ensemble(
                            problem,
                            nets,
                            device=cfg2.device,
                            grid_n_2d=int(args.grid_n_eval),
                            n_eval=int(args.n_eval),
                            eval_all_members=bool(args.eval_all_members),
                        )

                        run_row: Dict[str, object] = {
                            "run_id": run_id,
                            "cmd": "tune",
                            "problem": args.problem,
                            "ablation": name,
                            "alpha": cfg2.alpha,
                            "alpha_schedule": str(cfg2.alpha_schedule),
                            "alpha_start": float(cfg2.alpha_start) if cfg2.alpha_start is not None else float("nan"),
                            "alpha_start_factor": float(cfg2.alpha_start_factor),
                            "alpha_hold_frac": float(cfg2.alpha_hold_frac),
                            "alpha_ramp_frac": float(cfg2.alpha_ramp_frac),
                            "seed": cfg2.seed,
                            "ensemble": args.ensemble,
                            "use_warm_start": int(cfg2.use_warm_start),
                            "use_boundary_loss": int(cfg2.use_boundary_loss),
                            "boundary_weight": cfg2.boundary_weight,
                            "epochs_warm": cfg2.epochs_warm,
                            "epochs_hjb": cfg2.epochs_hjb,
                            "train_time_sec": float(train_time),
                            "warm_mse": float(warm_stats.get("warm_mse", float("nan"))),
                            "J_mse": float(eval_summary.get("J_mse_mean", float("nan"))),
                            "J_mse_ens_std": float(eval_summary.get("J_mse_std", float("nan"))),
                            "u_mse": float(eval_summary.get("u_mse_mean", float("nan"))),
                            "u_mse_ens_std": float(eval_summary.get("u_mse_std", float("nan"))),
                            "res_rms": float(eval_summary.get("res_rms_mean", float("nan"))),
                            "res_rms_ens_std": float(eval_summary.get("res_rms_std", float("nan"))),
                            "res_max": float(eval_summary.get("res_max_mean", float("nan"))),
                            "res_max_ens_std": float(eval_summary.get("res_max_std", float("nan"))),
                        }

                        if member_train_stats:
                            finals = np.array([float(s.get("final_loss", float("nan"))) for s in member_train_stats], dtype=float)
                            run_row["train_final_loss"] = float(np.nanmean(finals))
                            run_row["train_final_loss_std"] = float(np.nanstd(finals))

                        run_rows.append(run_row)
                        member_eval_all.extend(
                            [
                                {
                                    "run_id": run_id,
                                    "ablation": name,
                                    "seed": cfg2.seed,
                                    "alpha": cfg2.alpha,
                                    "alpha_schedule": str(cfg2.alpha_schedule),
                                    "alpha_start_factor": float(cfg2.alpha_start_factor),
                                    "alpha_hold_frac": float(cfg2.alpha_hold_frac),
                                    "alpha_ramp_frac": float(cfg2.alpha_ramp_frac),
                                    **r,
                                }
                                for r in member_eval_rows
                            ]
                        )
                        history_all.extend(history_rows)

        # Aggregate over seeds for each (alpha, schedule, ablation)
        metric_fields = ["J_mse", "u_mse", "res_rms", "res_max"]
        summary_rows = group_mean_std(
            run_rows,
            key_fields=["alpha", "alpha_schedule", "alpha_start_factor", "alpha_hold_frac", "alpha_ramp_frac", "ablation"],
            metric_fields=metric_fields,
        )

        # Pick the best setting
        rank_by = str(args.rank_by)
        if rank_by == "auto":
            have_j = summary_rows and not all(np.isnan(float(r.get("J_mse_mean", float("nan")))) for r in summary_rows)
            rank_metric = "J_mse" if have_j else "res_rms"
        else:
            rank_metric = rank_by

        def _metric_mean(row: Dict[str, object], metric: str) -> float:
            return float(row.get(f"{metric}_mean", float("nan")))

        def _rank_score(row: Dict[str, object]) -> float:
            v = _metric_mean(row, rank_metric)
            return float("inf") if np.isnan(v) else v

        best_cfg = min(summary_rows, key=_rank_score) if summary_rows else None
        if best_cfg is not None:
            ordered = sorted(summary_rows, key=_rank_score)
            print("\nTune top configurations (sorted by", rank_metric, "):" )

            print("rank	alpha	sched	ablation	" + rank_metric + "(mean±std)	J_MSE	res_RMS	res_max	n")

            def _sched_label(r: Dict[str, object]) -> str:
                sched = str(r.get("alpha_schedule", "constant"))
                sf = float(r.get("alpha_start_factor", 1.0))
                hold = float(r.get("alpha_hold_frac", 0.0))
                ramp = float(r.get("alpha_ramp_frac", 1.0))
                if _normalize_alpha_schedule(sched) == "constant":
                    return "constant"
                lab = f"{sched}@{sf:g}"
                if abs(hold) > 1e-12:
                    lab += f"_h{hold:g}"
                if abs(ramp - 1.0) > 1e-12:
                    lab += f"_r{ramp:g}"
                return lab

            for i, rr in enumerate(ordered[: min(10, len(ordered))], start=1):
                def _fmt(mean_key: str, std_key: str) -> str:
                    m = float(rr.get(mean_key, float("nan")))
                    s = float(rr.get(std_key, float("nan")))
                    if np.isnan(m):
                        return "nan"
                    if np.isnan(s):
                        return f"{m:.3e}"
                    return f"{m:.3e}±{s:.1e}"

                rm = _fmt(f"{rank_metric}_mean", f"{rank_metric}_std")
                jm = _fmt("J_mse_mean", "J_mse_std")
                rrms = _fmt("res_rms_mean", "res_rms_std")
                rmx = _fmt("res_max_mean", "res_max_std")
                print(
                    f"{i}	{float(rr.get('alpha')):g}	{_sched_label(rr)}	{rr.get('ablation')}	{rm}	{jm}	{rrms}	{rmx}	{int(rr.get('n'))}"
                )

            print("\nBest configuration (by", rank_metric, "):" )
            # (patched)
            print(best_cfg)

        # Save CSV artifacts
        if args.save_csv:
            write_rows_csv(os.path.join(args.outdir, "tune_runs.csv"), run_rows)
            write_rows_csv(os.path.join(args.outdir, "tune_members.csv"), member_eval_all)
            write_rows_csv(os.path.join(args.outdir, "tune_summary.csv"), summary_rows)
            write_json(os.path.join(args.outdir, "tune_train_config.json"), dataclasses.asdict(cfg))
            if best_cfg is not None:
                ab = str(best_cfg.get("ablation"))
                use_warm = int(ab in {"warm+boundary", "warm_only"})
                use_boundary = int(ab in {"warm+boundary", "boundary_only"})
                write_json(
                    os.path.join(args.outdir, "tune_best_config.json"),
                    {
                        "rank_metric": rank_metric,
                        "problem": args.problem,
                        "alpha": float(best_cfg.get("alpha")),
                        "alpha_schedule": str(best_cfg.get("alpha_schedule")),
                        "alpha_start_factor": float(best_cfg.get("alpha_start_factor")),
                        "alpha_hold_frac": float(best_cfg.get("alpha_hold_frac")),
                        "alpha_ramp_frac": float(best_cfg.get("alpha_ramp_frac")),
                        "ablation": ab,
                        "use_warm_start": use_warm,
                        "use_boundary_loss": use_boundary,
                        "boundary_weight": float(cfg.boundary_weight),
                        "epochs_warm": int(cfg.epochs_warm),
                        "epochs_hjb": int(cfg.epochs_hjb),
                        "ensemble": int(args.ensemble),
                        "seeds": [int(s) for s in seeds],
                    },
                )

        if args.save_history:
            write_rows_csv(os.path.join(args.outdir, "tune_loss_history.csv"), history_all)

        # Plots
        if args.make_plots and summary_rows:
            plot_dir = os.path.join(args.outdir, "tune_plots")
            _safe_makedirs(plot_dir)

            ablation_names = [a[0] for a in ablations]

            # Unique schedule variants in summary (so we can facet plots)
            sched_variants = sorted(
                {
                    (
                        str(r.get("alpha_schedule", "constant")),
                        float(r.get("alpha_start_factor", 1.0)),
                        float(r.get("alpha_hold_frac", 0.0)),
                        float(r.get("alpha_ramp_frac", 1.0)),
                    )
                    for r in summary_rows
                }
            )

            def _sched_tag(sched: str, sf: float, hold: float, ramp: float) -> str:
                sched_n = _normalize_alpha_schedule(sched)
                if sched_n == "constant":
                    return "constant"
                tag = f"{sched_n}@{sf:g}"
                if abs(hold) > 1e-12:
                    tag += f"_h{hold:g}"
                if abs(ramp - 1.0) > 1e-12:
                    tag += f"_r{ramp:g}"
                # filesystem safe
                return tag.replace("-", "m").replace(".", "p")

            def _close(x: float, y: float, tol: float = 1e-12) -> bool:
                return abs(float(x) - float(y)) <= tol * max(1.0, abs(float(x)), abs(float(y)))

            # One heatmap per schedule variant
            for sched, sf, hold, ramp in sched_variants:
                tag = _sched_tag(sched, sf, hold, ramp)
                mat = np.full((len(ablation_names), len(alphas)), np.nan, dtype=float)

                for r in summary_rows:
                    if str(r.get("alpha_schedule")) != str(sched):
                        continue
                    if not _close(float(r.get("alpha_start_factor", 1.0)), sf):
                        continue
                    if not _close(float(r.get("alpha_hold_frac", 0.0)), hold):
                        continue
                    if not _close(float(r.get("alpha_ramp_frac", 1.0)), ramp):
                        continue

                    # Find alpha index with tolerance
                    a_val = float(r.get("alpha"))
                    ai = next((i for i, aa in enumerate(alphas) if _close(float(aa), a_val, tol=1e-10)), None)
                    bi = ablation_names.index(str(r.get("ablation"))) if str(r.get("ablation")) in ablation_names else None
                    if ai is None or bi is None:
                        continue
                    mat[bi, ai] = float(r.get(f"{rank_metric}_mean", np.nan))

                plt.figure(figsize=(7, 4))
                im = plt.imshow(mat, origin="lower", aspect="auto")
                plt.colorbar(im)
                plt.xticks(range(len(alphas)), [f"{a:g}" for a in alphas], rotation=20)
                plt.yticks(range(len(ablation_names)), ablation_names)
                plt.xlabel("alpha")
                plt.ylabel("ablation")
                plt.title(f"Tune grid: {rank_metric} (sched={sched}, sf={sf:g})")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"tune_heatmap_{rank_metric}_{tag}.png"), dpi=200)
                plt.close()

                # Metric vs alpha curves for each ablation under this schedule
                plt.figure(figsize=(7, 4))
                for ab_name in ablation_names:
                    ys = []
                    for a in alphas:
                        rr = next(
                            (
                                r
                                for r in summary_rows
                                if _close(float(r.get("alpha")), float(a), tol=1e-10)
                                and str(r.get("ablation")) == ab_name
                                and str(r.get("alpha_schedule")) == str(sched)
                                and _close(float(r.get("alpha_start_factor", 1.0)), sf)
                                and _close(float(r.get("alpha_hold_frac", 0.0)), hold)
                                and _close(float(r.get("alpha_ramp_frac", 1.0)), ramp)
                            ),
                            None,
                        )
                        ys.append(float(rr.get(f"{rank_metric}_mean", np.nan)) if rr else np.nan)
                    plt.plot(alphas, ys, marker="o", label=ab_name)
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("alpha")
                plt.ylabel(f"{rank_metric}")
                plt.title(f"Tune: {rank_metric} vs alpha (sched={sched}@{sf:g})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"tune_{rank_metric}_vs_alpha_{tag}.png"), dpi=200)
                plt.close()

            # Loss curves
            if args.plot_runs != "none" and run_rows:
                if args.plot_runs == "all":
                    plot_ids = [str(r.get("run_id")) for r in run_rows]
                else:
                    plot_ids = []
                    if best_cfg is not None:
                        a_best = float(best_cfg.get("alpha"))
                        b_best = str(best_cfg.get("ablation"))
                        s_best = str(best_cfg.get("alpha_schedule"))
                        sf_best = float(best_cfg.get("alpha_start_factor"))
                        hold_best = float(best_cfg.get("alpha_hold_frac"))
                        ramp_best = float(best_cfg.get("alpha_ramp_frac"))

                        def _run_score(rr: Dict[str, object]) -> float:
                            if not _close(float(rr.get("alpha", np.nan)), a_best, tol=1e-10):
                                return float("inf")
                            if str(rr.get("ablation")) != b_best:
                                return float("inf")
                            if str(rr.get("alpha_schedule")) != s_best:
                                return float("inf")
                            if not _close(float(rr.get("alpha_start_factor", 1.0)), sf_best):
                                return float("inf")
                            if not _close(float(rr.get("alpha_hold_frac", 0.0)), hold_best):
                                return float("inf")
                            if not _close(float(rr.get("alpha_ramp_frac", 1.0)), ramp_best):
                                return float("inf")

                            v = float(rr.get(rank_metric if rank_metric != "J_mse" else "J_mse", float("nan")))
                            if np.isnan(v):
                                return float("inf")
                            return v

                        best_run = min(run_rows, key=_run_score) if run_rows else None
                        if best_run is not None and _run_score(best_run) < float("inf"):
                            plot_ids = [str(best_run.get("run_id"))]

                for rid in plot_ids:
                    rr = next((r for r in run_rows if r.get("run_id") == rid), None)
                    title_extra = ""
                    if rr is not None:
                        title_extra = f"sched={rr.get('alpha_schedule')}@{float(rr.get('alpha_start_factor', 1.0)):g} "
                    plot_loss_curves_for_run(
                        history_all,
                        run_id=rid,
                        outdir=plot_dir,
                        title_prefix="tune " + title_extra,
                        alpha=float(rr.get("alpha", float("nan"))) if rr else None,
                        boundary_weight=float(rr.get("boundary_weight", float("nan"))) if rr else None,
                    )
    elif args.cmd == "study":
        # Orchestrate: tune -> train CPINN(best) -> baselines -> optional RL, then collate.
        study_id = f"study_{args.problem}_{time.strftime('%Y%m%d_%H%M%S')}"
        study_dir = os.path.join(args.outdir, study_id)
        os.makedirs(study_dir, exist_ok=True)

        script_path = os.path.abspath(__file__)
        py = sys.executable

        def _run(cmd: List[str]) -> None:
            print("[study] running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

        def _find_single(pattern: str, folder: str) -> str:
            import glob

            matches = sorted(glob.glob(os.path.join(folder, pattern)))
            if not matches:
                raise FileNotFoundError(f"No files matching {pattern} in {folder}")
            # If multiple, return newest
            matches.sort(key=lambda p: os.path.getmtime(p))
            return matches[-1]

        # Common CPINN/baseline args we forward to sub-commands
        common = [
            "--problem",
            str(args.problem),
            "--device",
            str(args.device),
            "--act",
            str(args.act),
            "--lr_warm",
            str(args.lr_warm),
            "--lr_hjb",
            str(args.lr_hjb),
            "--epochs_warm",
            str(args.epochs_warm),
            "--epochs_hjb",
            str(args.epochs_hjb),
            "--batch_interior",
            str(args.batch_interior),
            "--batch_boundary",
            str(args.batch_boundary),
            "--low",
            str(args.low),
            "--high",
            str(args.high),
            "--boundary_weight",
            str(args.boundary_weight),
            "--grid_n_eval",
            str(args.grid_n_eval),
            "--n_eval",
            str(args.n_eval),
            "--eval_all_members",
            str(args.eval_all_members),
            "--save_csv",
            str(args.save_csv),
            "--save_history",
            str(args.save_history),
            "--record_every",
            str(args.record_every),
            "--make_plots",
            str(args.make_plots),
        ]
        # Hidden layers are variadic
        for h in args.hidden:
            common.extend(["--hidden", str(h)])

        # Forward pendulum knobs (harmless for other problems)
        common.extend(
            [
                "--pendulum_theta_low",
                str(args.pendulum_theta_low),
                "--pendulum_theta_high",
                str(args.pendulum_theta_high),
                "--pendulum_omega_low",
                str(args.pendulum_omega_low),
                "--pendulum_omega_high",
                str(args.pendulum_omega_high),
                "--pendulum_u_max",
                str(args.pendulum_u_max),
                "--pendulum_vi_grid",
                str(args.pendulum_vi_grid),
                "--pendulum_vi_iters",
                str(args.pendulum_vi_iters),
                "--pendulum_vi_dt",
                str(args.pendulum_vi_dt),
                "--pendulum_vi_u_points",
                str(args.pendulum_vi_u_points),
                "--pendulum_vi_force",
                str(args.pendulum_vi_force),
                "--pendulum_g",
                str(args.pendulum_g),
                "--pendulum_m",
                str(args.pendulum_m),
                "--pendulum_l",
                str(args.pendulum_l),
                "--pendulum_cache_tag",
                str(args.pendulum_cache_tag),
            ]
        )

        # 1) Tune
        tune_dir = os.path.join(study_dir, "cpinn_tune")
        os.makedirs(tune_dir, exist_ok=True)
        cmd_tune = [py, script_path, "tune", "--outdir", tune_dir] + common + [
            "--ensemble",
            str(args.ensemble),
            "--rank_by",
            str(args.rank_by),
        ]
        # alpha grid
        cmd_tune += ["--alphas"] + [str(a) for a in args.alphas]
        # alpha schedules (optional)
        if args.alpha_schedules is not None:
            cmd_tune += ["--alpha_schedules"] + [str(s) for s in args.alpha_schedules]
        # seeds
        if args.seeds is not None:
            cmd_tune += ["--seeds"] + [str(s) for s in args.seeds]
        _run(cmd_tune)

        best_path = os.path.join(tune_dir, "tune_best_config.json")
        best = json.load(open(best_path, "r"))
        print("[study] best config:", best)

        # Parse warm/boundary switches from ablation string
        ablation = best.get("ablation", "warm+boundary")
        warm = "warm" in ablation
        boundary = "boundary" in ablation

        # 2) Train CPINN with best config (fresh run saved to its own folder)
        cpinn_dir = os.path.join(study_dir, "cpinn_best")
        os.makedirs(cpinn_dir, exist_ok=True)
        cmd_cpinn = [py, script_path, "train_cpinn", "--outdir", cpinn_dir] + common + [
            "--ensemble",
            str(args.ensemble),
            "--seed",
            str(args.seed),
            "--alpha",
            str(best.get("alpha", args.alpha)),
            "--alpha_schedule",
            str(best.get("alpha_schedule", args.alpha_schedule)),
            "--alpha_start_factor",
            str(best.get("alpha_start_factor", args.alpha_start_factor)),
            "--alpha_hold_frac",
            str(best.get("alpha_hold_frac", args.alpha_hold_frac)),
            "--alpha_ramp_frac",
            str(best.get("alpha_ramp_frac", args.alpha_ramp_frac)),
            "--use_warm_start",
            str(int(warm)),
            "--use_boundary_loss",
            str(int(boundary)),
        ]
        _run(cmd_cpinn)

        # 3) Baselines (each in its own folder)
        rows: List[Dict[str, object]] = []

        # CPINN summary
        try:
            cpinn_sum_path = _find_single("*_run_summary.csv", cpinn_dir)
            with open(cpinn_sum_path, "r", encoding="utf-8") as f:
                import csv

                rdr = csv.DictReader(f)
                cpinn_row = next(rdr)
            cpinn_row["method"] = "cpinn"
            rows.append(cpinn_row)
        except Exception as e:
            print("[study] warning: failed to parse CPINN run_summary:", e)

        if args.do_tfc:
            tfc_dir = os.path.join(study_dir, "tfc")
            os.makedirs(tfc_dir, exist_ok=True)
            cmd = [py, script_path, "benchmarks", "--which", "tfc", "--outdir", tfc_dir] + common + [
                "--seed",
                str(args.seed),
                "--alpha",
                str(best.get("alpha", args.alpha)),
                "--alpha_schedule",
                str(best.get("alpha_schedule", args.alpha_schedule)),
                "--alpha_start_factor",
                str(best.get("alpha_start_factor", args.alpha_start_factor)),
                "--alpha_hold_frac",
                str(best.get("alpha_hold_frac", args.alpha_hold_frac)),
                "--alpha_ramp_frac",
                str(best.get("alpha_ramp_frac", args.alpha_ramp_frac)),
            ]
            _run(cmd)
            try:
                summ = _find_single("*_summary.json", tfc_dir)
                d = json.load(open(summ, "r"))
                d["method"] = "tfc"
                rows.append(d)
            except Exception as e:
                print("[study] warning: failed to parse TFC summary:", e)

        if args.do_bellman:
            bell_dir = os.path.join(study_dir, "bellman")
            os.makedirs(bell_dir, exist_ok=True)
            cmd = [py, script_path, "benchmarks", "--which", "bellman", "--outdir", bell_dir] + common + [
                "--seed",
                str(args.seed),
            ]
            _run(cmd)
            try:
                summ = _find_single("*_summary.json", bell_dir)
                d = json.load(open(summ, "r"))
                d["method"] = "bellman"
                rows.append(d)
            except Exception as e:
                print("[study] warning: failed to parse Bellman summary:", e)

        # Determine state dimension without instantiating the problem object.
        # (Instantiating PendulumVI here can trigger expensive VI reference computation.)
        if args.problem in {"nonlinear2d", "pendulum", "pendulum_lqr"}:
            state_dim = 2
        elif args.problem == "cartpole_lqr":
            state_dim = 4
        else:
            # lqr / cubic_nd
            state_dim = int(getattr(args, "state_dim", 4))

        if args.do_vi and state_dim == 2:
            vi_dir = os.path.join(study_dir, "vi")
            os.makedirs(vi_dir, exist_ok=True)
            cmd = [py, script_path, "benchmarks", "--which", "vi", "--outdir", vi_dir] + common + [
                "--seed",
                str(args.seed),
                "--vi_grid",
                str(args.vi_grid),
                "--vi_iters",
                str(args.vi_iters),
            ]
            _run(cmd)
            try:
                summ = _find_single("*_summary.json", vi_dir)
                d = json.load(open(summ, "r"))
                d["method"] = "vi"
                rows.append(d)
            except Exception as e:
                print("[study] warning: failed to parse VI summary:", e)

        elif args.do_vi and state_dim != 2:
            print("[study] skipping VI baseline: only implemented for 2D problems.")

        # 4) RL baselines (only really meaningful for Pendulum-v1)
        if args.do_rl and args.problem.startswith("pendulum"):
            rl_dir = os.path.join(study_dir, "rl_ppo")
            os.makedirs(rl_dir, exist_ok=True)
            cmd = [py, script_path, "rl_ppo", "--outdir", rl_dir, "--env", str(args.env), "--steps", str(args.rl_steps), "--seed", str(args.seed), "--device", str(args.device)]
            _run(cmd)
            try:
                summ = _find_single("*_summary.json", rl_dir)
                d = json.load(open(summ, "r"))
                d["method"] = "ppo"
                rows.append(d)
            except Exception as e:
                print("[study] warning: failed to parse PPO summary:", e)

            hyb_dir = os.path.join(study_dir, "rl_pinn_hybrid")
            os.makedirs(hyb_dir, exist_ok=True)
            cmd = [py, script_path, "rl_pinn_hybrid", "--outdir", hyb_dir, "--env", str(args.env), "--steps", str(args.rl_steps), "--physics_weight", str(args.physics_weight), "--seed", str(args.seed), "--device", str(args.device)]
            _run(cmd)
            try:
                summ = _find_single("*_summary.json", hyb_dir)
                d = json.load(open(summ, "r"))
                d["method"] = "ppo+pinn"
                rows.append(d)
            except Exception as e:
                print("[study] warning: failed to parse PPO+PINN summary:", e)

        # 5) Collate
        out_csv = os.path.join(study_dir, "study_comparison.csv")
        if rows:
            write_rows_csv(out_csv, rows)
            print("[study] wrote:", out_csv)
        else:
            print("[study] no rows to write")

        # Simple comparison plots (if possible)
        try:
            # J_mse bar
            jmse = [(r.get("method", ""), float(r.get("J_mse", r.get("J_mse_mean", float('nan'))))) for r in rows]
            jmse = [(m, v) for m, v in jmse if math.isfinite(v)]
            if jmse:
                plt.figure(figsize=(7, 4))
                plt.bar([m for m, _ in jmse], [v for _, v in jmse])
                plt.ylabel("J_mse")
                plt.title(f"Comparison J_mse ({args.problem})")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(study_dir, "comparison_J_mse.png"), dpi=160)
                plt.close()

            # RL return bar
            ret = [(r.get("method", ""), float(r.get("avg_return_last20", float('nan')))) for r in rows]
            ret = [(m, v) for m, v in ret if math.isfinite(v)]
            if ret:
                plt.figure(figsize=(7, 4))
                plt.bar([m for m, _ in ret], [v for _, v in ret])
                plt.ylabel("avg_return_last20")
                plt.title(f"RL comparison ({args.env})")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(study_dir, "comparison_return.png"), dpi=160)
                plt.close()
        except Exception as e:
            print("[study] warning: plotting failed:", e)

        print("[study] done. outputs in:", study_dir)


    elif args.cmd == "benchmarks":
        bench_id = f"benchmark_{args.which}_{args.problem}_{time.strftime('%Y%m%d_%H%M%S')}"
        if args.which == "tfc":
            tfc, train_stats = train_tfc(problem, cfg)
            member_rows, eval_summary = evaluate_ensemble(
                problem,
                [tfc],
                device=cfg.device,
                grid_n_2d=int(args.grid_n_eval),
                n_eval=int(args.n_eval),
                eval_all_members=False,
            )
            print("[tfc] train_stats:", train_stats)
            print("[tfc] eval:", eval_summary)
            if args.save_csv:
                write_json(os.path.join(args.outdir, f"{bench_id}_tfc_train.json"), train_stats)
                write_rows_csv(os.path.join(args.outdir, f"{bench_id}_tfc_eval_member.csv"), member_rows)
                write_rows_csv(
                    os.path.join(args.outdir, f"{bench_id}_tfc_eval_summary.csv"),
                    [{"bench_id": bench_id, "method": "tfc", **eval_summary, **train_stats}],
                )
                # A compact, study-friendly summary JSON (used by `study` to collate results)
                summary = {
                    "bench_id": bench_id,
                    "method": "tfc",
                    "problem": args.problem,
                    "seed": int(args.seed),
                    "J_mse": float(eval_summary.get("J_mse_mean", float("nan"))),
                    "u_mse": float(eval_summary.get("u_mse_mean", float("nan"))),
                    "res_rms": float(eval_summary.get("res_rms_mean", float("nan"))),
                    "res_max": float(eval_summary.get("res_max_mean", float("nan"))),
                    "res_mean": float(eval_summary.get("res_mean_mean", float("nan"))),
                }
                summary.update({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in train_stats.items()})
                write_json(os.path.join(args.outdir, f"{bench_id}_summary.json"), summary)


        elif args.which == "bellman":
            bcfg = BellmanConfig(seed=args.seed, device=args.device, hidden=tuple(args.hidden), act=args.act, epochs=args.epochs_hjb)
            critic, actor, train_stats = train_bellman_nn(problem, bcfg)
            eval_summary = evaluate_actor_critic(
                problem,
                critic,
                actor,
                device=bcfg.device,
                grid_n_2d=int(args.grid_n_eval),
                n_eval=int(args.n_eval),
            )
            print("[bellman] train_stats:", train_stats)
            print("[bellman] eval:", eval_summary)
            if args.save_csv:
                cfg_row = dataclasses.asdict(bcfg)
                write_json(os.path.join(args.outdir, f"{bench_id}_bellman_config.json"), cfg_row)
                write_json(os.path.join(args.outdir, f"{bench_id}_bellman_train.json"), train_stats)
                write_rows_csv(
                    os.path.join(args.outdir, f"{bench_id}_bellman_eval_summary.csv"),
                    [{"bench_id": bench_id, "method": "bellman", **eval_summary, **train_stats}],
                )
                summary = {
                    "bench_id": bench_id,
                    "method": "bellman",
                    "problem": args.problem,
                    "seed": int(args.seed),
                    "J_mse": float(eval_summary.get("J_mse", float("nan"))),
                    "u_mse": float(eval_summary.get("u_mse", float("nan"))),
                    "res_rms": float(eval_summary.get("res_rms", float("nan"))),
                    "res_max": float(eval_summary.get("res_max", float("nan"))),
                    "res_mean": float(eval_summary.get("res_mean", float("nan"))),
                }
                summary.update({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in train_stats.items()})
                write_json(os.path.join(args.outdir, f"{bench_id}_summary.json"), summary)


        elif args.which == "vi":
            if problem.state_dim != 2:
                raise ValueError("Value iteration baseline is only implemented for 2D problems.")
            out = value_iteration_hjb_2d(problem, grid_n=args.vi_grid, iters=args.vi_iters)
            plot_heatmap(out["x1"], out["x2"], out["V"], "Value iteration baseline (V)", outpath=f"{args.outdir}/{bench_id}_vi_V.png")

            # Evaluate VI against analytic/reference if available
            eval_row: Dict[str, float] = {
                "J_mse": float("nan"),
                "u_mse": float("nan"),
                "res_rms": float("nan"),
                "res_max": float("nan"),
            }
            if problem.has_analytic():
                X1, X2 = np.meshgrid(out["x1"], out["x2"], indexing="xy")
                pts = np.stack([X1.ravel(), X2.ravel()], axis=1)

                # NOTE: problem.J_star/u_star are defined on NumPy arrays for both analytic and VI-reference problems.
                J_true = problem.J_star(pts).reshape(out["V"].shape)
                eval_row["J_mse"] = float(np.mean((out["V"] - J_true) ** 2))

                if "U" in out:
                    u_true = problem.u_star(pts).reshape(out["U"].shape)
                    eval_row["u_mse"] = float(np.mean((out["U"] - u_true) ** 2))

            if args.save_csv:
                np.savez_compressed(os.path.join(args.outdir, f"{bench_id}_vi_arrays.npz"), **out)
                write_rows_csv(
                    os.path.join(args.outdir, f"{bench_id}_vi_summary.csv"),
                    [{"bench_id": bench_id, "method": "vi", **eval_row, "vi_grid": int(args.vi_grid), "vi_iters": int(args.vi_iters)}],
                )
                summary = {
                    "bench_id": bench_id,
                    "method": "vi",
                    "problem": args.problem,
                    "seed": int(args.seed),
                    "J_mse": float(eval_row.get("J_mse", float("nan"))),
                    "u_mse": float(eval_row.get("u_mse", float("nan"))),
                    "res_rms": float(eval_row.get("res_rms", float("nan"))),
                    "res_max": float(eval_row.get("res_max", float("nan"))),
                    "vi_grid": int(args.vi_grid),
                    "vi_iters": int(args.vi_iters),
                }
                write_json(os.path.join(args.outdir, f"{bench_id}_summary.json"), summary)


            print("[vi] eval:", eval_row)

        else:
            raise ValueError(args.which)


    elif args.cmd == "rl_ppo":
        rcfg = PPOConfig(seed=args.seed, device=args.device, steps=args.steps)
        run_id = f"rl_ppo_{args.env.replace('/', '-')}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
        stats = run_ppo(args.env, rcfg, physics_weight=0.0, record_history=bool(args.save_history or args.make_plots or args.save_csv))
        print("[ppo] done:", {k: stats[k] for k in stats if k not in {'ep_returns','episode_history','train_history'}})

        if args.save_csv:
            write_json(os.path.join(args.outdir, f"{run_id}_config.json"), dataclasses.asdict(rcfg))
            write_rows_csv(os.path.join(args.outdir, f"{run_id}_summary.csv"), [{"run_id": run_id, "env": args.env, "seed": args.seed, "steps": stats.get('steps'), "time_sec": stats.get('time_sec'), "avg_return_last20": stats.get('avg_return_last20')}])
            write_json(
                os.path.join(args.outdir, f"{run_id}_summary.json"),
                {
                    "run_id": run_id,
                    "method": "ppo",
                    "env": args.env,
                    "seed": int(args.seed),
                    "steps": int(stats.get("steps", 0) or 0),
                    "time_sec": float(stats.get("time_sec", float("nan"))),
                    "avg_return_last20": float(stats.get("avg_return_last20", float("nan"))),
                },
            )
            # Episode returns
            if stats.get("episode_history"):
                write_rows_csv(os.path.join(args.outdir, f"{run_id}_episode_returns.csv"), [{"run_id": run_id, **r} for r in stats["episode_history"]])
            # Training history
            if stats.get("train_history"):
                write_rows_csv(os.path.join(args.outdir, f"{run_id}_train_history.csv"), [{"run_id": run_id, **r} for r in stats["train_history"]])

        if args.make_plots:
            plot_rl_curves(stats, run_id=run_id, outdir=args.outdir, title_prefix="PPO ")

    elif args.cmd == "rl_pinn_hybrid":
        rcfg = PPOConfig(seed=args.seed, device=args.device, steps=args.steps)
        run_id = f"rl_pinnhyb_{args.env.replace('/', '-')}_seed{args.seed}_w{float(args.physics_weight):g}_{time.strftime('%Y%m%d_%H%M%S')}"
        stats = run_ppo(args.env, rcfg, physics_weight=float(args.physics_weight), record_history=bool(args.save_history or args.make_plots or args.save_csv))
        print("[ppo+pinn] done:", {k: stats[k] for k in stats if k not in {'ep_returns','episode_history','train_history'}})

        if args.save_csv:
            cfg_row = dataclasses.asdict(rcfg)
            cfg_row["physics_weight"] = float(args.physics_weight)
            write_json(os.path.join(args.outdir, f"{run_id}_config.json"), cfg_row)
            write_rows_csv(
                os.path.join(args.outdir, f"{run_id}_summary.csv"),
                [{"run_id": run_id, "env": args.env, "seed": args.seed, "physics_weight": float(args.physics_weight), "steps": stats.get('steps'), "time_sec": stats.get('time_sec'), "avg_return_last20": stats.get('avg_return_last20')}],
            )
            write_json(
                os.path.join(args.outdir, f"{run_id}_summary.json"),
                {
                    "run_id": run_id,
                    "method": "ppo+pinn",
                    "env": args.env,
                    "seed": int(args.seed),
                    "physics_weight": float(args.physics_weight),
                    "steps": int(stats.get("steps", 0) or 0),
                    "time_sec": float(stats.get("time_sec", float("nan"))),
                    "avg_return_last20": float(stats.get("avg_return_last20", float("nan"))),
                },
            )

            if stats.get("episode_history"):
                write_rows_csv(os.path.join(args.outdir, f"{run_id}_episode_returns.csv"), [{"run_id": run_id, **r} for r in stats["episode_history"]])
            if stats.get("train_history"):
                write_rows_csv(os.path.join(args.outdir, f"{run_id}_train_history.csv"), [{"run_id": run_id, **r} for r in stats["train_history"]])

        if args.make_plots:
            plot_rl_curves(stats, run_id=run_id, outdir=args.outdir, title_prefix=f"PPO+PINN (w={float(args.physics_weight):g}) ")

    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
