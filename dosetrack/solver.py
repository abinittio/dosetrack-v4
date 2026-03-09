"""
Fixed-step RK4 ODE integrator.

Why RK4 over scipy.integrate: we own every line of the numerics. Fixed-step
RK4 is transparent, stable for moderately stiff systems like Michaelis-Menten
pharmacokinetics, and trivially fast at the timescales we simulate (24-168h).
No external dependency for the core math.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

StateVec = np.ndarray  # 1-D float array


def rk4_step(
    f: Callable[[float, StateVec], StateVec],
    t: float,
    y: StateVec,
    dt: float,
) -> StateVec:
    """Advance state y from t to t+dt using fourth-order Runge-Kutta."""
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(
    f: Callable[[float, StateVec], StateVec],
    y0: StateVec,
    t_span: tuple[float, float],
    dt: float = 0.01,
    dt_output: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate dy/dt = f(t, y) over t_span with fixed step dt.

    Parameters
    ----------
    f : ODE right-hand side, signature f(t, y) -> dy/dt
    y0 : initial state vector
    t_span : (t_start, t_end) in hours
    dt : internal step size in hours (default 0.01h = 36s)
    dt_output : output resolution. If None, returns every internal step.

    Returns
    -------
    t_out : 1-D array of output times
    y_out : 2-D array, shape (len(t_out), len(y0))
    """
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / dt))
    dt_actual = (t_end - t_start) / n_steps  # adjust for exact endpoint

    # Pre-allocate internal trajectory
    t_all = np.linspace(t_start, t_end, n_steps + 1)
    y_all = np.empty((n_steps + 1, len(y0)))
    y_all[0] = y0

    y = y0.copy()
    for i in range(n_steps):
        y = rk4_step(f, t_all[i], y, dt_actual)
        # Clamp any state that should be non-negative
        np.maximum(y, 0.0, out=y)
        y_all[i + 1] = y

    if dt_output is None or dt_output <= dt_actual:
        return t_all, y_all

    # Decimate to output resolution by nearest-neighbour lookup
    t_out = np.arange(t_start, t_end + dt_output / 2, dt_output)
    indices = np.searchsorted(t_all, t_out, side="left")
    indices = np.clip(indices, 0, len(t_all) - 1)
    return t_out, y_all[indices]
