import numpy as np
import pandas as pd

def rk4_step(f, y, t, dt, args):
    """Classical 4th-order Runge-Kutta step (works for non-autonomous ODEs)"""
    k1 = f(y, t, *args)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt, *args)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt, *args)
    k4 = f(y + dt*k3, t + dt, *args)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def mean_field_ode(y, t, beta_func, delta_func):
    """
    Non-autonomous mean-field SIM equations
    y = [S, I, M]
    beta_func(t)  -> instantaneous transmission rate
    delta_func(t) -> instantaneous quarantine rate
    """
    S, I, M = y
    N = S + I + M
    if N <= 0:
        return np.zeros(3)
    
    beta = beta_func(t)
    delta = delta_func(t)
    
    dS = -beta * S * I / N
    dI =  beta * S * I / N - delta * I
    dM =  delta * I
    return np.array([dS, dI, dM])

def simulate_rk4_timevarying(
    num_agents=100,
    t_final=15.0,
    dt=0.05,
    beta_func=None,
    delta_func=None,
    # convenience defaults for stationary cases
    global_trust=0.85,
    damping=0.0,
    delta=0.2,
    k_avg=15.26,
    alpha=0.9
):
    """
    Integrate the non-stationary SIM system with RK4.
    
    If beta_func / delta_func are supplied they are used directly.
    Otherwise a stationary beta is computed from the classic parameters.
    """
    if beta_func is None:
        beta_const = k_avg * global_trust * alpha * (1.0 - damping)
        beta_func = lambda t: beta_const
    if delta_func is None:
        delta_func = lambda t: delta

    y = np.array([num_agents - 1.0, 1.0, 0.0], dtype=float)
    times = np.arange(0.0, t_final + dt, dt)
    
    history = {"t": [], "S": [], "I": [], "M": [], "beta": [], "delta": [], "R0": []}
    
    for t in times:
        b = beta_func(t)
        d = delta_func(t)
        R0 = b / d if d > 0 else np.inf
        
        history["t"].append(t)
        history["S"].append(y[0])
        history["I"].append(y[1])
        history["M"].append(y[2])
        history["beta"].append(b)
        history["delta"].append(d)
        history["R0"].append(R0)
        
        y = rk4_step(mean_field_ode, y, t, dt, args=(beta_func, delta_func))
        y = np.maximum(y, 0.0)          # enforce non-negativity
    
    return pd.DataFrame(history)

# ----------------------------------------------------------------------
# Example 1 – Original stationary cases (backward compatible)
# ----------------------------------------------------------------------
df_def = simulate_rk4_timevarying(global_trust=0.85, damping=0.0, delta=0.2)
df_mit = simulate_rk4_timevarying(global_trust=0.06, damping=0.85, delta=0.2)

print("=== Stationary Deficient ===")
print(f"Peak I = {df_def['I'].max():.2f}   Final R0 = {df_def['R0'].iloc[-1]:.2f}")
print("=== Stationary Mitigated ===")
print(f"Peak I = {df_mit['I'].max():.2f}   Final R0 = {df_mit['R0'].iloc[-1]:.2f}")

# ----------------------------------------------------------------------
# Example 2 – Time-varying (non-stationary) demonstration
# ----------------------------------------------------------------------
# Scenario: damping slowly degrades (silent decay of γ) while trust stays high
def beta_decay(t):
    # γ(t) = 0.1 + 0.7 * (1 - exp(-0.15*t))   → damping rises, 1-γ falls
    gamma_t = 0.10 + 0.70 * (1.0 - np.exp(-0.15 * t))
    k_avg, T, alpha = 15.26, 0.85, 0.9
    return k_avg * T * alpha * (1.0 - gamma_t)

def delta_const(t):
    return 0.20

df_drift = simulate_rk4_timevarying(
    t_final=20.0,
    beta_func=beta_decay,
    delta_func=delta_const
)

print("\n=== Non-stationary drift (damping decay) ===")
print(f"Initial R0 = {df_drift['R0'].iloc[0]:.2f}")
print(f"Final   R0 = {df_drift['R0'].iloc[-1]:.2f}")
print(f"Peak I     = {df_drift['I'].max():.2f}")
print(df_drift[["t", "I", "R0"]].iloc[::40].to_string(index=False))
