import numpy as np
import pandas as pd

def rk4_step(f, y, t, dt, args):
    """Classical 4th-order Runge-Kutta step (non-autonomous safe)"""
    k1 = f(y, t, *args)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt, *args)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt, *args)
    k4 = f(y + dt*k3, t + dt, *args)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def heterogeneous_mean_field_ode(y, t, params):
    """
    Two-class heterogeneous SIM mean-field ODEs
    Classes: Hub (index 0) and Workers (index 1)
    
    y = [S_h, I_h, M_h, S_w, I_w, M_w]
    """
    S_h, I_h, M_h, S_w, I_w, M_w = y
    N_h = S_h + I_h + M_h
    N_w = S_w + I_w + M_w
    N   = N_h + N_w

    # Unpack time-dependent or constant parameters
    beta_hh = params["beta_hh"](t)   # hub → hub
    beta_hw = params["beta_hw"](t)   # hub → worker
    beta_wh = params["beta_wh"](t)   # worker → hub
    beta_ww = params["beta_ww"](t)   # worker → worker
    delta_h = params["delta_h"](t)
    delta_w = params["delta_w"](t)

    # Force of infection on each class
    lambda_h = (beta_hh * I_h + beta_wh * I_w) / max(N, 1e-12)
    lambda_w = (beta_hw * I_h + beta_ww * I_w) / max(N, 1e-12)

    dS_h = -lambda_h * S_h
    dI_h =  lambda_h * S_h - delta_h * I_h
    dM_h =  delta_h * I_h

    dS_w = -lambda_w * S_w
    dI_w =  lambda_w * S_w - delta_w * I_w
    dM_w =  delta_w * I_w

    return np.array([dS_h, dI_h, dM_h, dS_w, dI_w, dM_w])

def simulate_heterogeneous_rk4(
    N_hub=1,
    N_workers=99,
    t_final=15.0,
    dt=0.05,
    # time-varying callables (or constants via lambda t: value)
    beta_hh=None, beta_hw=None, beta_wh=None, beta_ww=None,
    delta_h=None, delta_w=None,
    # convenient stationary defaults (hub-and-spoke style)
    T_hub=0.90, T_worker=0.15,
    alpha_hub=0.95, alpha_worker=0.80,
    gamma_hub=0.10, gamma_worker=0.25,
    delta_hub=0.18, delta_worker=0.22,
    k_hub=99.0,          # hub connected to all workers
    k_worker_hub=1.0,    # each worker connected back to hub
    k_worker_worker=4.0  # sparse worker mesh
):
    """
    Heterogeneous two-class RK4 integrator.
    Returns a DataFrame with full compartment trajectories + instantaneous R0 proxy.
    """
    # Build default time-constant rate functions if none supplied
    def make_beta(k, T, alpha, gamma):
        return lambda t: k * T * alpha * (1.0 - gamma)

    if beta_hh is None:
        beta_hh = make_beta(0.0, T_hub, alpha_hub, gamma_hub)          # usually no hub-hub
    if beta_hw is None:
        beta_hw = make_beta(k_hub, T_hub, alpha_hub, gamma_worker)
    if beta_wh is None:
        beta_wh = make_beta(k_worker_hub, T_worker, alpha_worker, gamma_hub)
    if beta_ww is None:
        beta_ww = make_beta(k_worker_worker, T_worker, alpha_worker, gamma_worker)

    if delta_h is None:
        delta_h = lambda t: delta_hub
    if delta_w is None:
        delta_w = lambda t: delta_worker

    params = {
        "beta_hh": beta_hh, "beta_hw": beta_hw,
        "beta_wh": beta_wh, "beta_ww": beta_ww,
        "delta_h": delta_h, "delta_w": delta_w
    }

    # Initial condition: one worker infected
    y = np.array([
        N_hub, 0.0, 0.0,               # Hub: fully susceptible
        N_workers-1.0, 1.0, 0.0        # Workers: 1 infected
    ], dtype=float)

    times = np.arange(0.0, t_final + dt, dt)
    history = {
        "t": [], "S_h": [], "I_h": [], "M_h": [],
        "S_w": [], "I_w": [], "M_w": [],
        "I_total": [], "R0_proxy": []
    }

    for t in times:
        # Simple instantaneous R0 proxy (dominant eigenvalue approximation)
        b_hw = beta_hw(t)
        b_wh = beta_wh(t)
        b_ww = beta_ww(t)
        d_h  = delta_h(t)
        d_w  = delta_w(t)
        # rough next-generation estimate
        R0_proxy = 0.5 * (b_ww/d_w + np.sqrt((b_ww/d_w)**2 + 4*b_hw*b_wh/(d_h*d_w)))

        history["t"].append(t)
        history["S_h"].append(y[0])
        history["I_h"].append(y[1])
        history["M_h"].append(y[2])
        history["S_w"].append(y[3])
        history["I_w"].append(y[4])
        history["M_w"].append(y[5])
        history["I_total"].append(y[1] + y[4])
        history["R0_proxy"].append(R0_proxy)

        y = rk4_step(heterogeneous_mean_field_ode, y, t, dt, args=(params,))
        y = np.maximum(y, 0.0)

    return pd.DataFrame(history)

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Stationary hub-and-spoke (high risk)
    df_high = simulate_heterogeneous_rk4(
        T_hub=0.90, T_worker=0.15,
        gamma_hub=0.05, gamma_worker=0.20
    )

    # Mitigated (stronger worker damping + lower hub trust)
    df_safe = simulate_heterogeneous_rk4(
        T_hub=0.25, T_worker=0.08,
        gamma_hub=0.40, gamma_worker=0.55,
        delta_hub=0.30, delta_worker=0.35
    )

    print("=== High-risk heterogeneous regime ===")
    print(f"Peak total I : {df_high['I_total'].max():.1f}")
    print(f"Final R0 proxy: {df_high['R0_proxy'].iloc[-1]:.2f}")
    print(df_high[["t", "I_h", "I_w", "I_total"]].iloc[::50].to_string(index=False))

    print("\n=== Mitigated heterogeneous regime ===")
    print(f"Peak total I : {df_safe['I_total'].max():.1f}")
    print(f"Final R0 proxy: {df_safe['R0_proxy'].iloc[-1]:.2f}")
