import numpy as np
import pandas as pd

def rk4_step(f, y, t, dt, args):
    """Classical 4th-order Runge-Kutta step"""
    k1 = f(y, t, *args)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt, *args)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt, *args)
    k4 = f(y + dt*k3, t + dt, *args)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def mean_field_ode(y, t, beta, delta):
    """
    Continuous-time mean-field SIM equations
    y = [S, I, M]
    beta = effective transmission rate = <k> * T * alpha * (1-gamma)
    delta = quarantine rate
    """
    S, I, M = y
    N = S + I + M
    dS = -beta * S * I / N
    dI =  beta * S * I / N - delta * I
    dM =  delta * I
    return np.array([dS, dI, dM])

def simulate_rk4(num_agents=100, t_final=15.0, dt=0.05,
                 global_trust=0.85, damping=0.0, delta=0.2, k_avg=15.26, alpha=0.9):
    """
    Integrate the mean-field SIM system with RK4
    """
    beta = k_avg * global_trust * alpha * (1.0 - damping)   # effective transmission rate
    
    # Initial condition: 1 infected, rest susceptible
    y = np.array([num_agents-1.0, 1.0, 0.0])
    
    times = np.arange(0, t_final+dt, dt)
    history = {"t": [], "S": [], "I": [], "M": []}
    
    for t in times:
        history["t"].append(t)
        history["S"].append(y[0])
        history["I"].append(y[1])
        history["M"].append(y[2])
        y = rk4_step(mean_field_ode, y, t, dt, args=(beta, delta))
        # Keep non-negative
        y = np.maximum(y, 0.0)
    
    return pd.DataFrame(history), beta

# -------------------------------------------------
# Scenario A: Oversight-Deficient (R0 ≈ 58.37)
df_def, beta_def = simulate_rk4(global_trust=0.85, damping=0.0, delta=0.2)
R0_def = beta_def / 0.2

# Scenario B: Zero-Trust Mitigated (R0 ≈ 0.62)
df_mit, beta_mit = simulate_rk4(global_trust=0.06, damping=0.85, delta=0.2)
R0_mit = beta_mit / 0.2

print(f"Deficient  beta = {beta_def:.2f}   R0 = {R0_def:.2f}")
print(f"Mitigated  beta = {beta_mit:.2f}   R0 = {R0_mit:.2f}")
print("\n=== Deficient (RK4) – first/last rows ===")
print(df_def.head(3).to_string(index=False))
print("...")
print(df_def.tail(3).to_string(index=False))
print(f"\nPeak I (deficient): {df_def['I'].max():.1f}")

print("\n=== Mitigated (RK4) – first/last rows ===")
print(df_mit.head(3).to_string(index=False))
print("...")
print(df_mit.tail(3).to_string(index=False))
print(f"\nPeak I (mitigated): {df_mit['I'].max():.1f}")
