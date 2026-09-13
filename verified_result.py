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
    """Continuous-time mean-field SIM equations"""
    S, I, M = y
    N = S + I + M
    if N <= 0:
        return np.zeros(3)
    dS = -beta * S * I / N
    dI =  beta * S * I / N - delta * I
    dM =  delta * I
    return np.array([dS, dI, dM])

def simulate_rk4(num_agents=100, t_final=15.0, dt=0.05,
                 global_trust=0.85, damping=0.0, delta=0.2,
                 k_avg=15.26, alpha=0.9):
    beta = k_avg * global_trust * alpha * (1.0 - damping)
    y = np.array([num_agents - 1.0, 1.0, 0.0])
    times = np.arange(0, t_final + dt, dt)
    
    history = {"t": [], "S": [], "I": [], "M": []}
    for t in times:
        history["t"].append(t)
        history["S"].append(y[0])
        history["I"].append(y[1])
        history["M"].append(y[2])
        y = rk4_step(mean_field_ode, y, t, dt, args=(beta, delta))
        y = np.maximum(y, 0.0)  # enforce non-negativity
    
    return pd.DataFrame(history), beta

# ============================================================
# Five architectural regimes from Section 4.4.3
# ============================================================
regimes = [
    ("I.   Unmitigated Monolithic Mesh",       15.26, 0.85, 0.00, 0.20),
    ("II.  Dense Token-Filtered Swarm",        15.26, 0.85, 0.70, 0.20),
    ("III. Segmented / High-Trust Topology",    4.50, 0.85, 0.00, 0.20),
    ("IV.  Supercritical Controlled Border",   15.26, 0.20, 0.85, 0.20),
    ("V.   Zero-Trust Engineered Frontier",    15.26, 0.06, 0.85, 0.20),
]

print(f"{'Regime':<42} {'β':>8} {'R₀':>8} {'Peak I':>10} {'Final S':>10}")
print("-" * 82)

for name, k, T, gamma, delta in regimes:
    df, beta = simulate_rk4(
        k_avg=k,
        global_trust=T,
        damping=gamma,
        delta=delta
    )
    R0 = beta / delta
    peak_I = df["I"].max()
    final_S = df["S"].iloc[-1]
    
    print(f"{name:<42} {beta:8.4f} {R0:8.4f} {peak_I:10.2f} {final_S:10.2f}")
