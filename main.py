import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARÁMETROS FÍSICOS Y CONFIGURACIÓN (Tabla 1)
# ==========================================
L = 0.005           # Espesor [m]
Ta = 37.0 + 273.15  # Temp. arterial [K] (310.15 K)
T_inf = 23.0 + 273.15 # Temp. ambiente [K] (296.15 K)
h = 10.0            # Coef. convección [W/m^2/K]

rho_b = 1060.0      # Densidad sangre [kg/m^3]
cb = 3617.0         # Calor esp. sangre [J/kg/K]

# Valores Sano vs Tumoral
properties = {
    'healthy': {
        'rho': 1085.0, 'c': 3680.0, 'k': 0.47,
        'wb': 0.00105, 'qm': 631.0
    },
    'tumor': {
        'rho': 1085.0, 'c': 3680.0, 'k': 0.47,
        'wb': 0.00525, 'qm': 6310.0
    }
}

def get_grouped_params(tissue_type):
    p = properties[tissue_type]
    A = p['rho'] * p['c']
    B = p['k']
    C = p['wb'] * rho_b * cb
    D = p['qm']
    return np.array([A, B, C, D])

# Malla numérica
N_vol = 27
dx = L / N_vol
dt = 0.05
t_final_reheat = 50.0
N_steps_reheat = int(t_final_reheat / dt)
time_vec = np.linspace(0, t_final_reheat, N_steps_reheat)

# ==========================================
# 2. SOLUCIONADOR DE VOLÚMENES FINITOS IMPLÍCITO (1D)
# ==========================================
def solve_pennes_1d(P, cool_time=15.0, reheat_time=50.0, dt=0.05):
    A, B, C, D = P
    r = B / (dx**2)
    
    # 1. Estado Estacionario Inicial
    M_ss = np.copy(np.diag((2*r + C) * np.ones(N_vol)))
    M_ss += np.diag(-r * np.ones(N_vol-1), -1) + np.diag(-r * np.ones(N_vol-1), 1)
    M_ss[0, 0] = 3*r + C
    
    alpha_conv = h / (dx/2 + B/h)
    M_ss[-1, -1] = r + C + alpha_conv
    
    rhs_ss = (C * Ta + D) * np.ones(N_vol)
    rhs_ss[0] += 2 * r * Ta
    rhs_ss[-1] += alpha_conv * T_inf
    
    T_curr = np.linalg.solve(M_ss, rhs_ss)

    # 2. Fase de Enfriamiento (15 s a T_surf = 273.15 K)
    N_steps_cool = int(cool_time / dt)
    T_cool_surf = 273.15
    
    M_cool = np.copy(np.diag(((A/dt) + 2*r + C) * np.ones(N_vol)))
    M_cool += np.diag(-r * np.ones(N_vol-1), -1) + np.diag(-r * np.ones(N_vol-1), 1)
    M_cool[0, 0] = (A/dt) + 3*r + C
    M_cool[-1, -1] = (A/dt) + 3*r + C
    M_cool_inv = np.linalg.inv(M_cool)
    
    for _ in range(N_steps_cool):
        rhs = (A / dt) * T_curr + C * Ta + D
        rhs[0] += 2 * r * Ta
        rhs[-1] += 2 * r * T_cool_surf
        T_curr = M_cool_inv @ rhs
        
    # 3. Fase de Recalentamiento (50 s observados)
    M_reheat = np.copy(np.diag(((A/dt) + 2*r + C) * np.ones(N_vol)))
    M_reheat += np.diag(-r * np.ones(N_vol-1), -1) + np.diag(-r * np.ones(N_vol-1), 1)
    M_reheat[0, 0] = (A/dt) + 3*r + C
    M_reheat[-1, -1] = (A/dt) + r + C + alpha_conv
    M_inv = np.linalg.inv(M_reheat)

    T_surface = []
    for _ in range(N_steps_reheat):
        rhs = (A / dt) * T_curr + C * Ta + D
        rhs[0] += 2 * r * Ta
        rhs[-1] += alpha_conv * T_inf
        T_curr = M_inv @ rhs
        T_surface.append(T_curr[-1])
        
    return np.array(T_surface)

# ==========================================
# 3. GENERACIÓN DE DATOS SINTÉTICOS
# ==========================================
np.random.seed(42)
P_healthy_true = get_grouped_params('healthy')
P_tumor_true = get_grouped_params('tumor')

T_exact_healthy = solve_pennes_1d(P_healthy_true)
T_exact_tumor = solve_pennes_1d(P_tumor_true)

sigma_noise = 0.3
Y_healthy = T_exact_healthy + np.random.normal(0, sigma_noise, size=len(T_exact_healthy))
Y_tumor = T_exact_tumor + np.random.normal(0, sigma_noise, size=len(T_exact_tumor))

# ==========================================
# 4. INFERENCIA MCMC (METROPOLIS-HASTINGS)
# ==========================================
def log_prior(P, P_mu):
    A, B, C, D = P
    for j, val in enumerate([A, B, D]):
        mu = P_mu[j if j < 2 else 3]
        if val < 0.9 * mu or val > 1.1 * mu:
            return -np.inf
    if C < 0 or C > 40257.21:
        return -np.inf
        
    log_p = 0.0
    for j, val in enumerate([A, B, D]):
        mu = P_mu[j if j < 2 else 3]
        sig = 0.0005 * mu
        log_p += -0.5 * ((val - mu) / sig)**2
    return log_p

def run_mcmc(Y_obs, P_init, P_mu, w_steps, n_samples=20000):
    chain = np.zeros((n_samples, 4))
    chain[0] = P_init
    
    T_curr = solve_pennes_1d(chain[0])
    log_like_curr = -0.5 * np.sum(((Y_obs - T_curr) / sigma_noise)**2)
    log_prior_curr = log_prior(chain[0], P_mu)
    
    accepted = 0
    for t in range(1, n_samples):
        r = np.random.uniform(0, 1, size=4)
        P_cand = chain[t-1] + w_steps * (2 * r - 1)
        
        log_prior_cand = log_prior(P_cand, P_mu)
        if np.isinf(log_prior_cand):
            chain[t] = chain[t-1]
            continue
            
        T_cand = solve_pennes_1d(P_cand)
        log_like_cand = -0.5 * np.sum(((Y_obs - T_cand) / sigma_noise)**2)
        
        log_alpha = (log_like_cand + log_prior_cand) - (log_like_curr + log_prior_curr)
        
        if np.log(np.random.uniform(0, 1)) < log_alpha:
            chain[t] = P_cand
            log_like_curr = log_like_cand
            log_prior_curr = log_prior_cand
            accepted += 1
        else:
            chain[t] = chain[t-1]
            
    print(f"Tasa de Aceptación MCMC: {accepted / n_samples:.2%}")
    return chain

w_tumor = np.array([79.86, 9.4e-6, 1.41, 0.403])
print("Ejecutando muestreo MCMC...")
full_chain_tumor = run_mcmc(Y_tumor, P_init=P_healthy_true, P_mu=P_healthy_true, 
                            w_steps=w_tumor, n_samples=20000)

burn_in = 5000
converged_chain = full_chain_tumor[burn_in:]

# ==========================================
# 5. VISUALIZACIÓN DE RESULTADOS (MATPLOTLIB)
# ==========================================

# ------------------------------------------
# FIGURA 1: Temperaturas Exactas vs Medidas (Eq. Fig. 2 del Paper)
# ------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(time_vec, T_exact_healthy, 'k-', label='Temperatura Exacta')
plt.plot(time_vec, Y_healthy, 'r.', markersize=3, alpha=0.6, label='Temperatura Simulada')
plt.title('a) Tejido Sano')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [K]')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time_vec, T_exact_tumor, 'k-', label='Temperatura Exacta')
plt.plot(time_vec, Y_tumor, 'r.', markersize=3, alpha=0.6, label='Temperatura Simulada')
plt.title('b) Tejido Tumoral')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [K]')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()

# ------------------------------------------
# FIGURA 2: Transición de la Cadena MCMC con Burn-in (Eq. Fig. 9 del Paper)
# ------------------------------------------
plt.figure(figsize=(8, 4))
c_rel_full = full_chain_tumor[:, 2] / P_tumor_true[2]
plt.plot(c_rel_full, color='tab:blue', linewidth=0.8)
plt.axvline(x=burn_in, color='red', linestyle='--', label=f'Fin de Burn-in ({burn_in})')
plt.title('Evolución de la Cadena Markoviana para el Parámetro C (Perfusión)')
plt.xlabel('Estados en la Cadena Markoviana')
plt.ylabel('C / C_exacto')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------
# FIGURA 3: Cadenas Convergidas y Posteriores Marginales (Eq. Fig. 10 a 13)
# ------------------------------------------
param_names = ['A (Capacidad Calorífica)', 'B (Conductividad)', 'C (Perfusión)', 'D (Metabolismo)']
fig, axes = plt.subplots(4, 2, figsize=(12, 10))

for j in range(4):
    rel_vals = converged_chain[:, j] / P_tumor_true[j]
    
    # Traceplot (Cadena Convergida)
    axes[j, 0].plot(rel_vals, color='gray', linewidth=0.5)
    axes[j, 0].set_title(f'Cadena Convergida: {param_names[j]}')
    axes[j, 0].set_xlabel('Estados')
    axes[j, 0].set_ylabel(f'P_{j+1} / P_{j+1}_exacto')
    axes[j, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Histograma (Distribución a Posteriori)
    axes[j, 1].hist(rel_vals, bins=20, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    mean_val = np.mean(rel_vals)
    std_val = np.std(rel_vals)
    axes[j, 1].set_title(f'Posterior Marginal {param_names[j]}\n(Media = {mean_val:.4f}, SD = {std_val:.6f})')
    axes[j, 1].set_xlabel(f'P_{j+1} / P_{j+1}_exacto')
    axes[j, 1].set_ylabel('Densidad')
    axes[j, 1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()