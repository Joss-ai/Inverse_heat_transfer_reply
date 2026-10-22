"""
pennes_mcmc.py
==============
Biblioteca para replicar Rojczyk et al. (2015), "Inverse heat transfer problems:
an application to bioheat transfer", CAMES 22(4):365-383, doi:10.24423/cames.20.

Contenido
---------
* Problema directo: ecuacion de Pennes 1D en una placa de espesor L,
  volumenes finitos centrados en celda, Euler implicito (como en el paper:
  "central difference, fully implicit finite volume method", 27 volumenes,
  dt = 0.05 s, 50 s de recalentamiento).
    - forward_eig    : solucion exacta del esquema discreto via descomposicion
                       espectral (rapida, la usa el MCMC).
    - forward_march  : la misma discretizacion integrada paso a paso
                       (solver tridiagonal); sirve para verificar forward_eig.
* Coeficientes de sensibilidad reducidos J_Pj = Pj dT/dPj (Ec. 18).
* Verosimilitud gaussiana (Ec. 5), priors (Ec. 20 y uniforme para C).
* Metropolis-Hastings con propuesta de paseo aleatorio uniforme (Ec. 21).
* Criterio de Geweke (Ecs. 22-23).

Parametros estimados: P = [A, B, C, D] = [rho*c, k, w_b*rho_b*c_b, q_m] (Ec. 19).
h, T_inf y T_a se consideran conocidos (como en el paper).

Unidades: temperaturas internas en degC; las salidas de temperatura van en K
(como en las figuras del paper).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, solve_banded

KELVIN = 273.15

# ----------------------------------------------------------------------------
# Tabla 1 del paper
# ----------------------------------------------------------------------------
TABLE1 = dict(
    L=0.005, Ta=37.0, h=10.0, Tinf=23.0,
    rho=1085.0, c=3680.0, k=0.47,
    cb=3617.0, rhob=1060.0,
    wb_t=0.00105, wb_tumor=0.00525,
    qm_t=631.0, qm_tumor=6310.0,
)


def exact_params(tissue: str = "healthy") -> np.ndarray:
    """Vector P = [A, B, C, D] 'exacto' de la Tabla 1."""
    t = TABLE1
    A = t["rho"] * t["c"]
    B = t["k"]
    if tissue == "healthy":
        C = t["wb_t"] * t["rhob"] * t["cb"]
        D = t["qm_t"]
    elif tissue == "tumor":
        C = t["wb_tumor"] * t["rhob"] * t["cb"]
        D = t["qm_tumor"]
    else:
        raise ValueError(tissue)
    return np.array([A, B, C, D], dtype=float)


class Setup:
    """Configuracion numerica y experimental (todo lo que no se estima)."""

    def __init__(self, n_vol=27, dt=0.05, t_final=50.0, t_cool=15.0,
                 T_cool=0.0, meas_every=1, L=None, h=None, Tinf=None, Ta=None):
        t = TABLE1
        self.n = int(n_vol)
        self.L = t["L"] if L is None else L
        self.h = t["h"] if h is None else h
        self.Tinf = t["Tinf"] if Tinf is None else Tinf
        self.Ta = t["Ta"] if Ta is None else Ta
        self.dt = float(dt)
        self.n_steps = int(round(t_final / dt))
        self.n_cool = int(round(t_cool / dt))
        self.T_cool = float(T_cool)
        self.dx = self.L / self.n
        # instantes de medicion: t_n = n*dt, n = meas_every, 2*meas_every, ...
        self.meas_idx = np.arange(meas_every, self.n_steps + 1, meas_every)
        self.t_meas = self.meas_idx * self.dt

    # --- matrices del operador espacial: A dT/dt = -K T + s ----------------
    def operator(self, P, surface="robin"):
        """Devuelve (K, s, sfc) con K simetrica (n x n) y s (n,).

        surface='robin'    : k dT/dx + h T = h T_inf   (Ec. 17_3)
        surface='dirichlet': T(L) = T_cool              (periodo de enfriamiento)
        sfc(Tvec) devuelve la temperatura de la superficie x = L en degC.
        """
        A, B, C, D = P
        n, dx = self.n, self.dx
        g = B / dx**2
        main = np.full(n, 2.0 * g)
        off = np.full(n - 1, -g)
        s = np.full(n, C * self.Ta + D)
        # x = 0: Dirichlet T = Ta a media celda
        main[0] = 3.0 * g            # g (vecino) + 2g (media celda)
        s[0] += 2.0 * g * self.Ta
        if surface == "robin":
            U = 1.0 / (dx / (2.0 * B) + 1.0 / self.h)   # conductancia W/m2K
            main[-1] = g + U / dx
            s[-1] += U / dx * self.Tinf

            def sfc(T):
                q = U * (T[..., -1] - self.Tinf)
                return T[..., -1] - q * dx / (2.0 * B)
        elif surface == "dirichlet":
            main[-1] = 3.0 * g
            s[-1] += 2.0 * g * self.T_cool

            def sfc(T):
                return np.full(np.shape(T)[:-1], self.T_cool)
        else:
            raise ValueError(surface)
        main += C                     # termino de perfusion
        K = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
        return K, s, sfc, (main, off)


# ----------------------------------------------------------------------------
# Problema directo
# ----------------------------------------------------------------------------
def initial_condition(P, S: Setup, return_profiles=False):
    """Estado estacionario (Robin) -> 15 s de contacto con T_cool -> F(x)."""
    Kr, sr, _, _ = S.operator(P, "robin")
    T_ss = np.linalg.solve(Kr, sr)
    Kc, sc, _, _ = S.operator(P, "dirichlet")
    lam, V = eigh(Kc)
    Tc_star = np.linalg.solve(Kc, sc)
    mu = 1.0 / (1.0 + S.dt * lam / P[0])
    F = Tc_star + V @ (mu**S.n_cool * (V.T @ (T_ss - Tc_star)))
    if return_profiles:
        return F, T_ss
    return F


def forward_eig(P, S: Setup):
    """Temperatura superficial (K) en S.t_meas. Exacta para el esquema
    implicito con operador constante: T^n = T* + V diag(mu^n) V^T (F - T*)."""
    P = np.asarray(P, dtype=float)
    F = initial_condition(P, S)
    Kr, sr, sfc, _ = S.operator(P, "robin")
    lam, V = eigh(Kr)
    T_star = np.linalg.solve(Kr, sr)
    mu = 1.0 / (1.0 + S.dt * lam / P[0])
    coef = V.T @ (F - T_star)                       # (n,)
    # temperatura de la ultima celda en cada instante
    powers = np.exp(np.outer(S.meas_idx, np.log(mu)))   # (m, n)
    T_last = T_star[-1] + powers @ (V[-1, :] * coef)
    # superficie a partir de la ultima celda (relacion lineal de Robin)
    U = 1.0 / (S.dx / (2.0 * P[1]) + 1.0 / S.h)
    Ts = T_last - U * (T_last - S.Tinf) * S.dx / (2.0 * P[1])
    return Ts + KELVIN


def forward_march(P, S: Setup, return_field=False):
    """Misma discretizacion, integrada paso a paso (Thomas/banded)."""
    P = np.asarray(P, dtype=float)
    A = P[0]
    Kr, sr, sfc_r, (mr, orr) = S.operator(P, "robin")
    Kc, sc, _, (mc, oc) = S.operator(P, "dirichlet")
    T = np.linalg.solve(Kr, sr)

    def banded(main, off):
        ab = np.zeros((3, S.n))
        ab[0, 1:] = S.dt / A * off
        ab[1, :] = 1.0 + S.dt / A * main
        ab[2, :-1] = S.dt / A * off
        return ab

    abc = banded(mc, oc)
    for _ in range(S.n_cool):
        T = solve_banded((1, 1), abc, T + S.dt / A * sc)
    abr = banded(mr, orr)
    out = np.empty(S.n_steps + 1)
    field = np.empty((S.n_steps + 1, S.n))
    out[0] = sfc_r(T)
    field[0] = T
    for i in range(1, S.n_steps + 1):
        T = solve_banded((1, 1), abr, T + S.dt / A * sr)
        out[i] = sfc_r(T)
        field[i] = T
    Ts = out[S.meas_idx] + KELVIN
    if return_field:
        return Ts, field + KELVIN
    return Ts


def forward_reference(P, n_vol=400, dt=0.0025):
    """Solucion de referencia de alta resolucion (Crank-Nicolson, malla fina)
    usada en lugar de ANSYS para verificar el problema directo."""
    S = Setup(n_vol=n_vol, dt=dt)
    P = np.asarray(P, dtype=float)
    A = P[0]
    Kr, sr, sfc_r, (mr, orr) = S.operator(P, "robin")
    Kc, sc, _, (mc, oc) = S.operator(P, "dirichlet")
    T = np.linalg.solve(Kr, sr)

    def cn(main, off, s, T, nsteps, record=None):
        a = 0.5 * dt / A
        ab = np.zeros((3, S.n))
        ab[0, 1:] = a * off
        ab[1, :] = 1.0 + a * main
        ab[2, :-1] = a * off
        rec = []
        for i in range(nsteps):
            rhs = T - a * (main * T)
            rhs[1:] -= a * off * T[:-1]
            rhs[:-1] -= a * off * T[1:]
            rhs += 2 * a * s
            T = solve_banded((1, 1), ab, rhs)
            if record is not None:
                rec.append(record(T))
        return T, np.array(rec)

    T, _ = cn(mc, oc, sc, T, S.n_cool)
    _, ts = cn(mr, orr, sr, T, S.n_steps, record=sfc_r)
    t = np.arange(1, S.n_steps + 1) * dt
    return t, ts + KELVIN


# ----------------------------------------------------------------------------
# Sensibilidades reducidas (Ec. 18)
# ----------------------------------------------------------------------------
def reduced_sensitivities(P, S: Setup, rel_step=1e-4):
    P = np.asarray(P, dtype=float)
    J = np.empty((S.t_meas.size, P.size))
    for j in range(P.size):
        dp = rel_step * P[j]
        Pp, Pm = P.copy(), P.copy()
        Pp[j] += dp
        Pm[j] -= dp
        J[:, j] = P[j] * (forward_eig(Pp, S) - forward_eig(Pm, S)) / (2 * dp)
    return J


# ----------------------------------------------------------------------------
# Inferencia bayesiana
# ----------------------------------------------------------------------------
class Posterior:
    """log pi(P|Y) = log pi(Y|P) + log pi(P) (constantes omitidas).

    Prior por defecto (Sec. 4 del paper):
      A, B, D ~ Normal(mu_j, (sig_rel*mu_j)^2) truncada en [0.9 mu_j, 1.1 mu_j]
      C ~ Uniforme(0, C_max)
    """

    def __init__(self, Y, S: Setup, mu_prior, sigma_noise=0.3,
                 sig_rel=0.0005, trunc=(0.9, 1.1), C_max=40257.21):
        self.Y = np.asarray(Y, float)
        self.S = S
        self.mu = np.asarray(mu_prior, float)
        self.sig = sig_rel * self.mu
        self.lo = trunc[0] * self.mu
        self.hi = trunc[1] * self.mu
        self.sn = sigma_noise
        self.Cmax = C_max
        self.gauss_idx = [0, 1, 3]

    def log_prior(self, P):
        if not (0.0 < P[2] < self.Cmax):
            return -np.inf
        lp = 0.0
        for j in self.gauss_idx:
            if not (self.lo[j] <= P[j] <= self.hi[j]):
                return -np.inf
            lp -= 0.5 * ((P[j] - self.mu[j]) / self.sig[j]) ** 2
        return lp

    def log_lik(self, P):
        r = self.Y - forward_eig(P, self.S)
        return -0.5 * np.dot(r, r) / self.sn**2

    def __call__(self, P):
        lp = self.log_prior(P)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_lik(P)


def metropolis_hastings(logpost, P0, w, n_states, rng, verbose_every=0):
    """Paseo aleatorio uniforme P* = P + w (2r - 1) aplicado a los 4
    parametros a la vez (Ec. 21). Propuesta simetrica => alpha = min(1, ratio)."""
    P = np.asarray(P0, float).copy()
    w = np.asarray(w, float)
    lp = logpost(P)
    if not np.isfinite(lp):
        raise ValueError("Estado inicial fuera del soporte del prior")
    chain = np.empty((n_states, P.size))
    logp = np.empty(n_states)
    acc = 0
    chain[0], logp[0] = P, lp
    for t in range(1, n_states):
        Pstar = P + w * (2.0 * rng.random(P.size) - 1.0)
        lps = logpost(Pstar)
        if np.log(rng.random()) <= lps - lp:
            P, lp = Pstar, lps
            acc += 1
        chain[t], logp[t] = P, lp
        if verbose_every and t % verbose_every == 0:
            print(f"  estado {t}/{n_states}  tasa aceptacion {acc / t:.3f}",
                  flush=True)
    return chain, logp, acc / (n_states - 1)


def geweke_means(chain, sa=0.1, sb=0.5):
    """Medias al inicio (primer 10%) y al final (ultimo 50%) de la cadena
    convergida (Ecs. 22-23) y el estadistico z de Geweke con varianza
    espectral aproximada por lotes."""
    n = chain.shape[0]
    a = chain[: int(sa * n)]
    b = chain[n - int(sb * n):]

    def batch_var(x, nb=20):
        m = x.shape[0] // nb
        bm = x[: m * nb].reshape(nb, m, -1).mean(axis=1)
        return bm.var(axis=0, ddof=1) / nb

    z = (a.mean(0) - b.mean(0)) / np.sqrt(batch_var(a) + batch_var(b))
    return a.mean(0), b.mean(0), z


def ess(x, max_lag=None):
    """Tamano efectivo de muestra (autocorrelacion con corte de Geyer simple)."""
    x = np.asarray(x, float) - np.mean(x)
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    max_lag = n // 2 if max_lag is None else max_lag
    s = 0.0
    for k in range(1, max_lag, 2):
        pair = acf[k] + (acf[k + 1] if k + 1 < n else 0.0)
        if pair < 0:
            break
        s += pair
    tau = 1.0 + 2.0 * s
    return n / tau
