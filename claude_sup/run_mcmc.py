"""
run_mcmc.py -- genera las medidas simuladas y ejecuta las cadenas de
Metropolis-Hastings con la configuracion del paper (Sec. 4, Tabla 2).

Uso:
    python run_mcmc.py healthy            # 200 000 estados, semilla por defecto
    python run_mcmc.py tumor --n 200000 --seed 7
    python run_mcmc.py healthy --tag tuned --wscale auto   # propuesta ajustada

Salida: results/chain_<tissue>[_<tag>].npz con la cadena completa, las medidas,
la semilla y la tasa de aceptacion.
"""
import argparse
import os
import time

import numpy as np

from pennes_mcmc import Setup, exact_params, forward_eig, Posterior, \
    metropolis_hastings, reduced_sensitivities

# Tabla 2: variacion maxima w_j de la propuesta (A, B, C, D)
W_TABLE2 = {
    "healthy": np.array([79.86, 9.4e-6, 0.28, 0.081]),
    "tumor": np.array([79.86, 9.4e-6, 1.41, 0.403]),
}
SIGMA_NOISE = 0.3          # K (Sec. 4)
DATA_SEED = {"healthy": 2015, "tumor": 2016}


def simulate_data(tissue, S, seed=None):
    rng = np.random.default_rng(DATA_SEED[tissue] if seed is None else seed)
    T_exact = forward_eig(exact_params(tissue), S)
    return T_exact, T_exact + SIGMA_NOISE * rng.standard_normal(T_exact.size)


def laplace_sd(post, P):
    """Desviaciones de la aproximacion de Laplace (gaussiana local) del
    posterior: (J^T J / s^2 + prior precision)^-1, sin truncamiento."""
    S = post.S
    Jred = reduced_sensitivities(P, S)
    J = Jred / P                       # dT/dP
    H = J.T @ J / post.sn**2
    for j in post.gauss_idx:
        H[j, j] += 1.0 / post.sig[j]**2
    cov = np.linalg.inv(H)
    return np.sqrt(np.diag(cov)), cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tissue", choices=["healthy", "tumor"])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=12345, help="semilla del MCMC")
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--wscale", default="table2",
                    help="'table2' (paper) o 'auto' (w = 2*sd de Laplace)")
    ap.add_argument("--sig-rel", type=float, default=0.0005)
    ap.add_argument("--meas-every", type=int, default=1)
    args = ap.parse_args()

    S = Setup(meas_every=args.meas_every)
    T_exact, Y = simulate_data(args.tissue, S, args.data_seed)
    Pex = exact_params(args.tissue)
    post = Posterior(Y, S, mu_prior=Pex, sigma_noise=SIGMA_NOISE,
                     sig_rel=args.sig_rel)

    # Estado inicial (decision documentada en el informe):
    #   sano  : valores exactos (el texto dice que la cadena de C empezo en el
    #           valor exacto; Fig. 7 empieza en C/C_exact = 1)
    #   tumor : A, B exactos; C en el valor del tejido sano; D en la media del
    #           prior (el valor sano 631 esta fuera del soporte del prior de D)
    P0 = Pex.copy()
    if args.tissue == "tumor":
        P0[2] = exact_params("healthy")[2]

    if args.wscale == "table2":
        w = W_TABLE2[args.tissue]
    else:
        sd, _ = laplace_sd(post, Pex)
        w = 2.0 * sd
    print(f"[{args.tissue}{args.tag}] w = {w}", flush=True)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    chain, logp, acc = metropolis_hastings(post, P0, w, args.n, rng,
                                           verbose_every=20_000)
    dt = time.time() - t0
    os.makedirs("results", exist_ok=True)
    name = f"results/chain_{args.tissue}{('_' + args.tag) if args.tag else ''}.npz"
    np.savez_compressed(name, chain=chain, logp=logp, acc=acc, Y=Y,
                        T_exact=T_exact, t=S.t_meas, P_exact=Pex, P0=P0, w=w,
                        seed=args.seed, sig_rel=args.sig_rel, runtime=dt)
    print(f"[{args.tissue}{args.tag}] listo en {dt:.0f} s, aceptacion {acc:.3f}"
          f" -> {name}", flush=True)


if __name__ == "__main__":
    main()
