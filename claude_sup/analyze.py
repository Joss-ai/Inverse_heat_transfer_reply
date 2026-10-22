"""
analyze.py -- figuras, tablas y experimentos de sensibilidad de la replicacion.

Uso (despues de run_mcmc.py):
    python analyze.py            # todo
    python analyze.py --quick    # sin el experimento de realizaciones de ruido

Escribe figures/*.png y results/summary.json.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from pennes_mcmc import (Setup, exact_params, forward_eig, forward_march,
                         forward_reference, reduced_sensitivities, Posterior,
                         geweke_means, ess, initial_condition, KELVIN)
from run_mcmc import simulate_data, laplace_sd, SIGMA_NOISE

BURN = 60_000
NAMES = ["A", "B", "C", "D"]
PAPER = {  # valores reportados (relativos al exacto)
    "healthy": {"mean": [1.0238, 1.0034, 1.3807, 1.0005],
                "std": [0.00030534, 0.000317, 0.00050118, 0.00051205]},
    "tumor": {"mean": [0.9784, 1.0343, 0.9607, 1.0023],
              "std": [0.00039629, 0.0004455, 0.0003152, 0.00041685]},
}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, C1, C2, C3 = "#222222", "#2a6fbb", "#c4572a", "#6b8e23"


def fig_forward(S, out):
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for tis, col in [("healthy", C1), ("tumor", C2)]:
        P = exact_params(tis)
        a, b = forward_eig(P, S), forward_march(P, S)
        t, r = forward_reference(P)
        ri = np.interp(S.t_meas, t, r)
        res[tis] = dict(eig_vs_march=float(np.abs(a - b).max()),
                        vs_ref_max=float(np.abs(a - ri).max()),
                        vs_ref_after_1s=float(np.abs(a - ri)[S.t_meas >= 1].max()),
                        T_0p05=float(a[0]), T_50=float(a[-1]))
        lab = "sano" if tis == "healthy" else "tumor"
        ax[0].plot(S.t_meas, a, color=col, label=f"{lab}: 27 vol., dt=0.05 s")
        ax[0].plot(t, r, "--", color=INK, lw=0.8)
        ax[1].plot(S.t_meas, a - ri, color=col, label=lab)
    ax[0].plot([], [], "--", color=INK, lw=0.8, label="referencia 400 vol., CN")
    ax[0].set(xlabel="t [s]", ylabel="T superficie [K]",
              title="Problema directo: recalentamiento")
    ax[0].legend(frameon=False)
    ax[1].set(xlabel="t [s]", ylabel="diferencia [K]", ylim=(-0.1, 0.6),
              title="27 vol. menos referencia")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return res


def fig_profiles(S, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    x = (np.arange(S.n) + 0.5) * S.dx * 1e3
    for tis, col in [("healthy", C1), ("tumor", C2)]:
        F, Tss = initial_condition(exact_params(tis), S, True)
        lab = "sano" if tis == "healthy" else "tumor"
        ax.plot(x, Tss + KELVIN, color=col, label=f"{lab}: estacionario")
        ax.plot(x, F + KELVIN, "--", color=col, label=f"{lab}: F(x) tras 15 s")
    ax.set(xlabel="x [mm] (x = 5 mm es la superficie)", ylabel="T [K]",
           title="Perfiles iniciales")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_sensitivities(S, out):
    res = {}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, tis in zip(axs, ["healthy", "tumor"]):
        P = exact_params(tis)
        J = reduced_sensitivities(P, S)
        res[tis] = {"max_abs": np.abs(J).max(0).tolist(),
                    "at_50s": J[-1].tolist(),
                    "corr_AB": float(np.corrcoef(J[:, 0], J[:, 1])[0, 1]),
                    "corr_matrix_J": np.corrcoef(J.T).tolist()}
        for j, (nm, col, ls) in enumerate(zip(NAMES, [INK, C1, C2, C3],
                                              ["-", "--", "-", ":"])):
            ax.plot(S.t_meas, J[:, j], ls, color=col, label=nm)
        ax.axhline(0, color="#999", lw=0.5)
        ax.set(xlabel="t [s]", ylabel="J = P dT/dP [K]",
               title="Tejido sano" if tis == "healthy" else "Tejido tumoral")
        ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return res


def fig_data(S, out):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, tis in zip(axs, ["healthy", "tumor"]):
        Te, Y = simulate_data(tis, S)
        ax.plot(S.t_meas, Y, "x", ms=2, color="#888", label="medidas simuladas")
        ax.plot(S.t_meas, Te, color=INK, lw=1.2, label="temperatura exacta")
        ax.set(xlabel="t [s]", title="Tejido sano" if tis == "healthy"
               else "Tejido tumoral", ylim=(270, 303))
        ax.legend(frameon=False, loc="lower right")
    axs[0].set_ylabel("T superficie [K]")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def chain_stats(tis, tag=""):
    f = f"results/chain_{tis}{('_' + tag) if tag else ''}.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f)
    ch = d["chain"]
    Pex = d["P_exact"]
    rel = ch / Pex
    post = rel[BURN:]
    ga, gb, gz = geweke_means(post)
    return dict(file=f, acc=float(d["acc"]), runtime=float(d["runtime"]),
                mean=post.mean(0).tolist(), std=post.std(0).tolist(),
                mean_dim=(post.mean(0) * Pex).tolist(),
                std_dim=(post.std(0) * Pex).tolist(),
                q025=np.quantile(post, 0.025, axis=0).tolist(),
                q975=np.quantile(post, 0.975, axis=0).tolist(),
                ess=[float(ess(post[:, j])) for j in range(4)],
                geweke_a=ga.tolist(), geweke_b=gb.tolist(),
                geweke_z=gz.tolist(), w=d["w"].tolist()), rel


def fig_chains(tis, rel, out_chain, out_full):
    post = rel[BURN:]
    fig, axs = plt.subplots(4, 2, figsize=(9, 9))
    for j in range(4):
        axs[j, 0].plot(np.arange(BURN, rel.shape[0]), post[:, j], lw=0.3,
                       color=C1)
        axs[j, 0].set_ylabel(f"{NAMES[j]}/{NAMES[j]}exacto")
        axs[j, 1].hist(post[:, j], bins=40, color="#9bb7d9",
                       edgecolor="white")
        m, s = PAPER[tis]["mean"][j], PAPER[tis]["std"][j]
        axs[j, 1].axvline(post[:, j].mean(), color=C1, lw=1.2,
                          label=f"esta replica: {post[:, j].mean():.4f}")
        axs[j, 1].axvline(m, color=C2, lw=1.2, ls="--",
                          label=f"paper: {m:.4f}")
        axs[j, 1].axvline(1.0, color=INK, lw=0.8, ls=":", label="exacto")
        axs[j, 1].legend(frameon=False, fontsize=7)
    axs[3, 0].set_xlabel("estado de la cadena")
    axs[3, 1].set_xlabel("valor relativo")
    fig.suptitle("Tejido sano" if tis == "healthy" else "Tejido tumoral")
    fig.tight_layout()
    fig.savefig(out_chain)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(rel[:, 2], lw=0.6, color=C2)
    ax.axvline(BURN, color="#999", ls="--", lw=0.8)
    ax.axhline(1.0, color=INK, ls=":", lw=0.8)
    ax.set(xlabel="estado", ylabel="C / C exacto",
           title="Cadena completa de C (" + ("sano)" if tis == "healthy"
                                              else "tumor)"))
    fig.tight_layout()
    fig.savefig(out_full)
    plt.close(fig)


def map_estimate(Y, S, Pex, sig_rel=0.0005, x0=None):
    """MAP (maximo a posteriori) con el mismo prior que el MCMC."""
    sig = sig_rel
    x0 = np.ones(4) if x0 is None else x0

    def res(x):
        P = x * Pex
        r = (Y - forward_eig(P, S)) / SIGMA_NOISE
        pr = (x[[0, 1, 3]] - 1.0) / sig
        return np.concatenate([r, pr])

    lb = [0.9, 0.9, 1e-6, 0.9]
    ub = [1.1, 1.1, 40257.21 / Pex[2], 1.1]
    sol = least_squares(res, x0, bounds=(lb, ub), x_scale=[sig, sig, 0.01, sig])
    return sol.x


def noise_experiment(S, n_real=200):
    out = {}
    for tis in ["healthy", "tumor"]:
        Pex = exact_params(tis)
        xs = []
        for s in range(n_real):
            _, Y = simulate_data(tis, S, seed=10_000 + s)
            xs.append(map_estimate(Y, S, Pex))
        xs = np.array(xs)
        out[tis] = dict(mean=xs.mean(0).tolist(), std=xs.std(0).tolist(),
                        q01=np.quantile(xs, 0.01, axis=0).tolist(),
                        q99=np.quantile(xs, 0.99, axis=0).tolist(),
                        maxC=float(xs[:, 2].max()), minC=float(xs[:, 2].min()),
                        samplesC=xs[:, 2].tolist())
    return out


def fig_noise(ne, out):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for tis, col, ref in [("healthy", C1, 1.0), ("tumor", C2, 5.0)]:
        c = np.array(ne[tis]["samplesC"]) * ref   # en unidades de C_sano
        ax.hist(c, bins=30, color=col, alpha=0.7,
                label="sano" if tis == "healthy" else "tumor")
    ax.axvline(1.3807, color=C1, ls="--", lw=1, label="paper, sano (1.381)")
    ax.axvline(0.9607 * 5, color=C2, ls="--", lw=1,
               label="paper, tumor (0.961 x 5)")
    ax.set(xlabel="C estimado / C del tejido sano",
           ylabel="realizaciones", title="MAP de C en 200 realizaciones del ruido")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def design_sensitivity(S0):
    """Desviaciones de Laplace de C para variantes de diseno."""
    rows = []
    variants = [("paper (1000 medidas, prior 0.05%)", dict(), 0.0005),
                ("100 medidas (cada 0.5 s)", dict(meas_every=10), 0.0005),
                ("prior A,B,D 5%", dict(), 0.05),
                ("prior A,B,D 10%", dict(), 0.10)]
    for name, kw, sr in variants:
        S = Setup(**kw)
        row = {"variante": name}
        for tis in ["healthy", "tumor"]:
            Pex = exact_params(tis)
            Te = forward_eig(Pex, S)
            post = Posterior(Te, S, Pex, SIGMA_NOISE, sig_rel=sr)
            sd, cov = laplace_sd(post, Pex)
            row[tis] = (sd / Pex).tolist()
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    S = Setup()
    summ = {}
    summ["forward"] = fig_forward(S, "figures/fig_forward_verification.png")
    fig_profiles(S, "figures/fig_initial_profiles.png")
    summ["sensitivity"] = fig_sensitivities(S, "figures/fig1_sensitivities.png")
    fig_data(S, "figures/fig2_simulated_data.png")

    for tis in ["healthy", "tumor"]:
        Pex = exact_params(tis)
        _, Y = simulate_data(tis, S)
        post = Posterior(Y, S, Pex, SIGMA_NOISE)
        sd, cov = laplace_sd(post, Pex)
        summ[f"laplace_{tis}"] = {"sd_rel": (sd / Pex).tolist(),
                                  "corr": (cov / np.outer(sd, sd)).tolist()}
        summ[f"map_{tis}"] = map_estimate(Y, S, Pex).tolist()
        for tag in ["", "tuned"]:
            r = chain_stats(tis, tag)
            if r is None:
                continue
            st, rel = r
            key = f"mcmc_{tis}" + (f"_{tag}" if tag else "")
            summ[key] = st
            suf = f"_{tag}" if tag else ""
            fig_chains(tis, rel, f"figures/chains_{tis}{suf}.png",
                       f"figures/chainC_full_{tis}{suf}.png")

    summ["design"] = design_sensitivity(S)
    # desviacion de las estimaciones del paper respecto del prior (en sigmas)
    summ["paper_prior_z"] = {
        tis: [(PAPER[tis]["mean"][j] - 1.0) / 0.0005 for j in (0, 1, 3)]
        for tis in PAPER}
    if not args.quick:
        ne = noise_experiment(S)
        summ["noise_experiment"] = {k: {kk: vv for kk, vv in v.items()
                                        if kk != "samplesC"}
                                    for k, v in ne.items()}
        fig_noise(ne, "figures/fig_noise_realizations.png")
    with open("results/summary.json", "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items()
                      if not k.startswith("sens")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
