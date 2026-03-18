import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import jax
import jax.numpy as jnp
from fit_model import (
    expfunc,
    logfunc,
    polynomial_func,
    mlp,
    sol_sol_func,
    equations,
    find_root,
    get_root_error,
    li_free_energy,
    conc_factor,
    rescale_input_params,
    energetics,
)
from tqdm import tqdm
import copy

from matplotlib import rcParams

fontsize = 20
plot_settings = {
    # "font.family": "times new roman",
    "axes.labelsize": fontsize,
    "axes.labelweight": "bold",
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.size": fontsize,
    "axes.linewidth": 2.0,
    "lines.dashed_pattern": [5, 2.5],
    "lines.markersize": 10,
    "lines.linewidth": 2,
    "lines.markeredgewidth": 1,
    # "lines.markeredgecolor": "k",
    "legend.fontsize": fontsize,
    "legend.frameon": False,
    "figure.figsize": [6, 6],
}

solvent_map_dict = pd.read_csv('/nfs/turbo/coe-venkvis/zhaohc/ising-electrolyte/volume-dft/solvent_map_dict.csv').set_index('molecule').to_dict(orient='index')

# Update rcParams with settings from JSON file
rcParams.update(plot_settings)


def sol_dn_func_contour(npoints=1000, sol_params_dn=None, params_conc_sol=None):
    """
    plot how solvent DN affects output
    """
    x_pred = jnp.linspace(1.0, 40.0, npoints)
    m_pred = jnp.linspace(0.1, 0.8, npoints)
    x_pred, m_pred = jnp.meshgrid(x_pred, m_pred)
    x_pred = x_pred.reshape(-1, 1)
    m_pred = m_pred.reshape(-1, 1)
    dn_eff = x_pred * conc_factor(
        m_pred,
        0.14,
        solvent_map_dict["DME"]['volume A3'],
        solvent_map_dict["TFSI"]['volume A3'],
        params_conc_sol,
    )
    y_pred = expfunc(dn_eff, sol_params_dn[:4]) + logfunc(m_pred, sol_params_dn[4])
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.contourf(
        x_pred.reshape(npoints, npoints),
        m_pred.reshape(npoints, npoints),
        y_pred.reshape(npoints, npoints),
        levels=100,
        cmap="plasma",
    )
    plt.xlabel("Solvent DN")
    plt.ylabel("Solvent Molar Ratio")
    plt.colorbar(label="$h_{Li^+-sol}$")
    plt.savefig("sol_dn_func.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    x_pred = x_pred.flatten()
    m_pred = m_pred.flatten()
    y_pred = y_pred.flatten()
    df = pd.DataFrame(
        {
            "solvent_dn": x_pred,
            "molar_ratio": m_pred,
            "energy": y_pred,
        }
    )
    df.to_csv("sol_dn_func_contour_data.csv", index=False)


# @jax.jit
def sol_dn_func_data_to_csv(sol_params_dn, conc_factor_sol):
    """save the data for functional form analysis"""
    npoints = 100
    x_pred = jnp.linspace(1.0, 40.0, npoints)
    m_pred = jnp.linspace(0.1, 0.8, npoints)
    xs = []
    ms = []
    ys = []
    for x in tqdm(x_pred):
        for m in m_pred:
            dn_eff = x * conc_factor(
                m,
                0.14,
                solvent_map_dict["DME"]['volume A3'],
                solvent_map_dict["TFSI"]['volume A3'],
                params_conc_sol,
            )
            y = expfunc(dn_eff, sol_params_dn[:4]) + logfunc(m, sol_params_dn[4])
            xs.append(x)
            ms.append(m)
            ys.append(y)
    xs = np.array(xs)
    ms = np.array(ms)
    ys = np.array(ys)
    print(xs.shape, ms.shape, ys.shape)
    df = pd.DataFrame(
        {
            "solvent_dn": xs,
            "molar_ratio": ms,
            "energy": ys,
        }
    )
    df.to_csv("sol_dn_func_data.csv", index=False)


def sol_dn_func_fixed_m(sol_params_dn, params_conc_factor_sol):
    """plot how DN affects output, fixing molar ratio"""
    x_pred = jnp.linspace(0, 40, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    for m in np.arange(0.2, 0.6, 0.1):
        m_pred = m * jnp.ones_like(x_pred)
        dn_eff = x_pred * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )
        y_pred = expfunc(dn_eff, sol_params_dn[:4]) + logfunc(m_pred, sol_params_dn[4])
        plt.plot(
            x_pred.flatten(), y_pred.flatten(), label=r"$x_{solvent}$" + f"={m:.1f}"
        )
    plt.xlabel("Solvent DN")
    plt.ylabel("$h_{Li^+-sol}$")
    plt.legend()
    plt.savefig("sol_dn_func_fixed_m.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sol_dn_func_no_conc_factor(sol_params_dn):
    x_pred = jnp.linspace(0, 40, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    y_pred = expfunc(x_pred, sol_params_dn[:4])
    plt.plot(x_pred.flatten(), y_pred.flatten())
    # plot the half wave potential exp data points
    df = pd.read_csv(
        "/nfs/turbo/coe-venkvis/zhaohc/ising-electrolyte/LHCE/Ising-model-fitting/DN_trade_offs.csv"
    )
    plt.plot(df["x"], df["y"], "o", label="Exp Li|Li$^+$ half-wave potential")
    plt.xlabel("Solvent DN")
    plt.ylabel("$h_{Li^+-sol}$")
    plt.legend()
    plt.savefig("sol_dn_func_no_conc_factor.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sol_an_func_no_conc_factor(sol_params_an):
    x_pred = jnp.linspace(5, 40, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    dns = [-6.2, 7.3, 11.2]
    labels = ["PF6", "BF4", "TFSI"]
    for i, dn in enumerate(dns):
        dn_pred = dn * jnp.ones_like(x_pred)
        y_pred = sol_sol_func((dn_pred, x_pred), sol_params_an[:5])
        plt.plot(x_pred.flatten(), y_pred.flatten(), label=labels[i])
    plt.xlabel("Solvent AN")
    plt.ylabel("$J_{sol-anion}$")
    plt.legend()
    # plt.legend()
    plt.savefig("sol_an_func_no_conc_factor.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sol_dn_func_fixed_dn(sol_params_dn, params_conc_factor_sol):
    """plot how molar ratio affects output, fixing DN"""
    x_pred = jnp.arange(0.1, 0.8, 0.01).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    for dn in [1, 5, 10, 15, 20, 25, 30, 35]:
        dn_pred = dn * jnp.ones_like(x_pred)
        dn_eff = dn_pred * conc_factor(
            x_pred,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )
        y_pred = expfunc(dn_eff, sol_params_dn[:4]) + logfunc(x_pred, sol_params_dn[4])
        plt.plot(x_pred.flatten(), y_pred.flatten(), label=f"DN = {dn:d}")
    plt.xlabel("Molar Ratio")
    plt.ylabel("$h_{Li^+-sol}$")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig("sol_dn_func_fixed_dn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def an_m_func_contour(params_an, params_conc_factor_sol):
    # plot how solvent AN and M affects output, 2d contour plot
    an_pred = jnp.linspace(5, 40, npoints)
    m_pred = jnp.linspace(0.2, 0.8, npoints)
    an_pred, m_pred = jnp.meshgrid(an_pred, m_pred)
    an_pred = an_pred.reshape(-1, 1)
    m_pred = m_pred.reshape(-1, 1)
    an_eff = an_pred * conc_factor(
        m_pred,
        0.14,
        solvent_map_dict["DME"]['volume A3'],
        solvent_map_dict["TFSI"]['volume A3'],
        params_conc_factor_sol,
    )
    dn = 11.2  # TFSI anion DN
    y_pred = sol_sol_func(
        (jnp.ones_like(an_eff) * dn, an_eff), params_an[:5]
    ) + logfunc(m_pred, params_an[5]) + logfunc(0.14, params_an[5])  # assuming salt molar ratio is 0.14
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.contourf(
        an_pred.reshape(npoints, npoints),
        m_pred.reshape(npoints, npoints),
        y_pred.reshape(npoints, npoints),
        levels=100,
        cmap="viridis",
    )
    plt.xlabel("Solvent AN")
    plt.ylabel("Solvent Molar Ratio")
    plt.colorbar(label="$J_{sol-anion}$")
    plt.savefig("an_m_func.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    an_pred = an_pred.flatten()
    m_pred = m_pred.flatten()
    y_pred = y_pred.flatten()
    df = pd.DataFrame(
        {
            "solvent_an": an_pred,
            "molar_ratio": m_pred,
            "energy": y_pred,
        }
    )
    df.to_csv("an_m_func_contour_data.csv", index=False)


# @jax.jit
def sol_an_func_data_to_csv(sol_params_an, params_conc_factor_sol):
    """save the data for functional form analysis"""
    npoints = 100
    x_pred = jnp.linspace(0.0, 100.0, npoints)
    m_pred = jnp.linspace(0.1, 0.8, npoints)
    xs = []
    ms = []
    ys = []
    for x in tqdm(x_pred):
        for m in m_pred:
            an_eff = x * conc_factor(
                m,
                0.14,
                solvent_map_dict["DME"]['volume A3'],
                solvent_map_dict["TFSI"]['volume A3'],
                params_conc_factor_sol,
            )
            y = (
                sol_sol_func((jnp.ones_like(an_eff) * 11.2, an_eff), sol_params_an[:5])
                + logfunc(m, sol_params_an[5])
                + logfunc(0.14, sol_params_an[5])
            )  # assuming salt molar ratio is 0.14
            xs.append(x)
            ms.append(m)
            ys.append(y)
    xs = np.array(xs)
    ms = np.array(ms)
    ys = np.array(ys)
    df = pd.DataFrame(
        {
            "solvent_an": xs,
            "molar_ratio": ms,
            "energy": ys,
        }
    )
    df.to_csv("sol_an_func_data.csv", index=False)


def salt_dn_func_fixed_c(salt_params_dn):
    """
    fix c, plot how solvent salt DN affects output
    """
    dn_pred = jnp.linspace(-10, 20, 1000).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    for c in np.arange(0.05, 0.25, 0.05):
        c_pred = c * jnp.ones_like(dn_pred)
        y_pred = expfunc(dn_pred, salt_params_dn[:4]) + logfunc(
            c_pred, salt_params_dn[4]
        )
        plt.plot(
            dn_pred.flatten(), y_pred.flatten(), label=r"$x_{anion}=$" + f"{c:.3f}"
        )
    plt.xlabel("Salt DN")
    plt.ylabel("$h_l$")
    plt.legend()
    plt.savefig("salt_dn_func_fixed_c.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def an_func_fixed_m(params_an, params_conc_factor_sol):
    """
    fix m, plot how solvent AN affects output
    """
    an_pred = jnp.linspace(5, 40, 1000).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    for m in np.arange(0.1, 0.5, 0.1):
        an_eff = an_pred * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        dn = 11.2  # TFSI anion DN
        y_pred = sol_sol_func(
            (jnp.ones_like(an_eff) * dn, an_eff), params_an[:5]
        ) + logfunc(m, params_an[5]) + logfunc(0.14, params_an[5])  # assuming salt molar ratio is 0.14
        plt.plot(
            an_pred.flatten(), y_pred.flatten(), label=r"$x_{solvent}$" + f"={m:.1f}"
        )
    plt.xlabel("Solvent AN")
    plt.ylabel("$J_{sol-anion}$")
    plt.legend()
    plt.savefig(f"an_func_fixed_m.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def an_func_fixed_an(params_an, params_conc_factor_sol):
    """fix an, plot how m affects output"""
    m_pred = jnp.arange(0.1, 0.8, 0.01).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    for an_pred in range(5, 25, 5):
        an_eff = an_pred * conc_factor(
            m_pred,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        dn = 11.2  # TFSI anion DN
        y_pred = sol_sol_func(
            (jnp.ones_like(an_eff) * dn, an_eff), params_an[:5]
        ) + logfunc(m_pred, params_an[5]) + logfunc(0.14, params_an[5])  # assuming salt molar ratio is 0.14
        plt.plot(m_pred.flatten(), y_pred.flatten(), label=f"AN = {an_pred:d}")
    plt.xlabel("Molar Ratio")
    plt.ylabel("$J_{sol-anion}$")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(f"an_func_fixed_an.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def salt_dn_func(salt_params_dn):
    # plot how solvent salt DN and c affects output
    dn_pred = jnp.linspace(-10, 40, npoints)
    c_pred = jnp.linspace(0.1, 0.8, npoints)
    dn_pred, c_pred = jnp.meshgrid(dn_pred, c_pred)
    dn_pred = dn_pred.reshape(-1, 1)
    c_pred = c_pred.reshape(-1, 1)
    y_pred = expfunc(dn_pred, salt_params_dn[:4]) + logfunc(c_pred, salt_params_dn[4])
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.contourf(
        dn_pred.reshape(npoints, npoints),
        c_pred.reshape(npoints, npoints),
        y_pred.reshape(npoints, npoints),
        levels=100,
        cmap="plasma",
    )
    plt.xlabel("Salt DN")
    plt.ylabel("Salt Molar Ratio")
    plt.colorbar(label="$h_l$")
    plt.savefig("salt_dn_func.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sol_sol_func_full_contour(params_sol, params_conc_factor_sol):
    """
    plot how DN and AN affects sol-sol interaction
    """
    ms = np.arange(0.3, 0.81, 0.1)
    npoints = 100
    dn_lin = jnp.linspace(0, 40, npoints)
    an_lin = jnp.linspace(0, 40, npoints)
    # pre‐compute global min/max of f so all panels share the same color‐scale
    all_vals = []
    for m in ms:
        dn_eff = dn_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        an_eff = an_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        m_arr = m * jnp.ones_like(dn_eff)
        DN, AN = jnp.meshgrid(dn_eff, an_eff)
        dn_eff = dn_eff.reshape(-1, 1)
        an_eff = an_eff.reshape(-1, 1)
        val = (
            sol_sol_func(jnp.hstack([dn_eff, an_eff]).T, params_sol[:5])
            + sol_sol_func(jnp.hstack([dn_eff, an_eff]).T, params_sol[:5])
            + sol_sol_func(jnp.hstack([dn_eff, dn_eff]).T, params_sol[5:10])
            + sol_sol_func(jnp.hstack([an_eff, an_eff]).T, params_sol[10:15])
            + 2.0 * logfunc(m_arr, params_sol[15])
        )
        all_vals.append(val)
    all_vals = jnp.concatenate(all_vals)
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    maxrow, maxcol = 3, 2
    fig, axes = plt.subplots(
        maxrow, maxcol,
        figsize=(12, 10),
        sharex=True,
        sharey=True,
        gridspec_kw={"right": 0.85},
        constrained_layout=False,
    )
    for (row, col), ax in np.ndenumerate(axes):
        m = ms[row * axes.shape[1] + col]
        dn_eff = dn_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )
        an_eff = an_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )
        DN, AN = jnp.meshgrid(dn_lin, an_lin)
        dn_eff, an_eff = jnp.meshgrid(dn_eff, an_eff)
        Z = (
            sol_sol_func((dn_eff, an_eff), params_sol[:5])
            + sol_sol_func((dn_eff, an_eff), params_sol[:5])
            + sol_sol_func((dn_eff, dn_eff), params_sol[5:10])
            + sol_sol_func((an_eff, an_eff), params_sol[10:15])
            + 2.0 * logfunc(m, params_sol[15])
        )
        cf = ax.contourf(DN, AN, Z, levels=100, cmap="plasma", vmin=vmin, vmax=vmax)
        ax.set_title(r"$x_{solvent}$" + f"={m:.1f}")

        # only label the bottom row
        if row == maxrow - 1:
            ax.set_xlabel("Solvent DN")
        # only label the leftmost column
        if col == 0:
            ax.set_ylabel("Solvent AN")
    # single colorbar for all
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cb = fig.colorbar(cf, cax=cbar_ax, label="$J_{sol-sol}$")
    plt.savefig(
        "sol_sol_func_3x3.png",
        dpi=300,
    )
    plt.close(fig)


def sol_sol_func_dn_an_contour(params_sol, params_conc_factor_sol):
    """
    plot how DN and AN affects sol-sol interaction
    """
    ms = np.arange(0.3, 0.81, 0.1)
    npoints = 100
    dn_lin = jnp.linspace(0, 40, npoints)
    an_lin = jnp.linspace(0, 40, npoints)
    # pre‐compute global min/max of f so all panels share the same color‐scale
    all_vals = []
    for m in ms:
        dn_eff = dn_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        an_eff = an_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        m_arr = m * jnp.ones_like(dn_eff)
        DN, AN = jnp.meshgrid(dn_eff, an_eff)
        dn_eff = dn_eff.reshape(-1, 1)
        an_eff = an_eff.reshape(-1, 1)
        val = sol_sol_func(jnp.hstack([dn_eff, an_eff]).T, params_sol[:5])
        all_vals.append(val)
    all_vals = jnp.concatenate(all_vals)
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    maxrow, maxcol = 3, 2
    fig, axes = plt.subplots(
        maxrow, maxcol,
        figsize=(12, 10),
        sharex=True,
        sharey=True,
        gridspec_kw={"right": 0.85},
        constrained_layout=False,
    )
    for (row, col), ax in np.ndenumerate(axes):
        m = ms[row * axes.shape[1] + col]
        dn_eff = dn_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        an_eff = an_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )
        DN, AN = jnp.meshgrid(dn_lin, an_lin)
        dn_eff, an_eff = jnp.meshgrid(dn_eff, an_eff)
        Z = sol_sol_func((dn_eff, an_eff), params_sol[:5])
        cf = ax.contourf(DN, AN, Z, levels=100, cmap="plasma", vmin=vmin, vmax=vmax)
        ax.set_title(r"$x_{solvent}$" + f"={m:.1f}")

        # only label the bottom row
        if row == maxrow - 1:
            ax.set_xlabel("Solvent DN")
        # only label the leftmost column
        if col == 0:
            ax.set_ylabel("Solvent AN")
    # single colorbar for all
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cb = fig.colorbar(cf, cax=cbar_ax, label="$J_{sol-sol}$")
    plt.savefig(
        "sol_sol_func_dn_an_3x3.png",
        dpi=300,
    )
    plt.close(fig)

def sol_sol_func_dn_dn_contour(params_sol, params_conc_factor_sol):
    """
    plot how DN and AN affects sol-sol interaction
    """
    ms = np.arange(0.3, 0.81, 0.1)
    npoints = 100
    dn1_lin = jnp.linspace(0, 40, npoints)
    dn2_lin = jnp.linspace(0, 40, npoints)
    # pre‐compute global min/max of f so all panels share the same color‐scale
    all_vals = []
    for m in ms:
        dn1_eff = dn1_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        dn2_eff = dn2_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        m_arr = m * jnp.ones_like(dn1_eff)
        DN1, DN2 = jnp.meshgrid(dn1_eff, dn2_eff)
        dn1_eff = dn1_eff.reshape(-1, 1)
        dn2_eff = dn2_eff.reshape(-1, 1)
        val = sol_sol_func(jnp.hstack([dn1_eff, dn2_eff]).T, params_sol[5:10])
        all_vals.append(val)
    all_vals = jnp.concatenate(all_vals)
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    maxrow, maxcol = 3, 2
    fig, axes = plt.subplots(
        maxrow, maxcol,
        figsize=(12, 10),
        sharex=True,
        sharey=True,
        gridspec_kw={"right": 0.85},
        constrained_layout=False,
    )
    for (row, col), ax in np.ndenumerate(axes):
        m = ms[row * axes.shape[1] + col]
        dn1_eff = dn1_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        ) # assuming salt molar ratio is 0.14
        dn2_eff = dn2_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        ) # assuming salt molar ratio is 0.14
        DN1, DN2 = jnp.meshgrid(dn1_lin, dn2_lin)
        dn1_eff, dn2_eff = jnp.meshgrid(dn1_eff, dn2_eff)
        Z = sol_sol_func((dn1_eff, dn2_eff), params_sol[5:10])
        cf = ax.contourf(DN1, DN2, Z, levels=100, cmap="plasma", vmin=vmin, vmax=vmax)
        ax.set_title(r"$x_{solvent}$" + f"={m:.1f}")

        # only label the bottom row
        if row == maxrow - 1:
            ax.set_xlabel("DN$_1$")
        # only label the leftmost column
        if col == 0:
            ax.set_ylabel("DN$_2$")
    # single colorbar for all
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cb = fig.colorbar(cf, cax=cbar_ax, label="$J_{sol-sol}$")
    plt.savefig(
        "sol_sol_func_dn_dn_3x3.png",
        dpi=300,
    )
    plt.close(fig)

def sol_sol_func_an_an_contour(params_sol, params_conc_factor_sol):
    """
    plot how DN and AN affects sol-sol interaction
    """
    ms = np.arange(0.3, 0.81, 0.1)
    npoints = 100
    an1_lin = jnp.linspace(0, 40, npoints)
    an2_lin = jnp.linspace(0, 40, npoints)
    # pre‐compute global min/max of f so all panels share the same color‐scale
    all_vals = []
    for m in ms:
        an1_eff = an1_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        an2_eff = an2_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        m_arr = m * jnp.ones_like(an1_eff)
        AN1, AN2 = jnp.meshgrid(an1_eff, an2_eff)
        val = sol_sol_func(jnp.hstack([an1_eff.reshape(-1, 1), an2_eff.reshape(-1, 1)]).T, params_sol[10:15])
        all_vals.append(val)
    all_vals = jnp.concatenate(all_vals)
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    maxrow, maxcol = 3, 2
    fig, axes = plt.subplots(
        maxrow, maxcol,
        figsize=(12, 10),
        sharex=True,
        sharey=True,
        gridspec_kw={"right": 0.85},
        constrained_layout=False,
    )
    for (row, col), ax in np.ndenumerate(axes):
        m = ms[row * axes.shape[1] + col]
        an1_eff = an1_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )
        an2_eff = an2_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )
        AN1, AN2 = jnp.meshgrid(an1_lin, an2_lin)
        an1_eff, an2_eff = jnp.meshgrid(an1_eff, an2_eff)
        Z = sol_sol_func((an1_eff, an2_eff), params_sol[10:15])
        cf = ax.contourf(AN1, AN2, Z, levels=100, cmap="plasma", vmin=vmin, vmax=vmax)
        ax.set_title(r"$x_{solvent}$" + f"={m:.1f}")

        # only label the bottom row
        if row == maxrow - 1:
            ax.set_xlabel("AN$_1$")
        # only label the leftmost column
        if col == 0:
            ax.set_ylabel("AN$_2$")
    # single colorbar for all
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cb = fig.colorbar(cf, cax=cbar_ax, label="$J_{sol-sol}$")
    plt.savefig(
        "sol_sol_func_an_an_3x3.png",
        dpi=300,
    )
    plt.close(fig)

def sol_sol_func_save_data(params_sol, params_conc_factor_sol):
    """save the data for functional form analysis"""
    ms = np.linspace(0.1, 0.8, 8)
    npoints = 100
    dn_lin = jnp.linspace(0, 40, npoints)
    an_lin = jnp.linspace(0, 40, npoints)
    DN, AN = jnp.meshgrid(dn_lin, an_lin)
    dns = []
    ans = []
    dns_eff = []
    ans_eff = []
    ys = []
    ms_df = []
    for m in tqdm(ms):
        m_arr = m * jnp.ones_like(dn_lin)
        dn_eff = dn_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        an_eff = an_lin * conc_factor(
            m,
            0.14,
            solvent_map_dict["DME"]['volume A3'],
            solvent_map_dict["TFSI"]['volume A3'],
            params_conc_factor_sol,
        )  # assuming salt molar ratio is 0.14
        dn_eff = dn_eff.reshape(-1, 1)
        an_eff = an_eff.reshape(-1, 1)
        y_pred = sol_sol_func(jnp.hstack([dn_eff, an_eff]).T, params_sol[:5]) + logfunc(
            m_arr, params_sol[5]
        )
        dns.extend(dn_lin.flatten())
        ans.extend(an_lin.flatten())
        dns_eff.extend(dn_eff.flatten())
        ans_eff.extend(an_eff.flatten())
        ys.extend(y_pred.flatten())
        ms_df.extend([m for _ in range(len(dn_lin))])
    df = pd.DataFrame(
        {
            "solvent_dn": dns,
            "solvent_an": ans,
            "solvent_dn_eff": dns_eff,
            "solvent_an_eff": ans_eff,
            "energy": ys,
            "molar_ratio": ms_df,
        }
    )
    df.to_csv("sol_sol_func_data.csv", index=False)

def sol_sol_func_fixed_m_an(params_sol, params_conc_factor_sol):
    """
    plot how solvent DN affects output, fixing molar ratio and AN
    """
    """
    Draw a 3×3 grid of plots, each one showing f vs. solvent DN
    at a fixed AN (10 through 90), and for m=0.1…0.5.
    """
    # the nine AN values
    AN_vals = np.arange(5, 50, 5)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    npoints = 100
    # shrink the subplots on the right to make room for a single legend
    fig.subplots_adjust(
        right=0.80,  # leave 20% of width on the right
        bottom=0.07,  # a little extra room at bottom
        top=0.95,
    )  # a little extra room at top

    dn_lin = jnp.linspace(0, 40, npoints)

    # first panel will be used to gather legend handles
    legend_handles = []
    legend_labels = []

    for ax, an in zip(axes.flatten(), AN_vals):
        an_pred = an * jnp.ones_like(dn_lin).flatten()

        for m in np.arange(0.1, 0.5, 0.1):
            dn_eff = dn_lin * conc_factor(
                m,
                0.14,
                solvent_map_dict["DME"]['volume A3'],
                solvent_map_dict["TFSI"]['volume A3'],
                params_conc_factor_sol,
            )  # assuming salt molar ratio is 0.14
            an_eff = an_pred * conc_factor(
                m,
                0.14,
                solvent_map_dict["DME"]['volume A3'],
                solvent_map_dict["TFSI"]['volume A3'],
                params_conc_factor_sol,
            )  # assuming salt molar ratio is 0.14
            y_pred = sol_sol_func((dn_eff, an_eff), params_sol[:5]) + logfunc(
                m, params_sol[5]
            )
            (h,) = ax.plot(
                dn_lin.flatten(), y_pred.flatten(), label=f"m={m:.1f}", linewidth=1.5
            )
            # save handles from the very first axes
            if f"m={m:.1f}" not in legend_labels:
                legend_handles.append(h)
                legend_labels.append(f"m={m:.1f}")
        ax.set_title(f"AN = {an}", fontsize=12)
        ax.grid(True)

    # global x/y labels
    fig.text(0.5, 0.02, "Solvent DN", ha="center", va="center", fontsize=14)
    fig.text(
        0.04,
        0.5,
        "$J_{sol-sol}$",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    # put one legend in the right margin
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        title="m values",
        frameon=False,
        fontsize=12,
    )
    plt.savefig("sol_sol_func_fixed_m_an.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def sol_sol_func_fixed_m_dn(params_sol, params_conc_factor_sol):
    """
    plot how solvent DN affects output, fixing molar ratio and DN
    """
    """
    Draw a 3×3 grid of plots, each one showing f vs. solvent AN
    at a fixed DN (10 through 90), and for m=0.1…0.5.
    """
    # the nine DN values
    DN_vals = [1, 2, 5, 10, 15, 20, 25, 30, 35]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)

    # shrink the subplots on the right to make room for a single legend
    fig.subplots_adjust(
        right=0.80,  # leave 20% of width on the right
        bottom=0.07,  # a little extra room at bottom
        top=0.95,
    )  # a little extra room at top

    an_lin = jnp.linspace(0, 40, npoints).reshape(-1, 1)

    # first panel will be used to gather legend handles
    legend_handles = []
    legend_labels = []

    for ax, dn in zip(axes.flatten(), DN_vals):
        dn_pred = dn * jnp.ones_like(an_lin)

        for m in np.arange(0.1, 0.5, 0.1):
            dn_eff = dn_pred * conc_factor(
                m,
                0.14,
                solvent_map_dict["DME"]['volume A3'],
                solvent_map_dict["TFSI"]['volume A3'],
                params_conc_factor_sol,
            )  # assuming salt molar ratio is 0.14
            an_eff = an_lin * conc_factor(
                m,
                0.14,
                solvent_map_dict["DME"]['volume A3'],
                solvent_map_dict["TFSI"]['volume A3'],
                params_conc_factor_sol,
            )
            y_pred = sol_sol_func((dn_eff, an_eff), params_sol[:5]) + logfunc(
                m, params_sol[5]
            )
            (h,) = ax.plot(
                an_lin.flatten(), y_pred.flatten(), label=f"m={m:.1f}", linewidth=1.5
            )
            # save handles from the very first axes
            if f"m={m:.1f}" not in legend_labels:
                legend_handles.append(h)
                legend_labels.append(f"m={m:.1f}")

        ax.set_title(f"DN = {dn}", fontsize=12)
        ax.grid(True)

    # global x/y labels
    fig.text(0.5, 0.02, "Solvent AN", ha="center", va="center", fontsize=14)
    fig.text(
        0.04,
        0.5,
        "$J_{sol-sol}$",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=14,
    )
    # put one legend in the right margin
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        title="m values",
        frameon=False,
        fontsize=12,
    )
    plt.savefig("sol_sol_func_fixed_m_dn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def frac_occupation(input_params):
    # diluent fractional occupation as a function of solvent and diluent DN
    # solvent: diluent: salt = 3:6:1
    npoints = 50
    dn1 = jnp.linspace(0, 40, npoints)
    dn2 = jnp.linspace(0, 40, npoints)
    dn1, dn2 = jnp.meshgrid(dn1, dn2)
    dn1_flat = dn1.flatten()
    dn2_flat = dn2.flatten()
    dn_anion = 11.2 * jnp.ones_like(dn1_flat)
    x0 = 0.24 * jnp.ones_like(dn1_flat)
    x1 = 0.48 * jnp.ones_like(dn1_flat)
    an0 = 10.2 * jnp.ones_like(dn1_flat)
    an1 = 10.2 * jnp.ones_like(dn1_flat)
    x_anion = 0.14 * jnp.ones_like(dn1_flat)
    z = 3.72 / 2.0 * jnp.ones_like(dn1_flat)
    solvent_volume = solvent_map_dict["DME"]['volume A3'] * jnp.ones_like(dn1_flat)
    diluent_volume = solvent_map_dict["TTE"]['volume A3'] * jnp.ones_like(dn1_flat)
    anion_volume = solvent_map_dict["TFSI"]['volume A3'] * jnp.ones_like(dn1_flat)
    input_constants = jnp.vstack(
        [
            dn1_flat,
            dn2_flat,
            dn_anion,
            x0,
            x1,
            an0,
            an1,
            x_anion,
            solvent_volume,
            diluent_volume,
            anion_volume,
            z,
        ]
    )
    # run find_root function on each row of input_constants and whole of trained_params
    find_root_batched = jax.vmap(find_root, in_axes=(None, 1))
    roots, _ = find_root_batched(input_params, input_constants)
    # reshape roots to 100x100
    roots = roots.reshape(npoints, npoints, 3)

    # make all the sub-zero values in the roots array to 0
    # roots = jnp.where(roots < 0, 0, roots)
    def update_array(arr):
        # Compute the average of the 4-neighbors using jnp.roll
        up = jnp.roll(arr, shift=1, axis=0)
        down = jnp.roll(arr, shift=-1, axis=0)
        left = jnp.roll(arr, shift=1, axis=1)
        right = jnp.roll(arr, shift=-1, axis=1)
        neighbor_avg = (up + down + left + right) / 4.0
        # Replace values > 1 with the computed neighbor average
        arr = jnp.where(arr > 1, neighbor_avg, arr)
        # Replace values < 0 with the computed neighbor average
        arr = jnp.where(arr < 0, neighbor_avg, arr)
        return arr

    roots = jnp.where(roots < 0, 0, roots)
    roots = update_array(roots)
    # y_pred = roots
    y_pred = roots[:, :, 1]
    # subfigure of 1*3
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=True)
    vmin = np.min(roots)
    vmax = np.max(roots)
    axes[0].set_title("Solvent 1 Occupation")
    cs0 = axes[0].contourf(
        dn1.reshape(npoints, npoints),
        dn2.reshape(npoints, npoints),
        roots[:, :, 0],
        levels=100,
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )
    axes[1].set_title("Solvent 2 Occupation")
    cs1 = axes[1].contourf(
        dn1.reshape(npoints, npoints),
        dn2.reshape(npoints, npoints),
        roots[:, :, 1],
        levels=100,
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )
    axes[2].set_title("Anion Occupation")
    cs2 = axes[2].contourf(
        dn1.reshape(npoints, npoints),
        dn2.reshape(npoints, npoints),
        roots[:, :, 2],
        levels=100,
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )
    for ax in axes:
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.set_yticks(np.arange(0, 31, 5))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
    # use shared x and y labels, xlabel in the middle of the whole figure
    axes[0].set_xlabel("Solvent 1 DN")
    axes[1].set_xlabel("Solvent 1 DN")
    axes[2].set_xlabel("Solvent 1 DN")
    axes[0].set_ylabel("Solvent 2 DN")
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="plasma")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical")
    cbar.set_label("Fractional Occupation")
    plt.savefig("occupations.jpg", dpi=300, bbox_inches="tight")
    # save the root array to numpy readable format
    roots = np.array(roots)
    np.save("roots.npy", roots)

def occupations_dn2_an2_csv(trained_params):
    dn0 = 20.0
    an0 = 10.2
    dn1 = jnp.linspace(1.0, 40, 100)
    an1 = jnp.linspace(1.0, 40, 100)
    dn1, an1 = jnp.meshgrid(dn1, an1)
    dn1_flat = dn1.flatten()
    an1_flat = an1.flatten()
    dn_anion = 11.2
    x0, x1 = 0.24, 0.48
    x_anion = 0.14
    solvent_volume = solvent_map_dict["DME"]['volume A3']
    diluent_volume = solvent_map_dict["TTE"]['volume A3']
    anion_volume = solvent_map_dict["TFSI"]['volume A3']
    z = 3.72 / 2.0

    def predict(d1, a1):
        input_constants = jnp.array(
            [
                dn0,
                d1,
                dn_anion,
                x0,
                x1,
                an0,
                a1,
                x_anion,
                solvent_volume,
                diluent_volume,
                anion_volume,
                z,
            ]
        )
        roots, _ = find_root(trained_params, input_constants)
        return roots    
    batched_predict = jax.vmap(predict)
    y_pred_flat = batched_predict(dn1_flat, an1_flat)
    y_pred = y_pred_flat.reshape(100, 100, 3)
    fig = plt.figure(figsize=(18, 5))
    axes = fig.subplots(1, 3, sharex=False, sharey=True)
    axes[0].set_title("Solvent 1 Occupation")
    axes[0].contourf(
        dn1.reshape(100, 100),
        an1.reshape(100, 100),
        y_pred[:, :, 0].reshape(100, 100),
        levels=100,
        cmap="plasma",
    )
    axes[1].set_title("Solvent 2 Occupation")
    axes[1].contourf(
        dn1.reshape(100, 100),
        an1.reshape(100, 100),
        y_pred[:, :, 1].reshape(100, 100),
        levels=100,
        cmap="plasma",
    )
    axes[2].set_title("Anion Occupation")
    axes[2].contourf(
        dn1.reshape(100, 100),
        an1.reshape(100, 100),
        y_pred[:, :, 2].reshape(100, 100),
        levels=100,
        cmap="plasma",
    )
    for ax in axes:
        ax.set_xlabel("Solvent 2 DN")
        ax.set_ylabel("Solvent 2 AN")
    cbar = fig.colorbar(axes[0].collections[0], ax=axes, orientation="vertical")
    cbar.set_label("Fractional Occupation")
    plt.savefig("occpupations_dn2_an2_contour.png", dpi=300, bbox_inches="tight")
    plt.close(fig=fig)
    # save the root array to dataframe
    dn1_flat = np.array(dn1_flat)
    an1_flat = np.array(an1_flat)
    m_pred = np.array(y_pred[:,:,0]).flatten()
    n_pred = np.array(y_pred[:,:,1]).flatten()
    p_pred = np.array(y_pred[:,:,2]).flatten()
    df = pd.DataFrame(
        {
            "solvent2_dn": dn1_flat,
            "solvent2_an": an1_flat,
            "solvent1_occupation": m_pred,
            "solvent2_occupation": n_pred,
            "anion_occupation": p_pred,
        }
    )
    df.to_csv("occupations_dn2_an2_contour.csv", index=False)

def occupations_dn2_an2_contour(trained_params):
    dn0, an0 = 20.0, 10.2
    dn1 = jnp.linspace(1.0, 40, 100)
    an1 = jnp.array([5,10,15,20])
    fig, axs = plt.subplots(1,3,figsize=(18,5), sharex=False, sharey=False)
    for a1 in an1:
        dn1_flat = dn1.flatten()
        an1_flat = a1 * jnp.ones_like(dn1_flat)
        dn_anion = 11.2
        x0, x1 = 0.24, 0.48
        x_anion = 0.14
        solvent_volume = solvent_map_dict["DME"]['volume A3']
        diluent_volume = solvent_map_dict["TTE"]['volume A3']
        anion_volume = solvent_map_dict["TFSI"]['volume A3']
        z = 3.72 / 2.0

        def predict(d1, a1):
            input_constants = jnp.array(
                [
                    dn0,
                    d1,
                    dn_anion,
                    x0,
                    x1,
                    an0,
                    a1,
                    x_anion,
                    solvent_volume,
                    diluent_volume,
                    anion_volume,
                    z,
                ]
            )
            roots, _ = find_root(trained_params, input_constants)
            return roots    
        batched_predict = jax.vmap(predict)
        y_pred_flat = batched_predict(dn1_flat, an1_flat)
        y_pred = y_pred_flat.reshape(100, 3)
        axs[0].plot(dn1, y_pred[:,0], label=f"AN={a1}")
        axs[1].plot(dn1, y_pred[:,1], label=f"AN={a1}")
        axs[2].plot(dn1, y_pred[:,2], label=f"AN={a1}")
    axs[0].set_ylabel("Solvent 1 Occupation")
    axs[1].set_ylabel("Solvent 2 Occupation")
    axs[2].set_ylabel("Anion Occupation")
    for ax in axs:
        ax.set_xlabel("Solvent 2 DN")
        ax.legend()
    plt.savefig("occpupations_dn2_an2.png", bbox_inches="tight")

def visualize_li_free_energy_contour(trained_params):
    # Li solvation free energy as a function of solvent 1 and solvent 2, salt being anion
    dn1 = jnp.linspace(1.0, 40, 100)
    dn2 = jnp.linspace(1.0, 40, 100)
    dn1, dn2 = jnp.meshgrid(dn1, dn2)
    dn1_flat = dn1.flatten()
    dn2_flat = dn2.flatten()
    dn_anion = 11.2
    x0, x1 = 0.24, 0.48
    an0, an1 = 10.2, 10.2
    x_anion = 0.14
    solvent_volume = solvent_map_dict["DME"]['volume A3']
    diluent_volume = solvent_map_dict["TTE"]['volume A3']
    anion_volume = solvent_map_dict["TFSI"]['volume A3']
    z = 3.72 / 2.0

    def predict(d1, d2):
        input_constants = jnp.array(
            [
                d1,
                d2,
                dn_anion,
                x0,
                x1,
                an0,
                an1,
                x_anion,
                solvent_volume,
                diluent_volume,
                anion_volume,
                z,
            ]
        )
        return li_free_energy(trained_params, input_constants)

    batched_predict = jax.vmap(predict)
    y_pred_flat = batched_predict(dn1_flat, dn2_flat)
    y_pred = y_pred_flat.reshape(100, 100)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.contourf(
        dn1.reshape(100, 100),
        dn2.reshape(100, 100),
        y_pred.reshape(100, 100),
        levels=100,
        cmap="plasma",
    )
    plt.xlabel("Solvent 1 DN")
    plt.ylabel("Solvent 2 DN")
    plt.colorbar(label="Energy (eV)")
    plt.savefig("li_free_energy_contour.png", dpi=300, bbox_inches="tight")

def visualize_li_free_energy_dn_an_contour(trained_params):
    dn0 = 20.0
    an0 = 10.2
    dn1 = jnp.linspace(5.0, 40, 100)
    an1 = jnp.linspace(1.0, 40, 100)
    dn1, an1 = jnp.meshgrid(dn1, an1)
    dn1_flat = dn1.flatten()
    an1_flat = an1.flatten()
    dn_anion = 11.2
    x0, x1 = 0.24, 0.48
    x_anion = 0.14
    solvent_volume = solvent_map_dict["DME"]['volume A3']
    diluent_volume = solvent_map_dict["TTE"]['volume A3']
    anion_volume = solvent_map_dict["TFSI"]['volume A3']
    z = 3.72 / 2.0

    def predict(d1, a1):
        input_constants = jnp.array(
            [
                dn0,
                d1,
                dn_anion,
                x0,
                x1,
                an0,
                a1,
                x_anion,
                solvent_volume,
                diluent_volume,
                anion_volume,
                z,
            ]
        )
        return li_free_energy(trained_params, input_constants)

    batched_predict = jax.vmap(predict)
    y_pred_flat = batched_predict(dn1_flat, an1_flat)
    y_pred = y_pred_flat.reshape(100, 100)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.contourf(
        dn1.reshape(100, 100),
        an1.reshape(100, 100),
        y_pred.reshape(100, 100),
        levels=100,
        cmap="plasma",
    )
    plt.xlabel("Solvent 2 DN")
    plt.ylabel("Solvent 2 AN")
    plt.colorbar(label="Energy (eV)")
    plt.savefig("li_free_energy_dn_an_contour.png", dpi=300, bbox_inches="tight")


def salt_salt_func(params_salt):
    """
    plot how salt DN and c_anion affects output
    """
    npoints = 100
    dn_pred = jnp.linspace(-10, 40, npoints)
    c_pred = jnp.linspace(0, 0.5, npoints)
    dn_pred, an_pred = jnp.meshgrid(dn_pred, c_pred)
    dn_pred = dn_pred.reshape(-1, 1)
    c_pred = an_pred.reshape(-1, 1)
    y_pred = expfunc(dn_pred, params_salt[:4]) + logfunc(c_pred, params_salt[4])
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.contourf(
        dn_pred.reshape(npoints, npoints),
        c_pred.reshape(npoints, npoints),
        y_pred.reshape(npoints, npoints),
        levels=100,
        cmap="plasma",
    )
    plt.xlabel("Salt DN")
    plt.ylabel("Salt Molar Ratio")
    plt.colorbar(label="$J_{anion-anion}$")
    plt.savefig("salt_salt_func.png", dpi=300, bbox_inches="tight")


def salt_salt_func_fix_c(params_salt):
    """
    plot how salt DN affects output, fixing c
    """
    npoints = 100
    dn_pred = jnp.linspace(-10, 40, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    for c in np.arange(0.05, 0.25, 0.05):
        c_pred = c * jnp.ones_like(dn_pred)
        y_pred = expfunc(dn_pred, params_salt[:4]) + logfunc(c_pred, params_salt[4:])
        plt.plot(
            dn_pred.flatten(), y_pred.flatten(), label=r"$x_{anion}$" + f"={c:.2f}"
        )
    plt.xlabel("Salt DN")
    plt.ylabel("$J_{anion-anion}$")
    plt.legend()
    plt.savefig("salt_salt_func_fixed_c.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def salt_salt_func_fixed_dn(params_salt):
    """
    plot how molar ratio affects output, fixing DN
    """
    npoints = 100
    c_pred = jnp.linspace(0.05, 0.5, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    salt_dn_dict = {"PF6": -6.2, "BF4": 7.3, "TFSI": 11.2}
    for key in salt_dn_dict.keys():
        dn = salt_dn_dict[key]
        dn_pred = dn * jnp.ones_like(c_pred)
        y_pred = expfunc(dn_pred, params_salt[:4]) + logfunc(c_pred, params_salt[4:])
        plt.plot(c_pred.flatten(), y_pred.flatten(), label=f"{key}$^-$")
    plt.xlabel("Molar Ratio")
    plt.ylabel("$J_{anion-anion}$")
    plt.legend()
    plt.savefig("salt_salt_func_fixed_dn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_conc_factor_x(params_conc_sol):
    # plot how solvent salt DN and c affects output
    x_salts = jnp.arange(0.01, 0.3, 0.01)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    phi_sol_list = []
    y_pred_list = []
    for x_salt in x_salts:
        npoints = 100
        x_sol = jnp.linspace(0.2, 0.7, npoints)
        x_anion = x_salt * jnp.ones_like(x_sol)
        phi_sol = x_sol / x_salt
        y_pred = expfunc(phi_sol, params_conc_sol[:4])
        phi_sol_list.extend(phi_sol.flatten().tolist())
        y_pred_list.extend(y_pred.flatten().tolist())
        plt.plot(
            phi_sol.flatten(),
            y_pred.flatten(),
            label=f"x_salt={x_salt:.2f}",
            color="tab:blue",
        )
    # plt.xlabel('Solvent Mole Fraction')
    plt.xlabel(r"$x_{sol}/x_{li}$")
    plt.ylabel("Concentration Factor")
    # plt.legend()
    plt.savefig("conc_factor_sol_x.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    df = pd.DataFrame(
        {
            "phi_sol": phi_sol_list,
            "conc_factor": y_pred_list,
        }
    )
    df.to_csv("conc_factor_sol_x.csv", index=False)


def visualize_conc_factor_V(params_conc_sol):
    # plot how solvent salt DN and c affects output
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    phi_sol_list = []
    y_pred_list = []
    npoints = 100
    volumes = jnp.linspace(60, 200, npoints)
    phi_sol = solvent_map_dict["TFSI"]['volume A3'] / volumes
    y_pred = expfunc(phi_sol, params_conc_sol[4:8])
    plt.plot(phi_sol, y_pred.flatten())
    plt.xlabel(r"$V_{anion} / V_{sol}$")
    plt.ylabel("Concentration Factor")
    # plt.legend()
    plt.savefig("conc_factor_sol_V.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def visualize_conc_factor_x_V(params_conc_sol):
    fig, ax = plt.subplots(figsize=(6,5))
    xs = jnp.arange(1.0, 10, 0.1)
    vs = jnp.arange(0.8, 3.5, 0.1)
    X, V = jnp.meshgrid(xs, vs)
    output = expfunc(X, params_conc_sol[:4]) * expfunc(V, params_conc_sol[4:8])
    plt.contourf(X, V, output, levels=100, cmap="plasma")
    plt.xlabel(r"$x_{sol}/x_{anion}$")
    plt.ylabel(r"$V_{anion} / V_{sol}$")
    plt.colorbar(label="Correction Factor")
    plt.savefig("conc_factor_sol_x_V.png", dpi=300, bbox_inches="tight")
    plt.close(fig)