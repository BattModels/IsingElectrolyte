import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from ..interactions import expfunc, logfunc, sol_sol_func
from ..conc_vol_correction import conc_factor_sep_x_v_sigmoid as conc_factor
from . import solvent_map_dict


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

def salt_salt_func(params_anion_anion):
    """Self-interaction J(anion, anion) as a function of anion DN and molar ratio.

    Uses the self-interaction form (DN_i = DN_j), sweeping DN and molar ratio
    as a 2D contour. For the cross-interaction contour (DN_i vs DN_j), see
    salt_salt_func_cross_contour.
    """
    npoints = 100
    dn_lin = jnp.linspace(-10, 40, npoints)
    c_lin  = jnp.linspace(0, 0.5, npoints)
    DN, C  = jnp.meshgrid(dn_lin, c_lin)
    dn_pred = DN.reshape(-1, 1)
    c_pred  = C.reshape(-1, 1)
    # Self-interaction: both anions have the same DN and the same molar ratio
    y_pred = (
        sol_sol_func((dn_pred, dn_pred), params_anion_anion[:5])
        + 2.0 * logfunc(c_pred, params_anion_anion[5])
    )
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.contourf(
        dn_pred.reshape(npoints, npoints),
        c_pred.reshape(npoints, npoints),
        y_pred.reshape(npoints, npoints),
        levels=100,
        cmap="plasma",
    )
    plt.xlabel("Anion DN")
    plt.ylabel("Anion Molar Ratio")
    plt.colorbar(label="$J_{anion-anion}$")
    plt.savefig("salt_salt_func.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def salt_salt_func_fixed_c(params_anion_anion):
    """Self-interaction J(anion, anion) vs. anion DN for fixed molar ratios."""
    npoints = 100
    dn_pred = jnp.linspace(-10, 40, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    for c in np.arange(0.05, 0.25, 0.05):
        c_pred = c * jnp.ones_like(dn_pred)
        y_pred = (
            sol_sol_func((dn_pred, dn_pred), params_anion_anion[:5])
            + 2.0 * logfunc(c_pred, params_anion_anion[5])
        )
        plt.plot(
            dn_pred.flatten(), y_pred.flatten(), label=r"$x_{anion}$" + f"={c:.2f}"
        )
    plt.xlabel("Anion DN")
    plt.ylabel("$J_{anion-anion}$")
    plt.legend()
    plt.savefig("salt_salt_func_fixed_c.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def salt_salt_func_fixed_dn(params_anion_anion):
    """Self-interaction J(anion, anion) vs. molar ratio for fixed anion DNs."""
    npoints = 100
    c_pred = jnp.linspace(0.05, 0.5, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    salt_dn_dict = {"PF6": -6.2, "BF4": 7.3, "TFSI": 11.2}
    for name, dn in salt_dn_dict.items():
        dn_pred = dn * jnp.ones_like(c_pred)
        y_pred = (
            sol_sol_func((dn_pred, dn_pred), params_anion_anion[:5])
            + 2.0 * logfunc(c_pred, params_anion_anion[5])
        )
        plt.plot(c_pred.flatten(), y_pred.flatten(), label=f"{name}$^-$")
    plt.xlabel("Anion Molar Ratio")
    plt.ylabel("$J_{anion-anion}$")
    plt.legend()
    plt.savefig("salt_salt_func_fixed_dn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def salt_salt_func_cross_contour(params_anion_anion):
    """Cross-interaction J(anion_i, anion_j) as a 2D contour over both anion DNs.

    Produces one panel per fixed molar ratio (shared for both anions).
    This complements salt_salt_func (which shows the self-interaction diagonal).
    """
    x_vals  = [0.05, 0.10, 0.15, 0.20]
    npoints = 100
    dn_lin  = jnp.linspace(-10, 40, npoints)
    DN_i, DN_j = jnp.meshgrid(dn_lin, dn_lin)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, x in zip(axes.flatten(), x_vals):
        # Cross-interaction: logfunc(x_i, ...) + logfunc(x_j, ...) with x_i = x_j = x
        Z = (
            sol_sol_func((DN_i, DN_j), params_anion_anion[:5])
            + 2.0 * logfunc(x * jnp.ones_like(DN_i), params_anion_anion[5])
        )
        cf = ax.contourf(DN_i, DN_j, Z, levels=100, cmap="plasma")
        ax.set_title(f"$x_{{anion}}$ = {x:.2f}")
        fig.colorbar(cf, ax=ax, label="$J_{anion-anion}$")
    for ax in axes[-1]:
        ax.set_xlabel("Anion $i$ DN")
    for ax in axes[:, 0]:
        ax.set_ylabel("Anion $j$ DN")
    plt.savefig("salt_salt_func_cross_contour.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
