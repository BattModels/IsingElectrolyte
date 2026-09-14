import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from ..interactions import expfunc, logfunc, sol_sol_func
from ..conc_vol_correction import conc_factor_sep_x_v_sigmoid as conc_factor
from . import solvent_map_dict


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
                conc_factor_sol,
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
