import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

from ..interactions import expfunc
from . import solvent_map_dict


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
