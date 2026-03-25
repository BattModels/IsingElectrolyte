import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import jax
import jax.numpy as jnp

from ..model import find_root, li_free_energy
from . import solvent_map_dict


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
