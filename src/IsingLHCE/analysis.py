import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import jax
import jax.numpy as jnp
from functions import (
    expfunc, 
    logfunc, 
    polynomial_func, 
    mlp, 
    sol_sol_func, 
)
from fit_model import (
    equations, 
    find_root, 
    get_root_error, 
    li_free_energy, 
    conc_factor,
)
from ocnc_factor_func import *
from tqdm import tqdm

solvent_map_dict = {
    'DMSO': [29.8, 19.3, 679, 104.62906905680619, 78.14], 
    'DCM': [1.0, 20.4, 6344, 88.23001018449519, 84.93], 
    'DCD': [3.2, 16.7, 19869, 143.47530021417475, 156.95], 
    'DEA': [32.2, 13.6, 12703, 177.36010258286555, 115.17], 
    'MPL': [27.3, 13.3, 13387, 139.65543911138744, 99.13], 
    'DME': [20.0, 10.2, 8071, 135.53372116858102, 90.12], 
    'DMA': [27.8, 13.6, 31374, 130.98940364974786, 87.12],
    'ACN': [14.1,18.9, 6342, 67.23175073152937,41.05],
    'TFSI':[11.2, None, 4176748, 209.7630622823716, 280.15],
    'FSI':[None, None, 5257893, 141.08788702493268, 180.14],
    'PF6':[-6.2, None, 9886, 98.22421574221886, 144.964],
    'BF4':[7.3, None, 26255, 72.88470609916, 86.81],
}

def sol_dn_func_contour(npoints=1000, sol_params_dn=None, params_conc_sol=None, conc_factor = conc_factor_sep_x_v_sigmoid):
    """
    plot how solvent DN affects output
    """
    x_pred = jnp.linspace(1.0, 40.0, npoints)
    m_pred = jnp.linspace(0.1, 0.8, npoints)
    x_pred, m_pred = jnp.meshgrid(x_pred, m_pred)
    x_pred = x_pred.reshape(-1, 1)
    m_pred = m_pred.reshape(-1, 1)
    dn_eff = x_pred * conc_factor(m_pred, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3],params_conc_sol)
    y_pred = expfunc(dn_eff, sol_params_dn[:4]) + logfunc(m_pred, sol_params_dn[4])
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    plt.contourf(x_pred.reshape(npoints, npoints), m_pred.reshape(npoints, npoints), y_pred.reshape(npoints, npoints), levels=100, cmap='coolwarm')
    plt.xlabel('Solvent DN')
    plt.ylabel('Solvent Molar Ratio')
    plt.colorbar(label='Energy (eV)')
    plt.savefig('sol_dn_func.png', dpi=300, bbox_inches='tight')

# @jax.jit
def sol_dn_func_data_to_csv(sol_params_dn, params_conc_sol, conc_factor = conc_factor_sep_x_v_sigmoid):
    """save the data for functional form analysis"""
    npoints = 100
    x_pred = jnp.linspace(1.0,40.0,npoints)
    m_pred = jnp.linspace(0.1, 0.8, npoints)
    xs = []
    ms = []
    ys = []
    for x in tqdm(x_pred):
        for m in m_pred:
            dn_eff = x * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3],params_conc_factor_sol)
            y = expfunc(dn_eff, sol_params_dn[:4]) + logfunc(m, sol_params_dn[4])
            xs.append(x)
            ms.append(m)
            ys.append(y)
    xs = np.array(xs)
    ms = np.array(ms)
    ys = np.array(ys)
    print(xs.shape, ms.shape, ys.shape)
    df = pd.DataFrame({
        'solvent_dn': xs,
        'molar_ratio': ms,
        'energy': ys,
    })
    df.to_csv('sol_dn_func_data.csv', index=False)    

def sol_dn_func_fixed_m(sol_params_dn, params_conc_factor_sol):
    """plot how DN affects output, fixing molar ratio"""
    x_pred = jnp.linspace(0, 40, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    for m in np.arange(0.2, 0.6, 0.1):
        m_pred = m * jnp.ones_like(x_pred)
        dn_eff = x_pred * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)
        y_pred = expfunc(dn_eff, sol_params_dn[:4]) + logfunc(m_pred, sol_params_dn[4:])
        plt.plot(x_pred.flatten(), y_pred.flatten(),label=r"$x_{solvent}$"+f'={m:.1f}')
    plt.xlabel('Solvent DN')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.savefig('sol_dn_func_fixed_m.png', dpi=300, bbox_inches='tight')    

def sol_dn_func_no_conc_factor(sol_params_dn):
    x_pred = jnp.linspace(0, 40, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    y_pred = expfunc(x_pred, sol_params_dn[:4])
    plt.plot(x_pred.flatten(), y_pred.flatten())
    # plot the half wave potential exp data points
    df = pd.read_csv('/nfs/turbo/coe-venkvis/zhaohc/ising-electrolyte/LHCE/Ising-model-fitting/DN_trade_offs.csv')
    plt.plot(df['x'],df['y'],'o', label='Exp Data')
    plt.xlabel('Solvent DN')
    plt.ylabel('Energy (eV)')
    # plt.legend()
    plt.savefig('sol_dn_func_no_conc_factor.png', dpi=300, bbox_inches='tight')    

def sol_an_func_no_conc_factor(sol_params_an):
    x_pred = jnp.linspace(0, 100, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    y_pred = expfunc(x_pred, sol_params_an[:4])
    plt.plot(x_pred.flatten(), y_pred.flatten())
    plt.xlabel('Solvent AN')
    plt.ylabel('Energy (eV)')
    # plt.legend()
    plt.savefig('sol_an_func_no_conc_factor.png', dpi=300, bbox_inches='tight')    


def sol_dn_func_fixed_dn(sol_params_dn, params_conc_factor_sol):
    """plot how molar ratio affects output, fixing DN"""
    x_pred = jnp.arange(0.1,0.8,0.01).reshape(-1, 1)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    for dn in [1,5,10,15,20,25,30,35]:
        dn_pred = dn * jnp.ones_like(x_pred)
        dn_eff = dn_pred * conc_factor(x_pred, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)
        y_pred = expfunc(dn_eff, sol_params_dn[:4]) + logfunc(x_pred, sol_params_dn[4:])
        plt.plot(x_pred.flatten(), y_pred.flatten(),label=f'DN = {dn:d}')
    plt.xlabel('Molar Ratio')
    plt.ylabel('Energy (eV)')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('sol_dn_func_fixed_dn.png', dpi=300, bbox_inches='tight')

def an_m_func_contour(params_an, params_conc_factor_sol):
    # plot how solvent AN and M affects output, 2d contour plot
    an_pred = jnp.linspace(0.0, 100, npoints)
    m_pred = jnp.linspace(0.1, 0.8, npoints)
    an_pred, m_pred = jnp.meshgrid(an_pred, m_pred)
    an_pred = an_pred.reshape(-1, 1)
    m_pred = m_pred.reshape(-1, 1)
    an_eff = an_pred * conc_factor(m_pred, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)
    y_pred = expfunc(an_eff, params_an[:4]) + logfunc(m_pred, params_an[4])
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    plt.contourf(an_pred.reshape(npoints, npoints), m_pred.reshape(npoints, npoints), y_pred.reshape(npoints, npoints), levels=100, cmap='coolwarm')
    plt.xlabel('Solvent AN')
    plt.ylabel('Solvent Molar Ratio')
    plt.colorbar(label='Energy (eV)')
    plt.savefig('an_m_func.png', dpi=300, bbox_inches='tight')


# @jax.jit
def sol_an_func_data_to_csv(sol_params_an, params_conc_factor_sol):
    """save the data for functional form analysis"""
    npoints = 100
    x_pred = jnp.linspace(0.0,100.0,npoints)
    m_pred = jnp.linspace(0.1,0.8,npoints)
    xs = []
    ms = []
    ys = []
    for x in tqdm(x_pred):
        for m in m_pred:
            an_eff = x * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)
            y = expfunc(an_eff, sol_params_an[:4]) + logfunc(m, sol_params_an[4]) + logfunc(0.1, sol_params_an[5])  # assuming salt molar ratio is 0.1
            xs.append(x)
            ms.append(m)
            ys.append(y)
    xs = np.array(xs)
    ms = np.array(ms)
    ys = np.array(ys)
    df = pd.DataFrame({
        'solvent_an': xs,
        'molar_ratio': ms,
        'energy': ys,
    })
    df.to_csv('sol_an_func_data.csv', index=False)    

def salt_dn_func_fixed_c(salt_params_dn):
    """
    fix c, plot how solvent salt DN affects output
    """
    dn_pred = jnp.linspace(-10, 20, 1000).reshape(-1, 1)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    for c in np.arange(0.05, 0.25, 0.05):
        c_pred = c * jnp.ones_like(dn_pred)
        y_pred = expfunc(dn_pred, salt_params_dn[:4]) + logfunc(c_pred, salt_params_dn[4:])
        plt.plot(dn_pred.flatten(), y_pred.flatten(),label=r"$x_{anion}=$"+f'{c:.3f}')
    plt.xlabel('Salt DN')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.savefig('salt_dn_func_fixed_c.png', dpi=300, bbox_inches='tight')

def an_func_fixed_m(params_an, params_conc_factor_sol):
    """
    fix m, plot how solvent AN affects output
    """
    an_pred = jnp.linspace(0, 60, 1000).reshape(-1, 1)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    for m in np.arange(0.1, 0.5, 0.1):
        an_eff = an_pred * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)  # assuming salt molar ratio is 0.1
        y_pred = expfunc(an_eff, params_an[:4]) + logfunc(m * jnp.ones_like(an_pred), params_an[4])
        plt.plot(an_pred.flatten(), y_pred.flatten(),label=r'$x_{solvent}$'+f'={m:.1f}')
    plt.xlabel('Solvent AN')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.savefig(f'an_func_fixed_m.png', dpi=300, bbox_inches='tight')

def an_func_fixed_an(params_an, params_conc_factor_sol):
    """fix an, plot how m affects output"""
    m_pred = jnp.arange(0.1, 0.8, 0.01).reshape(-1, 1)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    for an_pred in range(10, 100, 10):    
        an_eff = an_pred * conc_factor(m_pred, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)  # assuming salt molar ratio is 0.1
        y_pred = expfunc(an_eff, params_an[:4]) + logfunc(m_pred, params_an[4]) + logfunc(0.1, params_an[5])
        plt.plot(m_pred.flatten(), y_pred.flatten(),label=f'AN = {an_pred:d}')
    plt.xlabel('Molar Ratio')
    plt.ylabel('Energy (eV)')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'an_func_fixed_an.png', dpi=300, bbox_inches='tight')    

def salt_dn_func(salt_params_dn):
    # plot how solvent salt DN and c affects output
    dn_pred = jnp.linspace(-10, 40, npoints)
    c_pred = jnp.linspace(0.1, 0.8, npoints)
    dn_pred, c_pred = jnp.meshgrid(dn_pred, c_pred)
    dn_pred = dn_pred.reshape(-1, 1)
    c_pred = c_pred.reshape(-1, 1)
    y_pred = expfunc(dn_pred, salt_params_dn[:4]) + logfunc(c_pred, salt_params_dn[4:])
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    plt.contourf(dn_pred.reshape(npoints, npoints), c_pred.reshape(npoints, npoints), y_pred.reshape(npoints, npoints), levels=100, cmap='coolwarm')
    plt.xlabel('Salt DN')
    plt.ylabel('Salt Molar Ratio')
    plt.colorbar(label='Energy (eV)')
    plt.savefig('salt_dn_func.png', dpi=300, bbox_inches='tight')    

def sol_sol_func_contour(params_sol, params_conc_factor_sol):
    """
    plot how DN and AN affects sol-sol interaction
    """
    ms = np.linspace(0.1, 0.8, 8)
    npoints = 100
    dn_lin = jnp.linspace(0, 40, npoints)
    an_lin = jnp.linspace(0, 40, npoints)
    # pre‐compute global min/max of f so all panels share the same color‐scale
    all_vals = []
    for m in ms:
        dn_eff = dn_lin * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)  # assuming salt molar ratio is 0.1
        an_eff = an_lin * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)  # assuming salt molar ratio is 0.1
        m_arr = m * jnp.ones_like(dn_eff)
        DN, AN = jnp.meshgrid(dn_eff, an_eff)
        dn_eff = dn_eff.reshape(-1, 1)
        an_eff = an_eff.reshape(-1, 1)
        val = sol_sol_func(jnp.hstack([dn_eff, an_eff]).T,params_sol[:6]) + logfunc(m_arr, params_sol[6])
        all_vals.append(val)
    all_vals = jnp.concatenate(all_vals)
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, sharey=True,gridspec_kw={"right": 0.85},constrained_layout=False)
    for (row, col), ax in np.ndenumerate(axes):
        m = ms[row * axes.shape[1] + col]
        dn_eff = dn_lin * m / 0.1  # assuming salt molar ratio is 0.1
        an_eff = an_lin * m / 0.1  # assuming salt molar ratio is 0.1
        DN, AN = jnp.meshgrid(dn_lin, an_lin)   
        dn_eff, an_eff = jnp.meshgrid(dn_eff, an_eff)
        Z = sol_sol_func((dn_eff, an_eff), params_sol[:6]) + logfunc(m, params_sol[6])
        print(Z.shape, sol_sol_func((dn_eff, an_eff), params_sol[:6]).shape, logfunc(m, params_sol[6]).shape)
        cf = ax.contourf(
            DN, AN, Z,
            levels=100,
            cmap="coolwarm",
            vmin=vmin, vmax=vmax
        )
        ax.set_title(r"$x_{solvent}$"+f"={m:.1f}")

        # only label the bottom row
        if row == 2:
            ax.set_xlabel("Solvent DN")
        # only label the leftmost column
        if col == 0:
            ax.set_ylabel("Solvent AN")
    # single colorbar for all
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cb = fig.colorbar(cf, cax=cbar_ax, label="Energy (eV)")
    plt.savefig("sol_sol_func_3x3.png", dpi=300,)    

def sol_sol_func_save_data(params_sol, params_conc_factor_sol):
    """ save the data for functional form analysis"""
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
        dn_eff = dn_lin * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)  # assuming salt molar ratio is 0.1
        an_eff = an_lin * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)  # assuming salt molar ratio is 0.1
        dn_eff = dn_eff.reshape(-1, 1)
        an_eff = an_eff.reshape(-1, 1)
        y_pred = sol_sol_func(jnp.hstack([dn_eff, an_eff]).T, params_sol[:6]) + logfunc(m_arr, params_sol[6])
        dns.extend(dn_lin.flatten())
        ans.extend(an_lin.flatten())
        dns_eff.extend(dn_eff.flatten())
        ans_eff.extend(an_eff.flatten())
        ys.extend(y_pred.flatten())
        ms_df.extend([m for _ in range(len(dn_lin))])
    df = pd.DataFrame({
        'solvent_dn': dns,
        'solvent_an': ans,
        'solvent_dn_eff': dns_eff,
        'solvent_an_eff': ans_eff,
        'energy': ys,
        'molar_ratio': ms_df,
    })
    df.to_csv('sol_sol_func_data.csv', index=False)

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
    fig.subplots_adjust(right=0.80,   # leave 20% of width on the right
                        bottom=0.07,  # a little extra room at bottom
                        top=0.95)     # a little extra room at top

    dn_lin = jnp.linspace(0, 40, npoints)

    # first panel will be used to gather legend handles
    legend_handles = []
    legend_labels  = []

    for ax, an in zip(axes.flatten(), AN_vals):
        an_pred = an * jnp.ones_like(dn_lin).flatten()

        for m in np.arange(0.1, 0.5, 0.1):
            dn_eff = dn_lin * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)  # assuming salt molar ratio is 0.1
            an_eff = an_pred * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)  # assuming salt molar ratio is 0.1
            y_pred = sol_sol_func((dn_eff, an_eff), params_sol[:6]) + logfunc(m, params_sol[6])
            h, = ax.plot(dn_lin.flatten(),
                         y_pred.flatten(),
                         label=f"m={m:.1f}",
                         linewidth=1.5)
            # save handles from the very first axes
            if f"m={m:.1f}" not in legend_labels:
                legend_handles.append(h)
                legend_labels.append(f"m={m:.1f}")
        ax.set_title(f"AN = {an}", fontsize=12)
        ax.grid(True)

    # global x/y labels
    fig.text(0.5, 0.02, "Solvent DN", ha="center", va="center", fontsize=14)
    fig.text(0.04, 0.5, "Energy (eV)", ha="center", va="center",
             rotation="vertical", fontsize=14)

    # put one legend in the right margin
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        title="m values",
        frameon=False,
        fontsize=12
    )
    plt.savefig('sol_sol_func_fixed_m_an.png', dpi=300, bbox_inches='tight')

def sol_sol_func_fixed_m_dn(params_sol, params_conc_factor_sol):
    """
    plot how solvent DN affects output, fixing molar ratio and DN
    """
    """
    Draw a 3×3 grid of plots, each one showing f vs. solvent AN
    at a fixed DN (10 through 90), and for m=0.1…0.5.
    """
    # the nine DN values
    DN_vals = [1,2,5,10,15,20,25,30,35]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)

    # shrink the subplots on the right to make room for a single legend
    fig.subplots_adjust(right=0.80,   # leave 20% of width on the right
                        bottom=0.07,  # a little extra room at bottom
                        top=0.95)     # a little extra room at top

    an_lin = jnp.linspace(0, 40, npoints).reshape(-1, 1)

    # first panel will be used to gather legend handles
    legend_handles = []
    legend_labels  = []

    for ax, dn in zip(axes.flatten(), DN_vals):
        dn_pred = dn * jnp.ones_like(an_lin)

        for m in np.arange(0.1, 0.5, 0.1):
            dn_eff = dn_pred * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)  # assuming salt molar ratio is 0.1
            an_eff = an_lin * conc_factor(m, 0.1, solvent_map_dict['DME'][3], solvent_map_dict['TFSI'][3], params_conc_factor_sol)
            y_pred = sol_sol_func((dn_eff, an_eff), params_sol[:6]) + logfunc(m, params_sol[6])
            h, = ax.plot(an_lin.flatten(),
                         y_pred.flatten(),
                         label=f"m={m:.1f}",
                         linewidth=1.5)
            # save handles from the very first axes
            if f"m={m:.1f}" not in legend_labels:
                legend_handles.append(h)
                legend_labels.append(f"m={m:.1f}")

        ax.set_title(f"DN = {dn}", fontsize=12)
        ax.grid(True)

    # global x/y labels
    fig.text(0.5, 0.02, "Solvent AN", ha="center", va="center", fontsize=14)
    fig.text(0.04, 0.5, "Energy (eV)", ha="center", va="center",
             rotation="vertical", fontsize=14)
    # put one legend in the right margin
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        title="m values",
        frameon=False,
        fontsize=12
    )
    plt.savefig('sol_sol_func_fixed_m_dn.png', dpi=300, bbox_inches='tight')

def frac_occupation():
    # diluent fractional occupation as a function of solvent and diluent DN
    # solvent: diluent: salt = 3:6:1
    npoints = 50
    dn1 = jnp.linspace(0, 30, npoints)
    dn2 = jnp.linspace(0, 30, npoints)
    dn1, dn2 = jnp.meshgrid(dn1, dn2)
    dn1_flat = dn1.flatten()
    dn2_flat = dn2.flatten()
    dn_anion = 11.2 * jnp.ones_like(dn1_flat)
    x0 = 0.273 * jnp.ones_like(dn1_flat)
    x1 = 0.556 * jnp.ones_like(dn1_flat)
    an0 = 20.0 * jnp.ones_like(dn1_flat)
    an1 = 20.0 * jnp.ones_like(dn1_flat)
    x_anion = 0.1 * jnp.ones_like(dn1_flat)
    z = 3.72 / 2.0 * jnp.ones_like(dn1_flat)
    solvent_volume = solvent_map_dict['DME'][4] * jnp.ones_like(dn1_flat)
    diluent_volume = solvent_map_dict['DCM'][4] * jnp.ones_like(dn1_flat)
    anion_volume = solvent_map_dict['TFSI'][4] * jnp.ones_like(dn1_flat)
    input_constants = jnp.vstack([dn1_flat, dn2_flat, dn_anion, x0, x1, an0, an1, x_anion,solvent_volume, diluent_volume, anion_volume, z])
    # run find_root function on each row of input_constants and whole of trained_params
    find_root_batched = jax.vmap(find_root, in_axes=(None, 1))
    roots = find_root_batched(trained_params, input_constants)
    # reshape roots to 100x100
    roots = roots.reshape(npoints, npoints, 3)
    # make all the sub-zero values in the roots array to 0
    # roots = jnp.where(roots < 0, 0, roots)
    def update_array(arr):
        # Compute the average of the 4-neighbors using jnp.roll
        up    = jnp.roll(arr, shift=1, axis=0)
        down  = jnp.roll(arr, shift=-1, axis=0)
        left  = jnp.roll(arr, shift=1, axis=1)
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    axes[0].set_title('Solvent 1 Occupation')
    axes[0].contourf(dn1.reshape(npoints, npoints), dn2.reshape(npoints, npoints), roots[:, :, 0], levels=100, cmap='coolwarm')
    axes[1].set_title('Solvent 2 Occupation')
    axes[1].contourf(dn1.reshape(npoints, npoints), dn2.reshape(npoints, npoints), roots[:, :, 1], levels=100, cmap='coolwarm')
    axes[2].set_title('Anion Occupation')
    axes[2].contourf(dn1.reshape(npoints, npoints), dn2.reshape(npoints, npoints), roots[:, :, 2], levels=100, cmap='coolwarm')
    for ax in axes:
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.set_yticks(np.arange(0, 31, 5))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
    # use shared x and y labels, xlabel in the middle of the whole figure
    axes[1].set_xlabel('Solvent 1 Donor Number')
    axes[0].set_ylabel('Solvent 2 Donor Number')
    cbar = fig.colorbar(axes[0].collections[0], ax=axes, orientation='vertical')
    cbar.set_ticks(np.arange(0, 0.6, 0.1))
    cbar.set_label('Fractional Occupation')
    plt.savefig('occpupations.jpg', dpi=300, bbox_inches='tight')
    # save the root array to numpy readable format
    roots = np.array(roots)
    np.save('roots.npy', roots)    

def visualize_li_free_energy(trained_params):
    # Li solvation free energy as a function of solvent 1 and solvent 2, salt being anion
    dn1 = jnp.linspace(1.0, 40, 100)
    dn2 = jnp.linspace(1.0, 40, 100)
    dn1, dn2 = jnp.meshgrid(dn1, dn2)
    dn1_flat = dn1.flatten()
    dn2_flat = dn2.flatten()
    dn_anion = 11.2
    x0, x1 = 0.273, 0.556
    an0, an1 = 20.0, 20.0
    x_anion = 0.091
    solvent_volume = solvent_map_dict['DME'][4]
    diluent_volume = solvent_map_dict['DCM'][4]
    anion_volume = solvent_map_dict['TFSI'][4]
    z = 3.72 / 2.0
    def predict(d1,d2):
        input_constants = jnp.array([d1,d2,dn_anion, x0, x1, an0, an1, x_anion,solvent_volume, diluent_volume, anion_volume, z])
        return li_free_energy(trained_params, input_constants)
    batched_predict = jax.vmap(predict)
    y_pred_flat = batched_predict(dn1_flat,dn2_flat)
    y_pred = y_pred_flat.reshape(100,100)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    plt.contourf(dn1.reshape(100, 100), dn2.reshape(100, 100), y_pred.reshape(100, 100), levels=100, cmap='coolwarm')
    plt.xlabel('Solvent 1 DN')
    plt.ylabel('Solvent 2 DN')
    plt.colorbar(label='Energy (eV)')
    plt.savefig('li_free_energy.png', dpi=300, bbox_inches='tight')    

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
    y_pred = dn_func(jnp.hstack([dn_pred, c_pred]), params_salt)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    plt.contourf(dn_pred.reshape(npoints, npoints), c_pred.reshape(npoints, npoints), y_pred.reshape(npoints, npoints), levels=100, cmap='coolwarm')
    plt.xlabel('Salt DN')
    plt.ylabel('Salt Molar Ratio')
    plt.colorbar(label='Energy (eV)')
    plt.savefig('salt_salt_func.png', dpi=300, bbox_inches='tight')

def salt_salt_func_fix_c(params_salt):
    """
    plot how salt DN affects output, fixing c
    """
    npoints = 100
    dn_pred = jnp.linspace(-10, 40, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    for c in np.arange(0.05, 0.25, 0.05):
        c_pred = c * jnp.ones_like(dn_pred)
        y_pred = dn_func(jnp.hstack([dn_pred,c_pred]), params_salt)
        plt.plot(dn_pred.flatten(), y_pred.flatten(),label=r"$x_{anion}$"+f'={c:.3f}')
    plt.xlabel('Salt DN')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.savefig('salt_salt_func_fixed_c.png', dpi=300, bbox_inches='tight')

def salt_salt_func_fixed_dn(params_salt):
    """
    plot how molar ratio affects output, fixing DN
    """
    npoints = 100
    c_pred = jnp.linspace(0.05, 0.5, npoints).reshape(-1, 1)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    for dn in range(-10, 40, 5):
        dn_pred = dn * jnp.ones_like(c_pred)
        y_pred = dn_func(jnp.hstack([dn_pred,c_pred]), params_salt)
        plt.plot(c_pred.flatten(), y_pred.flatten(),label=f'DN = {dn:d}')
    plt.xlabel('Molar Ratio')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.savefig('salt_salt_func_fixed_dn.png', dpi=300, bbox_inches='tight')

def visualize_conc_factor_x(params_conc_sol):
    # plot how solvent salt DN and c affects output
    x_salts = jnp.arange(0.01,0.3,0.01)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
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
        plt.plot(phi_sol.flatten(), y_pred.flatten(), label=f'x_salt={x_salt:.2f}', color='tab:blue')
    # plt.xlabel('Solvent Mole Fraction')
    plt.xlabel(r'$x_{sol}/x_{li}$')
    plt.ylabel('Concentration Factor')
    # plt.legend()
    plt.savefig("conc_factor_sol_x.png", dpi=300, bbox_inches="tight")
    df = pd.DataFrame({
        'phi_sol': phi_sol_list,
        'conc_factor': y_pred_list,
    })
    df.to_csv('conc_factor_sol_x.csv', index=False)

def visualize_conc_factor_V(params_conc_sol):
    # plot how solvent salt DN and c affects output
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0,0,1,1])
    phi_sol_list = []
    y_pred_list = []
    npoints = 100
    volumes = jnp.linspace(60, 200, npoints)
    phi_sol = solvent_map_dict['TFSI'][3] / volumes
    y_pred = expfunc(phi_sol, params_conc_sol[4:8])
    plt.plot(phi_sol, y_pred.flatten())
    plt.xlabel(r'$V_{anion} / V_{sol} (\AA^3)$')
    plt.ylabel('Concentration Factor')
    # plt.legend()
    plt.savefig("conc_factor_sol_V.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    with open('trained_params.pkl', 'rb') as f:
        trained_params = pickle.load(f)

    num_params = 0
    for key in trained_params:
        for ii in trained_params[key]:
            num_params += ii.size
    print(f'Total number of parameters: {num_params}')    

    sol_params_dn = trained_params["sol_params_dn"]
    salt_params_dn = trained_params["salt_params_dn"]
    params_an = trained_params["params_sol_salt_an"]
    params_sol = trained_params["params_sol_sol"]
    params_salt = trained_params["params_salt"]
    params_conc_sol = trained_params["conc_factor_sol"]

    npoints = 100

    sol_dn_func_contour(npoints, sol_params_dn, params_conc_sol)
    sol_dn_func_data_to_csv(sol_params_dn, params_conc_sol[:4])
    sol_dn_func_fixed_m(sol_params_dn, params_conc_sol)
    sol_dn_func_fixed_dn(sol_params_dn, params_conc_sol)
    an_m_func_contour(params_an, params_conc_sol)
    salt_dn_func_fixed_c(salt_params_dn)
    an_func_fixed_m(params_an, params_conc_sol)
    an_func_fixed_an(params_an, params_conc_sol)
    sol_an_func_data_to_csv(params_an, params_conc_sol)
    salt_dn_func(salt_params_dn)
    sol_sol_func_contour(params_sol, params_conc_sol)
    sol_sol_func_save_data(params_sol, params_conc_sol)
    sol_sol_func_fixed_m_an(params_sol, params_conc_sol)
    sol_sol_func_fixed_m_dn(params_sol, params_conc_sol)
    frac_occupation()
    visualize_li_free_energy(trained_params)
    sol_dn_func_no_conc_factor(sol_params_dn)
    sol_an_func_no_conc_factor(params_an)
    visualize_conc_factor_x(params_conc_sol)
    visualize_conc_factor_V(params_conc_sol)