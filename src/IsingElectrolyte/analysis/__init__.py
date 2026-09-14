import os

import pandas as pd
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

rcParams.update(plot_settings)

# DN, AN (Gutmann, kcal/mol) and DFT molecular volume (A^3) for the solvents and
# anions used in the paper, keyed by molecule name.
_DATA_DIR = os.path.dirname(__file__)
solvent_map_dict = (
    pd.read_csv(os.path.join(_DATA_DIR, "solvent_map_dict.csv"))
    .set_index("molecule")
    .to_dict(orient="index")
)

from .h_terms import (
    sol_dn_func_contour,
    sol_dn_func_data_to_csv,
    sol_dn_func_fixed_m,
    sol_dn_func_no_conc_factor,
    sol_dn_func_fixed_dn,
    salt_dn_func_fixed_c,
    salt_dn_func,
)
from .j_terms import (
    sol_an_func_no_conc_factor,
    an_m_func_contour,
    sol_an_func_data_to_csv,
    an_func_fixed_m,
    an_func_fixed_an,
    sol_sol_func_full_contour,
    sol_sol_func_dn_an_contour,
    sol_sol_func_dn_dn_contour,
    sol_sol_func_an_an_contour,
    sol_sol_func_save_data,
    sol_sol_func_fixed_m_an,
    sol_sol_func_fixed_m_dn,
    salt_salt_func,
    salt_salt_func_fixed_c,
    salt_salt_func_fixed_dn,
)
from .free_energy import (
    frac_occupation,
    occupations_dn2_an2_csv,
    occupations_dn2_an2_contour,
    visualize_li_free_energy_contour,
    visualize_li_free_energy_dn_an_contour,
)
from .conc_vol_viz import (
    visualize_conc_factor_x,
    visualize_conc_factor_V,
    visualize_conc_factor_x_V,
)
