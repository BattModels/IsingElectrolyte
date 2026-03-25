# IsingLHCE Overview
This is the code base for the IsingLHCE model, which uses a mean-field Ising model to rapidly predict the solvation structure of Li+ in localized high concentration electrolytes (LHCEs). The model takes as input the molecular properties of the solvent, diluent and salt, including donor number (DN), acceptor number (AN), concentration and volume, and predicts the average solvent, diluent and anion coordination numbers of Li+ in the system. The model is built using JAX to allow for end-to-end differentiability and efficient parameterization directly from MD or experimental solvation structure data. More details of the model setup can be found in this paper.

The code is organized into several modules, including interactions.py for defining interaction terms, conc_vol_correction.py for correcting for concentration and volume effects, model.py for building the IsingLHCE model and calculating energetics, train.py for training the model, and analysis/ subpackage for post-processing the results of the model.

# Code Architecture
## interactions.py
Includes helper functions for defining interaction terms.
- mlp: Multi-layer perceptron for modeling interactions.
- langmuirfunc: Langmuir isotherm function for modeling interactions.
- linearfunc: Linear function for modeling interactions.
- expfunc: Sigmoid-based function for modeling interactions. Default for modeling Li-sol and Li-anion enthalpic interactions.
- logfunc: Log function for modeling interactions, default for modeling Li-sol and Li-anion entropic interactions.
- polynomialfunc: Polynomial function for modeling interactions.
- sol_sol_func: Pairwise interaction function for solvent-solvent interactions.
## conc_vol_correction.py
Includes helper functions for correcting for concentration and volume effects in the model.
- conc_factor_sep_x_v_sigmoid: Sigmoid-based function, multiplication of two sigmoids, one for concentration and one for volume. This is the default correction used in the model.
- logfunc_conc_factor: Log-based concentration correction function.
- conc_factor_sep_x_v_taylor: Taylor expansion-based function, multiplication of two Taylor expansions, one for concentration and one for volume.
- conc_factor_power: Power function, multiplication of two power functions, one for concentration and one for volume.
- conc_factor_wrapped_sigmoid: Sigmoid-based function, lumps concentration and volume into a single sigmoid function.
## model.py
Includes the building of the IsingLHCE model.
- rescale_input_params: Rescales input parameters to ensure monotonicity, optional.
- energetics: Builds the IsingLHCE model and calculates the energetics of the system. This function is organized as follows:
    - unwrap input parameters.
    - unwrap input constants, including molecular properties (DN, AN, concentration, volume) and coordination number of Li.
    - calculate effective DN, AN by incorporating concentration and volume effects using the functions in conc_vol_correction.py.
    - define solvent, diluent and anion average coordination as vars.
    - calculate the Li-sol and Li-anion interactions and save as h terms.
    - calculate the solvent-solvent, solvent-diluent, diluent-diluent, solvent-anion, diluent-anion and anion-anion interactions and save as J terms.
- equations: computes the partition function and formulates the problem as a root-finding problem.
- find_root: uses a Broyden solver to find the root of the equations.
- get_root_error: calculates how far the root found by the Broyden solver is by calculating f(n) - n vs 0.
- li_free_energy: computes the Li solvation free energy for a given set of input parameters and constants.
## ref_params.pkl
Default parameter file used by initialize_params in "from_file" mode.

## train.py
Includes training functions for the IsingLHCE model.
- objective_single: objective function for single data point, using root mean squared error between predicted and true Li+ coordination numbers. returns both the loss and new initial guess for the next solving step.
- total_objective: jnp vectorized version of objective_single, for training on multiple data points.
- initialize_params: initializes the parameters of the model, including interaction parameters and concentration/volume correction parameters. Two modes: "from_scratch", which initializes parameters from pre-defined ranges, and "from_file", which initializes parameters from a saved pkl file.
- update: update function for training
- train: main training loop for the model, which iteratively updates the parameters using the objective function and a specified optimizer.
- parity_results: function for calculating parity results, which compares predicted and true Li+ coordination numbers and returns a dataframe with true values and predicted values.
## analysis/
Subpackage for post-processing and visualizing model results. Organized by physical quantity.

### h_terms.py — $Li^+$-interactions visualizations
#### h(Li-sol)
- sol_dn_func_contour: contour plot of h(Li-sol) as a function of solvent DN and concentration.
- sol_dn_func_data_to_csv: calculates h(Li-sol) and saves to csv.
- sol_dn_func_fixed_m: h(Li-sol) vs. solvent DN for a fixed concentration.
- sol_dn_func_no_conc_factor: h(Li-sol) vs. solvent DN without concentration correction.
- sol_dn_func_fixed_dn: h(Li-sol) vs. concentration for a fixed solvent DN.
#### h(Li-anion)
- salt_dn_func: contour plot of h(Li-anion) as a function of salt DN and concentration.
- salt_dn_func_fixed_c: h(Li-anion) vs. salt DN for a fixed concentration.

### j_terms.py — Non-Li interactions visualizations
#### J(sol-anion)
- sol_an_func_no_conc_factor: J(sol-anion) vs. solvent AN without concentration correction.
- an_m_func_contour: contour plot of J(sol-anion) as a function of solvent AN and concentration.
- sol_an_func_data_to_csv: calculates J(sol-anion) and saves to csv.
- an_func_fixed_m: J(sol-anion) vs. solvent AN for a fixed concentration.
- an_func_fixed_an: J(sol-anion) vs. concentration for a fixed solvent AN.
#### J(sol-sol)
- sol_sol_func_full_contour: total J(sol-sol) (sum of DN-AN cross, DN-DN self, AN-AN self, and entropic terms) vs. solvent DN and AN, subfigures by concentration.
- sol_sol_func_dn_an_contour: J(sol-sol) DN-AN cross interaction sub-term vs. solvent DN and AN, subfigures by concentration.
- sol_sol_func_dn_dn_contour: J(sol-sol) DN-DN self interaction sub-term vs. solvent 1 DN and solvent 2 DN, subfigures by concentration.
- sol_sol_func_an_an_contour: J(sol-sol) AN-AN self interaction sub-term vs. solvent 1 AN and solvent 2 AN, subfigures by concentration.
- sol_sol_func_save_data: calculates J(sol-sol) and saves to csv.
- sol_sol_func_fixed_m_an: J(sol-sol) vs. solvent DN, with AN and concentration fixed.
- sol_sol_func_fixed_m_dn: J(sol-sol) vs. solvent AN, with DN and concentration fixed.
#### J(salt-salt)
- salt_salt_func: contour plot of J(salt-salt) as a function of salt DN and concentration.
- salt_salt_func_fixed_c: J(salt-salt) vs. salt DN for a fixed concentration.
- salt_salt_func_fixed_dn: J(salt-salt) vs. concentration for a fixed salt DN.

### free_energy.py — solvation shell occupation and energetics
- frac_occupation: diluent fractional occupation as a function of solvent and diluent DN (solvent:diluent:anion:Li=24:48:14:14).
- occupations_dn2_an2_contour: contour plot of solvation shell occupation vs. diluent DN and AN, subfigures for solvent, diluent and anion.
- occupations_dn2_an2_csv: calculates solvation shell occupation vs. diluent DN and AN, saves to csv.
- visualize_li_free_energy_contour: contour plot of Li solvation free energy vs. solvent 1 DN and solvent 2 (diluent) DN, with AN fixed.
- visualize_li_free_energy_dn_an_contour: contour plot of Li solvation free energy vs. diluent DN and diluent AN, with solvent DN and AN fixed.
- visualize_li_free_energy_dn_an: line plot of Li h-term free energy (h·m + h·n + h·l at equilibrium) vs. diluent DN, with separate curves for different diluent AN values, solvent DN and AN fixed.

### conc_vol_viz.py — concentration-volume correction visualization
- visualize_conc_factor_x: correction factor vs. concentration.
- visualize_conc_factor_V: correction factor vs. volume.
- visualize_conc_factor_x_V: correction factor vs. concentration and volume (contour).
