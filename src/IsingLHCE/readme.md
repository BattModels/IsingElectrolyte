# Structure of IsingLHCE code
## interactions.py
Includes helper functions for defining interaction terms.
- mlp: Multi-layer perceptron for modeling interactions.
- langmuirfunc: Langmuir isotherm function for modeling interactions.
- linearfunc: Linear function for modeling interactions.
- expfunc: Sigmoid-based function for modeling interactions. Default for modeling Li-sol and Li-anion enthalpic interactions.
- logfunc: Log function for modeling interactions, default for modeling Li-sol and Li-anion entropic interactions.
- polynomialfunc: Polynomial function for modeling interactions.
## conc_vol_correction.py
Includes helper functions for correcting for concentration and volume effects in the model.
- conc_factor_sep_x_v_sigmoid: Sigmoid-based function, multiplication of two sigmoids, one for concentration and one for volume.
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
## train.py
Includes training functions for the IsingLHCE model.
<!-- - dn_loss: Loss function for training the model, using root mean squared error between predicted and true Li+ coordination numbers. -->
- objective_single: objective function for single data point, using root mean squared error between predicted and true Li+ coordination numbers. returns both the loss and new initial guess for the next solving step.
- total_objective: jnp vectorized version of objective_single, for training on multiple data points.
- initialize_params: initializes the parameters of the model, including interaction parameters and concentration/volume correction parameters. Two modes: "from_scratch", which initializes parameters from pre-defined ranges, and "from_file", which initializes parameters from a saved pkl file.
- update: update function for training
- train: main training loop for the model, which iteratively updates the parameters using the objective function and a specified optimizer.
- parity_results: function for calculating parity results, which compares predicted and true Li+ coordination numbers and returns a dataframe with true values and predicted values.
## analysis.py
Includes helper functions for post-processing the results of the model.
- sol_dn_func_contour: function for plotting h(Li-sol) as a function of solvent DN and concentration, for a given set of parameters.
- sol_dn_func_data_to_csv: function for calculating h(Li-sol) for a given set of parameters and saving the results to a csv file.
- sol_dn_func_fixed_m: function for plotting h(Li-sol) as a function of solvent DN for a fixed concentration, for a given set of parameters.
- sol_dn_func_no_conc_factor: function for plotting h(Li-sol) as a function of solvent DN without concentration correction, for a given set of parameters.
- sol_an_func_no_conc_factor: function for plotting J(sol-anion) as a function of solvent AN without concentration correction, for a given set of parameters.
- solv_dn_func_fixed_dn: function for plotting h(Li-sol) as a function of concentration for a fixed solvent DN, for a given set of parameters.
- an_m_func_contour: function for plotting J(sol-anion) as a function of solvent AN and concentration, for a given set of parameters.
- sol_an_func_data_to_csv: function for calculating J(sol-anion) for a given set of parameters and saving the results to a csv file.
- salt_dn_func_fixed_c: function for plotting h(Li-anion) as a function of salt DN for a fixed concentration, for a given set of parameters.
- an_func_fixed_m: function for plotting J(sol-anion) as a function of solvent AN for a fixed concentration, for a given set of parameters.
- an_func_fixed_an: function for plotting J(sol-anion) as a function of concentration for a fixed solvent AN, for a given set of parameters.
- salt_dn_func: contour plot of h(Li-anion) as a function of salt DN and concentration, for a given set of parameters.
- sol_sol_func_dn_an_contour: function for plotting J(sol-sol) as a function of solvent DN and AN, with each subfigure representing a different concentration, for a given set of parameters.
- sol_sol_func_dn_dn_contour: function for plotting J(sol-sol) as a function of solvent 1 DN and solvent 2 DN, with each subfigure representing a different concentration, for a given set of parameters.
- sol_sol_func_an_an_contour: function for plotting J(sol-sol) as a function of solvent 1 AN and solvent 2 AN, with each subfigure representing a different concentration, for a given set of parameters.
- sol_sol_func_save_data: function for calculating J(sol-sol) for a given set of parameters and saving the results to a csv file.
- sol_sol_func_fixed_m_an: function for plotting J(sol-sol) as a function of solvent Dn, with solvent AN and concentration fixed, for a given set of parameters.
- sol_sol_func_fixed_m_dn: function for plotting J(sol-sol) as a function of solvent AN, with solvent DN and concentration fixed, for a given set of parameters.
- frac_occupation: diluent fractional occupation as a function of solvent and diluent DN, with solvent:diluent:anion:Li=24:48:14:14, for a given set of parameters.
- occupations_dn2_an2_contour: contour plot of solvation shell occupation as a function of solvent 2 (diluent) DN and AN, each subfigure representing solvent, diluent and anion.
- occupations_dn2_an2_csv: calculates solvation shell occupation as a function of solvent 2 (diluent) DN and AN, and saves the results to a csv file.
- visualize_li_free_energy_contour: contour plot of Li solvation energetics as a function of solvent 2 (diluent) DN and AN.
- salt_salt_func: contour plot of J(salt-salt) as a function of salt DN and concentration, for a given set of parameters.
- salt_salt_func_fixed_c: function for plotting J(salt-salt) as a function of salt DN for a fixed concentration, for a given set of parameters.
- salt_salt_func_fixed_dn: function for plotting J(salt-salt) as a function of concentration for a fixed salt DN, for a given set of parameters.
- visualize_conc_factor_x: plot conentration-volume correction scheme as a function of concentration, for a given set of parameters.
- visualize_conc_factor_V: plot conentration-volume correction scheme as a function of volume
- visualize_conc_factor_x_V: plot conentration-volume correction scheme as a function of concentration and volume, for a given set of parameters.

## Code Structure
```
src/IsingLHCE/
├── __init__.py
├── interactions.py
├── conc_vol_correction.py
├── model.py
├── train.py
├── analysis/
│   ├── __init__.py
│   ├── h_terms.py
│   ├── j_terms.py
│   ├── free_energy.py
│   └── conc_vol_viz.py
├── readme.md
└── ref_params.pkl
```
