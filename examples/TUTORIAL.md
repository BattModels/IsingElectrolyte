# IsingLHCE Package Tutorial

This document is a guided tour of the `IsingLHCE` package for someone who knows the old
`fit_model.py` well. It explains what each part of the package does, how it maps to the
old monolithic script, and when to use the legacy vs. new interface.

---

## Package Layout

```
src/IsingLHCE/
├── interactions.py        # primitive math functions + rescale companions
├── conc_vol_correction.py # concentration/volume correction (conc_factor)
├── model.py               # Ising physics: energetics, equations, find_root, free energy
├── train.py               # training loop: objective, update, train, parity_results
└── analysis/              # plotting helpers (h-terms, J-terms, free energy, conc-vol viz)

examples/
├── train_base_model.py    # legacy 2-sol+1-anion replication (params_salt interface)
├── train_lhce.py          # new generic interface with YAML config
├── config.yaml            # YAML config for train_lhce.py
└── TUTORIAL.md            # this file
```

---

## 1. Primitive Functions — `interactions.py`

In the old `fit_model.py`, helper functions like `expfunc`, `logfunc`, `sol_sol_func` were
defined at the top of the file and called directly inside `energetics()`. In the new package
these live in `interactions.py` and are imported where needed.

| Old `fit_model.py` | `interactions.py` | Notes |
|--------------------|-------------------|-------|
| `expfunc(x, params)` | `expfunc(x, params)` | identical |
| `logfunc(x, params)` | `logfunc(x, params)` | identical |
| `sol_sol_func(X, params)` | `sol_sol_func(X, params)` | identical |
| `conc_factor(x0, x_ref, V0, V_ref, params)` | `conc_factor_sep_x_v_sigmoid(...)` in `conc_vol_correction.py` | same math, renamed |

On top of the primitives, `interactions.py` also defines **default term functions** that
wrap the primitives into the standard h / J physics:

```python
default_h_sol(props, params)       # h(Li–solvent): expfunc(dn_eff) + logfunc(x)
default_h_anion(props, params)     # h(Li–anion):   expfunc(dn_an) + logfunc(x)
default_J_sol_sol(props_i, props_j, params)    # 16-param solvent-solvent coupling
default_J_sol_anion(props_sol, props_anion, params)  # 6-param solvent-anion coupling
default_J_anion_anion(props_i, props_j, params)      # 6-param anion-anion coupling
```

Each of these has a **rescale companion** (`default_h_sol_rescale`, etc.) that applies
signed softplus to enforce monotonicity constraints — more on that in §4.

---

## 2. Physics Core — `model.py`

`model.py` contains the mean-field Ising solver. It has **two parallel interfaces**:

### 2a. Legacy interface (`_old` suffix) — exact replication of `fit_model.py`

These functions use the same parameter keys, functional forms, and hardcoded monotonicity
constraints as the old script. Use them whenever you need bit-for-bit reproducibility.

| Old `fit_model.py` | `model.py` | Param key differences |
|--------------------|------------|----------------------|
| `rescale_input_params(params)` | `rescale_input_params_old(params)` | uses `params_salt` (5 elem) |
| `energetics(vars, params, constants)` | `energetics_old(vars, params, constants)` | j22 uses `expfunc` (old) |
| `equations(vars, params, constants)` | `equations_old(vars, params, constants)` | identical logic |
| `find_root(params, constants, ...)` | `find_root_old(params, constants, ...)` | identical logic |
| `get_root_error(m,n,l, params, constants)` | `get_root_error_old(...)` | identical |
| `li_free_energy(params, constants)` | `li_free_energy_old(params, constants)` | identical |

**Old parameter dict keys** (used by `_old` functions):
```python
{
    "sol_params_dn":      # shape (5,)  — h(Li–solvent)
    "salt_params_dn":     # shape (5,)  — h(Li–anion)
    "params_sol_salt_an": # shape (6,)  — J(solvent–anion)
    "params_sol_sol":     # shape (16,) — J(solvent–solvent)
    "params_salt":        # shape (5,)  — J(anion–anion), uses expfunc  ← old key
    "conc_factor_sol":    # shape (8,)  — concentration/volume factor
}
```

The `input_constants` argument is a flat 12-element list/array (same order as in the old
`energetics` function):
```
[dn0, dn1, dn_anion, x0, x1, an0, an1, x_anion,
 solvent_volume, diluent_volume, anion_volume, z]
```

### 2b. New generic interface

The new interface supports **N solvents + M anions** and uses injectable h/J functions.
Species are passed as nested dicts instead of flat tuples:

```python
solvents = {
    "EC":  {"dn": 16.4, "an": 18.4, "x": 0.6, "volume": 66.0},
    "DMC": {"dn": 15.1, "an": 16.0, "x": 0.3, "volume": 84.8},
}
anions = {
    "LiFSI": {"dn": 5.0, "x": 0.1, "volume": 100.0},
}
occupations, found_valid = find_root(params, solvents, anions, z=1.86)
```

**New parameter dict keys** (used by the generic interface):
```python
{
    "sol_params_dn":      # shape (5,)  — h(Li–solvent)
    "salt_params_dn":     # shape (5,)  — h(Li–anion)
    "params_sol_salt_an": # shape (6,)  — J(solvent–anion)
    "params_sol_sol":     # shape (16,) — J(solvent–solvent)
    "params_anion_anion": # shape (6,)  — J(anion–anion), uses sol_sol_func  ← new key
    "conc_factor_sol":    # shape (8,)  — concentration/volume factor
}
```

> **Key difference:** `params_salt` (old, 5 elem, `expfunc` functional form for j22) was
> replaced by `params_anion_anion` (new, 6 elem, `sol_sol_func` for j22). The two have
> different functional forms, so an old checkpoint cannot be dropped into the new
> interface *unchanged*.
>
> **However — with a single anion (M=1) old checkpoints ARE usable without re-training.**
> The anion-anion term then only ever appears as the self-interaction `j22`, so injecting
> the old functional form reproduces it exactly (verified: `h` matches `energetics_old`
> to 0.0). This is how the HCE and HEE studies reuse the trial-25 models:
>
> ```python
> from IsingLHCE.interactions import expfunc, logfunc, _apply_softplus
>
> def old_J_anion_anion(props_i, props_j, p):         # legacy j22
>     return expfunc(props_i["dn"], p[:4]) + logfunc(props_i["x"], p[4])
>
> def old_J_anion_anion_rescale(p, monotonicity):
>     return _apply_softplus(p, jnp.array([0, +1, +1, +1, -1]))
>
> params["params_anion_anion"] = params["params_salt"]   # 5-elem; the func uses [:5]
> occ, ok = find_root(params, solvents, anions, z=3.72/2.0,
>                     J_anion_anion_func=old_J_anion_anion,
>                     rescale_J_anion_anion=old_J_anion_anion_rescale,
>                     monotonicity_dict=LEGACY_MONOTONICITY, groups=...)
> ```
>
> With **two or more anions** the forms genuinely diverge and re-training is required.
>
> The legacy monotonicity constraints map onto the new presets exactly:
> `sol_params_dn`, `salt_params_dn`, `params_sol_salt_an`, `params_sol_sol` → `"decrease"`;
> `conc_factor_sol` → `"increase"`; `params_anion_anion` uses the custom rescale above.

### 2c. Species groups (`groups`) — required reading for N >= 3 solvents

`sol_sol_func` weights its two inputs differently (`a1 != a2`), so the raw
solvent-solvent block is **not symmetric** — invalid for a pair coupling. The generic
interface repairs this at assembly time using a `groups` argument.

`groups` is an integer rank per species, length `N+M`, **solvents first then anions**:

- **same rank** = interchangeable partners → their coupling is averaged over both
  orderings (symmetric *and* independent of how you list them);
- **different rank** = distinct roles → the lower-ranked species takes the first slot
  (symmetric, and follows the role rather than the list position).

```python
find_root(params, solvents, anions, z=1.86, groups=(0, 1, 2))          # LHCE: solvent, diluent, anion
find_root(params, solvents, anions, z=1.86, groups=(0, 0, 0, 0, 0, 1)) # HEE: 5 interchangeable solvents + salt
```

`groups=None` (the default) means all-distinct, which reproduces the legacy
`fit_model.py` ordering exactly — so **existing results do not shift unless you opt in.**
Pass `pair_symmetry="none"` to restore the pre-fix behaviour for comparison.

Rules of thumb:

- **N=1** (e.g. HCE) — `groups` is irrelevant; there are no off-diagonal solvent pairs.
- **N=2 with real roles** (LHCE solvent/diluent) — use all-distinct; it is bit-exact
  against the fitted model.
- **N>=3 interchangeable** (HEE) — you *must* pass a shared group, or the answer will
  depend on the arbitrary order you listed the solvents in.

See `docs/asymmetry-fix.md` for the derivation, measured magnitudes, and the open
question about how groups should be declared for additive formulations.

---

## 3. Training Loop — `train.py`

The training loop follows the same structure as the old `fit_model.py` but is written against
the new generic interface. It is **not** used by `train_base_model.py` (which re-implements
the loop locally against the `_old` physics).

| Old `fit_model.py` | `train.py` | Notes |
|--------------------|------------|-------|
| `initialize_params(mode, file_path, seed)` | `initialize_params(mode, file_path, seed, param_sizes)` | new key `params_anion_anion`; rejects old checkpoints with `params_salt` |
| `objective_single(params, constants, m, n, l, guess)` | `objective_single(params, sol_props, anion_props, z, targets, guess, ...)` | new species-dict interface |
| `total_objective(params, data)` | `total_objective(params, data, monotonicity_dict, ...)` | `data` uses `sol_props`/`anion_props` keys |
| `update(params, opt_state, data)` | `update(params, opt_state, data, monotonicity_dict, ...)` | module-level `optimizer` global |
| `train(params, epochs, ...)` | `train(params, epochs, ..., monotonicity_dict, ...)` | same early-stopping logic |
| `parity_results(data_dict, params)` | `parity_results(data_dict, params, monotonicity_dict, ...)` | same CSV+plot outputs |

The `data` dict expected by the new `train.py` functions:
```python
data = {
    "sol_props":   {"dn": ..., "an": ..., "x": ..., "v": ...},  # each shape (n, N_sol)
    "anion_props": {"dn": ..., "x": ..., "v": ...},              # each shape (n, N_an)
    "z":           ...,   # shape (n,)
    "targets":     ...,   # shape (n, N_sol + N_an)
    "init_guess":  ...,   # shape (n, N_sol + N_an)
}
```

---

## 4. Monotonicity / Rescaling

In the old `fit_model.py`, parameter signs were enforced by `rescale_input_params()`, which
applied hardcoded `jnn.softplus` calls at fixed indices. This is replaced by a flexible
system in the new package.

### How it works

Each term function has a **rescale companion** in `interactions.py`:
```python
default_h_sol_rescale(params, monotonicity)
default_h_anion_rescale(params, monotonicity)
default_J_sol_sol_rescale(params, monotonicity)
default_J_sol_anion_rescale(params, monotonicity)
default_J_anion_anion_rescale(params, monotonicity)
default_conc_factor_rescale(params, monotonicity)
```

`monotonicity` is a string: `"decrease"`, `"increase"`, or `"none"`.

A **`monotonicity_dict`** controls the constraint for each parameter group:
```python
monotonicity_dict = {
    "sol_params_dn":      "decrease",  # h(Li–sol) strictly decreasing with DN
    "salt_params_dn":     "decrease",
    "params_sol_salt_an": "decrease",
    "params_sol_sol":     "decrease",
    "params_anion_anion": "increase",
    "conc_factor_sol":    "increase",
}
```

The default (`DEFAULT_MONOTONICITY` in `model.py`) is **all `"none"`** — no constraints
applied. This is the correct default when starting from scratch or when constraints are not
needed. The old `rescale_input_params_old()` is equivalent to using the `"decrease"` /
`"increase"` pattern above but with the constraints baked in.

### `_old` vs new comparison

```python
# Old — hardcoded constraints, always applied
input_params = rescale_input_params_old(input_params)

# New — injectable, off by default
input_params = rescale_input_params(input_params, monotonicity_dict={
    "sol_params_dn": "decrease", ...
})
```

---

## 5. Which Script to Use

| Goal | Use |
|------|-----|
| Reproduce old `fit_model.py` results exactly | `examples/train_base_model.py` |
| Train with new generic interface (YAML config) | `examples/train_lhce.py` + `config.yaml` |
| Use physics in a notebook (old interface) | `from IsingLHCE.model import find_root_old, li_free_energy_old` |
| Use physics in a notebook (new interface) | `from IsingLHCE.model import find_root, li_free_energy` |
| Extend to 3 solvents or 2 anions | New interface only (`find_root`, `energetics`) |

---

## 6. `train_base_model.py` Walkthrough

This script is a clean rewrite of `fit_model.py` using the package's `_old` backend.
The physics are **identical** to the old script — only the boilerplate is reorganised.

### `initialize_params_old(mode, file_path, random_seed)`

Same hand-tuned starting values as old `fit_model.py`, same `params_salt` key (5 elem),
same 5% multiplicative noise. Two modes:
- `"from_scratch"` — use hard-coded starting values
- `"from_file"` — load a `.pkl` checkpoint and add noise

### `total_objective_old(params, data)` and `update_old(params, opt_state, data)`

These wrap `find_root_old` from the package. The `data` dict uses the same flat keys as the
old script (`dn0`, `dn1`, `x0`, `x1`, etc.). The module-level `optimizer` variable must be
set before calling `update_old`, just as in the old code.

### `train_old(params, num_epochs, ...)`

Verbatim replication of the old `train()` logic:
- Full-batch training (batch_size = n_train)
- Early stopping after 100 epochs without validation improvement
- Stops early if both train and val loss drop below 0.04
- Returns best params (by validation loss) and loss logs

### `parity_results_old(data_dict, input_params)`

Iterates over each split, calls `find_root_old` per row, writes:
- `ising_results_{split}.csv` — predicted vs target fractional CN
- `ising_{split}.png` — parity plot (solvent / diluent / anion)
- `free_solvent_{split}.png` — free solvent fraction parity plot

### `__main__` block

Reads `train.csv`, `test.csv`, `val.csv` from the **current working directory**. Runs up to
10 random-seed trials; keeps the parameters that achieve the best validation loss across
trials. After training, saves:
- `trained_params.pkl`
- `train_val_loss_log.csv`
- `train_val_loss.png`
- `time_log.txt`

---

## 7. Quick-Start Recipes

### Reproduce a trial-25 run

```bash
cd /path/to/trial-25/base-model
python /path/to/IsingLHCE/examples/train_base_model.py
```

### Load a trained checkpoint and inspect predictions

```python
import pickle
import jax.numpy as jnp
from IsingLHCE.model import find_root_old, li_free_energy_old

with open("trained_params.pkl", "rb") as f:
    params = pickle.load(f)

# 2-solvent + 1-anion example
input_constants = [
    16.4, 5.0, 5.0,       # dn0, dn1, dn_anion
    0.6, 0.3,              # x0, x1
    18.4, 16.0,            # an0, an1
    0.1,                   # x_anion
    66.0, 84.8, 100.0,     # solvent_volume, diluent_volume, anion_volume
    3.72 / 2.0,            # z
]
(m, n, l), found = find_root_old(params, input_constants)
print(f"Solvent: {m:.3f}, Diluent: {n:.3f}, Anion: {l:.3f}, Valid: {found}")
```

### Use the new generic interface

```python
from IsingLHCE.model import find_root
from IsingLHCE.train import initialize_params

params = initialize_params(mode="from_scratch", random_seed=42)

solvents = {
    "EC":  {"dn": 16.4, "an": 18.4, "x": 0.6, "volume": 66.0},
    "DMC": {"dn": 5.0,  "an": 16.0, "x": 0.3, "volume": 84.8},
}
anions = {
    "LiFSI": {"dn": 5.0, "x": 0.1, "volume": 100.0},
}
occupations, found = find_root(params, solvents, anions, z=1.86)
# occupations: jnp.array([m_EC, m_DMC, l_LiFSI])
```

### Run training with the new interface (YAML config)

```bash
# edit examples/config.yaml to point to your data files, then:
python examples/train_lhce.py --config examples/config.yaml

# or override individual settings from the CLI:
python examples/train_lhce.py --config examples/config.yaml --epochs 2000 --learning_rate 0.001
```

---

## 8. Dependency Injection (advanced)

The new interface lets you swap out any h or J function without touching the package source.
This is useful for experimenting with different functional forms.

```python
import jax.numpy as jnp
from IsingLHCE.model import find_root

def my_h_sol(props, params):
    """Custom field term — linear in DN."""
    dn_eff = props["dn"] * props["x"]
    a0, a1 = params[:2]
    return a0 + a1 * dn_eff

# Pass your function; the package won't touch it — JAX resolves it before JIT
occupations, found = find_root(
    params, solvents, anions, z=1.86,
    h_sol_func=my_h_sol,
)
```

The resolver helper in `train_lhce.py` (`_resolve_func`) also accepts **strings** that map
to function names in `interactions.py`, so you can reference custom functions from YAML
configs once they are added to `interactions.py`.
