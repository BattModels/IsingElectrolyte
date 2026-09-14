# IsingElectrolyte tutorial

This tutorial goes one level deeper than the [README](../README.md). It walks from the
physics of the model, through prediction and screening, to training your own parameters
and writing your own interaction terms. No prior knowledge of the code is assumed.

Every Python block runs as-is from the repository root, and the blocks build on each other,
so run them in order (e.g. paste them into one Jupyter notebook). `tests/test_tutorial.py`
executes them to keep this document in sync with the code.

**Contents**

1. [The model in one page](#1-the-model-in-one-page)
2. [Predicting with the paper model](#2-predicting-with-the-paper-model)
3. [Sweeps and screening](#3-sweeps-and-screening)
4. [More than two solvents or one salt](#4-more-than-two-solvents-or-one-salt)
5. [Training on your own data](#5-training-on-your-own-data)
6. [Custom interaction terms](#6-custom-interaction-terms)
7. [Inspecting and plotting the interaction terms](#7-inspecting-and-plotting-the-interaction-terms)
8. [Reproducing the paper exactly (legacy interface)](#8-reproducing-the-paper-exactly-legacy-interface)

---

## 1. The model in one page

The first solvation shell of a Li⁺ ion is treated as a small lattice of sites. Each site is
occupied by exactly one species: one of the N solvents or one of the M anions. The quantity
the model predicts is the **occupation** ⟨σᵢ⟩ of each species *i*, i.e. the fraction of the
shell it occupies (the fractional coordination number). Occupations sum to 1.

Two kinds of energy (in eV) decide the occupations:

- **h — Li⁺–species interaction.** How strongly species *i* binds to Li⁺. For solvents it
  is a sigmoid of the donor number (DN) plus a log term in the mole fraction; for anions the
  same form in the anion DN.
- **J — species–species interaction.** The interaction between two species sharing (or
  neighboring) a shell. Solvent–solvent terms depend on DN and acceptor number (AN) of both
  partners; solvent–anion terms on anion DN and solvent AN; anion–anion terms on anion DN.

Solvent DN and AN are not used raw: they are multiplied by a **concentration–volume
correction**, a product of two sigmoids in (x_solvent / x_anion) and (V_anion / V_solvent).
This lets the same parameters describe dilute and concentrated electrolytes and small and
large molecules.

The **mean-field approximation** replaces each neighbor by its average occupation. The
energy of placing species *i* on a site is then

  Eᵢ = hᵢ + (z/2)·(Σⱼ Jᵢⱼ⟨σⱼ⟩ + Jᵢᵢ⟨σᵢ⟩),

and the occupations must be self-consistent with a Boltzmann distribution,

  ⟨σᵢ⟩ = exp(−Eᵢ/kT) / Σₖ exp(−Eₖ/kT),  kT = 0.0257 eV.

`find_root` solves these equations with a Broyden solver, restarting from several initial
guesses until it finds a solution with every occupation in [0, 1].

**z** is the mean-field coordination parameter. The paper uses z = N/2 = 1.86, where
N = 3.72 is the average Li⁺ coordination number from MD (`PAPER_Z`).

The model is written in JAX, so everything above is differentiable with respect to the
parameters. That is what makes fitting to MD solvation structures a gradient-descent problem.
The full derivation is in the paper (arXiv:2609.05671) and its Supporting Information.

---

## 2. Predicting with the paper model

### Inputs

Each species is a small dict:

| Key | Meaning | Units |
|---|---|---|
| `dn` | Gutmann donor number | kcal/mol |
| `an` | Gutmann acceptor number (solvents only) | — |
| `x` | mole fraction | — |
| `volume` | molecular volume (the paper uses DFT volumes) | Å³ per molecule |

**Mole fractions count Li⁺ as its own species.** For a salt LiA dissolved in solvents,
x_solvent,1 + … + x_anion + x_Li = 1 with x_Li = x_anion. A helper that converts a
formulation (salt molality and solvent molar ratios) to these mole fractions:

```python
import numpy as np
import jax.numpy as jnp

np.set_printoptions(precision=3, suppress=True)


def mole_fractions(molality, ratios, masses):
    """Mole fractions with Li+ counted.

    molality: mol of salt per kg of all solvents together
    ratios:   molar ratio of the solvents, e.g. [1, 2]
    masses:   molar masses of the solvents in g/mol
    Returns (x_solvents, x_anion).
    """
    n = np.asarray(ratios, dtype=float)
    n_salt = molality * float(n @ np.asarray(masses, dtype=float)) / 1000.0
    total = 2.0 * n_salt + n.sum()  # salt contributes Li+ and the anion
    return n / total, n_salt / total


x_sol, x_an = mole_fractions(1.0, ratios=[1, 2], masses=[90.12, 232.07])  # 1.0 m LiTFSI in DME:TTE 1:2
print(x_sol, x_an)
```

`src/IsingElectrolyte/analysis/solvent_map_dict.csv` lists DN, AN, DFT volume and molar mass
for the molecules used in the paper. Several diluents (e.g. TTE, BTFE) have no measured AN;
the paper's analyses used AN = 20 for those, and so does this tutorial.

### A single prediction

```python
from IsingElectrolyte.model import find_root, li_free_energy
from IsingElectrolyte.pretrained import PAPER_Z, load_paper_model

params, model_kwargs = load_paper_model(fold=0)

solvents = {
    "DME": {"dn": 20.0, "an": 10.2, "x": x_sol[0], "volume": 135.53},
    "TTE": {"dn": 1.9, "an": 20.0, "x": x_sol[1], "volume": 187.76},
}
anions = {"TFSI": {"dn": 11.2, "x": x_an, "volume": 209.76}}

occupations, found_valid = find_root(params, solvents, anions, PAPER_Z, **model_kwargs)
assert found_valid, "no physical solution; do not use the occupations"
print(dict(zip([*solvents, *anions], np.asarray(occupations).round(3).tolist())))
```

This prints roughly `{'DME': 0.389, 'TTE': 0.0, 'TFSI': 0.611}`: an anion-rich shell, with
the diluent excluded, which is what makes this formulation an LHCE.

Three rules:

- **Always pass `**model_kwargs`.** The paper model was trained with a specific anion–anion
  term and monotonicity constraints. `load_paper_model` returns them; without them the same
  parameters give wrong numbers.
- **Always check `found_valid`.** If the solver finds no physical solution the occupations
  are NaN, and NaN propagates silently into anything computed from them.
- Occupations come back in order: solvents in dict order, then anions.

### Free solvent and Li⁺ solvation energy

Low free-solvent content matters for oxidative stability. With ⟨m⟩ the solvent occupation
and N = 2z the Li⁺ coordination number, the free-solvent molar ratio is
x_solvent − x_Li·⟨m⟩·N (the quantity whose accuracy the paper reports):

```python
N_cn = 2 * PAPER_Z
free_solvent = max(x_sol[0] - x_an * float(occupations[0]) * N_cn, 0.0)
print(f"free DME molar ratio {free_solvent:.3f}  ({free_solvent / x_sol[0]:.0%} of all DME is uncoordinated)")

G = li_free_energy(params, solvents, anions, PAPER_Z, **model_kwargs)
print(f"Li+ solvation energy {float(G):.3f} eV")
```

`li_free_energy` returns z·Σᵢ ⟨σᵢ⟩ hᵢ: lower (more negative) means Li⁺ is more strongly
solvated.

### Uncertainty from the five cross-validation models

The paper fitted one parameter set per cross-validation fold. Their spread is a cheap
uncertainty estimate:

```python
from IsingElectrolyte.pretrained import load_paper_ensemble

params_list, model_kwargs = load_paper_ensemble()
ensemble = np.array([np.asarray(find_root(p, solvents, anions, PAPER_Z, **model_kwargs)[0])
                     for p in params_list])
print("mean", ensemble.mean(axis=0), " std", ensemble.std(axis=0))
```

---

## 3. Sweeps and screening

Because a prediction takes milliseconds after the first (compiling) call, scanning a design
space is just a loop. The first call with a new number of species is slow while JAX
compiles; later calls reuse the compiled solver.

### Salt concentration sweep

```python
def predict(solvent_props, ratios, anion_props, molality, params=params):
    """solvent_props: {name: (dn, an, volume, molar_mass)}; anion_props: (dn, volume)."""
    x_s, x_a = mole_fractions(molality, ratios, [v[3] for v in solvent_props.values()])
    solv = {name: {"dn": dn, "an": an, "x": x, "volume": vol}
            for (name, (dn, an, vol, _)), x in zip(solvent_props.items(), x_s)}
    anion = {"anion": {"dn": anion_props[0], "x": x_a, "volume": anion_props[1]}}
    occ, ok = find_root(params, solv, anion, PAPER_Z, **model_kwargs)
    return np.asarray(occ) if ok else np.full(len(solv) + 1, np.nan)


DME, TTE, TFSI = (20.0, 10.2, 135.53, 90.12), (1.9, 20.0, 187.76, 232.07), (11.2, 209.76)
for molality in [0.5, 1.0, 1.5, 2.0]:
    print(f"{molality:.1f} m  DME/TTE/TFSI occupations", predict({"DME": DME, "TTE": TTE}, [1, 2], TFSI, molality))
```

The anion share of the shell grows with salt concentration, and the diluent only starts to
enter the shell at high molality.

### Screening diluents

Section 5 of the paper designs an LHCE around LiTFSI in tetraglyme (G4) by asking which
diluent DN keeps the diluent out of the shell. Holding everything else fixed and scanning the
diluent DN (volume and molar mass of fluorobenzene, AN = 20):

```python
G4 = (16.6, 20.0, 310.88, 222.28)
for dn in [0, 2, 4, 6, 8, 10]:
    diluent = (float(dn), 20.0, 123.60, 96.10)
    occ = predict({"G4": G4, "diluent": diluent}, [1, 2], TFSI, molality=1.0)
    print(f"diluent DN {dn:>2}: G4 {occ[0]:.2f}  diluent {occ[1]:.2f}  TFSI {occ[2]:.2f}")
```

The diluent stays out of the shell up to DN ≈ 2–3 and displaces G4 and TFSI from DN ≈ 4 on,
consistent with the paper's conclusion that diluents need DN < 5 for this solvent.

---

## 4. More than two solvents or one salt

`find_root` takes any number of solvents and anions. With three or more solvents you must
also tell it which species play the same role, through `groups`.

The solvent–solvent J term weighs its two partners differently, so for a pair of *different*
species the model has to decide whose descriptors go in which slot. `groups` gives one
integer rank per species (solvents first, then anions):

- **same rank**: the species are interchangeable, and their couplings are averaged over both
  orderings, so the result does not depend on the order you list them in;
- **different rank**: the species have distinct roles (e.g. solvent vs. diluent), and the
  lower rank takes the first slot.

`groups=None` (the default) treats every species as distinct. That is right for an LHCE
(solvent, diluent, anion) but wrong for a mixture of interchangeable solvents, such as a
high-entropy electrolyte (HEE):

```python
hee = {  # name: (dn, an, volume, molar mass); AN = 20 where no measured value exists
    "DME": (20.0, 10.2, 135.53, 90.12), "EA": (17.1, 9.3, 126.54, 88.11),
    "MA": (16.5, 20.0, 102.92, 74.08), "PP": (17.0, 20.0, 92.92, 58.08),
    "THF": (20.0, 8.0, 108.85, 72.11),
}
x_s, x_a = mole_fractions(1.0, [1] * 5, [v[3] for v in hee.values()])
tfsi = {"TFSI": {"dn": 11.2, "x": x_a, "volume": 209.76}}


def hee_occupations(order, groups):
    solv = {k: {"dn": hee[k][0], "an": hee[k][1], "x": x_s[0], "volume": hee[k][2]} for k in order}
    occ, ok = find_root(params, solv, tfsi, PAPER_Z, groups=groups, **model_kwargs)
    assert ok
    return dict(zip(order, np.asarray(occ)))


forward, backward = list(hee), list(reversed(hee))
for groups in [(0, 0, 0, 0, 0, 1), None]:
    a, b = hee_occupations(forward, groups), hee_occupations(backward, groups)
    shift = max(abs(a[k] - b[k]) for k in hee)
    print(f"groups={groups}: max change when the solvent list is reversed = {shift:.1e}")
```

With a shared group, reversing the list changes nothing (~1e-10, solver tolerance). Without
it, the answer moves by ~6e-3 purely because of listing order.

Rules of thumb:

| System | `groups` |
|---|---|
| 1 solvent (e.g. HCE) | irrelevant |
| solvent + diluent + salt (LHCE) | `None` (default), solvent listed first |
| N interchangeable solvents + salt (HEE) | `(0,) * N + (1,)` |
| solvents + additives + salt | e.g. `(0, 0, 1, 2)`: interchangeable within each band |

**Two or more anions.** The paper model's anion–anion term only describes an anion
interacting with itself, so it is exact for a single salt. Mixed-salt systems need a model
trained with the default anion–anion term (next section).

The derivation and the measured magnitudes are in [`docs/asymmetry-fix.md`](../docs/asymmetry-fix.md).

---

## 5. Training on your own data

### Data format

Training data is one CSV per split (train / validation / test), one row per formulation with
two solvents and one salt. `examples/data/lhce_md/fold_{0..4}/` holds the paper's MD dataset
in exactly this format.

| Column | Meaning |
|---|---|
| `Solvent DN`, `Diluent DN`, `Anion DN` | donor numbers |
| `Solvent AN`, `Diluent AN` | acceptor numbers |
| `solvent molar ratio`, `diluent molar ratio`, `anion molar ratio` | mole fractions (Li⁺ counted) |
| `solvent volume`, `diluent volume`, `anion volume` | molecular volumes (Å³) |
| `Fractional CN solvent`, `Fractional CN diluent`, `Fractional CN anion` | targets: occupations from MD |

### The quick way: `train_lhce.py`

```bash
python examples/train_lhce.py examples/config.yaml                     # fold 0 of the paper data
python examples/train_lhce.py examples/config.yaml --epochs 200 --trials 2
```

`examples/config.yaml` documents every option (data paths, optimizer, number of random
restarts, interaction terms, monotonicity constraints, initialization). The script writes
`trained_params.pkl`, the loss log, parity plots and per-split predictions **to the current
directory**, so run it from a scratch folder if you do not want them next to your code.

### The programmatic way

`IsingElectrolyte.train.train` expects arrays of shape (formulations, species):

```python
import optax
import pandas as pd

import IsingElectrolyte.train as ising_train
from IsingElectrolyte.train import initialize_params, train


def load_split(path):
    df = pd.read_csv(path)
    col = lambda c: jnp.array(df[c].to_numpy(dtype=float))
    pair = lambda a, b: jnp.stack([col(a), col(b)], axis=1)
    return {
        "dn_sol": pair("Solvent DN", "Diluent DN"),
        "an_sol": pair("Solvent AN", "Diluent AN"),
        "x_sol": pair("solvent molar ratio", "diluent molar ratio"),
        "v_sol": pair("solvent volume", "diluent volume"),
        "dn_an": col("Anion DN")[:, None],
        "x_an": col("anion molar ratio")[:, None],
        "v_an": col("anion volume")[:, None],
        "z": jnp.full(len(df), PAPER_Z),
        "targets": jnp.stack([col("Fractional CN solvent"), col("Fractional CN diluent"),
                              col("Fractional CN anion")], axis=1),
    }


fold = "examples/data/lhce_md/fold_0"
train_data, val_data, test_data = (load_split(f"{fold}/{s}.csv") for s in ("train", "val", "test"))
```

**Monotonicity constraints** encode physics the parameters must respect, e.g. that Li⁺–solvent
binding gets stronger (h decreases) with DN. Each parameter group takes `"decrease"`,
`"increase"` or `"none"`; the constraint is enforced by passing the relevant parameters through
a signed softplus, so the optimizer can never violate it. The package default is no
constraints; these are the physically motivated ones:

```python
monotonicity = {
    "sol_params_dn": "decrease",       # h(Li-solvent) decreases with DN
    "salt_params_dn": "decrease",      # h(Li-anion) decreases with anion DN
    "params_sol_salt_an": "decrease",  # J(solvent-anion)
    "params_sol_sol": "decrease",      # J(solvent-solvent)
    "params_anion_anion": "increase",  # J(anion-anion)
    "conc_factor_sol": "increase",     # concentration-volume correction
}
```

Training is non-convex, so like the paper we run several random restarts and keep the one with
the lowest validation loss. `train` stops a restart early if its validation loss stops
improving, and immediately if a loss becomes NaN (some random starting points have no physical
solution for some formulations); such restarts are simply discarded. Only a few epochs here —
use about 1000 for a real fit:

```python
EPOCHS = 3  # ~1000 for a real fit

best = {"val_loss": np.inf}
for seed in [42, 52]:
    params0 = initialize_params(random_seed=seed)
    # train() reads the optimizer from this module-level variable; set it before training.
    ising_train.optimizer = optax.adam(optax.exponential_decay(0.002, transition_steps=1000, decay_rate=0.95))
    fitted, train_loss, val_loss = train(
        params0, EPOCHS, train_data, test_data, val_data,
        opt_state=ising_train.optimizer.init(params0),
        monotonicity_dict=monotonicity, random_seed=seed,
    )
    if len(val_loss) and float(np.min(val_loss)) < best["val_loss"]:
        best = {"val_loss": float(np.min(val_loss)), "params": fitted, "seed": seed}
print(f"best restart: seed {best['seed']}, validation loss {best['val_loss']:.4f}")
```

A learning rate much above 0.002 tends to push the root solver into NaN.

### Using what you trained

Predict with the same term functions and constraints you trained with. Parameters are a plain
dict of arrays, so `pickle` is enough to save them:

```python
import pickle
import tempfile
from pathlib import Path

checkpoint = Path(tempfile.mkdtemp()) / "my_params.pkl"
checkpoint.write_bytes(pickle.dumps(best["params"]))

my_params = pickle.loads(checkpoint.read_bytes())  # or initialize_params(file_path=checkpoint) to fine-tune
occ, ok = find_root(my_params, solvents, anions, PAPER_Z, monotonicity_dict=monotonicity)
print("barely trained model:", np.asarray(occ), bool(ok))
```

`initialize_params(file_path=...)` loads a checkpoint *with 5 % random noise added*, as a
starting point for further training; use plain `pickle.load` for predictions.

---

## 6. Custom interaction terms

Every h and J term is an ordinary JAX function you can replace, without editing the package.
Pass it to `find_root`, `li_free_energy` and `train` under these keyword arguments:

| Keyword | Parameters key | Signature |
|---|---|---|
| `h_sol_func` | `sol_params_dn` | `(dn_eff, x, params) -> eV` |
| `h_an_func` | `salt_params_dn` | `(dn_anion, x, params) -> eV` |
| `J_sol_sol_func` | `params_sol_sol` | `(dn_i, an_i, x_i, dn_j, an_j, x_j, params) -> eV` |
| `J_sol_an_func` | `params_sol_salt_an` | `(dn_anion, an_solvent, x_solvent, x_anion, params) -> eV` |
| `J_an_an_func` | `params_anion_anion` | `(dn_i, x_i, dn_j, x_j, params) -> eV` |

Each term has a **rescale companion**, `rescale_h_sol`, `rescale_h_an`, … with signature
`(params, monotonicity) -> params`, which maps raw parameters to constrained ones (section 5).
If your function uses a different number of parameters than the default, you must also pass
its companion, because the default one assumes the default layout.

Custom functions must use `jax.numpy`, not `numpy`. Keep energies bounded and in eV: they are
divided by kT = 0.0257 eV and exponentiated, and the concentration correction can multiply the
effective DN several-fold during training (to over 200 at some random starts). A term that
grows linearly with DN overflows; a saturating one does not.

Example: replace the sigmoid Li⁺–solvent term with a three-parameter `tanh`, with a slope that
is negative by construction:

```python
def tanh_h_sol(dn_eff, x, p):
    """h(Li-solvent) = p0 - softplus(p1) * tanh(DN_eff / 20) + p2 * ln(x)."""
    return p[0] - jnp.logaddexp(0.0, p[1]) * jnp.tanh(dn_eff / 20.0) + p[2] * jnp.log(x)


def tanh_h_sol_rescale(p, monotonicity):
    return p  # decreasing in DN already guaranteed by the softplus inside the term


custom = dict(h_sol_func=tanh_h_sol, rescale_h_sol=tanh_h_sol_rescale, monotonicity_dict=monotonicity)

# param_sizes changes the length of one parameter array; the others keep their defaults.
params_custom = initialize_params(random_seed=42, param_sizes={"sol_params_dn": 3})
params_custom["sol_params_dn"] = jnp.array([-0.8, 0.0, -0.3])  # a sensible start in eV

occ, ok = find_root(params_custom, solvents, anions, PAPER_Z, **custom)
print("untrained custom model:", np.asarray(occ), bool(ok))

ising_train.optimizer = optax.adam(optax.exponential_decay(0.002, transition_steps=1000, decay_rate=0.95))
fitted_custom, train_loss, val_loss = train(
    params_custom, EPOCHS, train_data, test_data, val_data,
    opt_state=ising_train.optimizer.init(params_custom), random_seed=42, **custom,
)
print("custom-term training losses:", np.asarray(train_loss))
```

Arrays with a non-default size start from zeros plus 5 % noise, which is rarely a good
starting point in eV; set them explicitly as above.

**From the YAML config.** `train_lhce.py` resolves `h_sol_func:`, `rescale_h_sol:` etc. by name
from `IsingElectrolyte.interactions`. Add your function there and reference it by name, or
assign it to the `H_SOL_FUNC` / `RESCALE_H_SOL` variables at the top of the script.

---

## 7. Inspecting and plotting the interaction terms

### h and J for a specific formulation

`energetics` returns the energies behind a prediction: h (one per species) and the J matrix,
evaluated at given occupations. Note it takes arrays rather than species dicts, and the
monotonicity settings as a frozenset:

```python
from IsingElectrolyte.model import energetics

arrays = lambda species, key: jnp.array([props[key] for props in species.values()])
h, J, kT = energetics(
    occupations, params,
    arrays(solvents, "dn"), arrays(solvents, "an"), arrays(solvents, "x"), arrays(solvents, "volume"),
    arrays(anions, "dn"), arrays(anions, "x"), arrays(anions, "volume"), PAPER_Z,
    J_an_an_func=model_kwargs["J_an_an_func"], rescale_J_an_an=model_kwargs["rescale_J_an_an"],
    monotonicity_dict=frozenset(model_kwargs["monotonicity_dict"].items()),
)
print("h [eV]:", np.asarray(h))
print("J [eV]:\n", np.asarray(J))
```

For DME/TTE/LiTFSI this shows DME binding Li⁺ most strongly (h ≈ −1.46 eV), TFSI next
(≈ −1.05 eV) and TTE weakest (≈ −0.63 eV): DN dominates the solvation energetics.

### Ready-made plots

`IsingElectrolyte.analysis` contains the plotting functions behind the paper's interaction-term
figures. They work on **constrained** parameter arrays, so pass the parameters through
`rescale_input_params` first. Use rescaled parameters only for plotting; never feed them back
into `find_root`, which would rescale them a second time. The functions save PNG/CSV files to
the current directory:

```python
import os

from IsingElectrolyte.analysis import sol_dn_func_contour, sol_sol_func_dn_dn_contour, visualize_conc_factor_x
from IsingElectrolyte.model import rescale_input_params

physical = rescale_input_params(
    params,
    monotonicity_dict=frozenset(model_kwargs["monotonicity_dict"].items()),
    rescale_J_an_an=model_kwargs["rescale_J_an_an"],
)

plot_dir = tempfile.mkdtemp()
cwd = os.getcwd()
os.chdir(plot_dir)
try:
    sol_dn_func_contour(npoints=200, sol_params_dn=physical["sol_params_dn"], params_conc_sol=physical["conc_factor_sol"])
    sol_sol_func_dn_dn_contour(physical["params_sol_sol"], physical["conc_factor_sol"])
    visualize_conc_factor_x(physical["conc_factor_sol"])
finally:
    os.chdir(cwd)
print(sorted(os.listdir(plot_dir)))
```

- `h_terms`: h(Li⁺–solvent) vs. DN and concentration (`sol_dn_func_*`), h(Li⁺–anion) (`salt_dn_func_*`).
- `j_terms`: J(solvent–anion) (`an_*`), J(solvent–solvent) (`sol_sol_func_*`), J(anion–anion) (`salt_salt_func_*`).
- `conc_vol_viz`: the concentration–volume correction.
- `free_energy`: occupation and Li⁺ solvation-energy maps over solvent DN/AN.

Current limitations:

- The `free_energy` maps take unscaled parameters but have no argument for a custom
  anion–anion term, so they work only for models trained with the default one (not the paper
  model). Pass their `monotonicity_dict` as a frozenset, e.g.
  `frac_occupation(my_params, monotonicity_dict=frozenset(monotonicity.items()))`.
- `sol_dn_func_fixed_m`, `sol_dn_func_no_conc_factor`, `salt_dn_func`,
  `sol_an_func_no_conc_factor`, `an_m_func_contour` and `sol_sol_func_fixed_m_dn` currently
  fail with an undefined `npoints`.

For anything not covered, sweep `find_root` / `energetics` yourself as in sections 3 and 7.

---

## 8. Reproducing the paper exactly (legacy interface)

The package also keeps the original fixed-species implementation used to fit the paper model,
for bit-for-bit reproduction. These functions carry an `_old` suffix (`find_root_old`,
`energetics_old`, `li_free_energy_old`, …), handle exactly two solvents and one anion, and take
a flat list of 12 numbers instead of species dicts:

```python
from IsingElectrolyte.model import find_root_old

constants = [20.0, 1.9, 11.2,          # DN of solvent, diluent, anion
             x_sol[0], x_sol[1],       # mole fractions of solvent, diluent
             10.2, 20.0,               # AN of solvent, diluent
             x_an,                     # mole fraction of the anion
             135.53, 187.76, 209.76,   # volumes of solvent, diluent, anion
             PAPER_Z]
(m, n, l), ok = find_root_old(params, constants)
print("legacy interface: ", [round(float(v), 6) for v in (m, n, l)], bool(ok))
print("generic interface:", np.asarray(occupations).round(6).tolist())
```

Both interfaces give the same occupations to solver tolerance (~1e-8).

- **Parameter layout.** Legacy checkpoints store the anion–anion parameters under
  `params_salt` (5 values). `load_paper_model` maps them for the generic interface; for your own
  legacy checkpoint, set `params["params_anion_anion"] = params["params_salt"]` and pass the same
  `model_kwargs` as the paper model. This is exact for a single anion.
- **Retraining a paper fold** with the original procedure (10 restarts × up to 1000 epochs,
  legacy physics) is `examples/train_base_model.py`; see the README. It reproduces the paper's
  5-fold accuracy (summed RMSE 10.7 %, R² 0.87).
- **Checking the published numbers** without training: `python examples/reproduce_paper_cv.py`.
