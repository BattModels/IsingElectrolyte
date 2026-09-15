# IsingElectrolyte

[![arXiv](https://img.shields.io/badge/arXiv-2609.05671-b31b1b.svg)](https://arxiv.org/abs/2609.05671)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A differentiable mean-field Ising model of the Li⁺ solvation shell, for electrolyte design.**

This is the code for the paper:

> Hancheng Zhao, Hongyi Lin, Celia Kelly, Venkatasubramanian Viswanathan.
> **Differentiable Solvation Shell Model for Rational Electrolyte Design.**
> arXiv:2609.05671 (2026). https://arxiv.org/abs/2609.05671

Model inputs involve properties of each solvent and salt in an electrolyte:
Gutmann donor number (DN), acceptor number (AN), mole fraction and molecular volume (here we use DFT-computed molecular size).
It predicts what fraction of the Li⁺ first solvation shell each species occupies.
The model is written in [JAX](https://github.com/jax-ml/jax), so it is differentiable
end to end. Its parameters are fitted directly to solvation structures from molecular
dynamics (MD) by gradient descent.

On localized high-concentration electrolytes (LHCEs) it reaches **10.7 % RMSE and
R² = 0.87** for shell composition in 5-fold cross-validation against MD. A prediction
takes under a second per formulation, compared with ~100 CPU-hours for MD.

The released parameters were fitted on LHCE data only, yet they transfer to
high-concentration electrolytes (HCEs) without refitting. The framework itself is not
LHCE-specific: given solvation data, it can be refitted to other electrolyte classes,
such as high-entropy electrolytes (HEEs).

This repository contains:

- the model (any number of solvents; one or more anions),
- the five cross-validation models from the paper, ready to use,
- the MD dataset of 182 LHCE formulations the paper was trained on,
- scripts to predict, reproduce the paper's accuracy, and train on your own data.

---

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/BattModels/IsingElectrolyte.git
cd IsingElectrolyte
pip install -e .            # add ".[dev]" to also install pytest
```

This installs the CPU build of JAX, which is all the model needs. For a GPU build, follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

Check the installation:

```bash
python examples/predict_solvation.py
```

---

## Quick start: predict a solvation shell

```python
from IsingElectrolyte.model import find_root
from IsingElectrolyte.pretrained import load_paper_model, PAPER_Z

params, model_kwargs = load_paper_model(fold=0)

# 1.0 m LiTFSI in DME:TTE = 1:2 (mol:mol)
solvents = {
    "DME": {"dn": 20.0, "an": 10.2, "x": 0.2434, "volume": 135.53},
    "TTE": {"dn": 1.9,  "an": 20.0, "x": 0.4868, "volume": 187.76},
}
anions = {
    "TFSI": {"dn": 11.2, "x": 0.1349, "volume": 209.76},
}

occupations, found_valid = find_root(params, solvents, anions, PAPER_Z, **model_kwargs)
assert found_valid
print(occupations)   # [DME, TTE, TFSI] share of the Li+ shell: ≈ [0.38, 0.00, 0.62]
```

### Inputs

| Key | Meaning | Units |
|---|---|---|
| `dn` | Gutmann donor number | kcal/mol |
| `an` | Gutmann acceptor number (solvents only) | — |
| `x` | mole fraction (see note) | — |
| `volume` | molecular volume | Å³ per molecule |

- **Mole fractions count Li⁺ as its own species.** For a salt LiA,
  `x_solvent_1 + x_solvent_2 + … + x_anion + x_Li = 1` and `x_Li = x_anion`. `examples/predict_solvation.py` shows how
  to convert salt molality and solvent:diluent ratio into these mole fractions.
- `PAPER_Z = 1.86` is the mean-field coordination number (half of the MD average Li⁺
  coordination number, 3.72).
- `examples/data/lhce_md/` and `src/IsingElectrolyte/analysis/solvent_map_dict.csv` list DN, AN
  and volume for the molecules used in the paper.

### Outputs

- `occupations` holds one value per species, solvents first (in dict order) and then
  anions. Each value is the fraction of the Li⁺ first solvation shell that species
  occupies, so the values sum to 1.
- `found_valid` is `False` if the solver found no physical solution. Always check it,
  because failed solves return NaN.

### Always pass `**model_kwargs`

`model_kwargs` tells `find_root` which anion–anion term and monotonicity constraints the
paper model was trained with. Without it, the same parameters give different, wrong numbers.

### Model uncertainty

The five cross-validation models give an uncertainty estimate. Average their predictions:

```python
import numpy as np
from IsingElectrolyte.pretrained import load_paper_ensemble

params_list, model_kwargs = load_paper_ensemble()
occ = np.array([find_root(p, solvents, anions, PAPER_Z, **model_kwargs)[0] for p in params_list])
print(occ.mean(axis=0), occ.std(axis=0))
```

### Li⁺ solvation free energy

`li_free_energy` returns the Li⁺ solvation energy G = z Σᵢ ⟨occupationᵢ⟩ hᵢ (eV), where
hᵢ is the Li⁺–species interaction:

```python
from IsingElectrolyte.model import li_free_energy

G = li_free_energy(params, solvents, anions, PAPER_Z, **model_kwargs)
```

---

## Beyond two solvents and one salt

`find_root` accepts any number of solvents.

- **Three or more solvents: pass `groups`.** Give each species an integer rank, solvents
  first and then anions. Species with the same rank are treated as interchangeable (e.g.
  the solvents of a high-entropy electrolyte). Different ranks mark distinct roles
  (e.g. solvent vs. diluent). If you leave `groups` out, every species is treated as
  distinct and the prediction depends on the order you list the solvents in.

  ```python
  # five interchangeable solvents + one salt
  find_root(params, solvents, anions, PAPER_Z, **model_kwargs, groups=(0, 0, 0, 0, 0, 1))
  ```

- **Two or more anions:** the paper model's anion–anion term is only exact for a single
  anion. Mixed-salt systems need a model retrained with the default anion–anion term
  (see below).

Note: accuracy is lower below ~0.5 m, likely because solvation there also depends on the dielectric constant, which the model does not include at the moment.

See [section 4 of the tutorial](examples/TUTORIAL.md#4-more-than-two-solvents-or-one-salt) for how `groups` works.

---

## Reproduce the paper

**Cross-validation accuracy** (about 10 seconds on a CPU). Each fold model predicts its own
held-out test set:

```bash
python examples/reproduce_paper_cv.py
# summed RMSE of fractional CN: 10.75 %   (paper: 10.75 %)
# R^2:                          0.87     (paper: 0.87)
```

**Retrain a fold** with the exact training procedure from the paper: 10 random restarts,
up to 1000 epochs each. The script reads `train.csv` / `val.csv` / `test.csv` from the
current directory and writes its results there:

```bash
cp -r examples/data/lhce_md/fold_0 my_fold_0 && cd my_fold_0
python ../examples/train_base_model.py
```

---

## Train on your own data

`examples/train_lhce.py` trains the model with a YAML config. Run it from the repository
root. The shipped config trains on fold 0 of the paper dataset:

```bash
python examples/train_lhce.py examples/config.yaml
python examples/train_lhce.py examples/config.yaml --epochs 200 --trials 2   # CLI flags override YAML
```

To use your own data, point `train`, `val` and `test` in the config at CSV files with these columns:

| Column | Meaning |
|---|---|
| `Solvent DN`, `Diluent DN`, `Anion DN` | donor numbers |
| `Solvent AN`, `Diluent AN` | acceptor numbers |
| `solvent molar ratio`, `diluent molar ratio`, `anion molar ratio` | mole fractions (Li⁺ counted) |
| `solvent volume`, `diluent volume`, `anion volume` | molecular volumes (Å³) |
| `Fractional CN solvent`, `Fractional CN diluent`, `Fractional CN anion` | MD targets: share of the Li⁺ first shell |

The script writes the best parameters (`trained_params.pkl`), a loss log, parity plots
and per-split predictions to the current directory. To use the trained model, load the
parameters and call `find_root` with the same term functions and `monotonicity_dict`
you trained with.

Everything in the config is documented in [`examples/config.yaml`](examples/config.yaml),
including interaction functions, monotonicity constraints and initialization.

---

## Custom interaction terms

The model's energy has two kinds of terms, each a plain JAX function you can replace:

- **h terms:** Li⁺–solvent and Li⁺–anion interactions.
- **J terms:** solvent–solvent, solvent–anion and anion–anion interactions.

Pass your own function to `find_root`, `li_free_energy` or `train`:

```python
def my_h_sol(dn_eff, x, params):          # Li+-solvent interaction, linear in DN
    return params[0] + params[1] * dn_eff

find_root(params, solvents, anions, PAPER_Z, h_sol_func=my_h_sol, ...)
```

Functions must use `jax.numpy`, not `numpy`. [`examples/TUTORIAL.md`](examples/TUTORIAL.md)
covers:

- the signature of every term,
- the companion rescale functions that enforce monotonicity,
- how to use your own functions from a YAML config.

---

## Repository layout

```
src/IsingElectrolyte/
├── model.py                 # mean-field equations, root finding (find_root), free energy
├── interactions.py          # h / J term functions and their monotonicity rescalings
├── conc_vol_correction.py   # concentration and molecular-size correction of DN / AN
├── train.py                 # loss, optimizer step, training loop, parity plots
├── pretrained.py            # load_paper_model / load_paper_ensemble
├── pretrained_models/       # the 5 cross-validation models from the paper
└── analysis/                # plotting of interaction terms and free-energy landscapes

examples/
├── predict_solvation.py     # predict an LHCE solvation shell vs. salt concentration
├── reproduce_paper_cv.py    # reproduce the paper's cross-validation accuracy
├── train_lhce.py            # train from a YAML config
├── config.yaml
├── train_base_model.py      # the paper's original training procedure
├── TUTORIAL.md              # detailed guide to the package
└── data/lhce_md/fold_{0..4}/{train,val,test}.csv

tests/                       # pytest suite
```

## Dataset

`examples/data/lhce_md/` contains the MD dataset from the paper:

- **Size:** 182 electrolyte formulations, each with 2 solvents and 1 salt.
- **Solvents and diluents (8):** DCD, DCM, DEA, DMA, DME, DMSO, MPL, THF.
- **Salts (3):** LiBF₄, LiPF₆, LiTFSI.
- **Concentrations:** 0.2–2.5 m.

It is split into the 5 cross-validation folds used in the paper. Each fold's `test.csv` is
disjoint from the others, and together they cover all 182 formulations. Besides the model
inputs and targets, each row holds:

- the formulation name and molality,
- PubChem CIDs and molar masses,
- absolute coordination numbers, free-species fractions and density from MD.

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow"   # about a minute: install, predictions, gradients, short training, paper accuracy
pytest                 # everything, adds a few minutes: every tutorial example and the training script
```

## Citation

If you use this code or data, please cite:

```bibtex
@article{zhao2026differentiable,
  title   = {Differentiable Solvation Shell Model for Rational Electrolyte Design},
  author  = {Zhao, Hancheng and Lin, Hongyi and Kelly, Celia and Viswanathan, Venkatasubramanian},
  journal = {arXiv preprint arXiv:2609.05671},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.05671}
}
```

## License

MIT — see [LICENSE](LICENSE). Questions and bug reports:
[GitHub issues](https://github.com/BattModels/IsingElectrolyte/issues).
