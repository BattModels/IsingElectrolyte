"""Pretrained parameters from the paper.

Zhao, Lin, Kelly & Viswanathan, "Differentiable Solvation Shell Model for Rational
Electrolyte Design", arXiv:2609.05671 (2026).

The paper fits the model with 5-fold cross-validation on 182 MD-simulated
LHCE formulations (2 solvents + 1 salt each). One parameter set per fold is shipped
here; together their test folds cover every formulation exactly once.

Usage::

    from IsingLHCE.model import find_root
    from IsingLHCE.pretrained import load_paper_model, PAPER_Z

    params, model_kwargs = load_paper_model(fold=0)
    occupations, found_valid = find_root(params, solvents, anions, PAPER_Z, **model_kwargs)

``model_kwargs`` selects the anion-anion term and monotonicity constraints the
paper model was trained with. Always pass it: with the package defaults the same
parameters give different (wrong) predictions.
"""

import os

import jax.numpy as jnp
import numpy as np

from .interactions import legacy_J_an_an, legacy_J_an_an_rescale

#: Mean-field coordination number used in the paper: z = N/2, with N = 3.72 the
#: average Li+ coordination number from MD.
PAPER_Z = 3.72 / 2.0

#: Number of cross-validation folds (and shipped parameter sets).
N_FOLDS = 5

#: Monotonicity constraints the paper model was trained with. The anion-anion
#: constraint is fixed inside ``legacy_J_an_an_rescale``.
PAPER_MONOTONICITY = {
    "sol_params_dn":      "decrease",
    "salt_params_dn":     "decrease",
    "params_sol_salt_an": "decrease",
    "params_sol_sol":     "decrease",
    "params_anion_anion": "none",
    "conc_factor_sol":    "increase",
}

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models")


def load_paper_model(fold=0):
    """Load one cross-validation model from the paper.

    Args:
        fold: integer in ``range(N_FOLDS)``.

    Returns:
        (params, model_kwargs):
            params:       parameter dict for ``find_root`` / ``li_free_energy``.
            model_kwargs: keyword arguments to pass alongside ``params``
                          (``J_an_an_func``, ``rescale_J_an_an``, ``monotonicity_dict``).

    The paper model is exact for a single anion (M = 1). It can be applied to any
    number of solvents; for N >= 3 pass ``groups`` as well (see TUTORIAL §2c).
    """
    if fold not in range(N_FOLDS):
        raise ValueError(f"fold must be an integer in 0..{N_FOLDS - 1}, got {fold!r}")
    with np.load(os.path.join(_MODEL_DIR, f"paper_fold{fold}.npz")) as f:
        params = {name: jnp.asarray(f[name]) for name in f.files}
    # The generic interface reads the anion-anion parameters from this key.
    params["params_anion_anion"] = params["params_salt"]
    model_kwargs = {
        "J_an_an_func":      legacy_J_an_an,
        "rescale_J_an_an":   legacy_J_an_an_rescale,
        "monotonicity_dict": dict(PAPER_MONOTONICITY),
    }
    return params, model_kwargs


def load_paper_ensemble():
    """Load all ``N_FOLDS`` paper models.

    Returns:
        (params_list, model_kwargs): a list of parameter dicts and the shared kwargs.
    """
    params_list = [load_paper_model(k)[0] for k in range(N_FOLDS)]
    return params_list, load_paper_model(0)[1]
