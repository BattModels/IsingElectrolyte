"""Does a short training run work?"""

import pickle

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest

import IsingElectrolyte.train as ising_train
from IsingElectrolyte.model import find_root
from IsingElectrolyte.pretrained import PAPER_Z
from IsingElectrolyte.train import initialize_params, train

from conftest import DATA


def load_split(path, rows):
    df = pd.read_csv(path)
    df = df if rows is None else df.head(rows)
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


@pytest.fixture(scope="module")
def small_splits():
    fold = DATA / "fold_0"
    return tuple(load_split(fold / f"{s}.csv", rows=8) for s in ("train", "val", "test"))


def test_initialize_params_shapes():
    params = initialize_params(random_seed=0)
    assert {k: v.shape for k, v in params.items()} == {
        "sol_params_dn": (5,), "salt_params_dn": (5,), "params_sol_salt_an": (6,),
        "params_sol_sol": (16,), "params_anion_anion": (6,), "conc_factor_sol": (8,),
    }
    custom = initialize_params(random_seed=0, param_sizes={"params_sol_sol": 9})
    assert custom["params_sol_sol"].shape == (9,)


def test_initialize_params_is_reproducible():
    a, b = initialize_params(random_seed=7), initialize_params(random_seed=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_rescaled_start_values_match_default_params(physical_monotonicity):
    """The hard-coded unconstrained start is the default start after the physical softplus."""
    from IsingElectrolyte import interactions as I
    from IsingElectrolyte.train import _DEFAULT_PARAMS, _DEFAULT_PARAMS_RESCALED

    rescale = {"sol_params_dn": I.default_h_sol_rescale, "salt_params_dn": I.default_h_an_rescale,
               "params_sol_salt_an": I.default_J_sol_an_rescale, "params_sol_sol": I.default_J_sol_sol_rescale,
               "conc_factor_sol": I.default_conc_factor_rescale}
    for key, fn in rescale.items():
        expected = fn(_DEFAULT_PARAMS[key], physical_monotonicity[key])
        np.testing.assert_allclose(np.asarray(_DEFAULT_PARAMS_RESCALED[key]), np.asarray(expected), rtol=1e-12, err_msg=key)


def test_constrained_start_is_unchanged(physical_monotonicity):
    """Passing the constraints you train with does not change the (raw) constrained start."""
    for seed in (0, 42):
        default, constrained = initialize_params(random_seed=seed), initialize_params(random_seed=seed, monotonicity_dict=physical_monotonicity)
        assert all(np.array_equal(default[k], constrained[k]) for k in default)


def test_start_is_chosen_per_group(physical_monotonicity):
    raw = initialize_params(random_seed=3)
    unconstrained = initialize_params(random_seed=3, monotonicity_dict={})
    partial = initialize_params(random_seed=3, monotonicity_dict={"sol_params_dn": "decrease"})
    np.testing.assert_array_equal(partial["sol_params_dn"], raw["sol_params_dn"])          # constrained group
    np.testing.assert_array_equal(partial["salt_params_dn"], unconstrained["salt_params_dn"])  # unlisted → "none"
    assert not np.allclose(unconstrained["salt_params_dn"], raw["salt_params_dn"])


def test_unconstrained_start_is_solvable():
    """Without constraints, the default start must give a valid root for every training formulation."""
    import jax

    from IsingElectrolyte.model import DEFAULT_MONOTONICITY, _find_root_impl, _freeze_mono

    data = load_split(DATA / "fold_0" / "train.csv", rows=None)
    mono = _freeze_mono(DEFAULT_MONOTONICITY)
    for seed in (42, 112):  # seed 112 had 56 unsolvable formulations before the anion-anion start was rescaled
        params = initialize_params(random_seed=seed, monotonicity_dict=DEFAULT_MONOTONICITY)
        solve = jax.vmap(lambda a, b, x, v, d, xa, va: _find_root_impl(
            params, a, b, x, v, d, xa, va, PAPER_Z, jnp.ones(3) / 3, max_tries=10, monotonicity_dict=mono))
        _, ok = solve(data["dn_sol"], data["an_sol"], data["x_sol"], data["v_sol"], data["dn_an"], data["x_an"], data["v_an"])
        assert bool(jnp.all(ok)), f"seed {seed}: {int((~ok).sum())} formulations without a valid root"


def test_short_training_run(small_splits, physical_monotonicity):
    train_data, val_data, test_data = small_splits
    params0 = initialize_params(random_seed=42)
    ising_train.optimizer = optax.adam(optax.exponential_decay(0.002, transition_steps=1000, decay_rate=0.95))
    params, train_loss, val_loss = train(
        params0, 2, train_data, test_data, val_data,
        opt_state=ising_train.optimizer.init(params0),
        monotonicity_dict=physical_monotonicity, random_seed=42,
    )
    assert len(train_loss) == 2 and len(val_loss) == 2
    assert np.all(np.isfinite(np.asarray(train_loss))) and np.all(np.isfinite(np.asarray(val_loss)))
    assert np.asarray(train_loss)[1] < np.asarray(train_loss)[0]  # the optimizer is making progress
    assert {k: v.shape for k, v in params.items()} == {k: v.shape for k, v in params0.items()}
    assert any(not np.allclose(params[k], params0[k]) for k in params)


def test_checkpoint_round_trip(tmp_path, lhce, physical_monotonicity):
    """A saved checkpoint loads for fine-tuning and gives predictions."""
    params = initialize_params(random_seed=42)
    path = tmp_path / "params.pkl"
    path.write_bytes(pickle.dumps(params))
    loaded = initialize_params(file_path=path, random_seed=0)  # from_file adds 5 % noise
    assert set(loaded) == set(params)
    occ, ok = find_root(pickle.loads(path.read_bytes()), *lhce, PAPER_Z, monotonicity_dict=physical_monotonicity)
    assert bool(ok)


def test_legacy_checkpoint_gives_helpful_error(tmp_path, paper_model):
    """Paper-format checkpoints (params_salt) are rejected with a pointer to load_paper_model."""
    params, _ = paper_model
    legacy = {k: v for k, v in params.items() if k != "params_anion_anion"}
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps(legacy))
    with pytest.raises(ValueError, match="load_paper_model"):
        initialize_params(file_path=path)
