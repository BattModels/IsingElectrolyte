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
    df = pd.read_csv(path).head(rows)
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
