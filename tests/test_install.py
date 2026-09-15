"""Is the package installed completely? (imports and bundled data files)"""

import importlib
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULES = [
    "IsingElectrolyte",
    "IsingElectrolyte.interactions",
    "IsingElectrolyte.conc_vol_correction",
    "IsingElectrolyte.model",
    "IsingElectrolyte.train",
    "IsingElectrolyte.pretrained",
    "IsingElectrolyte.analysis",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_jax_runs_in_double_precision():
    import jax.numpy as jnp

    import IsingElectrolyte.model  # noqa: F401  (enables x64 on import)

    assert jnp.ones(1).dtype == jnp.float64


def test_pretrained_models_are_bundled():
    from IsingElectrolyte.pretrained import N_FOLDS

    folder = resources.files("IsingElectrolyte") / "pretrained_models"
    for fold in range(N_FOLDS):
        with np.load(folder / f"paper_fold{fold}.npz") as f:
            assert set(f.files) == {"sol_params_dn", "salt_params_dn", "params_sol_salt_an",
                                    "params_sol_sol", "params_salt", "conc_factor_sol"}
            assert all(np.all(np.isfinite(f[k])) for k in f.files)


@pytest.mark.parametrize("name", ["solvent_map_dict.csv", "DN_trade_offs.csv"])
def test_analysis_data_is_bundled(name):
    df = pd.read_csv(resources.files("IsingElectrolyte") / "analysis" / name)
    assert len(df) > 0


def test_package_data_globs_cover_all_data_files(repo):
    """Data files must be listed in pyproject.toml, or `pip install .` silently drops them."""
    tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
    config = tomllib.loads((repo / "pyproject.toml").read_text())
    globs = config["tool"]["setuptools"]["package-data"]["IsingElectrolyte"]
    pkg = repo / "src" / "IsingElectrolyte"
    covered = {p for g in globs for p in pkg.glob(g)}
    data_files = {p for p in pkg.rglob("*") if p.suffix in {".npz", ".csv", ".pkl", ".json"}}
    assert data_files, "expected bundled data files"
    assert data_files <= covered, f"not covered by package-data: {sorted(map(str, data_files - covered))}"


def test_example_dataset_is_complete(repo):
    """The shipped MD dataset: 5 folds whose test sets together cover 182 formulations once."""
    data = repo / "examples" / "data" / "lhce_md"
    tests = [pd.read_csv(data / f"fold_{k}" / "test.csv") for k in range(5)]
    for k in range(5):
        for split in ("train", "val"):
            assert (data / f"fold_{k}" / f"{split}.csv").exists()
    all_tests = pd.concat(tests)
    assert len(all_tests) == 182
    assert not all_tests.duplicated(["formulation", "Anion", "molality"]).any()
