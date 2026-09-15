"""Do the example scripts in examples/ run and print what the README promises?"""

import re
import subprocess
import sys

import pytest

from conftest import DATA, EXAMPLES


def run_script(script, *args, cwd, check=True):
    result = subprocess.run([sys.executable, str(EXAMPLES / script), *map(str, args)],
                            cwd=cwd, capture_output=True, text=True, timeout=1800)
    if not check:
        return result
    assert result.returncode == 0, f"{script} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return result.stdout


def fold0_flags():
    fold = DATA / "fold_0"
    return ["--train", fold / "train.csv", "--val", fold / "val.csv", "--test", fold / "test.csv"]


def write_config(path, **extra):
    fold = DATA / "fold_0"
    lines = [f"train: {fold / 'train.csv'}", f"val: {fold / 'val.csv'}", f"test: {fold / 'test.csv'}"]
    lines += [f"{k}: {v}" for k, v in extra.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_predict_solvation(tmp_path):
    out = run_script("predict_solvation.py", cwd=tmp_path)
    rows = {float(m.group(1)): [float(v) for v in m.group(2).split()]
            for m in re.finditer(r"^\s*([\d.]+)\s+([\d.\s]+?)\s*$", out, flags=re.M)}
    assert set(rows) == {0.5, 1.0, 1.5, 2.0}
    dme, tte, tfsi, _free = rows[1.0]
    assert (dme, tte, tfsi) == pytest.approx((0.384, 0.000, 0.616), abs=2e-3)  # paper ensemble, 1.0 m


def test_reproduce_paper_cv(tmp_path):
    """The paper's 5-fold cross-validation accuracy, over all 182 formulations."""
    out = run_script("reproduce_paper_cv.py", cwd=tmp_path)
    assert "182 formulations" in out
    rmse = float(re.search(r"summed RMSE of fractional CN:\s*([\d.]+)", out).group(1))
    r2 = float(re.search(r"R\^2:\s*([\d.]+)", out).group(1))
    assert rmse == pytest.approx(10.75, abs=0.01)
    assert r2 == pytest.approx(0.87, abs=0.005)


@pytest.mark.slow
def test_train_lhce_script(tmp_path):
    """examples/train_lhce.py with the shipped config: one short restart, writes its outputs."""
    out = run_script("train_lhce.py", EXAMPLES / "config.yaml", *fold0_flags(),
                     "--epochs", 2, "--trials", 1, cwd=tmp_path)
    assert "Best params saved" in out
    for name in ["trained_params.pkl", "train_val_loss_log.csv", "ising_results_test.csv", "ising_test.png"]:
        assert (tmp_path / name).exists(), name


@pytest.mark.slow
def test_train_lhce_flags_only(tmp_path):
    """The documented CLI-only usage (no YAML) must train stably, i.e. default to the physical constraints."""
    out = run_script("train_lhce.py", *fold0_flags(), "--epochs", 2, "--trials", 1, cwd=tmp_path)
    assert "'sol_params_dn': 'decrease'" in out
    assert "diverged" not in out
    assert "Best params saved" in out
    assert (tmp_path / "trained_params.pkl").exists()


@pytest.mark.slow
def test_train_lhce_unconstrained(tmp_path):
    """monotonicity_dict: {} starts from the rescaled hand-tuned values and trains."""
    config = write_config(tmp_path / "unconstrained.yaml", monotonicity_dict="{}", epochs=2, trials=1)
    out = run_script("train_lhce.py", config, cwd=tmp_path)
    assert "'sol_params_dn': 'none'" in out
    assert "diverged" not in out
    assert (tmp_path / "trained_params.pkl").exists()


@pytest.mark.slow
def test_train_lhce_reports_when_all_trials_diverge(tmp_path):
    """A start that diverges (unconstrained, raw hand-tuned values forced): report it, no crash, no checkpoint."""
    config = write_config(tmp_path / "raw_start.yaml", monotonicity_dict="{}", epochs=2, trials=1,
                          initialize_params_kwargs="{mode: from_scratch, monotonicity_dict: null}")
    result = run_script("train_lhce.py", config, cwd=tmp_path, check=False)
    assert result.returncode != 0
    assert "all 1 trial(s) diverged" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "trained_params.pkl").exists()


def test_train_lhce_rejects_misspelled_monotonicity(tmp_path):
    config = write_config(tmp_path / "typo.yaml", monotonicity_dict="{sol_params_dn: decreasing}")
    result = run_script("train_lhce.py", config, cwd=tmp_path, check=False)
    assert result.returncode != 0
    assert "values must be 'increase', 'decrease' or 'none'" in result.stderr
    assert "Traceback" not in result.stderr
