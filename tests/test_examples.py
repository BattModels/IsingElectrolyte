"""Do the example scripts in examples/ run and print what the README promises?"""

import re
import subprocess
import sys

import pytest

from conftest import DATA, EXAMPLES


def run_script(script, *args, cwd):
    result = subprocess.run([sys.executable, str(EXAMPLES / script), *map(str, args)],
                            cwd=cwd, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, f"{script} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return result.stdout


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
    fold = DATA / "fold_0"
    out = run_script("train_lhce.py", EXAMPLES / "config.yaml",
                     "--train", fold / "train.csv", "--val", fold / "val.csv", "--test", fold / "test.csv",
                     "--epochs", 2, "--trials", 1, cwd=tmp_path)
    assert "Best params saved" in out
    for name in ["trained_params.pkl", "train_val_loss_log.csv", "ising_results_test.csv", "ising_test.png"]:
        assert (tmp_path / name).exists(), name
