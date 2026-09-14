"""Reproduce the paper's 5-fold cross-validation accuracy with the shipped models.

    python examples/reproduce_paper_cv.py

Each fold model predicts its own held-out test set (examples/data/lhce_md/fold_k/test.csv);
together the test sets cover all 182 formulations. Expected output:
summed RMSE 10.75 %, R^2 0.87 (paper, Fig. 2b).
"""

import os

import numpy as np
import pandas as pd

from IsingElectrolyte.model import find_root
from IsingElectrolyte.pretrained import N_FOLDS, PAPER_Z, load_paper_model

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lhce_md")


def row_to_species(row):
    """Map one dataset row to the solvents / anions dicts used by find_root."""
    solvents = {
        "solvent": {"dn": row["Solvent DN"], "an": row["Solvent AN"],
                    "x": row["solvent molar ratio"], "volume": row["solvent volume"]},
        "diluent": {"dn": row["Diluent DN"], "an": row["Diluent AN"],
                    "x": row["diluent molar ratio"], "volume": row["diluent volume"]},
    }
    anions = {
        "anion": {"dn": row["Anion DN"], "x": row["anion molar ratio"],
                  "volume": row["anion volume"]},
    }
    return solvents, anions


if __name__ == "__main__":
    predictions, targets = [], []
    for fold in range(N_FOLDS):
        params, model_kwargs = load_paper_model(fold)
        test = pd.read_csv(os.path.join(DATA_DIR, f"fold_{fold}", "test.csv"))
        for _, row in test.iterrows():
            solvents, anions = row_to_species(row)
            occ, found_valid = find_root(params, solvents, anions, PAPER_Z, **model_kwargs)
            if not bool(found_valid):
                print(f"warning: no valid root for {row['formulation']} {row['Anion']} {row['molality']} m")
            predictions.append(np.asarray(occ))
            targets.append([row["Fractional CN solvent"], row["Fractional CN diluent"],
                            row["Fractional CN anion"]])
        print(f"fold {fold}: {len(test)} test formulations")

    pred, target = np.array(predictions), np.array(targets)
    residual = pred - target
    rmse = np.sqrt((residual ** 2).sum(axis=1).mean())
    r2 = 1.0 - (residual ** 2).sum() / ((target - target.mean(axis=0)) ** 2).sum()
    print(f"\n{len(pred)} formulations")
    print(f"summed RMSE of fractional CN: {100 * rmse:.2f} %   (paper: 10.75 %)")
    print(f"R^2:                          {r2:.2f}     (paper: 0.87)")
