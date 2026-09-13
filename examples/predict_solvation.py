"""Predict the Li+ solvation shell of an LHCE with the pretrained paper model.

Example system: LiTFSI in DME:TTE = 1:2 (mol:mol), a common LHCE that is not in the
training set (paper Fig. 2d). Run from anywhere:

    python examples/predict_solvation.py

For each salt molality it prints the fraction of the Li+ first solvation shell
occupied by the solvent (DME), diluent (TTE) and anion (TFSI), averaged over the
5 cross-validation models, plus the fraction of solvent molecules not coordinated
to Li+ ("free solvent").
"""

import numpy as np

from IsingLHCE.model import find_root
from IsingLHCE.pretrained import PAPER_Z, load_paper_ensemble

# Molecular descriptors
#   dn, an : Gutmann donor / acceptor number (kcal/mol)
#   volume : DFT molecular volume (Angstrom^3 per molecule)
#   mass   : molar mass (g/mol), only used here to convert molality to mole fractions
DME = {"dn": 20.0, "an": 10.2, "volume": 135.53, "mass": 90.12}
TTE = {"dn": 1.9, "an": 20.0, "volume": 187.76, "mass": 232.07}
TFSI = {"dn": 11.2, "volume": 209.76}


def mole_fractions(molality, n_solvent, n_diluent, mass_solvent, mass_diluent):
    """Mole fractions in the convention the model was fitted with.

    Li+ and the anion are counted as separate species, so
    x_solvent + x_diluent + x_anion + x_Li = 1 and x_anion = x_Li.

    Args:
        molality: mol salt per kg of solvent + diluent.
        n_solvent, n_diluent: molar ratio of solvent to diluent (e.g. 1, 2).
        mass_solvent, mass_diluent: molar masses in g/mol.
    """
    n_salt = molality * (n_solvent * mass_solvent + n_diluent * mass_diluent) / 1000.0
    total = 2.0 * n_salt + n_solvent + n_diluent
    return n_solvent / total, n_diluent / total, n_salt / total


def predict(params_list, model_kwargs, molality):
    x_sol, x_dil, x_an = mole_fractions(molality, 1.0, 2.0, DME["mass"], TTE["mass"])
    solvents = {
        "DME": {"dn": DME["dn"], "an": DME["an"], "x": x_sol, "volume": DME["volume"]},
        "TTE": {"dn": TTE["dn"], "an": TTE["an"], "x": x_dil, "volume": TTE["volume"]},
    }
    anions = {"TFSI": {"dn": TFSI["dn"], "x": x_an, "volume": TFSI["volume"]}}

    occupations = []
    for params in params_list:
        occ, found_valid = find_root(params, solvents, anions, PAPER_Z, **model_kwargs)
        if not bool(found_valid):
            raise RuntimeError(f"No valid root at {molality} m")
        occupations.append(np.asarray(occ))
    occ = np.mean(occupations, axis=0)

    # Free solvent: solvent molecules not in any Li+ shell,
    # x_free = x_solvent - x_Li * <m> * N, with N = 2z the Li+ coordination number.
    free_solvent = (x_sol - x_an * occ[0] * 2.0 * PAPER_Z) / x_sol
    return occ, free_solvent


if __name__ == "__main__":
    params_list, model_kwargs = load_paper_ensemble()

    print("LiTFSI in DME:TTE = 1:2 (mol:mol)")
    print(f"{'molality':>9} {'DME':>7} {'TTE':>7} {'TFSI':>7} {'free DME':>9}")
    for molality in [0.5, 1.0, 1.5, 2.0]:
        occ, free = predict(params_list, model_kwargs, molality)
        print(f"{molality:>9.1f} {occ[0]:>7.3f} {occ[1]:>7.3f} {occ[2]:>7.3f} {free:>9.3f}")
