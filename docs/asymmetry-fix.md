# The `J` asymmetry fix (branch `asymmetry-fix`)

**Status:** implemented, verified, not merged.
**Scope:** `src/IsingLHCE/model.py` + `tests/test_asymmetry_fix.py`. No parameter
files change; no retraining is required.

This document exists so that a later cleanup pass can see *why* each line was
written, not just what changed.

---

## 1. The bug

The generic `energetics` builds the solvent-solvent block with

```python
J_ss[i, j] = default_J_sol_sol(dn_i, an_i, x_i, dn_j, an_j, x_j, p)
```

and `default_J_sol_sol` feeds `sol_sol_func` like this:

```python
sol_sol_func(jnp.array([dn_i, dn_j]), p[5:10])     # DN-DN block
sol_sol_func(jnp.array([an_i, an_j]), p[10:15])    # AN-AN block
```

But `sol_sol_func` is **not symmetric in its two inputs**:

```python
def sol_sol_func(X, params):
    dn, an = X
    L0, H0, a0, a1, a2 = params
    return L0 + H0 / (1.0 + jnp.exp(a0 + a1 * dn + a2 * an))
```

Slot 1 is weighted by `a1`, slot 2 by `a2`, and the fitted values differ — in the
trial-25 checkpoints `|a1 - a2|` reaches **0.48** in the DN-DN block and **1.17**
in the AN-AN block. So `J_ss[i,j] != J_ss[j,i]`.

That is invalid physics. `J_ij` is the interaction energy of an `(i, j)` pair; it
cannot depend on which partner is named first. The mean-field energy
`(z/2)·(J @ vars + diag(J)·vars)` assumes a symmetric coupling.

**Why the old `fit_model.py` never hit this:** it hardcoded the *same* argument
order into both cells —

```python
j01 = ... sol_sol_func(jnp.array([dn0, dn1]), p[5:10]) ...
j10 = ... sol_sol_func(jnp.array([dn0, dn1]), p[5:10]) ...   # same, not [dn1, dn0]
```

so `J` came out symmetric by fiat. The generalization to N solvents replaced
`(0, 1)` with `(i, j)` and lost that property.

---

## 2. Only two of four terms are affected

`default_J_sol_sol` is a sum of four terms. Decomposing them:

| term | form | symmetric under `i <-> j`? |
|---|---|---|
| **A** DN-AN cross `p[:5]` | `f(dn_i, an_j) + f(dn_j, an_i)` | **yes** — already a symmetric sum |
| **B** DN-DN `p[5:10]` | `g(dn_i, dn_j)` | **no** |
| **C** AN-AN `p[10:15]` | `g(an_i, an_j)` | **no** |
| **D** concentration `p[15]` | `logfunc(x_i) + logfunc(x_j)` | **yes** |

The distinction between A and B/C is the crux of the whole fix:

- In **A**, the two slots hold *genuinely different physical quantities* — a donor
  number and an acceptor number. `a1 != a2` is meaningful there, and the term is
  already written as a symmetric sum over both assignments. **Leave it alone.**
- In **B** and **C**, both slots hold *the same kind* of descriptor (two donor
  numbers, or two acceptor numbers) belonging to the two partners. `a1 != a2`
  there does not encode two physical channels; it only records *which partner was
  written first*. For interchangeable partners that is a pure labeling artifact.

Two consequences worth keeping in mind:

- The **diagonal is unaffected** by any symmetrization (`0.5*(J_ii + J_ii) == J_ii`).
  So **N=1 systems (e.g. HCE: one solvent + one anion) are bit-for-bit unchanged.**
- `J_sa` (solvent-anion) needs nothing: its slots hold different quantities, and
  the block is symmetrized structurally by `jnp.block([[J_ss, J_sa], [J_sa.T, J_aa]])`.

---

## 3. The design constraint that shaped the solution

There are three properties one might want. Treating the species list as *flat*,
you can only have two:

| scheme | symmetric | ordering-invariant | reproduces `fit_model.py` |
|---|---|---|---|
| as-shipped (no fix) | ✗ | ✓ | ✗ |
| mirror the `i<j` triangle | ✓ | ✗ | ✓ |
| average both orderings | ✓ | ✓ | ✗ |

Measured on the LHCE G4/FB pair, the same physical coupling evaluates to:

```
scheme      G4 listed first   FB listed first
none               1.430133          1.430133     (asymmetric matrix)
mirror             1.430133          1.454893     (order-dependent!)
average            1.442513          1.442513
```

A property-based canonical order (e.g. "higher DN first") does **not** rescue
this: in the training set, solvent DN > diluent DN in only **91.2%** of rows
(`DME/MPL`, `DME/DEA`, `DME/DMA` violate it), and AN runs the *opposite* way
(solvent AN < diluent AN in **85.2%**). The old convention is genuinely
*role-based* ("index 0 is the solvent"), not property-based.

**The resolution:** the trilemma is an artifact of the flat list. Once species
carry explicit group structure, "ordering-invariant" splits into two independent
requirements that can both be met:

- *in-group* ordering is meaningless (the five HEE solvents are interchangeable)
  → must be invariant;
- *cross-group* ordering is meaningful (solvent vs diluent are distinct roles)
  → must be preserved.

---

## 4. What was implemented

`symmetrize_pair_block(J, groups)` in `model.py`:

```python
grp  = jnp.asarray(groups)
same = grp[:, None] == grp[None, :]
lo   = grp[:, None] <  grp[None, :]
return jnp.where(same, 0.5 * (J + J.T),      # in-group: average
                 jnp.where(lo, J, J.T))      # cross-group: lower rank first
```

Applied to `J_ss` and `J_aa` at the **assembly point** inside `energetics`, not
inside the term functions. Two reasons:

1. Term functions receive property *values* (`dn_i`, `an_i`, `x_i`, ...), never
   indices, so a rule that depends on group rank cannot be expressed there.
2. Symmetrizing at assembly means any **custom injected** `J_sol_sol_func` is
   covered for free.

### API

`groups` — integer rank per species, length `N+M`, **solvents first then anions**.
Threaded as a static arg through `energetics` -> `equations` -> `_find_root_impl`
-> `find_root` / `get_root_error` / `li_free_energy`.

- `None` (default) = all-distinct `(0, 1, ..., N+M-1)`. This reproduces the legacy
  per-species ordering, so **existing results and LHCE reproduction are unchanged
  unless a caller opts in.**
- LHCE: `groups=(0, 1, 2)` — solvent, diluent, anion.
- HEE: `groups=(0, 0, 0, 0, 0, 1)` — five interchangeable solvents, then the salt.

`pair_symmetry` — `"group"` (default) applies the above; `"none"` restores the
pre-fix behaviour for A/B comparison.

---

## 5. Why this needs no retraining

- **N <= 2** — the only regime ever fitted. With all-distinct groups the output is
  bit-identical to `energetics_old` (verified: max |ΔJ| = **0.0**). The fit stands
  untouched; published LHCE parity plots still describe the running model.
- **N >= 3** — no fit ever constrained a third solvent's off-diagonal couplings.
  Choosing to average there is not a deviation from a fitted quantity; it is a
  choice of how to *extend* the functional form into a regime with no training
  signal. Both options are extensions; neither is "what was trained".

Monotonicity constraints also survive: the rescaler forces `p[8], p[9], p[13],
p[14]` negative, and averaging two sigmoids each decreasing in both arguments
stays decreasing in both.

**Magnitude of the change** (5-model ensemble, HEE 5-solvent mixture): switching
from as-shipped to symmetrized moves occupations by **<= 0.25 pp**, against an
ensemble spread of ~2.8 pp. Small, but systematic — and the pre-fix matrix was
not a valid pair coupling at all.

---

## 6. Verification

`tests/test_asymmetry_fix.py` (needs `pip install -e ".[dev]"` for pytest; no
environment on this cluster currently has both pytest and jax, so the assertions
were also run standalone):

| check | result |
|---|---|
| all-distinct groups reproduce `energetics_old` | max diff **0.0** |
| end-to-end `find_root` vs `find_root_old`, **real trial-25 params**, all 5 models, LHCE G4/FB/TFSI at `x_anion=0.15` | occupations agree to **<= 2.3e-10** (Broyden tolerance), `found_valid=True` both paths |
| shared group, sampled orderings | couplings **invariant** |
| all-distinct, sampled orderings | order-dependent *by design* |
| **real system**: HEE `DME-EA-MA-PP-THF` + LiTFSI at 1.0 m, 5-model ensemble, solvent list permuted — `groups=(0,0,0,0,0,1)` | occupations shift by **3.9e-10** (solver tolerance) |
| same system with `groups=(0,1,2,3,4,5)` | occupations shift by **1.3e-3** — a pure listing-order artifact |
| `J == J.T` for several groupings | passes |
| `pair_symmetry="none"` still asymmetric | passes (guards against vacuous tests) |
| N=1 identical under every grouping/mode | passes |
| diagonal preserved, off-diagonal averaged | passes |
| `groups` length validated | passes |

---

## 7. Open question — top priority for the next generalization pass

The group rule is implemented; **how groups get declared is not settled.**

Planned formulations put additives into the picture:

- LHCE + additives: `0 solvent, 1 diluent, 2 anion, >=3 additives`
- HEE + additives: `solvents, salts, additives` — interchangeable within each band

Both are expressible today by passing the right `groups` tuple, and the scheme was
checked against the LHCE+2-additive layout (`groups=(0,1,2,2)`): swapping the two
additives leaves every coupling unchanged while the solvent-diluent ordering is
preserved. What remains undecided:

1. **Declaration.** Callers currently hand-build a positional tuple, which is easy
   to misalign with the `solvents` / `anions` dicts. A role-name mapping
   (`{"solvent": 0, "diluent": 1, ...}`) or per-species `"group"` keys in the
   existing species dicts would be safer.
2. **Cross-group rank semantics.** Rank is currently an integer whose *order*
   selects which partner takes slot 1. Whether that ordering should stay
   index-based, or be derived from named roles, matters as soon as a formulation
   omits a role (e.g. an HCE with no diluent — is the anion rank 1 or 2?). Today
   the caller decides, and nothing validates the choice.
3. **Should additives be their own group at all**, or share the solvent group when
   chemically similar? This is a modeling question, not a code question.

Until (1)/(2) are settled, treat any `groups` tuple longer than 3 as a
deliberate, documented choice at the call site.
