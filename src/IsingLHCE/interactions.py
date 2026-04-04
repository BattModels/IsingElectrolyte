import jax
import jax.numpy as jnp

@jax.jit
def mlp(x, params):
    """
    A simple MLP.
    x is the input vector.
    """
    W1, b1, W2, b2, W3, b3 = params
    # W1, b1, W2, b2 = params_dn
    hidden1 = jax.nn.sigmoid(jnp.dot(x, W1) + b1)
    hidden2 = jax.nn.sigmoid(jnp.dot(hidden1, W2) + b2)
    output = jnp.dot(hidden2, W3) + b3
    return output

@jax.jit
def langmuirfunc(x, params):
    """
    A simple Langmuir isotherm function.
    x is the input vector.
    params is an array of parameters
    """
    fmax, n, K, x0, b = params
    output = fmax * (x - x0) ** n / (K**n + (x - x0) ** n) + b
    return output

@jax.jit
def linearfunc(x, params):
    """
    A simple linear function.
    x is the input vector.
    params is an array of parameters
    """
    a0, a1 = params
    output = a0 + a1 * x
    return output

@jax.jit
def expfunc(x, params):
    """
    A simple exp function.
    x is the input vector.
    """
    a0, a1, a2, a3 = params
    output = a0 + a1 / (1 + a2 * jnp.exp(-(a3 * x)))
    return output

@jax.jit
def logfunc(x, params):
    """
    A simple log function.
    x is the input vector.
    """
    a0 = params
    output = a0 * jnp.log(x)
    return output

@jax.jit
def sol_sol_func(X, params):
    """
    function computing solvent-solvent interaction.
    """
    dn, an = X
    L0, H0, a0, a1, a2 = params
    output = L0 + H0 / (1.0 + jnp.exp(a0 + a1 * dn + a2 * an))
    return output

@jax.jit
def polynomial_func(x, params, n=2):
    """
    A simple polynomial function.
    x is the input vector.
    params is an array of parameters
    """
    output = 0
    for i in range(n+1):
        output += params[i] * x**i
    return output


# ---------------------------------------------------------------------------
# Default h and J term functions — injectable defaults for energetics()
#
# Each function encodes one piece of the mean-field physics. Pass a
# replacement to find_root / energetics / li_free_energy to test a new
# hypothesis without touching any other code.
#
# Signatures are fixed contracts:
#   h_sol_func(props: dict, params: array) → scalar
#     props keys: 'dn' (dn_eff, conc-corrected), 'x', 'v', + any extras
#   h_anion_func(props: dict, params: array) → scalar
#     props keys: 'dn' (raw anion DN), 'x', 'v', + any extras
#   J_sol_sol_func(props_i: dict, props_j: dict, params: array) → scalar
#     props keys: 'dn' (dn_eff), 'an' (an_eff), 'x', + any extras
#   J_sol_anion_func(props_sol: dict, props_anion: dict, params: array) → scalar
#   J_anion_anion_func(props_i: dict, props_j: dict, params: array) → scalar
#
# Each default_* function is paired with a companion default_*_rescale
# that knows the sign pattern for that function's parameters.  Pass a
# custom *_rescale alongside a custom term function when injecting new
# physics — see rescale_input_params in model.py.
#
#   rescale_func(params, monotonicity: str) → params
#     monotonicity ∈ {"decrease", "increase", "none"}
# ---------------------------------------------------------------------------


def _apply_softplus(p, signs):
    """Vectorized signed-softplus transform.

    Replaces p[i] with signs[i]*softplus(p[i]) where signs[i] != 0,
    leaving p[i] unchanged where signs[i] == 0.  Single XLA op.

      signs[i] = +1  →  softplus(p[i])   (force positive)
      signs[i] = -1  → -softplus(p[i])   (force negative)
      signs[i] =  0  →  p[i]             (unconstrained)
    """
    return jnp.where(signs == 0, p, signs * jax.nn.softplus(p))

def default_h_sol(props, params):
    """Li-solvent h term: logistic function of effective DN + log of molar ratio.

    props: scalar dict with keys 'dn' (dn_eff, concentration-corrected), 'x'.
    params: sol_params_dn (5 elements) — params[:4] for expfunc, params[4] for logfunc.
    """
    return expfunc(props["dn"], params[:4]) + logfunc(props["x"], params[4])


def default_h_anion(props, params):
    """Li-anion h term: logistic function of anion DN + log of molar ratio.

    props: scalar dict with keys 'dn' (raw anion DN), 'x'.
    params: salt_params_dn (5 elements) — same layout as default_h_sol.
    """
    return expfunc(props["dn"], params[:4]) + logfunc(props["x"], params[4])


def default_J_sol_sol(props_i, props_j, params):
    """Solvent-solvent J term.

    A single unified formula that correctly handles both the self-interaction
    (i == j) and cross-interaction (i != j) cases: when i == j the two
    sol_sol_func DN-AN calls become identical (doubling) and the two logfunc
    calls collapse to 2*logfunc — matching the original explicit branching.

    props_i, props_j: scalar dicts with keys 'dn' (dn_eff), 'an' (an_eff), 'x'.
    params: params_sol_sol (16 elements)
      [:5]   DN-AN cross term
      [5:10] DN-DN term
      [10:15] AN-AN term
      [15]   logfunc concentration term
    """
    return (
        sol_sol_func(jnp.array([props_i["dn"], props_j["an"]]), params[:5])
        + sol_sol_func(jnp.array([props_j["dn"], props_i["an"]]), params[:5])
        + sol_sol_func(jnp.array([props_i["dn"], props_j["dn"]]), params[5:10])
        + sol_sol_func(jnp.array([props_i["an"], props_j["an"]]), params[10:15])
        + logfunc(props_i["x"], params[15])
        + logfunc(props_j["x"], params[15])
    )


def default_J_sol_anion(props_sol, props_anion, params):
    """Solvent-anion J term.

    props_sol: scalar dict with keys 'an' (an_eff), 'x'.
    props_anion: scalar dict with keys 'dn' (raw anion DN), 'x'.
    params: params_sol_salt_an (6 elements)
      [:5]  sol_sol_func term (anion DN × solvent AN)
      [5]   logfunc concentration term (applied to both x_sol and x_an)
    """
    return (
        sol_sol_func(jnp.array([props_anion["dn"], props_sol["an"]]), params[:5])
        + logfunc(props_sol["x"], params[5])
        + logfunc(props_anion["x"], params[5])
    )


def default_J_anion_anion(props_i, props_j, params):
    """Anion-anion J term.

    Unified formula for both self (i == j) and cross (i != j) interactions
    via sol_sol_func with both anion DNs as inputs.

    props_i, props_j: scalar dicts with keys 'dn', 'x'.
    params: params_anion_anion (6 elements)
      [:5]  sol_sol_func term (DN_i × DN_j)
      [5]   logfunc concentration term (applied to both x_i and x_j)
    """
    return (
        sol_sol_func(jnp.array([props_i["dn"], props_j["dn"]]), params[:5])
        + logfunc(props_i["x"], params[5])
        + logfunc(props_j["x"], params[5])
    )


# Keep old names as aliases for backwards compatibility
default_h_an = default_h_anion
default_J_sol_an = default_J_sol_anion
default_J_an_an = default_J_anion_anion


# ---------------------------------------------------------------------------
# Companion rescaling functions
#
# Each default_*_rescale(params, monotonicity) applies signed softplus to the
# parameter array for its term function.  The sign pattern encodes the physical
# constraint — which indices must be positive/negative to enforce the requested
# monotonicity direction.
#
# When you define a custom term function with different parameter semantics,
# define a matching custom_*_rescale and pass both to find_root / energetics.
# ---------------------------------------------------------------------------

def default_h_sol_rescale(params, monotonicity):
    """Rescaling for default_h_sol — 5 params: expfunc[:4] + logfunc[4].

    "decrease": h decreases with DN — params[3], params[4] forced negative.
    "increase": h increases with DN — params[3] forced positive.
    "none":     no constraint applied.
    """
    if monotonicity == "decrease":
        signs = jnp.array([ 0, +1, +1, -1, -1])
    elif monotonicity == "increase":
        signs = jnp.array([ 0, +1, +1, +1, -1])
    else:
        signs = jnp.zeros(5, dtype=int)
    return _apply_softplus(params, signs)


def default_h_anion_rescale(params, monotonicity):
    """Rescaling for default_h_anion — 5 params: same layout as default_h_sol."""
    if monotonicity == "decrease":
        signs = jnp.array([ 0, +1, +1, -1, -1])
    elif monotonicity == "increase":
        signs = jnp.array([ 0, +1, +1, +1, -1])
    else:
        signs = jnp.zeros(5, dtype=int)
    return _apply_softplus(params, signs)


# Backwards-compatible alias
default_h_an_rescale = default_h_anion_rescale


def default_J_sol_sol_rescale(params, monotonicity):
    """Rescaling for default_J_sol_sol — 16 params.

    Layout: [:5] DN-AN cross, [5:10] DN-DN, [10:15] AN-AN, [15] logfunc.
    "decrease": DN-AN cross decreasing; DN-DN and AN-AN increasing.
    "increase": all sub-terms increasing.
    """
    if monotonicity == "decrease":
        signs = jnp.array([ 0, +1, 0, +1, +1,  0, +1, 0, -1, -1,  0, +1, 0, -1, -1, -1])
    elif monotonicity == "increase":
        signs = jnp.array([ 0, +1, 0, -1, -1,  0, +1, 0, -1, -1,  0, +1, 0, -1, -1, -1])
    else:
        signs = jnp.zeros(16, dtype=int)
    return _apply_softplus(params, signs)


def default_J_sol_anion_rescale(params, monotonicity):
    """Rescaling for default_J_sol_anion — 6 params: sol_sol_func[:5] + logfunc[5].

    "decrease": J decreases with DN and AN.
    "increase": J increases with DN and AN.
    """
    if monotonicity == "decrease":
        signs = jnp.array([ 0, +1, 0, +1, +1, -1])
    elif monotonicity == "increase":
        signs = jnp.array([ 0, +1, 0, -1, -1, -1])
    else:
        signs = jnp.zeros(6, dtype=int)
    return _apply_softplus(params, signs)


def default_J_anion_anion_rescale(params, monotonicity):
    """Rescaling for default_J_anion_anion — 6 params: sol_sol_func[:5] + logfunc[5].

    "increase": J increases with DN (default physics for anion-anion).
    "decrease": J decreases with DN.
    """
    if monotonicity == "increase":
        signs = jnp.array([ 0, +1, 0, -1, -1, -1])
    elif monotonicity == "decrease":
        signs = jnp.array([ 0, +1, 0, +1, +1, -1])
    else:
        signs = jnp.zeros(6, dtype=int)
    return _apply_softplus(params, signs)


# Backwards-compatible aliases
default_J_sol_an_rescale = default_J_sol_anion_rescale
default_J_an_an_rescale = default_J_anion_anion_rescale


def default_conc_factor_rescale(params, monotonicity):
    """Rescaling for conc_factor_sol — 8 params.

    "increase": concentration factor increases with x and V (default).
    "decrease": concentration factor decreases with x and V.
    """
    if monotonicity == "increase":
        signs = jnp.array([ 0, -1, +1, -1,  0, +1, +1, +1])
    elif monotonicity == "decrease":
        signs = jnp.array([ 0, +1, +1, +1,  0, +1, +1, +1])
    else:
        signs = jnp.zeros(8, dtype=int)
    return _apply_softplus(params, signs)