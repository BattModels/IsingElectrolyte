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
#   h_sol_func(dn_eff, x, params)             → scalar
#   h_an_func(dn_an, x, params)               → scalar
#   J_sol_sol_func(dn_i, an_i, x_i,
#                  dn_j, an_j, x_j, params)   → scalar
#   J_sol_an_func(dn_an, an_sol, x_sol,
#                 x_an, params)               → scalar
#   J_an_an_func(dn_i, x_i, dn_j, x_j,
#                params)                      → scalar
# ---------------------------------------------------------------------------

def default_h_sol(dn_eff, x, params):
    """Li-solvent h term: logistic function of effective DN + log of molar ratio.

    params: sol_params_dn (5 elements) — params[:4] for expfunc, params[4] for logfunc.
    """
    return expfunc(dn_eff, params[:4]) + logfunc(x, params[4])


def default_h_an(dn_an, x, params):
    """Li-anion h term: logistic function of anion DN + log of molar ratio.

    params: salt_params_dn (5 elements) — same layout as default_h_sol.
    """
    return expfunc(dn_an, params[:4]) + logfunc(x, params[4])


def default_J_sol_sol(dn_i, an_i, x_i, dn_j, an_j, x_j, params):
    """Solvent-solvent J term.

    A single unified formula that correctly handles both the self-interaction
    (i == j) and cross-interaction (i != j) cases: when i == j the two
    sol_sol_func DN-AN calls become identical (doubling) and the two logfunc
    calls collapse to 2*logfunc — matching the original explicit branching.

    params: params_sol_sol (16 elements)
      [:5]   DN-AN cross term
      [5:10] DN-DN term
      [10:15] AN-AN term
      [15]   logfunc concentration term
    """
    return (
        sol_sol_func(jnp.array([dn_i, an_j]), params[:5])
        + sol_sol_func(jnp.array([dn_j, an_i]), params[:5])
        + sol_sol_func(jnp.array([dn_i, dn_j]), params[5:10])
        + sol_sol_func(jnp.array([an_i, an_j]), params[10:15])
        + logfunc(x_i, params[15])
        + logfunc(x_j, params[15])
    )


def default_J_sol_an(dn_an, an_sol, x_sol, x_an, params):
    """Solvent-anion J term.

    params: params_sol_salt_an (6 elements)
      [:5]  sol_sol_func term (anion DN × solvent AN)
      [5]   logfunc concentration term (applied to both x_sol and x_an)
    """
    return (
        sol_sol_func(jnp.array([dn_an, an_sol]), params[:5])
        + logfunc(x_sol, params[5])
        + logfunc(x_an, params[5])
    )


def default_J_an_an(dn_i, x_i, dn_j, x_j, params):
    """Anion-anion J term.

    Unified formula for both self (i == j) and cross (i != j) interactions
    via sol_sol_func with both anion DNs as inputs.

    params: params_anion_anion (6 elements)
      [:5]  sol_sol_func term (DN_i × DN_j)
      [5]   logfunc concentration term (applied to both x_i and x_j)
    """
    return (
        sol_sol_func(jnp.array([dn_i, dn_j]), params[:5])
        + logfunc(x_i, params[5])
        + logfunc(x_j, params[5])
    )