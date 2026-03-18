import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve, root
import jax, optax, os
import jax.numpy as jnp
from jaxopt import Broyden
from jax import grad
import pickle
import copy
from jax import random
from functools import partial
from IsingLHCE.functions import (
    expfunc,
    polynomial_func,
)

@jax.jit
def conc_factor_sep_x_v_sigmoid(x0, x_ref, V0, V_ref, params):
    output = expfunc((x0 / x_ref), params = params[:4]) * expfunc((V_ref / V0), params = params[4:8])
    return output

@jax.jit
def conc_factor_sep_x_v_taylor(x0, x_ref, V0, V_ref, params):
    output = polynomial_func((x0 / x_ref), params = params[:3], n=2) * polynomial_func((V_ref / V0), params = params[3:6], n=2)
    return output

@jax.jit
def conc_factor_power(x0, x_ref, V0, V_ref, params):
    a, b = params
    output = (x0 / x_ref) ** a * (V_ref / V0) ** b
    return output

@jax.jit
def conc_factor_wrapped_sigmoid(x0, x_ref, V0, V_ref, params):
    """
    Wrapped x and v into same input phi and pass to sigmoid function
    """
    phi = (x0 / x_ref) * (V_ref / V0)
    output = expfunc(phi, params = params)
    return output