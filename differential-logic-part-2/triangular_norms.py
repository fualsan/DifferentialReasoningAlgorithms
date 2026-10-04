import jax.numpy as jnp


def t_norm_min(x1, x2):
    """
    Minimum (Gödel) T-norm
    """
    return jnp.minimum(x1, x2)


def t_norm_product(x1, x2):
    """
    Product T-norm
    """
    return x1*x2


def t_norm_bounded(x1, x2):
    """
    Bounded (Łukasiewicz) T-norm
    """
    return jnp.maximum(0.0, x1+x2-1.0)


def t_conorm_maximum(x1, x2):
    """
    Maximum (Gödel) T-conorm
    """
    return jnp.maximum(x1, x2)


def t_conorm_sum(x1, x2):
    """
    Probabilistic Sum
    """
    return x1+x2-x1*x2


def t_conorm_bounded(x1, x2):
    """
    Bounded (Łukasiewicz) T-conorm
    """
    return jnp.minimum(x1+x2, 1.0)


def negation(x):
    return 1.0-x


VALID_TNORM_FUNCTIONS = {
    'minimum': t_norm_min,
    'product': t_norm_product,
    'bounded': t_norm_bounded,
}


VALID_TCONORM_FUNCTIONS = {
    'maximum': t_conorm_maximum,
    'probabilistic_sum': t_conorm_sum,
    'bounded': t_conorm_bounded
}


def diff_and(x1, x2, name='product'):
    _name = name.lower().strip()
    if _name not in VALID_TNORM_FUNCTIONS.keys():
        raise ValueError(f'{name} is a valid AND realization, select from: {VALID_TNORM_FUNCTIONS.keys()}')
    selected_realization = VALID_TNORM_FUNCTIONS[_name]
    return selected_realization(x1, x2)


def diff_or(x1, x2, name='probabilistic_sum'):
    _name = name.lower().strip()
    if _name not in VALID_TCONORM_FUNCTIONS.keys():
        raise ValueError(f'{name} is a valid AND realization, select from: {VALID_TCONORM_FUNCTIONS.keys()}')
    selected_realization = VALID_TCONORM_FUNCTIONS[_name]
    return selected_realization(x1, x2)


def diff_not(x):
    # (SINGLE REALIZATION OF NOT, BUT CAN BE ADDED LATER)
    # (THIS IS FOR FUTURE COMPATIBILITY)
    return negation(x)


def diff_implies(x1, x2, name_tconorm):
    return diff_or(diff_not(x1), x2, name_tconorm)


def diff_equivalence(x1, x2, name_tnorm, name_tconorm):
    return diff_and(diff_implies(x1, x2, name_tconorm), diff_implies(x2, x1, name_tconorm), name_tnorm)
