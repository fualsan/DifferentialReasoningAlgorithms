import jax
import jax.numpy as jnp


def sigmoid(x):
    return 1.0 / (1.0 + jnp.power(jnp.e, -x))


def logistic_sigmoid_relaxation_gt(x, condition, steepness):
    """
    True for points satisfying: x > condition
    """
    return sigmoid(steepness*(x-condition))


def logistic_sigmoid_relaxation_lt(x, condition, steepness):
    """
    True for points satisfying x < condition
    """
    return sigmoid(steepness*(condition-x))


def logistic_sigmoid_relaxation_interval_sub(x, condition_a, condition_b, steepness):
    """
    True for points satisfying condition_a < x < condition_b
    """
    return sigmoid(steepness * (x - condition_a)) - sigmoid(steepness * (x - condition_b))


def logistic_sigmoid_relaxation_prod(x, condition_a, condition_b, steepness):
    """
    True for points satisfying condition_a < x < condition_b
    """
    return sigmoid(steepness * (x - condition_a)) * sigmoid(steepness * (condition_b - x))