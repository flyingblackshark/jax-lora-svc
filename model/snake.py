import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from functools import partial
import operator
import warnings
import numpy as np
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax import lax

Array = Any

@jax.jit
def snake(x: Array, frequency: int = 1) -> Array:

    r"""Snake activation to learn periodic functions.

    Computes snake activation:

    $$
    \mathrm{snake}(x) = \mathrm{x} + \frac{1 - \cos(2 \cdot \mathrm{frequency} \cdot x)}{2 \cdot \mathrm{frequency}}.
    $$

    See [Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195).

    Usage:

    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> tfa.activations.snake(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.29192656,  0.        ,  1.7080734 ], dtype=float32)>

    Args:
        x: A `Tensor`.
        frequency: A scalar, frequency of the periodic part.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    #x = tf.convert_to_tensor(x)
    #frequency = tf.cast(frequency, x.dtype)

    return x + (1 - jnp.cos(2 * frequency * x)) / (2 * frequency)