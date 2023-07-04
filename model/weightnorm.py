import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence,Union,Any
def constant(value: float):
    """Constant initializer.
    """
    def init(_, shape, dtype=jnp.float32):
        # _ = key
        return jnp.full(shape, value, dtype)
    return init
class WeightStandardizedConvTranspose(nn.Module):
    """
    apply weight standardization  https://arxiv.org/abs/1903.10520
    """ 
    features: int
    kernel_size: Sequence[int] = 3
    strides: Union[None, int, Sequence[int]] = 1
    padding: Any = "SAME"
    kernel_init : Any = nn.initializers.lecun_normal()
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32


    @nn.compact
    def __call__(self, x):
        """
        Applies a weight standardized convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The convolved data.
        """
        x = x.astype(self.dtype)
        
        conv = nn.ConvTranspose(
            features=self.features, 
            kernel_size=self.kernel_size, 
            strides = self.strides,
            padding=self.padding, 
            kernel_init=self.kernel_init,
            dtype=self.dtype, 
            param_dtype = self.param_dtype,
            parent=None,precision="high")
        
        kernel_init = lambda  rng, x: conv.init(rng,x)['params']['kernel']
        bias_init = lambda  rng, x: conv.init(rng,x)['params']['bias']
        
        # standardize kernel
        kernel = self.param('kernel', kernel_init, x)
        kernel_norm = jnp.linalg.norm(kernel)
        # []
        norm = self.param('norm', constant(kernel_norm), [])
        # reduce over dim_out
        # redux = tuple(range(kernel.ndim - 1))
        # mean = jnp.mean(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        # var = jnp.var(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        standardized_kernel = norm * kernel / kernel_norm

        bias = self.param('bias',bias_init, x)

        return(conv.apply({'params': {'kernel': standardized_kernel, 'bias': bias}},x))
class WeightStandardizedConv(nn.Module):
    """
    apply weight standardization  https://arxiv.org/abs/1903.10520
    """ 
    features: int
    kernel_size: Sequence[int] = 3
    strides: Union[None, int, Sequence[int]] = 1
    padding: Any = "SAME"
    kernel_init : Any = nn.initializers.lecun_normal()
    kernel_dilation : Sequence[int] = 1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    feature_group_count : int = 1

    @nn.compact
    def __call__(self, x):
        """
        Applies a weight standardized convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The convolved data.
        """
        x = x.astype(self.dtype)
        
        conv = nn.Conv(
            features=self.features, 
            kernel_size=self.kernel_size, 
            strides = self.strides,
            padding=self.padding, 
            dtype=self.dtype, 
            kernel_dilation=self.kernel_dilation,
            param_dtype = self.param_dtype,
            kernel_init=self.kernel_init,
            feature_group_count=self.feature_group_count,
            parent=None,precision="highest")
        
        kernel_init = lambda  rng, x: conv.init(rng,x)['params']['kernel']
        bias_init = lambda  rng, x: conv.init(rng,x)['params']['bias']
        
        # standardize kernel
        kernel = self.param('kernel', kernel_init, x)
        kernel_norm = jnp.linalg.norm(kernel)
        # []
        norm = self.param('norm', constant(kernel_norm), [])
        # reduce over dim_out
        # redux = tuple(range(kernel.ndim - 1))
        # mean = jnp.mean(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        # var = jnp.var(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        standardized_kernel = norm * kernel / kernel_norm

        bias = self.param('bias',bias_init, x)

        return(conv.apply({'params': {'kernel': standardized_kernel, 'bias': bias}},x))