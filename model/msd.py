import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from .snake import snake
from .weightnorm import WeightStandardizedConv
class ScaleDiscriminator(nn.Module):
    def setup(self):
        self.convs = [
            WeightStandardizedConv(16, [15], 1),
            WeightStandardizedConv(64, [41], 4, feature_group_count =4),
            WeightStandardizedConv( 256, [41], 4, feature_group_count =16),
            WeightStandardizedConv( 1024, [41], 4, feature_group_count =64),
            WeightStandardizedConv( 1024, [41], 4, feature_group_count =256),
            WeightStandardizedConv( 1024, [5], 1),
        ]
       
        self.conv_post = WeightStandardizedConv( 1, [3], 1)

    def __call__(self, x,train=True):
        fmap = []
        for l in self.convs:
            x = l(x.transpose(0,2,1)).transpose(0,2,1)
            x = nn.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        fmap.append(x)
        x = jnp.reshape(x,[x.shape[0],-1])
        return [(fmap, x)]
