# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from .snake import snake
from .weightnorm import WeightStandardizedConv
class DiscriminatorP(nn.Module):
    hp:tuple
    period:tuple
    def setup(self):
        self.LRELU_SLOPE = self.hp.mpd.lReLU_slope

        kernel_size = self.hp.mpd.kernel_size
        stride = self.hp.mpd.stride
      

        self.convs = [
            WeightStandardizedConv(64, (kernel_size, 1), (stride, 1)),
            WeightStandardizedConv( 128, (kernel_size, 1), (stride, 1)),
            WeightStandardizedConv( 256, (kernel_size, 1), (stride, 1)),
            WeightStandardizedConv( 512, (kernel_size, 1), (stride, 1)),
            WeightStandardizedConv( 1024, (kernel_size, 1), 1),
        ]
        #self.norms = [nn.BatchNorm() for i in range(5)]
        self.conv_post = WeightStandardizedConv(1, (3, 1), 1)

    def __call__(self, x,train=True):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = jnp.pad(x, [(0,0),(0,0),(0, n_pad)], "reflect")
            t = t + n_pad
        x = jnp.reshape(x,[b, c, t // self.period, self.period])

        for l in self.convs:
            x = l(x.transpose(0,2,3,1)).transpose(0,3,1,2)
            x = nn.leaky_relu(x, self.LRELU_SLOPE)
            #x = snake(x)
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,3,1)).transpose(0,3,1,2)
        fmap.append(x)
        #x = torch.flatten(x, 1, -1)
        x = jnp.reshape(x,[x.shape[0],-1])

        return fmap, x


class MultiPeriodDiscriminator(nn.Module):
    hp:tuple
    def setup(self):
       # super(MultiPeriodDiscriminator, self).__init__()

        self.discriminators = [DiscriminatorP(self.hp, period) for period in self.hp.mpd.periods]
        

    def __call__(self, x,train=True):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x,train=train))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]
