import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
#from .alias.act import SnakeAlias

from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from .snake import snake
from .weightnorm import WeightStandardizedConv

class AMPBlock(nn.Module):
    channels:int
    kernel_size:int=3
    dilation:tuple=(1, 3, 5)
    def setup(self):
       
        self.convs1 =[
            WeightStandardizedConv(self.channels,[ self.kernel_size], 1, kernel_dilation=self.dilation[0],kernel_init=normal_init(0.01)),
            WeightStandardizedConv( self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[1],kernel_init=normal_init(0.01)
                               ),
            WeightStandardizedConv( self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[2],kernel_init=normal_init(0.01)
                               )
        ]


        self.convs2 = [
            WeightStandardizedConv( self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=normal_init(0.01)
                               ),
            WeightStandardizedConv( self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=normal_init(0.01)
                               ),
            WeightStandardizedConv(self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=normal_init(0.01)
                               )
        ]
        # total number of conv layers
        self.num_layers = len(self.convs1) + len(self.convs2)

    def __call__(self, x,train=True):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = snake(x)
            xt = c1(xt.transpose(0,2,1)).transpose(0,2,1)
            xt = snake(xt)
            xt = c2(xt.transpose(0,2,1)).transpose(0,2,1)
            x = xt + x
        return x
