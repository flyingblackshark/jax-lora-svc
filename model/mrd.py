import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from .snake import snake
from .weightnorm import WeightStandardizedConv
class DiscriminatorR(nn.Module):
    resolution:tuple
    hp:tuple
    def setup(self):
        self.LRELU_SLOPE = self.hp.mpd.lReLU_slope
        self.convs = [
            WeightStandardizedConv( 32, (3, 9)),
            WeightStandardizedConv( 32, (3, 9), strides=(1, 2)),
            WeightStandardizedConv( 32, (3, 9), strides=(1, 2)),
            WeightStandardizedConv( 32, (3, 9), strides=(1, 2)),
            WeightStandardizedConv( 32, (3, 3)),
        ]
        self.conv_post = WeightStandardizedConv( 1, (3, 3))
        
        

    def __call__(self, x,train=True):
        fmap = []

        x = self.spectrogram(x)
        x = jnp.expand_dims(x,1)
        for l in self.convs:
            x = l(x.transpose(0,2,3,1)).transpose(0,3,1,2)
            x = nn.leaky_relu(x, self.LRELU_SLOPE)
            #x = snake(x)
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,3,1)).transpose(0,3,1,2)
        fmap.append(x)
        x = jnp.reshape(x, [x.shape[0],-1])

        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = jnp.pad(x, [(0,0),(0,0),(int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2))], mode='reflect')
        x = x.squeeze(1)
        x = jax.scipy.signal.stft(x,fs=32000, nfft=n_fft, noverlap=win_length-hop_length, nperseg=win_length) #[B, F, TT, 2]
        mag = jnp.abs(x[2])
        return mag


class MultiResolutionDiscriminator(nn.Module):
    hp:tuple
    def setup(self):
        self.resolutions = eval(self.hp.mrd.resolutions)
        self.discriminators = [DiscriminatorR(resolution,self.hp) for resolution in self.resolutions]
        

    def __call__(self, x,train=True):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x,train=train))

        return ret  # [(feat, score), (feat, score), (feat, score)]
