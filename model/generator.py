


import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from .nsf import SourceModuleHnNSF
from .bigv import AMPBlock
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from .snake import snake
from .weightnorm import WeightStandardizedConvTranspose
# from .nsf import SourceModuleHnNSF
# from .bigv import  AMPBlock#, SnakeAlias

class SpeakerAdapter(nn.Module):
    speaker_dim : int
    adapter_dim : int
    epsilon : int = 1e-5
    def setup(self):
        self.W_scale = nn.Dense(features=self.adapter_dim,kernel_init=constant_init(0.),bias_init=constant_init(1.))
        self.W_bias = nn.Dense(features=self.adapter_dim,kernel_init=constant_init(0.),bias_init=constant_init(0.))


    def __call__(self, x, speaker_embedding):
        x = x.transpose(0,2,1)
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        std = jnp.sqrt(var + self.epsilon)
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= jnp.expand_dims(scale,1)
        y += jnp.expand_dims(bias,1)
        y = y.transpose(0,2,1)
        return y



class Generator(nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    hp:tuple
    def setup(self):
        self.num_kernels = len(self.hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(self.hp.gen.upsample_rates)
        # speaker adaper, 256 should change by what speaker encoder you use
        adapter = []
        # pre conv
        self.conv_pre = nn.Conv(self.hp.gen.upsample_initial_channel, (7,), 1)
        # nsf
        # self.f0_upsamp = torch.nn.Upsample(
        self.scale_factor=np.prod(self.hp.gen.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=self.hp.audio.sampling_rate)
        noise_convs = []
        # transposed conv-based upsamplers. does not apply anti-aliasing
        ups = []
        for i, (u, k) in enumerate(zip(self.hp.gen.upsample_rates, self.hp.gen.upsample_kernel_sizes)):
            # spk
            adapter.append(SpeakerAdapter(
                256, self.hp.gen.upsample_initial_channel // (2 ** (i + 1))))
            # print(f'ups: {i} {k}, {u}, {(k - u) // 2}')
            # base
            ups.append(
                    WeightStandardizedConvTranspose(
                        self.hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        (k,),
                        (u,),
                        kernel_init=normal_init(0.01))
            )
            # nsf
            if i + 1 < len(self.hp.gen.upsample_rates):
                stride_f0 = np.prod(self.hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                noise_convs.append(
                    nn.Conv(
                        self.hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=[stride_f0 * 2],
                        strides=stride_f0,
                        #padding=stride_f0 // 2,
                    )
                )
            else:
                noise_convs.append(
                    nn.Conv(self.hp.gen.upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=[1])
                )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        resblocks = []
        self.ups=ups
        for i in range(len(self.ups)):
            ch = self.hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(self.hp.gen.resblock_kernel_sizes, self.hp.gen.resblock_dilation_sizes):
                resblocks.append(AMPBlock(ch, k, d))

        # post conv
        #self.activation_post = nn.leaky_relu(ch)
        self.conv_post = nn.Conv(1, [7], 1, use_bias=False)
        self.adapter = adapter
       
        self.resblocks = resblocks
        self.noise_convs = noise_convs
  
        #self.norms2 = nn.BatchNorm()
        #self.ups_norm = ups_norm
        # weight initialization
    def __call__(self, spk, x, f0, train=True):
        #rng = jax.random.PRNGKey(1234)
        # nsf
        f0 = f0[:, None]
        B, H, W = f0.shape
        f0 = jax.image.resize(f0, shape=(B, H, W * self.scale_factor), method='nearest').transpose(0,2,1)
        #f0 = self.f0_upsamp(f0).transpose(1, 2)
        har_source = self.m_source(f0)
        har_source = har_source.transpose(0,2,1)
        # pre conv
        # if train:
        #     #x = x + torch.randn_like(x)     # Perturbation
        #     x = x + jax.random.normal(rng,x.shape)  
            
        x = x.transpose(0,2,1)      # [B, D, L]
        x = self.conv_pre(x.transpose(0,2,1)).transpose(0,2,1)

        x = x * nn.tanh(nn.softplus(x))

        for i in range(self.num_upsamples):
            # upsampling
            x = self.ups[i](x.transpose(0,2,1)).transpose(0,2,1)

            # adapter
            x = self.adapter[i](x, spk)
            # nsf
            x_source = self.noise_convs[i](har_source.transpose(0,2,1)).transpose(0,2,1)
            x = x + x_source
            
            
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x,train=train)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x,train=train)
            x = xs / self.num_kernels
        # post conv
        #x = self.activation_post(x)
        #x = nn.leaky_relu(x)
        x = snake(x)
        #x = self.norms2(x,use_running_average=not train)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        x = nn.tanh(x)
        return x

    # def remove_weight_norm(self):
    #     for l in self.ups:
    #         remove_weight_norm(l)
    #     for l in self.resblocks:
    #         l.remove_weight_norm()
    #     remove_weight_norm(self.conv_pre)

    # def eval(self, inference=False):
    #     super(Generator, self).eval()
    #     # don't remove weight norm while validation in training loop
    #     if inference:
    #         self.remove_weight_norm()

    def inference(self, spk, ppg, f0):
        MAX_WAV_VALUE = 32768.0
        audio = self.forward(spk, ppg, f0, False)
        audio = audio.squeeze()  # collapse all dimension except time axis
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio

    def pitch2wav(self, f0):
        MAX_WAV_VALUE = 32768.0
        # nsf
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)
        har_source = self.m_source(f0)
        audio = har_source.transpose(1, 2)
        audio = audio.squeeze()  # collapse all dimension except time axis
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio

    def train_lora(self):
        print("~~~train_lora~~~")
        for p in self.parameters():
           p.requires_grad = False
        for p in self.adapter.parameters():
           p.requires_grad = True
        for p in self.resblocks.parameters():
           p.requires_grad = True