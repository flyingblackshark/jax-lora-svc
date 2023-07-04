#import torch
import numpy as np
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn
#import torch.nn.functional as torch_nn_func


# class PulseGen(torch.nn.Module):
#     """Definition of Pulse train generator

#     There are many ways to implement pulse generator.
#     Here, PulseGen is based on SinGen. For a perfect
#     """

#     def __init__(self, samp_rate, pulse_amp=0.1, noise_std=0.003, voiced_threshold=0):
#         super(PulseGen, self).__init__()
#         self.pulse_amp = pulse_amp
#         self.sampling_rate = samp_rate
#         self.voiced_threshold = voiced_threshold
#         self.noise_std = noise_std
#         self.l_sinegen = SineGen(
#             self.sampling_rate,
#             harmonic_num=0,
#             sine_amp=self.pulse_amp,
#             noise_std=0,
#             voiced_threshold=self.voiced_threshold,
#             flag_for_pulse=True,
#         )

#     def forward(self, f0):
#         """Pulse train generator
#         pulse_train, uv = forward(f0)
#         input F0: tensor(batchsize=1, length, dim=1)
#                   f0 for unvoiced steps should be 0
#         output pulse_train: tensor(batchsize=1, length, dim)
#         output uv: tensor(batchsize=1, length, 1)

#         Note: self.l_sine doesn't make sure that the initial phase of
#         a voiced segment is np.pi, the first pulse in a voiced segment
#         may not be at the first time step within a voiced segment
#         """
#         with torch.no_grad():
#             sine_wav, uv, noise = self.l_sinegen(f0)

#             # sine without additive noise
#             pure_sine = sine_wav - noise

#             # step t corresponds to a pulse if
#             # sine[t] > sine[t+1] & sine[t] > sine[t-1]
#             # & sine[t-1], sine[t+1], and sine[t] are voiced
#             # or
#             # sine[t] is voiced, sine[t-1] is unvoiced
#             # we use torch.roll to simulate sine[t+1] and sine[t-1]
#             sine_1 = torch.roll(pure_sine, shifts=1, dims=1)
#             uv_1 = torch.roll(uv, shifts=1, dims=1)
#             uv_1[:, 0, :] = 0
#             sine_2 = torch.roll(pure_sine, shifts=-1, dims=1)
#             uv_2 = torch.roll(uv, shifts=-1, dims=1)
#             uv_2[:, -1, :] = 0

#             loc = (pure_sine > sine_1) * (pure_sine > sine_2) \
#                   * (uv_1 > 0) * (uv_2 > 0) * (uv > 0) \
#                   + (uv_1 < 1) * (uv > 0)

#             # pulse train without noise
#             pulse_train = pure_sine * loc

#             # additive noise to pulse train
#             # note that noise from sinegen is zero in voiced regions
#             pulse_noise = torch.randn_like(pure_sine) * self.noise_std

#             # with additive noise on pulse, and unvoiced regions
#             pulse_train += pulse_noise * loc + pulse_noise * (1 - uv)
#         return pulse_train, sine_wav, uv, pulse_noise


# class SignalsConv1d(torch.nn.Module):
#     """Filtering input signal with time invariant filter
#     Note: FIRFilter conducted filtering given fixed FIR weight
#           SignalsConv1d convolves two signals
#     Note: this is based on torch.nn.functional.conv1d

#     """

#     def __init__(self):
#         super(SignalsConv1d, self).__init__()

#     def forward(self, signal, system_ir):
#         """output = forward(signal, system_ir)

#         signal:    (batchsize, length1, dim)
#         system_ir: (length2, dim)

#         output:    (batchsize, length1, dim)
#         """
#         if signal.shape[-1] != system_ir.shape[-1]:
#             print("Error: SignalsConv1d expects shape:")
#             print("signal    (batchsize, length1, dim)")
#             print("system_id (batchsize, length2, dim)")
#             print("But received signal: {:s}".format(str(signal.shape)))
#             print(" system_ir: {:s}".format(str(system_ir.shape)))
#             sys.exit(1)
#         padding_length = system_ir.shape[0] - 1
#         groups = signal.shape[-1]

#         # pad signal on the left
#         signal_pad = torch_nn_func.pad(signal.permute(0, 2, 1), (padding_length, 0))
#         # prepare system impulse response as (dim, 1, length2)
#         # also flip the impulse response
#         ir = torch.flip(system_ir.unsqueeze(1).permute(2, 1, 0), dims=[2])
#         # convolute
#         output = torch_nn_func.conv1d(signal_pad, ir, groups=groups)
#         return output.permute(0, 2, 1)


# class CyclicNoiseGen_v1(torch.nn.Module):
#     """CyclicnoiseGen_v1
#     Cyclic noise with a single parameter of beta.
#     Pytorch v1 implementation assumes f_t is also fixed
#     """

#     def __init__(self, samp_rate, noise_std=0.003, voiced_threshold=0):
#         super(CyclicNoiseGen_v1, self).__init__()
#         self.samp_rate = samp_rate
#         self.noise_std = noise_std
#         self.voiced_threshold = voiced_threshold

#         self.l_pulse = PulseGen(
#             samp_rate,
#             pulse_amp=1.0,
#             noise_std=noise_std,
#             voiced_threshold=voiced_threshold,
#         )
#         self.l_conv = SignalsConv1d()

#     def noise_decay(self, beta, f0mean):
#         """decayed_noise = noise_decay(beta, f0mean)
#         decayed_noise =  n[t]exp(-t * f_mean / beta / samp_rate)

#         beta: (dim=1) or (batchsize=1, 1, dim=1)
#         f0mean (batchsize=1, 1, dim=1)

#         decayed_noise (batchsize=1, length, dim=1)
#         """
#         with torch.no_grad():
#             # exp(-1.0 n / T) < 0.01 => n > -log(0.01)*T = 4.60*T
#             # truncate the noise when decayed by -40 dB
#             length = 4.6 * self.samp_rate / f0mean
#             length = length.int()
#             time_idx = torch.arange(0, length, device=beta.device)
#             time_idx = time_idx.unsqueeze(0).unsqueeze(2)
#             time_idx = time_idx.repeat(beta.shape[0], 1, beta.shape[2])

#         noise = torch.randn(time_idx.shape, device=beta.device)

#         # due to Pytorch implementation, use f0_mean as the f0 factor
#         decay = torch.exp(-time_idx * f0mean / beta / self.samp_rate)
#         return noise * self.noise_std * decay

#     def forward(self, f0s, beta):
#         """Producde cyclic-noise"""
#         # pulse train
#         pulse_train, sine_wav, uv, noise = self.l_pulse(f0s)
#         pure_pulse = pulse_train - noise

#         # decayed_noise (length, dim=1)
#         if (uv < 1).all():
#             # all unvoiced
#             cyc_noise = torch.zeros_like(sine_wav)
#         else:
#             f0mean = f0s[uv > 0].mean()

#             decayed_noise = self.noise_decay(beta, f0mean)[0, :, :]
#             # convolute
#             cyc_noise = self.l_conv(pure_pulse, decayed_noise)

#         # add noise in invoiced segments
#         cyc_noise = cyc_noise + noise * (1.0 - uv)
#         return cyc_noise, pulse_train, sine_wav, uv, noise


class SineGen(nn.Module):
    
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)

    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)

    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """
    samp_rate:int
    harmonic_num:int=0
    sine_amp:float=0.1
    noise_std:float=0.003
    voiced_threshold:int=0
    flag_for_pulse:bool=False
    def setup(self):
        self.dim = self.harmonic_num + 1
        self.sampling_rate = self.samp_rate

    def _f02uv(self, f0):
        # generate uv signal
        uv = jnp.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        rng=jax.random.PRNGKey(1234)
        # initial phase noise (no noise for fundamental component)
        rand_ini = jax.random.uniform(rng,
           [f0_values.shape[0], f0_values.shape[2]]
        )
        rand_ini=rand_ini.at[:, 0].set(0)
        rad_values=rad_values.at[:, 0, :].set(rad_values[:, 0, :] + rand_ini)

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            tmp_over_one = jnp.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = jnp.zeros_like(rad_values)
            cumsum_shift=cumsum_shift.at[:, 1:, :].set(tmp_over_one_idx * -1.0)

            sines = jnp.sin(
                jnp.cumsum(rad_values + cumsum_shift, axis=1) * 2 * np.pi
            )
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = jnp.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = jnp.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = jnp.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = jnp.cos(i_phase * 2 * np.pi)
        return sines

    def __call__(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = jnp.zeros([f0.shape[0], f0.shape[1], self.dim])
        # fundamental component
        f0_buf=f0_buf.at[:, :, 0].set(f0[:, :, 0])
        for idx in np.arange(self.harmonic_num):
            # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
            f0_buf=f0_buf.at[:, :, idx + 1].set(f0_buf[:, :, 0] * (idx + 2))

        # generate sine waveforms
        sine_waves = self._f02sine(f0_buf) * self.sine_amp

        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        rng=jax.random.PRNGKey(1234)
        noise = noise_amp * jax.random.normal(rng,sine_waves.shape)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        # jax.debug.print("{}",sine_waves.shape)
        # jax.debug.print("{}",uv.shape)
        # jax.debug.print("{}",noise.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves


# class SourceModuleCycNoise_v1(torch.nn.Module):
#     """SourceModuleCycNoise_v1
#     SourceModule(sampling_rate, noise_std=0.003, voiced_threshod=0)
#     sampling_rate: sampling_rate in Hz

#     noise_std: std of Gaussian noise (default: 0.003)
#     voiced_threshold: threshold to set U/V given F0 (default: 0)

#     cyc, noise, uv = SourceModuleCycNoise_v1(F0_upsampled, beta)
#     F0_upsampled (batchsize, length, 1)
#     beta (1)
#     cyc (batchsize, length, 1)
#     noise (batchsize, length, 1)
#     uv (batchsize, length, 1)
#     """

#     def __init__(self, sampling_rate, noise_std=0.003, voiced_threshod=0):
#         super(SourceModuleCycNoise_v1, self).__init__()
#         self.sampling_rate = sampling_rate
#         self.noise_std = noise_std
#         self.l_cyc_gen = CyclicNoiseGen_v1(sampling_rate, noise_std, voiced_threshod)

#     def forward(self, f0_upsamped, beta):
#         """
#         cyc, noise, uv = SourceModuleCycNoise_v1(F0, beta)
#         F0_upsampled (batchsize, length, 1)
#         beta (1)
#         cyc (batchsize, length, 1)
#         noise (batchsize, length, 1)
#         uv (batchsize, length, 1)
#         """
#         # source for harmonic branch
#         cyc, pulse, sine, uv, add_noi = self.l_cyc_gen(f0_upsamped, beta)

#         # source for noise branch, in the same shape as uv
#         noise = torch.randn_like(uv) * self.noise_std / 3
#         return cyc, noise, uv


class SourceModuleHnNSF(nn.Module):
    sampling_rate:int=32000
    sine_amp:float=0.1
    add_noise_std:float=0.003
    voiced_threshod:int=0
    def setup(self):
        #super(SourceModuleHnNSF, self).__init__()
        harmonic_num = 8
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            self.sampling_rate, harmonic_num, self.sine_amp, self.add_noise_std, self.voiced_threshod
        )

        # to merge source harmonics into a single excitation
        #self.l_tanh = nn.tanh()
        self.merge_w=np.asarray([
            -0.1044, -0.4892, -0.4733, 0.4337, -0.2321,
           -0.1889, 0.1315, -0.1002, 0.0590,])
        self.merge_b=np.asarray([-0.2908])

    def __call__(self, x):
        """
        Sine_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        """
        # source for harmonic branch
        #jax.debug.print("{}",x.shape)
        sine_wavs = jax.lax.stop_gradient(self.l_sin_gen(x))
        sine_wavs = jnp.matmul(sine_wavs,jnp.transpose(self.merge_w)) + self.merge_b
        sine_wavs = jnp.expand_dims(sine_wavs,-1)
        sine_merge = nn.tanh(sine_wavs)#self.l_tanh(sine_wavs)
        return sine_merge
