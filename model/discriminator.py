import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from omegaconf import OmegaConf
from .msd import ScaleDiscriminator
from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator


class Discriminator(nn.Module):
    hp:tuple
    def setup(self):
        self.MRD = MultiResolutionDiscriminator(self.hp)
        self.MPD = MultiPeriodDiscriminator(self.hp)
        self.MSD = ScaleDiscriminator()

    def __call__(self, x,train=True):
        r = self.MRD(x,train=train)
        p = self.MPD(x,train=train)
        s = self.MSD(x,train=train)
        return r + p + s


# if __name__ == '__main__':
#     hp = OmegaConf.load('../config/maxgan.yaml')
#     model = Discriminator(hp)

#     x = torch.randn(3, 1, 16384)
#     print(x.shape)

#     output = model(x)
#     for features, score in output:
#         for feat in features:
#             print(feat.shape)
#         print(score.shape)

#     pytorch_total_params = sum(p.numel()
#                                for p in model.parameters() if p.requires_grad)
#     print(pytorch_total_params)
