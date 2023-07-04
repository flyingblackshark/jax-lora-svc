import os
import torch
import argparse
import numpy as np

from omegaconf import OmegaConf
from scipy.io.wavfile import write
from model.generator import Generator
from pitch import load_csv_pitch
import os
import time
import logging
import math
import tqdm
from flax.training import orbax_utils
import itertools
import traceback
import flax
import jax
import optax
import numpy as np
import orbax
from flax import linen as nn
from utils.dataloader import create_dataloader

from utils.writer import MyWriter
from utils.stft import TacotronSTFT
from utils.stft_loss import MultiResolutionSTFTLoss
from model.generator import Generator
from model.discriminator import Discriminator
import jax.numpy as jnp
from functools import partial
from typing import Any, Tuple
from flax.training import train_state
from flax.training.common_utils import shard, shard_prng_key
import torch
from flax.training import orbax_utils
from model.generator import Generator
from flax.training import train_state
class TrainState(train_state.TrainState):
    batch_stats: Any

# def load_svc_model(checkpoint_path, model):
#     assert os.path.isfile(checkpoint_path)
#     checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
#     saved_state_dict = checkpoint_dict["model_g"]
#     state_dict = model.state_dict()
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         try:
#             new_state_dict[k] = saved_state_dict[k]
#         except:
#             print("%s is not in the checkpoint" % k)
#             new_state_dict[k] = v
#     model.load_state_dict(new_state_dict)
#     return model


def main(args):
    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        os.system(f"python3 whisper/inference.py -w {args.wave} -p {args.ppg}")

    if (args.pit == None):
        args.pit = "svc_tmp.pit.csv"
        print(
            f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
        os.system(f"python3 pitch/inference.py -w {args.wave} -p {args.pit}")

    hp = OmegaConf.load(args.config)
    
    #@partial(jax.pmap, static_broadcasted_argnums=(1))
    def create_generator_state(rng, model_cls): 
        r"""Create the training state given a model class. """ 
        model = model_cls(hp=hp)
        tx = optax.lion(learning_rate=hp.train.adam.lr, b1=hp.train.adam.beta1,b2=hp.train.adam.beta2)
        fake_ppg = jnp.ones((hp.train.batch_size,40,1280))
        fake_pit = jnp.ones((hp.train.batch_size,40))
        fake_spk = jnp.ones((hp.train.batch_size,256))

        variables = model.init(rng, fake_spk,fake_ppg,fake_pit,train=False)

        state = TrainState.create(apply_fn=model.apply, tx=tx, 
            params=variables['params'],batch_stats=variables['batch_stats'])
        
        return state
    #@partial(jax.pmap, static_broadcasted_argnums=(1))
    def create_discriminator_state(rng, model_cls): 
        r"""Create the training state given a model class. """ 
        model = model_cls(hp=hp)
        fake_audio = jnp.ones((hp.train.batch_size,1,12800))
        tx = optax.lion(learning_rate=hp.train.adam.lr, b1=hp.train.adam.beta1,b2=hp.train.adam.beta2)
        variables = model.init(rng, fake_audio,train=False)
       
        state = TrainState.create(apply_fn=model.apply, tx=tx, 
            params=variables['params'], batch_stats=variables['batch_stats'])
        
        return state
    key = jax.random.PRNGKey(seed=1234)
    key_generator, key_discriminator, key = jax.random.split(key, 3)
    #key_generator = shard_prng_key(key_generator)
    #key_discriminator = shard_prng_key(key_discriminator)
    discriminator_state = create_discriminator_state(key_discriminator, Discriminator)
    generator_state = create_generator_state(key_generator, Generator)
    
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/lora-svc/', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'model_g': generator_state, 'model_d': discriminator_state}
        step = checkpoint_manager.latest_step()  # step = 4
        states=checkpoint_manager.restore(step,items=target)
        discriminator_state=states['model_d']
        generator_state=states['model_g']
    generator_state = flax.jax_utils.unreplicate(generator_state)
    # load_svc_model(args.model, model)
    # model.eval()
    # model.to(device)

    spk = np.load(args.spk)
    #spk = torch.FloatTensor(spk)
    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    #ppg = torch.FloatTensor(ppg)
    pit = np.array(load_csv_pitch(args.pit))
   
    # print("pitch shift: ", args.shift)
    # if (args.shift == 0):
    #     pass
    # else:

    #     pit = np.array(pit)
    #     source = pit[pit > 0]
    #     source_ave = source.mean()
    #     source_min = source.min()
    #     source_max = source.max()
    #     print(f"source pitch statics: mean={source_ave:0.1f}, \
    #             min={source_min:0.1f}, max={source_max:0.1f}")
    #     shift = args.shift
    #     shift = 2 ** (shift / 12)
    #     pit = pit * shift

    #pit = torch.FloatTensor(pit)

    len_pit = pit.shape[0]
    len_ppg = ppg.shape[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min]
    ppg = ppg[:len_min, :]
    pit=np.expand_dims(pit,0)
    ppg=np.expand_dims(ppg,0)
    spk=np.expand_dims(spk,0)
    #spk = np.broadcast_to(spk,[8,spk.shape[1]])
    #ppg = np.broadcast_to(ppg,[8,ppg.shape[1],ppg.shape[2]])
    #pit = np.broadcast_to(pit,[8,pit.shape[1]])
    #spk = shard(spk)
    #ppg = shard(ppg)
    #pit = shard(pit)
    
    #@partial(jax.pmap, axis_name='num_devices')   
    #def do_infer(generator: TrainState,ppg_val:jnp.ndarray,pit_val:jnp.ndarray,spk_val:jnp.ndarray):   
    audio = generator_state.apply_fn( {'params': generator_state.params,'batch_stats': generator_state.batch_stats},spk,ppg,pit,train=False, mutable=False)
    #return audio
    #audio = do_infer(generator_state,ppg,pit,spk)
    audio = np.asarray(audio)
    # with torch.no_grad():
    #     spk = spk.unsqueeze(0).to(device)
    #     ppg = ppg.unsqueeze(0).to(device)
    #     pit = pit.unsqueeze(0).to(device)
    #     audio = model.inference(spk, ppg, pit)
    #     audio = audio.cpu().detach().numpy()

    #     pitwav = model.pitch2wav(pit)
    #     pitwav = pitwav.cpu().detach().numpy()


    write("svc_out.wav", hp.audio.sampling_rate, audio)
    #write("svc_out_pitch.wav", hp.audio.sampling_rate, pitwav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/maxgan.yaml",
                        help="yaml file for config.")
    # parser.add_argument('--model', type=str, required=True,
    #                     help="path of model for evaluation")
    parser.add_argument('--wave', type=str,
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str, default="aurora.spk.npy",
                        help="Path of speaker.")
    parser.add_argument('--ppg', type=str,
                        help="Path of content vector.")
    parser.add_argument('--pit', type=str,
                        help="Path of pitch csv file.")
    parser.add_argument('--shift', type=int, default=0,
                        help="Pitch shift key.")
    args = parser.parse_args()

    main(args)
