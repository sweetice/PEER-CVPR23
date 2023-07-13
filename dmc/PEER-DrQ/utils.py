import math
import os
import random
from collections import deque

import numpy as np
import scipy.linalg as sp_la

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.util.shape import view_as_windows
from torch import distributions as pyd
import pickle
import pandas as pd


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class CFG():
    env = "cartpole_swingup"
    # IMPORTANT= if action_repeat is used the effective number of env steps needs to be
    # multiplied by action_repeat in the result graphs.
    # This is a common practice for a fair comparison.
    # See the 2nd paragraph in Appendix C of SLAC= https=//arxiv.org/pdf/1907.00953.pdf
    # See Dreamer TF2's implementation= https=//github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/dreamer.py#L340
    action_repeat = 4
    # train
    num_train_steps = 110000
    num_train_iters = 1
    num_seed_steps = 1000 # √ set to 2 for debug, default = 1000 TODO
    replay_buffer_capacity = 100000 # √
    seed = 1
    # eval
    eval_frequency = 10000
    num_eval_episodes = 10  # TODO Change to 10 as default
    # misc
    log_frequency_step = 10000
    log_save_tb = False
    save_video = False
    device = "cuda"
    # observation
    image_size = 84
    image_pad = 4
    frame_stack = 3
    # global params
    lr = 1e-3 # √
    # IMPORTANT= please use a batch size of 512 to reproduce the results in the paper. Hovewer with a smaller batch size it still works well.
    batch_size = 512 # √ # √ set to 128 for debug, default = 512 TODO

    obs_shape = 128
    action_shape = 128
    action_range = 128
    encoder_cfg = None
    critic_cfg = None
    actor_cfg = None
    discount = 0.99 # √
    init_temperature = 0.1 # √
    actor_update_frequency = 2 # √
    critic_tau = 0.01 # √
    critic_target_update_frequency = 2 # √

    hidden_dim = 1024 # √
    hidden_depth = 2 # √

    log_std_bounds = [-10, 2] #
    feature_dim = 50 # √
    def __init__(self, domain=None, task=None, seed=None, debug=False, algo=None, beta=None):
        self.domain = domain
        self.task = task
        self.env_name = f"{self.domain}-{self.task}"
        self.seed = seed
        self.algo = algo
        self.beta = beta

        if debug:
            self.num_seed_steps = 2
            self.batch_size = 2
            self.num_eval_episodes = 2
            self.eval_frequency = 2000
            self.logger_interval = 500