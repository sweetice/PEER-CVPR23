import os
import time
import numpy as np

import dmc2gym
from peer import PEER
import torch
import utils
from replay_buffer import ReplayBuffer
from utils import CFG
import argparse
from tqdm import trange

torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)

def make_env(cfg, seed=None):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0
    seed = cfg.seed if seed is None else seed
    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=cfg.image_size,
                       width=cfg.image_size,
                       frame_skip=cfg.action_repeat,
                       camera_id=camera_id)

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        if self.cfg.algo == "PEER":
            print("Agent: ", self.cfg.algo)
            self.agent = PEER(
                action_shape = self.env.action_space.shape,
                action_range = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
            ],
                device="cuda",
                discount=self.cfg.discount,
                init_temperature = self.cfg.init_temperature,
                lr=self.cfg.lr,
                actor_update_frequency=self.cfg.actor_update_frequency,
                critic_tau=self.cfg.critic_tau,
                critic_target_update_frequency=self.cfg.critic_target_update_frequency,
                batch_size=self.cfg.batch_size,
                env_name=self.cfg.env_name,
                seed=self.cfg.seed,
                obs_shape=self.env.observation_space.shape,
                feature_dim=self.cfg.feature_dim,
                hidden_dim=self.cfg.hidden_dim,
                hidden_depth=self.cfg.hidden_depth,

                # actor setting
                log_std_bounds=[-10, 2],
                beta=args.beta
            )
        else:
            raise  NotImplementedError("Unknown Agent")

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)
        self.step = 0


    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        for place_holder in trange(self.cfg.num_train_steps):
            if done:
                print(
                    f"Algorithm: {self.cfg.algo}, Env: {self.cfg.env_name}, Seed: {self.cfg.seed}, Num Step: {self.step}, Episode Reward: {episode_reward}")
                # evaluate agent periodically

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.step)


            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


def main(cfg):
    from main import Workspace as W
    workspace = W(cfg)
    workspace.run()

parser = argparse.ArgumentParser("DRQ-PEER")
parser.add_argument('--domain_name', default="ball_in_cup")
parser.add_argument('--task_name', default="catch")
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu_idx', default=0, type=int)
parser.add_argument('--algo', default="PEER", type=str)
parser.add_argument('--beta', default=5e-3, type=float)
# parser.add_argument('--debug', action=, type=bool)
args = parser.parse_args()
args.debug = False


cfg=CFG(domain=args.domain_name,
        task=args.task_name,
        seed=args.seed,
        debug=args.debug,
        algo=args.algo,
        beta=args.beta
        )

main(cfg)