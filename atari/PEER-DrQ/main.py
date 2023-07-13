import os
import time
import numpy as np
import atari
import torch
import utils
import argparse
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from tqdm import trange
from peer import PEER


torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = atari.make_env(cfg.env, cfg.seed, cfg.terminal_on_life_loss)
        self.eval_env = atari.make_env(cfg.env, cfg.seed + 1,
                                       cfg.terminal_on_life_loss)


        self.agent = PEER(
            obs_shape=self.env.observation_space.shape,
            num_actions=self.env.action_space.n,
            device=cfg.device,
            # encoder_cfg,
            # critic_cfg,
            discount=cfg.discount,
            lr=cfg.lr,
            beta_1=cfg.beta_1,
            beta_2=cfg.beta_2,
            weight_decay=cfg.weight_decay,
            adam_eps=cfg.adam_eps,
            max_grad_norm=cfg.max_grad_norm,
            critic_tau=cfg.critic_tau,
            critic_target_update_frequency=cfg.critic_target_update_frequency,
            batch_size=cfg.batch_size,
            multistep_return=cfg.multistep_return,
            eval_eps=cfg.eval_eps,
            double_q=cfg.double_q,
            prioritized_replay_beta0=cfg.prioritized_replay_beta0,
            prioritized_replay_beta_steps=cfg.prioritized_replay_beta_steps,
            env_name=cfg.env,
            seed=cfg.seed,
            hidden_dim=cfg.hidden_dim, hidden_depth=cfg.hidden_depth,
            dueling=cfg.dueling, aug_type=cfg.aug_type,
            image_pad=cfg.image_pad, intensity_scale=cfg.intensity_scale,
        )
        if cfg.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.env.observation_space.shape, cfg.replay_buffer_capacity,
                cfg.prioritized_replay_alpha, self.device)
        else:
            self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                              cfg.replay_buffer_capacity,
                                              self.device)

        self.step = 0

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True

        for placeholder in trange(self.cfg.num_train_steps):
        # while self.step < self.cfg.num_train_steps:
            if done:
                print(
                    f"Algorithm: DrQ PEER,  Env: {self.cfg.env}, Num Step: {self.step}, Episode Return: {episode_reward}")

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1


            steps_left = self.cfg.num_exploration_steps + self.cfg.start_training_steps - self.step
            bonus = (1.0 - self.cfg.min_eps
                     ) * steps_left / self.cfg.num_exploration_steps
            bonus = np.clip(bonus, 0., 1. - self.cfg.min_eps)
            self.agent.eps = self.cfg.min_eps + bonus


            # sample action for data collection
            if np.random.rand() < self.agent.eps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs)

            # run training update
            if self.step >= self.cfg.start_training_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, None,
                                      self.step)
                    # print("Train " * 30)

            next_obs, reward, terminal, info = self.env.step(action)

            time_limit = 'TimeLimit.truncated' in info
            done = info['game_over'] or time_limit

            terminal = float(terminal)
            terminal_no_max = 0 if time_limit else terminal

            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs,
                                   terminal_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


def main(cfg):
    # from train import Workspace as W
    workspace = Workspace(cfg)
    workspace.run()


parser = argparse.ArgumentParser("DrQ-Atari-100k")
parser.add_argument("--env", default="Breakout", type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu_idx', default=0, type=int)
# parser.add_argument('--debug', action="store_true", type=bool)
args = parser.parse_args()

args.debug = False
if args.debug == True:
    print("-" * 30)
    print("Mode: Debug")
    print("-"*30)
    print()
    print()
    time.sleep(3)

cfg=utils.CFG(env=args.env,
        seed=args.seed,
        debug=args.debug
        )

main(cfg)
