import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
import hydra
import kornia
import os

from replay_buffer import PrioritizedReplayBuffer


# from
# https://github.com/mlperf/inference/blob/master/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py#L40
class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] +
                            effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class SEEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape):
        super().__init__()

        self.feature_dim = 64 * 4 * 4

        self.conv1 = Conv2d_tf(obs_shape[0], 32, 5, stride=5, padding='SAME')
        self.conv2 = Conv2d_tf(32, 64, 5, stride=5, padding='SAME')

        self.outputs = dict()

    def forward(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        h = torch.relu(self.conv1(obs))
        self.outputs['conv1'] = h

        h = torch.relu(self.conv2(h))
        self.outputs['conv2'] = h

        out = h.view(h.size(0), -1)
        self.outputs['out'] = out

        assert out.shape[1] == self.feature_dim
        return out



class Encoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape):
        super().__init__()

        self.feature_dim = 64 * 11 * 11

        self.conv1 = Conv2d_tf(obs_shape[0], 32, 8, stride=4, padding='valid')
        self.conv2 = Conv2d_tf(32, 64, 4, stride=2, padding='valid')
        self.conv3 = Conv2d_tf(64, 64, 3, stride=1, padding='valid')

        self.outputs = dict()

    def forward(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        h = torch.relu(self.conv1(obs))
        self.outputs['conv1'] = h

        h = torch.relu(self.conv2(h))
        self.outputs['conv2'] = h

        h = torch.relu(self.conv3(h))
        self.outputs['conv3'] = h

        out = h.view(h.size(0), -1)
        self.outputs['out'] = out

        assert out.shape[1] == self.feature_dim

        return out


class Intensity(nn.Module):
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        noise = 1.0 + (self.scale * torch.randn(
            (x.size(0), 1, 1, 1), device=x.device).clamp_(-2.0, 2.0))
        return x * noise


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_shape, num_actions, hidden_dim, hidden_depth,
                 dueling, aug_type, image_pad, intensity_scale):
        super().__init__()

        AUGMENTATIONS = {
            'intensity':
            Intensity(scale=intensity_scale),
            'reflect_crop':
            nn.Sequential(nn.ReplicationPad2d(image_pad),
                          kornia.augmentation.RandomCrop((84, 84))),
            'crop_intensity':
            nn.Sequential(nn.ReplicationPad2d(image_pad),
                          kornia.augmentation.RandomCrop((84, 84)),
                          Intensity(scale=intensity_scale)),
            'zero_crop':
            nn.Sequential(nn.ZeroPad2d(image_pad),
                          kornia.augmentation.RandomCrop((84, 84))),
            'rotate':
            kornia.augmentation.RandomRotation(degrees=5.0),
            'h_flip':
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            'v_flip':
            kornia.augmentation.RandomVerticalFlip(p=0.5),
            'none':
            nn.Identity(),
            'all':
            nn.Sequential(nn.ReplicationPad2d(image_pad),
                          kornia.augmentation.RandomCrop((84, 84)),
                          kornia.augmentation.RandomHorizontalFlip(p=0.5),
                          kornia.augmentation.RandomVerticalFlip(p=0.5),
                          kornia.augmentation.RandomRotation(degrees=5.0))
        }

        assert aug_type in AUGMENTATIONS.keys()

        self.aug_trans = AUGMENTATIONS.get(aug_type)

        # self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.encoder = Encoder(obs_shape)

        self.dueling = dueling
        self.num_actions = num_actions

        if dueling:
            self.V = utils.mlp(self.encoder.feature_dim, hidden_dim, 1,
                               hidden_depth)
            self.A = utils.mlp(self.encoder.feature_dim, hidden_dim,
                               num_actions, hidden_depth)
        else:
            self.Q = utils.mlp(self.encoder.feature_dim, hidden_dim,
                               num_actions, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, use_aug=False):
        if use_aug:
            obs = self.aug_trans(obs)

        obs = self.encoder(obs)

        if self.dueling:
            v = self.V(obs)
            a = self.A(obs)
            q = v + a - a.mean(1, keepdim=True)
        else:
            q = self.Q(obs)

        self.outputs['q'] = q
        # return representation
        return obs, q


class PEER(object):
    """Data regularized Q-learning: Deep Q-learning."""
    def __init__(self, obs_shape, num_actions, device,
                 # encoder_cfg, critic_cfg,
                 discount, lr, beta_1, beta_2, weight_decay, adam_eps,
                 max_grad_norm, critic_tau, critic_target_update_frequency,
                 batch_size, multistep_return, eval_eps, double_q,
                 prioritized_replay_beta0, prioritized_replay_beta_steps,

                 # New add for logging
                 env_name, seed,
                 # encoder
                 # obs_shape
                 # decoder
                 hidden_dim, hidden_depth,
                 dueling, aug_type, image_pad, intensity_scale
                 ):

        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.num_actions = num_actions
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.eval_eps = eval_eps
        self.max_grad_norm = max_grad_norm
        self.multistep_return = multistep_return
        self.double_q = double_q
        assert prioritized_replay_beta0 <= 1.0
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_steps = prioritized_replay_beta_steps
        self.eps = 0

        self.critic = Critic(obs_shape, num_actions, hidden_dim, hidden_depth,
                 dueling, aug_type, image_pad, intensity_scale).to(self.device)
        self.critic_target = Critic(obs_shape, num_actions, hidden_dim, hidden_depth,
                 dueling, aug_type, image_pad, intensity_scale).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr,
                                                 betas=(beta_1, beta_2),
                                                 weight_decay=weight_decay,
                                                 eps=adam_eps)
        # PEER setting
        self.beta = 5e-4
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    def act(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0).contiguous()
            _, q = self.critic(obs)
            action = q.max(dim=1)[1].item()
        return action

    def update_critic(self, obs, action, reward, next_obs, not_done, weights,
                      logger, step):
        with torch.no_grad():
            discount = self.discount**self.multistep_return
            if self.double_q:
                next_repr, next_Q_target, = self.critic_target(next_obs, use_aug=True)
                _, next_Q = self.critic(next_obs, use_aug=True)
                next_action = next_Q.max(dim=1)[1].unsqueeze(1)
                next_Q = next_Q_target.gather(1, next_action)
                target_Q = reward + (not_done * discount * next_Q)
            else:
                next_repr, next_Q = self.critic_target(next_obs, use_aug=True)
                next_Q = next_Q.max(dim=1)[0].unsqueeze(1)
                target_Q = reward + (not_done * discount * next_Q)

        # get current Q estimates
        current_repr, current_Q = self.critic(obs, use_aug=True)
        current_Q = current_Q.gather(1, action)

        td_errors = current_Q - target_Q
        peer_loss = torch.einsum('ij,ij->i', [current_repr, next_repr]).mean() * self.beta
        critic_losses = F.smooth_l1_loss(current_Q, target_Q, reduction='none')  + peer_loss
        if weights is not None:
            critic_losses *= weights

        critic_loss = critic_losses.mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic.parameters(),
                                     self.max_grad_norm)
        self.critic_optimizer.step()


        return td_errors.squeeze(dim=1).detach().cpu().numpy()

    def update(self, replay_buffer, logger, step):

        prioritized_replay = type(replay_buffer) == PrioritizedReplayBuffer

        if prioritized_replay:
            fraction = min(step / self.prioritized_replay_beta_steps, 1.0)
            beta = self.prioritized_replay_beta0 + fraction * (
                1.0 - self.prioritized_replay_beta0)
            obs, action, reward, next_obs, not_done, weights, idxs = replay_buffer.sample_multistep(
                self.batch_size, beta, self.discount, self.multistep_return)
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_multistep(
                self.batch_size, self.discount, self.multistep_return)
            weights = None


        td_errors = self.update_critic(obs, action, reward, next_obs, not_done,
                                       weights, logger, step)

        if prioritized_replay:
            prios = np.abs(td_errors) + 1e-6
            replay_buffer.update_priorities(idxs, prios)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
