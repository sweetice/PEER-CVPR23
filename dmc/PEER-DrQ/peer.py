import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import os

class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds, obs_shape, feature_dim=50):
        super().__init__()

        self.encoder = Encoder(obs_shape=obs_shape, feature_dim=feature_dim)

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim,
                               2 * action_shape[0], hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist



class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, action_shape, hidden_dim, hidden_depth, obs_shape, feature_dim=50):
        super().__init__()

        # self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.encoder = Encoder(obs_shape=obs_shape, feature_dim=feature_dim)

        self.repr1 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, hidden_dim, hidden_depth-1)
        self.repr2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, hidden_dim, hidden_depth-1)

        self.Q1=torch.nn.Linear(hidden_dim, 1)
        self.Q2=torch.nn.Linear(hidden_dim, 1)
        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs, action], dim=-1)
        repr1, repr2 = self.repr1(obs_action), self.repr2(obs_action) # output_dim=32
        q1 = self.Q1(repr1)  # input_dim=1024
        q2 = self.Q2(repr2)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return repr1, repr2, q1, q2


class PEER(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, action_shape, action_range, device,
                discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, env_name, seed,

                 # critic setting
                 obs_shape,
                 feature_dim,
                 hidden_dim=1024,
                 hidden_depth=2,

                 # actor setting
                 log_std_bounds=[-10, 2],


                 # PEER
                 beta = None
                 ):
        assert beta is not None
        self.action_range = action_range
        self.device = "cuda"
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.actor = Actor(action_shape, hidden_dim, hidden_depth,
                 log_std_bounds, obs_shape, feature_dim=50).to(self.device)

        # self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic = Critic(action_shape, hidden_dim, hidden_depth, obs_shape, feature_dim).to(self.device)

        self.critic_target = Critic(action_shape, hidden_dim, hidden_depth, obs_shape, feature_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # PEER loss
        self.beta = beta

        # self.count = 0
        self.train()
        self.critic_target.train()


    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            dist = self.actor(obs)
            action = dist.sample() if sample else dist.mean
            action = action.clamp(*self.action_range)
            assert action.ndim == 2 and action.shape[0] == 1
            return utils.to_np(action[0])

    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_repr1, target_repr2, target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)
            target_repr1_aug, target_repr2_aug, target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug)
            # We should get the target repr
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_repr1, current_repr2, current_Q1, current_Q2 = self.critic(obs, action)

        PEER_loss1 = torch.einsum('ij,ij->i', [current_repr1, target_repr1])
        PEER_loss2 = torch.einsum('ij,ij->i', [current_repr2, target_repr2])
        critic_loss1 = F.mse_loss(current_Q1, target_Q)
        critic_loss2 = F.mse_loss(current_Q2, target_Q)
        critic_loss_original = critic_loss1 + critic_loss2
        PEER_loss1_beta = self.beta * PEER_loss1.mean()
        PEER_loss2_beta = self.beta * PEER_loss2.mean()
        critic_loss_before_aug = critic_loss_original + PEER_loss1_beta + PEER_loss2_beta
        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        #     current_Q2, target_Q) + self.beta * PEER_loss1.mean() + self.beta * PEER_loss2.mean()

        repr_aug1, repr_aug2, Q1_aug, Q2_aug = self.critic(obs_aug, action)
        PEER_aug1 = torch.einsum('ij,ij->i', [repr_aug1, target_repr1_aug])
        PEER_aug2 = torch.einsum('ij,ij->i', [repr_aug2, target_repr2_aug])
        PEER_aug1_beta = PEER_aug1.mean() * self.beta
        PEER_aug2_beta = PEER_aug2.mean() * self.beta
        aug_loss_without_peer = F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)
        critic_aug_loss = aug_loss_without_peer + PEER_aug1_beta + PEER_aug2_beta
        total_critic_loss = critic_loss_before_aug + critic_aug_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()


    def update_actor_and_alpha(self, obs):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        _, _, actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()

        alpha_loss.backward()
        self.log_alpha_optimizer.step()


    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
            self.batch_size)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
