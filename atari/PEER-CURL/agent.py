# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================

# PEER agent
from __future__ import division
import os
import kornia.augmentation as aug
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_

from model import DQN

random_shift = nn.Sequential(aug.RandomCrop((80, 80)), nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
aug = random_shift


class Agent():
    def __init__(self, args, env):

        self.args = args
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        self.coeff = 0.01 if args.game in ['pong', 'boxing', 'private_eye', 'freeway'] else 1.
        self.peer_coef = args.peer_coef
        self.online_net = DQN(args, self.action_space).to(device=args.device)
        self.momentum_net = DQN(args, self.action_space).to(device=args.device)


        self.online_net.train()
        self.initialize_momentum_net()
        self.momentum_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        for param in self.momentum_net.parameters():
            param.requires_grad = False
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            a, _ = self.online_net(state.unsqueeze(0))
            return (a * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def learn(self, mem):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
        aug_states_1 = aug(states).to(device=self.args.device)
        aug_states_2 = aug(states).to(device=self.args.device)
        # Calculate current state probabilities (online network noise already sampled)
        log_ps, _, current_representation = self.online_net(states, log=True, representation=True)  # Log probabilities log p(s_t, ·; θonline)
        _, z_anch = self.online_net(aug_states_1, log=True)
        _, z_target = self.momentum_net(aug_states_2, log=True)
        z_proj = torch.matmul(self.online_net.W, z_target.T)
        logits = torch.matmul(z_anch, z_proj)
        logits = (logits - torch.max(logits, 1)[0][:, None])
        logits = logits * 0.1
        labels = torch.arange(logits.shape[0]).long().to(device=self.args.device)
        moco_loss = (nn.CrossEntropyLoss()(logits, labels)).to(device=self.args.device)

        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns, _ = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(
                1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise()  # Sample new target net noise
            pns, _, target_representation = self.target_net(next_states, representation=True)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(
                self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(
                0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        peer_loss = torch.einsum('ij,ij->i', [current_representation, target_representation]).mean()
        # feature_loss2 = torch.einsum('ij,ij->i', [current_feature2, target_feature2]).mean()

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss = loss + (moco_loss * self.coeff) + self.peer_coef * peer_loss
        self.online_net.zero_grad()
        curl_loss = (weights * loss).mean()
        curl_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def initialize_momentum_net(self):
        for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(param_q.data)  # update
            param_k.requires_grad = False  # not update by gradient

    # Code for this function from https://github.com/facebookresearch/moco
    @torch.no_grad()
    def update_momentum_net(self, momentum=0.999):
        for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)  # update

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            a, _ = self.online_net(state.unsqueeze(0))
            return (a * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
