import numpy as np
import torch
import gym
import argparse
import os
import mujoco

import utils
import time
import random
import pybullet_envs


if __name__ == "__main__":

    begin_time = time.asctime(time.localtime(time.time()))
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="PEER")
    parser.add_argument("--env", default='HopperBulletEnv-v0')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=25e3, type=int)  #
    # default 25e3, for test 1e3
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--expl_noise", default=0.1)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--policy_noise", default=0.2)
    parser.add_argument("--noise_clip", default=0.5)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--save_model", default=True)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--eval_state_value", default=True)
    parser.add_argument("--gpu_idx", default=0, type=int)
    parser.add_argument("--gpu_num", default=0, type=int)
    parser.add_argument("--feature_coef", default=5e-4, type=float) # PRO coef

    args = parser.parse_args()

    num_thread = args.gpu_num
    gpu_idx = args.gpu_idx
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    random.seed(args.seed)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    if args.policy == "METD3":
        # Target policy smoothing is scaled wrt the action
        import metd3
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = metd3.METD3(**kwargs)

    elif args.policy == "PEER":
        # Target policy smoothing is scaled wrt the action
        import peer
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["feature_coef"] = args.feature_coef
        policy = peer.PEER(**kwargs)

    else:
        raise NotImplementedError("No policy named", args.policy)


    replay_buffer = utils.ReplayBufferMuJoCo(state_dim, action_dim)


    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state, dtype='float32'))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Algo: {args.policy} Env: {args.env} Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1