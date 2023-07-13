# Authors' implementations of PEER (CVPR'23)
[Paper: Frustratingly Easy Regularization on Representation Can Boost Deep Reinforcement Learning](https://openaccess.thecvf.com/content/CVPR2023/html/He_Frustratingly_Easy_Regularization_on_Representation_Can_Boost_Deep_Reinforcement_Learning_CVPR_2023_paper.html)

We provide readers with PEER implementations, which are performed on PyBullet, MuJoCo, DMControl, and Atari respectively.

```bash
.
├── README.md
├── atari
│   ├── PEER-CURL
│   │   ├── agent.py
│   │   ├── env.py
│   │   ├── main.py
│   │   ├── memory.py
│   │   ├── model.py
│   │   └── utils.py
│   └── PEER-DrQ
│       ├── atari.py
│       ├── main.py
│       ├── peer.py
│       ├── replay_buffer.py
│   └── utils.py
├── conda_env.yml
└── dmc
    ├── PEER-CURL
    │   ├── encoder.py
    │   ├── main.py
    │   ├── peer.py
    │   └── utils.py
    └── PEER-DrQ
        ├── main.py
        ├── peer.py
        ├── readme.md
        ├── replay_buffer.py
        └── utils.py

7 directories, 28 files

```


# Reproduce in local environment




1. Build Conda Environment

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
conda activate rl
```

2. Following instructions in [mujoco](https://github.com/openai/mujoco-py) and [atari-py](https://github.com/openai/atari-py) repo, and install it.

   

3. Instructions

Before running the code, you should enter the corresponding directory and activate peer conda enviornment.



# Logs

If you run Atari PEER-CURL, you could see logs that look like:

```bash
(PEER) python main.py
                          Options
                          id: peer-ms_pacman-0
                          seed: 0
                          disable_cuda: False
                          game: ms_pacman
                          T_max: 100000
                          max_episode_length: 108000
                          history_length: 4
                          architecture: data-efficient
                          hidden_size: 256
                          noisy_std: 0.1
                          atoms: 51
                          V_min: -10
                          V_max: 10
                          model: None
                          memory_capacity: 100000
                          replay_frequency: 1
                          priority_exponent: 0.5
                          priority_weight: 0.4
                          multi_step: 20
                          discount: 0.99
                          target_update: 2000
                          reward_clip: 1
                          learning_rate: 0.0001
                          adam_eps: 1.5e-05
                          batch_size: 32
                          norm_clip: 10
                          learn_start: 1600
                          evaluate: False
                          evaluation_interval: 10000
                          evaluation_episodes: 10
                          evaluation_size: 500
                          render: False
                          enable_cudnn: True
                          checkpoint_interval: 0
                          memory: None
                          disable_bzip_memory: False
                          peer_coef: 0.0005
  2%|█▋                                                    | 1506/100000 [00:09<05:30, 297.96it/s]
```

If you run Atari PEER-DrQ, you could see logs that look like:

```bash
Algorithm: DrQ PEER,  Env: Breakout, Num Step: 0, Episode Return: 0
  0%|          | 184/100000 [00:00<01:07, 1837.29it/s]
```

If you run Bullet/MuJoCo PEER, you could see logs that look like:
```bash
python main.py --env 'HopperBulletEnv-v0'
---------------------------------------
Policy: PEER, Env: HopperBulletEnv-v0, Seed: 0
---------------------------------------
pybullet build time: May 20 2022 19:44:17
Algo: PEER Env: HopperBulletEnv-v0 Total T: 12 Episode Num: 1 Episode T: 12 Reward: 19.889
Algo: PEER Env: HopperBulletEnv-v0 Total T: 24 Episode Num: 2 Episode T: 12 Reward: 19.425
Algo: PEER Env: HopperBulletEnv-v0 Total T: 32 Episode Num: 3 Episode T: 8 Reward: 19.589
```

```bash
 python3 main.py --env Hopper-v2
---------------------------------------
Policy: PEER, Env: Hopper-v2, Seed: 0
---------------------------------------
Algo: PEER Env: Hopper-v2 Total T: 17 Episode Num: 1 Episode T: 17 Reward: 13.496
Algo: PEER Env: Hopper-v2 Total T: 37 Episode Num: 2 Episode T: 20 Reward: 17.732
Algo: PEER Env: Hopper-v2 Total T: 50 Episode Num: 3 Episode T: 13 Reward: 9.743
Algo: PEER Env: Hopper-v2 Total T: 92 Episode Num: 4 Episode T: 42 Reward: 69.791

```

In your console,  if you run PEER-CURL DMControl task, you could see logs that look like:

```
Domain_name: ball_in_cup, Task_name: catch, Seed: 1, Num_steps: 0 Episode_length: 0, Episode_reward: 0
0%|          | 249/500000 [00:14<7:58:17, 17.41it/s]
```

In your console,  if you run PEER-DrQ DMControl task, you could see logs that look like:
```bash
Agent:  PEER
Algorithm: PEER, Env: ball_in_cup-catch, Seed: 1, Num Step: 250, Episode Reward: 162.64358073362703
  0%|          | 249/110000 [00:04<20:35, 88.83it/s]
```



For environment issues, see the Atari, DMControl PyBullet , and MuJoCO documentation.

Our implementations are based on [TD3](https://github.com/sfujim/TD3), [CURL](https://github.com/MishaLaskin/curl) and [DrQ](https://github.com/denisyarats/drq).

# Citation

If you use this code for your research, please cite our paper:

```
@InProceedings{He_2023_CVPR,
    author    = {He, Qiang and Su, Huangyuan and Zhang, Jieyu and Hou, Xinwen},
    title     = {Frustratingly Easy Regularization on Representation Can Boost Deep Reinforcement Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20215-20225}
}
```