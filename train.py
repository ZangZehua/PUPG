import os
import sys
import random
import datetime
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from algorithms import atari_agents
from atari.old.utils.utils import Logger


def make_save(cfg):
    time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y.%m.%d.%H.%M.%S"))
    cfg.save_path = os.path.join(cfg.base_path, time)
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    if cfg.model:
        cfg.model_path = os.path.join(cfg.save_path, "models")
        if not os.path.exists(cfg.model_path):
            os.makedirs(cfg.model_path)


def make_env(env_name, seed, ram=False, gary=True, fs=4, resize=84):
    def thunk():
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)

        if not ram:  # pixel obs
            env = gym.wrappers.ResizeObservation(env, (resize, resize))
            if gary:
                env = gym.wrappers.GrayScaleObservation(env)
            if fs > 0:
                env = gym.wrappers.FrameStack(env, fs)
        return env
    return thunk


def run(args, stdout):
    make_save(args)
    args.seed = random.randint(0, 100000)
    OmegaConf.save(args, os.path.join(args.save_path, "config.yaml"))
    sys.stdout = Logger(stdout, os.path.join(args.save_path, "logs.txt"))
    print("============================================================")
    print("saving at:", args.save_path)
    # check if ram
    if "ram" in args.env:
        args.ram = True
        args.bb = 0
    # create train env and eval env
    envs = gym.vector.SyncVectorEnv(
        [make_env("ALE/" + args.env, args.seed + i, args.ram, args.gray, args.fs, args.resize) for i in
         range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    eval_env = gym.vector.SyncVectorEnv(
        [make_env("ALE/" + args.env, args.seed, args.ram, args.gray, args.fs, args.resize)]
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.cuda_deterministic

    # create agent
    agent = atari_agents[args.algo.algo_name](args, envs, eval_env, device)
    agent.run()
    print("============================================================")
    print("saving at:", args.save_path)
    print("============================================================")
    sys.stdout.close()


@hydra.main(config_path='cfgs', config_name='config', version_base=None)
def main(cfg):
    stdout = sys.stdout
    for r in range(cfg.run):
        run(cfg, stdout)


if __name__ == "__main__":
    main()
