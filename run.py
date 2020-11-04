#!/usr/bin/env python
try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")

import gym
from gym.utils.seeding import hash_seed

import datetime
import os.path as osp
from mpi4py import MPI
import functools
from functools import partial
from trainer import Trainer
from utils import setup_mpi_gpus, setup_tensorflow_session

from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds

from wrappers import MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, make_atari_env, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit


def start_experiment(**args):
    
    make_env = partial(make_env_all_params, add_monitor=True, args=args)
    trainer = Trainer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'])
    log, tf_sess = get_experiment_environment(**args)
    with log, tf_sess:
        logdir = logger.get_dir()
        print("results will be saved to ", logdir)
        trainer.train()


def make_env_all_params(rank, add_monitor, args):
    if args["env_kind"] == 'atari':
        env = make_atari_env(args)
    elif args["env_kind"] == 'mario':
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env


def get_experiment_environment(**args):
    log_directory = osp.join('./output/'+ datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    logger_context = logger.scoped_configure(dir=log_directory,
                                             format_strs=['stdout', 'log',
                                                          'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])
    tf_context = setup_tensorflow_session()
    return logger_context, tf_context