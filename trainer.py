import functools
from functools import partial
import gym
#import tensorflow as tf
import tensorflow.compat.v1 as tf

from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from baselines import logger

class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()

        self.policy = CnnPolicy(
            scope='pol',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=tf.nn.leaky_relu)

        self.feature_extractor = {"none": FeatureExtractor,
                                  "idf": InverseDynamics,
                                  "vaesph": partial(VAE, spherical_obs=True),
                                  "vaenonsph": partial(VAE, spherical_obs=False),
                                  "pix2pix": JustPixels}[hps['feat_learning']]
        self.feature_extractor = self.feature_extractor(policy=self.policy,
                                                        features_shared_with_policy=False,
                                                        feat_dim=512,
                                                        layernormalize=hps['layernorm'])

        self.dynamics = Dynamics if hps['feat_learning'] != 'pix2pix' else UNet
        self.dynamics = self.dynamics(auxiliary_task=self.feature_extractor,
                                      predict_from_pixels=hps['dyn_from_pixels'],
                                      feat_dim=512)

        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamics=self.dynamics
        )

        self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
        self.agent.total_loss += self.agent.to_report['dyn_loss']
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def train(self):
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics=self.dynamics)
        while True:
            info = self.agent.step()
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                break

        self.agent.stop_interaction()