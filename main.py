import argparse
from run import start_experiment


def add_params(parser):

    # add environment params
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4',type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=4500, type=int)
    parser.add_argument('--env_kind', type=str, default="atari")
    parser.add_argument('--noop_max', type=int, default=30)

    # add optimization params 
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=int(1e8))

    # add rollout params 
    parser.add_argument('--nsteps_per_seg', type=int, default=128)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=128)
    parser.add_argument('--nlumps', type=int, default=1)

    # add other params 
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--feat_learning', type=str, default="none",
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])


if __name__ == '__main__':

	# parameter initialization
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_params(parser)
    args = parser.parse_args()
    
    # start running the experiment
    start_experiment(**args.__dict__)