import argparse
from worker import Agent, Environment
from brain import Brain, Optimiser
import gym
import time
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Get configurations for A3C")
    parser.add_argument("-e", "--env", default="CartPole-v0",
                        type=str, help="name of the environment")
    parser.add_argument("-t", "--time", default=480, type=float, help="training run time")

    # TD config
    parser.add_argument("-s", "--nstep", default=8, type=int, help="n-step return")
    parser.add_argument("-g", "--gamma", default=0.99, type=float, help="discount rate")

    # Multi threading
    parser.add_argument("-d", "--delay", default=0.001, type=float, help="thread delay")
    parser.add_argument("-n", "--threads", default=16, type=int, help="number of worker threads")
    parser.add_argument("-o", "--optimiser", default=2, type=int,
                        help="number of optimiser threads")

    # Epsilon Greedy config
    parser.add_argument("--eps_start", default=0.4, type=float, help="epsilon starting")
    parser.add_argument("--eps_end", default=0.1, type=float, help="epsilon ending")
    parser.add_argument("--eps_steps", default=75000, type=int, help="ratio of epsilon decay")

    # Optimiser config
    parser.add_argument("--learning_rate", default=3e-3, type=float, help="learning rate")
    parser.add_argument("--min_batch_size", default=32, type=int,
                        help="minimum training batch size")
    parser.add_argument("--loss_v", default=0.5, type=float, help="value loss coefficient")
    parser.add_argument("--loss_entropy", default=0.01, type=float, help="entropy loss coefficient")

    return parser.parse_args()


def get_config(args):
    config = {'ENV_NAME': args.env,
              'TRAINING_TIME': args.time,
              'N_STEPS': args.nstep,
              'GAMMA': args.gamma,
              'THREAD_DELAY': args.delay,
              'N_THREADS': args.threads,
              'N_OPTIMISERS': args.optimiser,
              'EPS_START': args.eps_start,
              'EPS_END': args.eps_end,
              'EPS_STEPS': args.eps_steps,
              'LEARNING_RATE': args.learning_rate,
              'MIN_BATCH_SIZE': args.min_batch_size,
              'LOSS_V': args.loss_v,
              'LOSS_ENTROPY': args.loss_entropy
              }
    e = gym.make(config['ENV_NAME'])
    config['STATE_SPACE'] = e.observation_space.shape[0]
    config['ACTION_SPACE'] = e.action_space.n
    config['GAMMA_N'] = config['GAMMA'] ** config['N_STEPS']
    e.close()
    return config


def add_brain_config(config, brain):
    config['decision_func'] = brain.act
    config['push_to_brain'] = brain.push_to_queue
    return config


if __name__ == "__main__":

    args = get_args()
    config = get_config(args)
    # Brain testing
    brain = Brain(config)
    config = add_brain_config(config, brain)
    test_env = Environment(config, render=True)
    envs = [Environment(config) for i in range(config['N_THREADS'])]
    opts = [Optimiser(brain.optimiser) for i in range(config['N_OPTIMISERS'])]

    for o in opts:
        o.start()

    for e in envs:
        e.start()

    time.sleep(config['TRAINING_TIME'])

    for e in envs:
        e.stop()
    for e in envs:
        e.join()

    for o in opts:
        o.stop()
    for o in opts:
        o.join()

    test_env.start()
