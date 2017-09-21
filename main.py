import argparse
from worker import Environment
from brain import Brain, Optimiser
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mute missing instructions errors


def get_args():
    parser = argparse.ArgumentParser(description="Get configurations for A3C")
    parser.add_argument("-e", "--env", default="CartPole-v0",
                        type=str, help="name of the environment")
    parser.add_argument("-t", "--time", default=240, type=float, help="training run time")

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
    parser.add_argument("--eps_ratio", default=0.1, type=float, help="ratio of epsilon decay")

    # Optimiser config
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--min_batch_size", default=128, type=int,
                        help="minimum training batch size")
    parser.add_argument("--loss_v", default=1, type=float, help="value loss coefficient")
    parser.add_argument("--loss_entropy", default=0.01, type=float, help="entropy loss coefficient")

    parser.add_argument("-b", "--debug", default=False, type=bool, help="debuging mode")
    parser.add_argument("-f", "--freq", default=60, type=int, help="output frequency")
    parser.add_argument("-w", action='store_true', help="new session")

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
              'EPS_RATIO': args.eps_ratio,
              'EPS_TIME': args.eps_ratio * args.time,
              'LEARNING_RATE': args.learning_rate,
              'MIN_BATCH_SIZE': args.min_batch_size,
              'LOSS_V': args.loss_v,
              'LOSS_ENTROPY': args.loss_entropy,
              'DEBUG': args.debug,
              'OUTPUT_FREQ': args.freq,
              'NEW_SESSION': args.w
              }
    e = gym.make(config['ENV_NAME'])
    config['STATE_SPACE'] = e.observation_space.shape[0]
    config['ACTION_SPACE'] = e.action_space.n
    config['GAMMA_N'] = config['GAMMA'] ** config['N_STEPS']
    e.close()
    return config


def add_brain_config(config, brain):
    config['decision_func'] = brain.decide
    config['push_to_brain'] = brain.push_to_queue
    config['get_epsilon'] = brain.get_epsilon
    return config


if __name__ == "__main__":
    # Configs
    args = get_args()
    config = get_config(args)
    brain = Brain(config)
    config = add_brain_config(config, brain)

    # File structure
    checkpoint_path = os.path.join('sessions', config['ENV_NAME'], 'ckpt')
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
        config['NEW_SESSION'] = True
    if tf.train.checkpoint_exists(checkpoint_path) and not config['NEW_SESSION']:
        brain.load_session(checkpoint_path)
    else:
        print('INFO: No existing sessons.')

    print('------- Configuration -------')
    for k in config:
        print('{}: {}'.format(k, config[k]))
    print('------- End of Configuration -------')

    test_env = Environment(config, render=True)
    envs = [Environment(config) for i in range(config['N_THREADS'])]
    opts = [Optimiser(brain.optimiser) for i in range(config['N_OPTIMISERS'])]

    for o in opts:
        o.start()

    for e in envs:
        e.start()

    # Output loop
    while time.time() - brain.init_time <= config['TRAINING_TIME']:
        time.sleep(config['OUTPUT_FREQ'])
        if len(Environment.scores) < 100:
            continue
        else:
            scores = Environment.scores[-100:]
        min_score = np.min(scores)
        max_score = np.max(scores)
        mean_scores = np.mean(scores)
        t = (time.time() - brain.init_time)
        p = t / config['TRAINING_TIME'] * 100
        print('{:.2f} {:.2f}% mean:{:.2f} max:{:.2f} min:{:.2f}'.format(
            t, p, mean_scores, max_score, min_score))

        # Save the session
        brain.save_session(checkpoint_path)

    for e in envs:
        e.stop()
    for e in envs:
        e.join()

    for o in opts:
        o.stop()
    for o in opts:
        o.join()

    brain.save_session(checkpoint_path)

    test_env.start()
