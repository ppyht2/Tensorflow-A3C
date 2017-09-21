import numpy as np
import threading
import gym
import time  # Thread delay
import random
from utils import init_state, update_state
from utils import save_pickle
import os


class Agent():
    """ An Agent class which collects experience for the brain
    Args:
      config: configuration dictionary
      config['decision_func']: function pointer for policy logits from the brain
      config['push_to_brain']: function pointer for appending training examples
      config['get_epsilon']: function pointer to the epsilon getter
    """

    def __init__(self, config):
        self.memory = []  # for n-step return
        self.G = 0.0  # return
        self.decision_func = config['decision_func']
        self.push_to_brain = config['push_to_brain']
        self.get_epsilon = config['get_epsilon']
        self.config = config

    def act(self, s):
        """ Act epsilon-greedly using epsilon from the brain"""
        eps = self.get_epsilon()
        rand = random.random()
        if rand < eps:
            return random.randint(0, self.config['ACTION_SPACE'] - 1)
        else:
            return self.decision_func(s)

    def get_sample(self, n):
        """ sample n-step predictions, states and returns
        """
        s, a, _, _ = self.memory[0]
        _, _, _, sp = self.memory[n - 1]
        return s, a, self.G, sp

    def train(self, s, a, r, sp):
        """ manipulate memory and push to brain for training
        Args:
          s: state
          a: actions
          r: immediate reward after a
          sp: immediate state after a
        """
        # Turn a into a one-hot vector and add to memory
        a_onehot = np.zeros(self.config['ACTION_SPACE'])
        a_onehot[a] = 1
        self.memory.append([s, a_onehot, r, sp])
        self.G = (self.G + r * self.config['GAMMA_N']) / self.config['GAMMA']

        # Calculate n-step return if the episode terminates
        # This calucate is not accurate
        if sp is None:
            # Terminal State this calculation is inaccurate
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, sp = self.get_sample(n)
                self.push_to_brain(s, a, r, sp)
                self.G = (self.G - self.memory[0][2]) / self.config['GAMMA']
                self.memory.pop(0)
            self.G = 0.0

        # Calulcate n-step return according to config
        if len(self.memory) >= self.config['N_STEPS']:
            s, a, r, sp = self.get_sample(self.config['N_STEPS'])
            self.push_to_brain(s, a, r, sp)
            self.G = self.G - self.memory[0][2]
            self.memory.pop(0)


class Environment(threading.Thread):
    """ An contrainer class with an instance of Gym environments and Agent each

    Args:
      config: configuration dictoinary
      render: bool, this will not work with multi-threading
      debug: bool, this will save details of each epsiode into a pickle file
    """
    env_id = 0  # Evnironment ID
    scores = []  # Global scores record for reporting. this may cause mem overflow

    def __init__(self, config, render=False, debug=False):
        threading.Thread.__init__(self)
        self.stop_signal = False
        self.render = render
        self.config = config
        self.env = gym.make(self.config['ENV_NAME'])
        self.id = Environment.env_id
        Environment.env_id += 1
        self.agent = Agent(config)
        self.episode_number = 0
        self.debug = debug
        self.debug_log = []

    def run_episode(self):
        """ Game loop """
        # reset
        obs = self.env.reset()
        s = init_state(obs)
        R = 0  # total reward this episode
        self.debug_log = []

        while True:
            time.sleep(self.config['THREAD_DELAY'])  # yield

            if self.render:
                self.env.render()

            a = self.agent.act(s)
            obs, r, done, info = self.env.step(a)
            sp = update_state(s, obs)

            if done:  # terminal state
                sp = None

            self.agent.train(s, a, r, sp)
            if self.debug:
                self.debug_log.append([s, a, r, sp, done])

            s = sp
            R += r

            if done or self.stop_signal:
                Environment.scores.append(R)
                self.episode_number += 1
                if self.debug:  # Save logs
                    # TODO: folder restructure
                    save_pickle(self.debug_log, os.path.join(
                        'debug_logs', "ENV{}_EPISODE{}".format(self.id, self.episode_number)))
                if self.render:  # Demo mode
                    print("ENV_{} INFO: total reward this episode: {}".format(self.id, R))
                break

    def run(self):
        while not self.stop_signal:
            self.run_episode()

    def stop(self):
        self.stop_signal = True
