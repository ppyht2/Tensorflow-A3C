import numpy as np
import threading
import gym
import time  # Thread delay
import random


class Agent():
    """ Act for the brain and collect experience for the brain to consume
    """

    def __init__(self, config):
        self.memory = []  # This is used for n-step return
        self.G = 0.0  # Return
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
        s, a, _, _ = self.memory[0]
        _, _, _, sp = self.memory[n - 1]
        return s, a, self.G, sp

    def train(self, s, a, r, sp):
        a_onehot = np.zeros(self.config['ACTION_SPACE'])
        a_onehot[a] = 1

        self.memory.append([s, a_onehot, r, sp])

        self.G = (self.G + r * self.config['GAMMA_N']) / self.config['GAMMA']

        if sp is None:
            # Terminal State this calculation is inaccurate
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, sp = self.get_sample(n)
                self.push_to_brain(s, a, r, sp)

                self.G = (self.G - self.memory[0][2]) / self.config['GAMMA']
                self.memory.pop(0)

            self.G = 0.0

        if len(self.memory) >= self.config['N_STEPS']:
            s, a, r, sp = self.get_sample(self.config['N_STEPS'])
            self.push_to_brain(s, a, r, sp)

            self.G = self.G - self.memory[0][2]
            self.memory.pop(0)


class Environment(threading.Thread):
    """ An class of the evnironment and agent"""
    env_id = 0
    scores = []

    def __init__(self, config, render=False):
        threading.Thread.__init__(self)
        self.stop_signal = False
        self.render = render
        self.config = config
        self.env = gym.make(self.config['ENV_NAME'])
        self.id = Environment.env_id
        self.agent = Agent(config)
        Environment.env_id += 1

    def run_episode(self):
        s = self.env.reset()
        R = 0
        while True:
            time.sleep(self.config['THREAD_DELAY'])  # yield

            if self.render:
                self.env.render()

            a = self.agent.act(s)
            sp, r, done, info = self.env.step(a)

            if done:  # terminal state
                sp = None

            self.agent.train(s, a, r, sp)

            s = sp
            R += r

            if done or self.stop_signal:
                Environment.scores.append(R)
                if self.render:
                    print("ENV_{} INFO: total reward this episode: {}".format(self.id, R))
                break

    def run(self):
        while not self.stop_signal:
            self.run_episode()

    def stop(self):
        self.stop_signal = True
