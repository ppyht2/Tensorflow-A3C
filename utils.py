import numpy as np
import scipy.misc  # scipy.misc.imresize
import gym
import matplotlib.pyplot as plt

import pickle  # Saving files

from collections import deque


frame_history = 4
frame_channels = 3
input_channels = frame_history * frame_channels
input_height = 84
input_width = 84


def get_luminance(rgb_array):
    """ Caluclate luminance from RGB image
    https://en.wikipedia.org/wiki/Relative_luminance

    Args:
      rbg_array: A numpy array of size H x W x C, where C represents the RGB
      channels respectively.
    """
    y = 0.2126 * rgb_array[:, :, 0] + 0.7152 * rgb_array[:, :, 1] + 0.0722 * rgb_array[:, :, 2]
    return y


def down_sample(image):
    """ Down sample the image intold_state[:, :, : - 1]o 84x84 pixels

    Args:
      image: A numpy array of size H x W x C.
    """
    return scipy.misc.imresize(image, (input_height, input_height))


def preprocess(obs):
    """ Preprocess gym observation for learning
    1. Convert into luminance
    2. Down sample
    3. Normalise

        Args:
          image: A numpy array of size H x W x C.
    """
    return down_sample(get_luminance(obs)) / 255.0


def init_state(obs):
    """ Creats an input state based on a single observation
    """
    state = np.zeros([input_height, input_height, input_channels])
    state[:, :, -1] = preprocess(obs)
    return state


def update_state(old_state, obs):
    """ Update the previouse input state using with a new observation
    """
    new_state = np.empty([input_height, input_width, input_channels])
    new_state[:, :, : - 1] = old_state[:, :, 1:]
    new_state[:, :,  - 1] = preprocess(obs)
    return new_state


def init_env(env_name):
    """ Initialise an GYM environment """
    try:
        env = gym.make(env_name)
    except:
        print('ERROR: Environment name {} is invalid.' .format(env_name))
        pass
    obs = env.reset()
    n_outputs = env.action_space.n
    print('INFO: The action space of {} is {}' .format(env_name, n_outputs))
    return env, obs, n_outputs


def plot_state(state, show=True):
    """ Visualise an input state
    """
    fig = plt.figure(figsize=[18, 9])
    for i in range(state.shape[-1]):
        plt.subplot(1, state.shape[-1], i + 1)
        plt.imshow(state[:, :, i], cmap='Greys')
    if show:
        plt.show()
    return fig


def rolling(x, w):
    """ Calculates the rolling average of x with a windows of w
    """
    y = []
    for i in range(len(x)):
        if i <= w:
            y.append(None)
        else:
            y.append(np.mean(x[i - w:i]))
    return np.array(y)


class EpsilonGreedy():
    """ Class for epsilon-greedy actions
    """

    def __init__(self, eps_min, eps_max, decay_steps, n_outputs):
        self.epsilon_min = eps_min
        self.epsilon_max = eps_max
        self.min_delay_step = decay_steps
        self.n_outputs = n_outputs

        pass

    def __calc_epsilon__(self, step):
        epsilon = max(self.epsilon_min,
                      self.epsilon_max - (self.epsilon_max -
                                          self.epsilon_min)
                      * step / self.min_delay_step)
        return epsilon

    def __call__(self, q_values, step):
        epsilon = self.__calc_epsilon__(step)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            return np.argmax(q_values)


class ExperienceReplay():
    """
    # state, action, reward, next_state, continue
    """

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque([], maxlen=self.memory_size)
        pass

    def load(self, path):
        f = open(path, 'rb')
        self.memory = pickle.load(f)
        f.close()

    def remember(self, state, action, reward, next_state, cont, q_values):
        self.memory.append((state, action, reward, next_state, cont, q_values))
        pass

    def __call__(self, batch_size):
        """ Samples reply memory for size of batch_size
        """
        indices = np.random.permutation(len(self.memory))[:batch_size]
        cols = [[], [], [], [], []]
        for idx in indices:
            memory = self.memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

    def save_memory(self, path):
        """ Savs the memory queue into a pickle file
        """
        f = open(path, 'wb')
        pickle.dump(self.memory, f)
        f.close()


def save_pickle(v, path):
    """ Savs variable v into a pickle file
    """
    f = open(path, 'wb')
    pickle.dump(v, f)
    f.close()


def load_pickle(path):
    f = open(path, 'rb')
    v = pickle.load(f)
    f.close()
    return v
