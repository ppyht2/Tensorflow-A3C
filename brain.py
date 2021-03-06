import tensorflow as tf
import numpy as np
import threading
import time
import os
from utils import save_pickle

# Brain Settings
INPUT_HEIGHT = 84
INPUT_WIDTH = 84
INPUT_CHANNELS = 3 * 4
CONV_N_MAPS = [32, 32, 64, 64]
CONV_KERNEL_SIZES = [(5, 5), (5, 5), (4, 4), (3, 3)]
CONV_STRIDES = [1, 1, 1, 1]
CONV_PADDINGS = ["SAME"] * 4
CONV_ACTIVATION = [tf.nn.relu] * 4
N_HIDDEN_IN = 64 * 11 * 11  # 84/8 arounded up
N_HIDDEN = 512
HIDDEN_ACTIVATION = tf.nn.relu
INITIALISER = tf.contrib.layers.variance_scaling_initializer()


class Brain():
    """ A class of the A3C brain
    A3c: Asynchronous Advantage Actor-Critic
    Paper: https://arxiv.org/abs/1602.01783

    Args:
      config: configuration dictionary
    """
    # TODO: debug functionality

    def __init__(self, config):
        self.config = config
        self.train_queue = [[], [], [], [], []]  # s, a, r, s', continoue
        # memories collected from all Agents
        self.lock_queue = threading.Lock()
        self.build_graph()
        self.init_time = time.time()
        self.eps_end_time = self.init_time + config['EPS_TIME']
        # time when epsilon will reach minimum
        pass

    def build_graph(self):
        """ Core TensorFlow graph
        HT: This shit is tragic, sort it out
        """
        tf.reset_default_graph()
        with tf.name_scope('inputs'):
            self.input_states = tf.placeholder(
                tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS], name='input_states')
            self.input_actions = tf.placeholder(
                tf.float32, [None, self.config['ACTION_SPACE']], name='input_actions')
            self.input_returns = tf.placeholder(tf.float32, [None, 1], name='input_returns')

        with tf.name_scope('hidden'):
            prev_layer = self.input_states
            conv_layers = []
            for n_maps, kernel_size, stride, padding, activation in zip(
                    CONV_N_MAPS, CONV_KERNEL_SIZES,
                    CONV_STRIDES, CONV_PADDINGS, CONV_ACTIVATION):
                prev_layer = tf.contrib.layers.conv2d(prev_layer, num_outputs=n_maps,
                                                      kernel_size=kernel_size, stride=stride,
                                                      padding=padding, activation_fn=activation,
                                                      weights_initializer=INITIALISER)
                prev_layer = tf.contrib.layers.max_pool2d(prev_layer, 2, padding='SAME')
                conv_layers.append(prev_layer)
            flatten_layer = tf.contrib.layers.flatten(prev_layer)
            hidden_layer = tf.layers.dense(flatten_layer,
                                           units=N_HIDDEN,
                                           activation=HIDDEN_ACTIVATION,
                                           kernel_initializer=INITIALISER)
            logits = tf.layers.dense(
                hidden_layer, self.config['ACTION_SPACE'], name='action_layer')

        with tf.name_scope('outputs'):
            self.policy = tf.nn.softmax(logits)
            self.value = tf.layers.dense(hidden_layer, 1, name='value_layer')

        with tf.name_scope('loss'):
            log_probs = tf.log(self.policy + 1e-6)
            log_pi_a_given_s = tf.reduce_sum(log_probs * self.input_actions, axis=1, keep_dims=True)
            advantage = tf.subtract(tf.stop_gradient(self.value),
                                    self.input_returns, name='advantage')
            # No importance for now
            policy_loss = tf.identity(log_pi_a_given_s * advantage, name='policy_loss')
            value_loss = self.config['LOSS_V'] * \
                tf.squared_difference(self.value, self.input_returns, name='value_loss')
            entropy = self.config['LOSS_ENTROPY'] * \
                tf.reduce_sum(self.policy * log_probs, name='xentropy_loss', axis=1,
                              keep_dims=True)

            loss_total = tf.reduce_mean(policy_loss + value_loss + entropy)
            tf.summary.scalar('total_loss', loss_total)
            self._record_mean(policy_loss, 'policy_loss')
            self._record_mean(entropy, 'entropy_loss')
            self._record_mean(value_loss, 'value_loss')

        with tf.name_scope('train'):
            #optimizer = tf.train.RMSPropOptimizer(self.config['LEARNING_RATE'], decay=.99)
            optimizer = tf.train.AdamOptimizer(self.config['LEARNING_RATE'], epsilon=1e-3)
            self.train_op = optimizer.minimize(loss_total)

        self.writer = tf.summary.FileWriter(os.path.join('tb', time.strftime('%H%M%S')))

        print('BRAIN INFO: graph completed.')
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        self.writer.add_graph(self.sess.graph)

        with tf.name_scope('mics'):
            self._record_mean(self.input_returns, 'input_returns')
            self._record_mean(self.value, 'values')
            self._record_mean(advantage, 'advantage')
            tf.summary.histogram('input_actions', self.input_actions)

        # Add summaries to tensorboards
        for e in tf.trainable_variables():
            tf.summary.histogram(e.name.replace(':', ''), e)

        self.merged_summary = tf.summary.merge_all()

        # Saver
        self.saver = tf.train.Saver()

    def _record_mean(self, v, name):
        """ private function for reporting in tensorboard """
        mean_v = tf.reduce_mean(v, name=name)
        tf.summary.scalar(name.format(name), mean_v)

    def optimiser(self):
        """ Core optimisation function """
        if len(self.train_queue[0]) < self.config['MIN_BATCH_SIZE']:
            time.sleep(0)
            pass

        with self.lock_queue:
            if len(self.train_queue[0]) < self.config['MIN_BATCH_SIZE']:
                return  # More thread could have passed without lock

            s, a, g, sp, sm = self.train_queue
            self.train_queue = [[], [], [], [], []]

            s = np.stack(s)
            a = np.vstack(a)
            g = np.vstack(g)
            sp = np.stack(sp)
            sm = np.vstack(sm)

            if self.config['TENSORBOARD']:
                # TODO: write to tensorboard may belong somewhere else?
                # Write to tensorboard
                t = int((time.time() - self.init_time) * 100)
                summary = self.sess.run(self.merged_summary,
                                        feed_dict={self.input_states: s,
                                                   self.input_actions: a,
                                                   self.input_returns: g})
                self.writer.add_summary(summary, t)

            if len(s) > 2 * self.config['MIN_BATCH_SIZE']:
                print('BRAIN INFO {}: training a batch of size: {}'.format(t, len(s)))

            v = self.sess.run(self.value, feed_dict={self.input_states: sp})
            g = g + self.config['GAMMA_N'] * v * sm  # set v to 0 when sp is terminal
            self.sess.run(self.train_op, feed_dict={self.input_states: s,
                                                    self.input_actions: a,
                                                    self.input_returns: g})

    def push_to_queue(self, s, a, r, sp):
        """ recieveing training examples from Agents """
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if sp is None:
                self.train_queue[3].append(np.zeros([INPUT_HEIGHT,
                                                     INPUT_WIDTH,
                                                     INPUT_CHANNELS]))
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(sp)
                self.train_queue[4].append(1.)

    def get_epsilon(self):
        """ epsilon getter for Agents """
        t = time.time()
        if (t >= self.eps_end_time):
            eps = self.config['EPS_END']
        else:
            eps = self.config['EPS_START'] + (self.config['EPS_END'] - self.config['EPS_START']
                                              ) * (t - self.init_time) / self.config['EPS_TIME']
        return eps

    def decide(self, s):
        """ policy function
        Args:
          s: state
        Returns:
          a: action
        """
        p = self.policy.eval(feed_dict={self.input_states: [s]}, session=self.sess)
        a = np.random.choice(self.config['ACTION_SPACE'], p=p[0, :])
        return a

    def save_session(self, path):
        """ saves existing session
        """
        self.saver.save(self.sess, path)

    def load_session(self, path):
        """ load from a pre-exsiting session
        """
        print('BRAIN INFO: loading session from {}'.format(path))
        self.saver.restore(self.sess, path)


class Optimiser(threading.Thread):
    """ A multi-thread wrapper for the optimiser
    """

    def __init__(self, func):
        threading.Thread.__init__(self)
        self.stop_signal = False
        self.func = func

    def run(self):
        while not self.stop_signal:
            self.func()

    def stop(self):
        self.stop_signal = True
