import tensorflow as tf
import numpy as np
import threading
import time
import random
import os


class Brain():
    """ A3C Class """

    def __init__(self, config):
        self.config = config
        self.train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
        self.lock_queue = threading.Lock()
        self.build_graph()
        self.frames = 0

        pass

    def build_graph(self):
        tf.reset_default_graph()
        with tf.name_scope('inputs'):
            self.input_states = tf.placeholder(
                tf.float32, [None, self.config['STATE_SPACE']], name='input_states')
            self.input_actions = tf.placeholder(
                tf.float32, [None, self.config['ACTION_SPACE']], name='input_actions')
            self.input_returns = tf.placeholder(tf.float32, [None, 1], name='input_returns')

        with tf.name_scope('hidden'):
            self.hidden_layer = tf.layers.dense(
                self.input_states, 16, tf.nn.relu, name='hidden_layer')
            self.output_layer = tf.layers.dense(
                self.hidden_layer, self.config['ACTION_SPACE'], name='output_layer')
        with tf.name_scope('outputs'):
            self.logits = tf.nn.softmax(self.output_layer)
            self.values = tf.layers.dense(self.output_layer, 1, name='value_layer')

        with tf.name_scope('loss'):
            log_prob = tf.log(tf.reduce_sum(self.logits * self.input_actions,
                                            axis=1, keep_dims=True) + 1e-6)
            advantage = self.input_returns - self.values
            loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
            loss_value = self.config['LOSS_V'] * tf.square(advantage)  # minimize value error
            entropy = self.config['LOSS_ENTROPY'] * tf.reduce_sum(self.logits * tf.log(self.logits + 1e-10), axis=1,
                                                                  keep_dims=True)  # maximize entropy (regularization)
            loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
            tf.summary.scalar('total_loss', loss_total)
            self._record_mean(loss_policy, 'policy_loss')
            self._record_mean(entropy, 'entropy_loss')
            self._record_mean(loss_value, 'value_loss')

        with tf.name_scope('train'):
            optimizer = tf.train.RMSPropOptimizer(self.config['LEARNING_RATE'], decay=.99)
            self.train_op = optimizer.minimize(loss_total)

        self.writer = tf.summary.FileWriter(os.path.join('tb', time.strftime('%H%M%S')))

        print('BRAIN INFO: graph completed.')
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        self.writer.add_graph(self.sess.graph)

        with tf.name_scope('mics'):
            self._record_mean(self.input_returns, 'input_returns')
            self._record_mean(self.values, 'values')
            self._record_mean(advantage, 'advantage')
            tf.summary.histogram('input_actions', self.input_actions)

        # Add summaries to tensorboards
        for e in tf.trainable_variables():
            tf.summary.histogram(e.name.replace(':', ''), e)

        self.merged_summary = tf.summary.merge_all()

    def _record_mean(self, v, name):
        mean_v = tf.reduce_mean(v, name=name)
        tf.summary.scalar(name.format(name), mean_v)

    def optimiser(self):
        if len(self.train_queue[0]) < self.config['MIN_BATCH_SIZE']:
            time.sleep(0)
            pass

        with self.lock_queue:
            if len(self.train_queue[0]) < self.config['MIN_BATCH_SIZE']:
                return  # More thread could have passed without lock

            s, a, g, sp, sm = self.train_queue
            self.train_queue = [[], [], [], [], []]
            self.frames += len(s)

            s = np.vstack(s)
            a = np.vstack(a)
            g = np.vstack(g)
            sp = np.vstack(sp)
            sm = np.vstack(sm)

            if len(s) > 2 * self.config['MIN_BATCH_SIZE']:
                print('BRAIN INFO {}: training a batch of size: {}'.format(self.frames, len(s)))

            v = self.sess.run(self.values, feed_dict={self.input_states: sp})
            g = g + self.config['GAMMA_N'] * v * sm  # set v to 0 when sp is terminal
            self.sess.run(self.train_op, feed_dict={
                          self.input_states: s, self.input_actions: a, self.input_returns: g})

            s = self.sess.run(self.merged_summary,
                              feed_dict={self.input_states: s, self.input_actions: a, self.input_returns: g})
            self.writer.add_summary(s, self.frames)

    def push_to_queue(self, s, a, r, sp):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if sp is None:
                self.train_queue[3].append(np.zeros([self.config['STATE_SPACE']]))
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(sp)
                self.train_queue[4].append(1.)

    def get_epsilon(self):

        if (self.frames >= self.config['EPS_STEPS']):
            eps = self.config['EPS_END']
        else:
            eps = self.config['EPS_START'] + self.frames * (self.config['EPS_END'] - self.config['EPS_START']) / \
                self.config['EPS_STEPS']  # linearly interpolate
        return eps

    def decide(self, s):
        p = self.logits.eval(feed_dict={self.input_states: [s]}, session=self.sess)
        a = np.random.choice(self.config['ACTION_SPACE'], p=p[0, :])
        return a

    def act(self, s):
        eps = self.get_epsilon()
        rand = random.random()
        if rand < eps:
            return random.randint(0, self.config['ACTION_SPACE'] - 1)
        else:
            return self.decide(s)


class Optimiser(threading.Thread):
    def __init__(self, func):
        threading.Thread.__init__(self)
        self.stop_signal = False
        self.func = func

    def run(self):
        while not self.stop_signal:
            self.func()

    def stop(self):
        self.stop_signal = True
