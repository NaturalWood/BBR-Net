import tensorflow as tf
import numpy as np
from networks import *

class PPONet(object):
    def __init__(self,a_dim,
                    s_dim,
                    a_lr=1e-3,
                    c_lr=1e-3,
                    update_freq=15,
                    epsilon=0.2,):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.c_lr = c_lr
        self.a_lr = a_lr
        self.update_freq = update_freq
        self.epsilon = epsilon
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        # critic
        w_init = tf.random_normal_initializer(0., .1)
        lc = tf.layers.dense(self.tfs, 200, tf.nn.relu, kernel_initializer=w_init, name='lc')
        self.v = tf.layers.dense(lc, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(self.c_lr).minimize(self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        self.oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)   # shape=(None, )
        oldpi_prob = tf.gather_nd(params=self.oldpi, indices=a_indices)  # shape=(None, )
        ratio = tf.exp(pi_prob - oldpi_prob)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(self.a_lr).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self,s,a,r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # update actor and critic in a update loop
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a.ravel(), self.tfadv: adv}) for _ in range(self.update_freq)]
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(self.update_freq)]


    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l_a = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            a_prob = tf.layers.dense(l_a, A_DIM, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def choose_action(self, s):  # run by a local
        prob_weights = self.sess.run(self.oldpi, feed_dict={self.tfs: s[None, :]})

        action = np.random.choice(range(prob_weights.shape[1]),
                                      p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
