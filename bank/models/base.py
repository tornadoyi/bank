import os
import numpy as np
import tensorflow as tf


class ModelBase(object):

    def __init__(self, sess, save_path, xdims, learning_rate=0.1):
        self._sess = sess
        self._save_path = save_path
        self._xdims = xdims
        self._learning_rate = learning_rate
        self._process = {}

        #  init statistics ops
        with tf.variable_scope('statistics'):
            self._init_statistics_ops()

        # create model
        with tf.variable_scope('model'):
            self._create_model()

        # create saver
        self._saver = tf.train.Saver(tf.global_variables(), filename='bank')



    @property
    def steps(self): return self._sess.run(self._op_steps)


    @property
    def accuracy(self): return self._sess.run(self._op_acc)


    @accuracy.setter
    def accuracy(self, v): self._sess.run(self._op_update_acc, feed_dict={self._op_set_acc: v})


    @property
    def process(self): return self._process


    def train(self, xs, ys):
        # steps +1
        self._sess.run(self._op_add_steps)

        return self._train(xs, ys)


    def predict(self, xs): return self._predict(xs)


    def _train(self, xs, ys): raise NotImplementedError('need to implement train')


    def _predict(self, xs): raise NotImplementedError('need to implement predict')


    def _create_model(self): raise NotImplementedError('need to implement predict')


    def save(self, *args, **kwargs):
        self._saver.save(self._sess, self._save_path, *args, **kwargs)
        print('save model to {}'.format(self._save_path))


    def restore(self):
        self._saver.restore(self._sess, self._save_path)
        print('restore model from {}'.format(self._save_path))



    def _init_statistics_ops(self):
        # steps
        self._op_steps = tf.Variable(0, dtype=tf.int64)
        self._op_add_steps = tf.assign_add(self._op_steps, 1)

        # accuracy
        self._op_acc = tf.Variable(0.0, dtype=tf.float32)
        self._op_set_acc = tf.placeholder(dtype=tf.float32)
        self._op_update_acc = tf.assign(self._op_acc, self._op_set_acc)


