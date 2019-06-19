import os
import numpy as np
import tensorflow as tf


class ModelBase(object):

    def __init__(self, sess, save_path, xdims, learning_rate=0.1):
        self._sess = sess
        self._save_path = save_path
        self._xdims = xdims
        self._learning_rate = learning_rate

        # create model
        self._create_model()

        # create saver
        self._saver = tf.train.Saver(tf.global_variables(), filename='bank')



    def train_step(self, xs, ys): raise NotImplementedError('need to implement train')


    def predict(self, xs): raise NotImplementedError('need to implement predict')


    def _create_model(self): raise NotImplementedError('need to implement predict')


    def save(self, *args, **kwargs):
        self._saver.save(self._sess, self._save_path, *args, **kwargs)
        print('save model to {}'.format(self._save_path))


    def restore(self):
        self._saver.restore(self._sess, self._save_path)
        print('restore model from {}'.format(self._save_path))
