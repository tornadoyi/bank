import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from .base import ModelBase

class LogisticsRegression(ModelBase):

    def __init__(self, *args, **kwargs):
        super(LogisticsRegression, self).__init__(*args, **kwargs)




    def train_step(self, xs, ys):
        _, loss, grads = self._sess.run(
            [self._op_train, self._loss, self._grads],
            feed_dict = {
                self._inputs: xs,
                self._labels: ys
            }
        )

        total_grads = 0
        for g in grads: total_grads += np.mean(np.abs(g))

        return loss, total_grads / len(grads)


    def predict(self, xs):
        probs = self._sess.run(
            [self._op_predict],
            feed_dict={
                self._inputs: xs
            })[0]
        return probs


    def _create_model(self):


        self._inputs = tf.placeholder(shape=[None, self._xdims], dtype=tf.float32)
        self._labels = tf.placeholder(shape=[None], dtype=tf.float32)

        l = self._inputs
        hidden_layers = [128, 1]
        for i in range(len(hidden_layers)):
            l = tf.layers.dense(
                l,
                hidden_layers[i],
                activation=tf.nn.tanh,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.1),
                #kernel_regularizer=slim.l2_regularizer(0.05),
                #bias_regularizer=slim.l2_regularizer(0.05),
            )

        lout = tf.reshape(l, shape=[-1])


        # loss
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=lout, labels=self._labels)
        self._loss = tf.reduce_mean(cross_entropy)


        # train
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        vars = tf.global_variables()
        self._grads = tf.gradients(self._loss, vars)
        self._op_train = optimizer.apply_gradients(zip(self._grads, vars))


        # predict
        self._op_predict = tf.nn.sigmoid(lout)






