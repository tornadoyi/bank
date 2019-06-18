import time
import numpy as np
import tensorflow as tf
from bank.datas import load_data
from bank.models import LogisticsRegression


def cmd_test(args):
    # load datas
    dl = load_data(args.data_path)

    # create model
    sess = tf.Session()
    model = LogisticsRegression(sess, args.save_path, dl.xdims, 1e-3)
    model.restore()

    # predict
    xs, ys = dl.datasets
    probs = model.predict(xs)

    # calculate
    cond = (probs >= 0.5) == (ys == 1.0)
    acc = np.sum(cond) / len(cond) * 100

    # save
    print(acc)