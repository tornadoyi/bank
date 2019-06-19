import time
import numpy as np
import tensorflow as tf
from bank.datas import load_training_data
from bank.models import LogisticsRegression

BATCH = 128

def cmd_train(args):
    # load datas
    dl = load_training_data()
    ds = rolling_dataset(*dl.training, BATCH)


    # create model
    sess = tf.Session()
    model = LogisticsRegression(sess, args.save_path, dl.xdims, 0.1)
    if args.restore:
        model.restore()
    else:
        sess.run(tf.global_variables_initializer())

    save_time = log_time = time.time()
    i = 0
    while True:
        i += 1
        if i >= args.nsteps: break
        xs, ys = ds.next_batch()
        loss, grad = model.train_step(xs, ys)

        # save
        if time.time() - save_time > 60.0:
            save_time = time.time()
            model.save()

        # output
        if time.time() - log_time > 1.0:
            log_time = time.time()

            xs, ys = dl.test
            probs = model.predict(xs)
            cond = (probs >= 0.5) == (ys == 1.0)
            acc = np.sum(cond) / len(cond) * 100

            print(loss, grad, acc)

    # save
    model.save()









def rolling_dataset(*args, **kwargs):
    class _Dataset(object):
        def __init__(self, xs, ys, batch):
            self.xs = xs
            self.ys = ys
            self.batch = batch
            self.index = 0

        def next_batch(self):

            last = self.index + self.batch
            if last <= len(self.xs):
                xs, ys = self.xs[self.index:last], self.ys[self.index:last]
                self.index = last
            else:
                xs, ys = self.xs[self.index:], self.ys[self.index:]
                remain = last - len(self.xs)

                xs = np.concatenate([xs, self.xs[0:remain]], axis=0)
                ys = np.concatenate([ys, self.ys[0:remain]], axis=0)

                self.index = remain

            return xs, ys

    return _Dataset(*args, **kwargs)