import sys
import time
import numpy as np
import tensorflow as tf
from bank.datas import load_training_data
from bank.models import LogisticsRegression
from prettytable import PrettyTable

BATCH = 128

def cmd_train(args):
    # load datas
    dl = load_training_data()
    ds = rolling_dataset(*dl.training, BATCH)

    # create model
    sess = tf.Session()
    model = LogisticsRegression(sess, args.save_path, dl.xdims, args.learning_rate)
    if args.restore:
        model.restore()
    else:
        sess.run(tf.global_variables_initializer())

    log_time = time.time()
    i = model.steps
    while True:
        # flush stdout
        sys.stdout.flush()

        # check steps
        i += 1
        if i >= args.nsteps: break

        # train
        xs, ys = ds.next_batch()
        model.train(xs, ys)

        # evaluate
        xs, ys = dl.test
        probs = model.predict(xs)
        cond = (probs >= 0.5) == (ys == 1.0)
        acc = np.sum(cond) / len(cond)
        best_acc = model.accuracy

        # output
        if time.time() - log_time > 1.0 or acc > best_acc:
            log_time = time.time()
            # output
            p = model.process
            t = PrettyTable(['steps', 'accuracy', 'loss', 'gradient'])
            t.add_row([
                i,
                acc * 100,
                p['loss'],
                p['grad']
            ])
            print(t)

        # save
        if acc > best_acc:
            model.accuracy = acc
            model.save()

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