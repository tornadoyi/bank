


class ModelBase(object):

    def __init__(self, sess):
        self.__sess = sess



    def train(self, xs, ys): raise NotImplementedError('need to implement train')


    def predict(self, xs): raise NotImplementedError('need to implement predict')

