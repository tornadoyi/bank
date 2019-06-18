import os
import numpy as np


# data_file_path = os.path.join(os.path.basename(__file__), 'bankTraining.csv')


class DataLoader(object):
    def __init__(self, data_file_path):
        self.__data_file_path = data_file_path

        self.__titles = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                         'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate',
                         'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']

        self.__xs, self.__ys = DataLoader.__load_embedding(self.__data_file_path)

        self.__train_xs, self.__train_ys, self.__test_xs, self.__test_ys = DataLoader.split_datasets(self.__xs, self.__ys)



    @property
    def datasets(self): return self.__xs, self.__ys

    @property
    def training(self): return self.__train_xs, self.__train_ys

    @property
    def test(self): return self.__test_xs, self.__test_ys



    @staticmethod
    def split_datasets(xs, ys):
        idx = int(len(xs) * 0.2)

        return xs[:, :-idx], ys[:, :-idx], xs[:, -idx:], ys[:, -idx:]




    @staticmethod
    def __load_raw(data_file_path):

        with open(data_file_path, 'r') as f:
            lines = f.readlines()

        splits = []
        for i in range(len(lines)):
            ln = lines[i]
            s = ln.rstrip('\n').rstrip('\r').split()
            splits.append(s)

        datas = splits[1:]

        xs, ys = [], []
        for i in range(len(datas)):
            d = datas[i]
            xs.append(d[:-1])
            ys.append(d[-1])

        return np.array(xs, dtype=np.str), np.array(ys, dtype=np.str)



    @staticmethod
    def __load_embedding(data_file_path):
        # load
        with open(data_file_path, 'r') as f:
            lines = f.readlines()

        splits = []
        for i in range(len(lines)):
            ln = lines[i]
            s = ln.rstrip('\n').rstrip('\r').split()
            splits.append(s)

        titles, datas = splits[0], splits[1:]


        # create embeded dict
        embeded_dict = DataLoader.__create_embedding_dict()

        # embedding
        em_datas = np.empty((len(datas), 0), dtype=np.float)
        for i in range(len(titles)):
            t = titles[i]
            f = embeded_dict[t]
            if f is None: continue
            _, x = f(datas[:, i])
            em_datas = np.vstack([em_datas, x])

        return em_datas[:, :-1], em_datas[:, -1]




    @staticmethod
    def __create_embedding_dict():

        def __float(ds): return None, ds.astype(np.float)

        def __norm(ds, base): return None, ds.astype(np.float) / base

        def __norm10(ds): return __norm(ds, 10.0)

        def __norm100(ds): return __norm(ds, 100.0)

        def __norm10000(ds): return __norm(ds, 10000.0)

        def __enum(ts, ds):
            fs = np.empty((len(ds), len(ts)), dtype=np.float)

            e = np.eye(len(ts), dtype=np.float)

            for i in range(len(ts)):
                t = ts[i]
                cond = (ds == t)
                fs[cond] = e[i]

            return ts, fs

        def __job(ds):
            ts = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed',
                  'services', 'student', 'technician', 'unemployed', 'unknown']
            return __enum(ts, ds)

        def __martial(ds):
            ts = ['divorced', 'married', 'single', 'unknown']
            return __enum(ts, ds)

        def __education(ds):
            ts = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course',
                  'university.degree', 'unknown']
            return __enum(ts, ds)

        def __check(ds): return __enum(['no', 'yes'], ds)

        def __check3(ds): return __enum(['no', 'unknown', 'yes'], ds)

        def __contact(ds): return __enum(['cellular', 'telephone'], ds)

        def __month(ds): return __enum(
            ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], ds)

        def __day_of_week(ds): __enum(['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'], ds)

        def __pdays(ds):
            ds = ds.astype(np.float)
            ds[ds == 999] = 0
            return None, ds / 100.0

        def __poutcome(ds): return __enum(["failure", "nonexistent", "success"], ds)

        embeded_dict = {
            'age': __norm100,
            'job': __job,
            'marital': __martial,
            'education': __education,
            'default': __check3,
            'housing': __check3,
            'loan': __check3,
            'contact': __contact,
            'month': __month,
            'day_of_week': __day_of_week,
            'duration': None,  # __norm10000,        # max 3643
            'campaign': __norm100,
            'pdays': __pdays,
            'previous': __norm10,
            'poutcome': __poutcome,
            'emp.var.rate': __float,
            'cons.price.idx': __norm100,
            'cons.conf.idx': __norm100,
            'euribor3m': __float,
            'nr.employed': __norm10000,
            'y': __check

        }

        return embeded_dict








