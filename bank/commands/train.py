
from bank.datas import load_training_data

def cmd_train(args):
    dl = load_training_data()


    print(dl)