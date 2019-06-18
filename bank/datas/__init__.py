import os
from .data import DataLoader

def load_training_data():
    return DataLoader(os.path.join(os.path.dirname(__file__), 'bankTraining.csv'))


def load_data(data_path):
    return DataLoader(data_path)