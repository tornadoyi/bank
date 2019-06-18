


def load_training_data():
    import os
    from .data import DataLoader

    return DataLoader(os.path.join(os.path.dirname(__file__), 'bankTraining.csv'))