import numpy as np 
from src.hmm_model import GaussianHMM

def train_hmm_models(train_data):
    digit_data = {i: [] for i in range(10)}

    for mfcc, digit in train_data:
        digit_data[digit].append(mfcc)

    models = {}

    for digit in range(10):
        hmm = GaussianHMM(n_states=3)
        hmm.initialize(digit_data[digit])
        hmm.baum_welch(digit_data[digit], n_iter=5)

        models[digit] = hmm

    return models   


def predict_digit_hmm(mfcc, models):
    scores = []

    for digit in range(10):
        hmm = models[digit]
        score = hmm.forward(mfcc)
        scores.append(score)

    return int(np.argmax(scores))
