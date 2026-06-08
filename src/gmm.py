import numpy as np 
from src.gmm_model import GaussianMixtureModel


def train_gmm_models(train_data):
    # grouping by digit
    digit_data = {i: [] for i in range(10)}
    for mfcc, digit in train_data:
        digit_data[digit].append(mfcc)

    models = {}

    for digit in range(10):
        X = np.vstack(digit_data[digit])

        gmm = GaussianMixtureModel(n_components=8, max_iter=50)
        gmm.fit(X)

        models[digit] = gmm

    return models



def predict_digit_gmm(mfcc, models):
    scores = []

    for digit in range(10):
        gmm = models[digit]
        ll = gmm.score(mfcc)
        scores.append(ll)

    return int(np.argmax(scores))


