import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from processing import data_processing, reshape_data

MODEL_PATH = '../../models'

def load_model(filename):
    """ Loads trained model from file  """

    with open(os.path.join(MODEL_PATH, filename), 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model

def standardizer(X):
    """ Standardizes features by removing mean and scaling to unit variance  """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

def intersection_over_union(confusion_matrix):
    """ Intersection-over-union metric for image segmentation """

    tn, fp, fn, tp = confusion_matrix.ravel()
    iou = tp / (tp + fn + fp)
    return iou

if __name__ == "__main__":

    # loading 2018 data
    X, y = data_processing.load_data(2018)

    # loading trained model
    log_reg = load_model('log_reg.pkl')

    # preprocessing
    X, y = data_processing.reshape_data(X, y)
    X_scaled = standardizer(X)

    # making predictions
    y_pred = log_reg.predict(X_scaled)

    # results
    confusion_matrix = confusion_matrix(y_val, y_pred)

    print('Accuracy: %.2f' % accuracy_score(y_val, y_pred))
    print('Intersection-over-Union: %.2f' % intersection_over_union(confusion_matrix))
    print('Confusion Matrix: \n', confusion_matrix)
    print('Classification report:\n', classification_report(y_val, y_pred))
