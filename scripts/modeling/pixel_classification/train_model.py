from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from processing import data_processing
import sys, os

MODEL_PATH = '../../models'

def build_pipeline():
    """ Builds logistic regression pipeline with standardization """

    pipeline = Pipeline([
        ('standarize', FunctionTransformer(standardizer, validate=False)),
        ('log_reg', LogisticRegression(solver='lbfgs', max_iter=300))])

    return pipeline

def standardizer(X):
    """ Standardizes features by removing mean and scaling to unit variance  """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def save_model(pipeline, filename):
    """ Saves model from pipeline to file """

    model = pipeline.named_steps['log_reg']
    pickle.dump(model, open(os.path.join(MODEL_PATH, filename), 'wb'))


def intersection_over_union(confusion_matrix):
    """ Intersection-over-union metric for image segmentation """

    tn, fp, fn, tp = confusion_matrix.ravel()
    iou = tp / (tp + fn + fp)

    return iou

if __name__ == "__main__":

    # loading and shaping data
    X, y = data_processing.load_data(year=2017)
    X, y = data_processing.reshape_data(X, y)

    # train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

    # training
    log_reg_pipeline = build_pipeline()
    log_reg_pipeline.fit(X_train, y_train)

    # results
    confusion_matrix = confusion_matrix(y_val, y_pred)

    print('Accuracy: %.2f' % accuracy_score(y_val, y_pred))
    print('Intersection-over-Union: %.2f' % intersection_over_union(confusion_matrix))
    print('Confusion Matrix: \n', confusion_matrix)
    print('Classification report:\n', classification_report(y_val, y_pred))
