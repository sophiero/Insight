from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from processing import data_processing
import sys

# Reshape data so that columns are spectral bands and rows are pixels
def reshape_data(X, y):
    X_reshaped = X.reshape(-1, X.shape[-1])
    y_reshaped = y.reshape(-1)

    return X_reshaped, y_reshaped

def standardizer(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def build_pipeline():
    pipeline = Pipeline([
        ('standarize', FunctionTransformer(standardizer, validate=False)),
        ('log_reg', LogisticRegression(solver='lbfgs', max_iter=300))])

    return pipeline

if __name__ == "__main__":

    # loading and shaping data
    X, y = data_processing.load_data(year=2017)
    X, y = reshape_data(X, y)

    # train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

    # training
    log_reg_pipeline = build_pipeline()
    log_reg_pipeline.fit(X_train, y_train)

    # results
    val_confusion_matrix = confusion_matrix(y_val, y_pred)

    print('Accuracy: %.2f' % accuracy_score(y_val, y_pred))
    print('Intersection-over-Union: %.2f' % iou(val_confusion_matrix))
    print('Confusion Matrix: \n', val_confusion_matrix)
    print('Classification report:\n', classification_report(y_val, y_pred))
