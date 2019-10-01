from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def standardizer(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Intersection-over-union
# Common metric for image segmentation problems
def iou(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    iou = tp / (tp + fn + fp)
    return iou

log_reg_pipeline = Pipeline([
    ('standarize', FunctionTransformer(standardizer, validate=False)),
    ('log_reg', LogisticRegression(solver='lbfgs', max_iter=300))])

log_reg_pipeline.fit(X_train, y_train)
y_pred = log_reg_pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Intersection-over-Union: %.2f' % iou(cm))
print('Confusion Matrix: \n', cm)
print('Classification report:\n', classification_report(y_test, y_pred))
