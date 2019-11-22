from keras_unet.metrics import iou
from keras.models import load_model
from processing import data_processing, image_processing

import os

MODEL_PATH = '../../models'

def standardize(X):

    # flatten X to 2D array
    X_reshaped = X.reshape(-1, X.shape[-1])

    # standardize on columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # reshape X to original shape
    X_final = X_scaled.reshape(X.shape)

    return X_final

if __name__ == "__main__":

    X, y = data_processing.load_data(year=2018)
    X_scaled = standardize(X)

    X_crops, y_crops = image_processing.get_img_patches(X_scaled, y)

    # load trained model and predict
    dependencies = { 'iou': iou }
    model = load_model(os.path.join(MODEL_PATH,'segm_model_v3.h5'), custom_objects=dependencies)

    y_pred = model.predict(X_crops)
    y_pred_binary = np.round_(y_pred, 0) # Rounding probabilities to 0/1

    y_true_flat, y_pred_flat = y_crops.flatten(), y_pred_binary.flatten()

    # results
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    print('Accuracy: %.2f' % accuracy_score(y_true_flat, y_pred_flat))
    print('Intersection-over-Union: %.2f' % iou(cm))
    print('Confusion Matrix: \n', cm)
    print('Classification report:\n', classification_report(y_true_flat, y_pred_flat))
