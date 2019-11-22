import numpy as np

from sklearn.preprocessing import StandardScaler
from processing import data_processing, image_processing

import keras_unet
from keras_unet.utils import get_augmented
from keras_unet.models import custom_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

def reshape_crops(X, y):

    X_reshaped = X.reshape(X.shape[0] * X.shape[1],
                            X.shape[2],
                            X.shape[3],
                            X.shape[4])

    y_reshaped = y.reshape(y.shape[0] * y.shape[1],
                            y.shape[2],
                            y.shape[3],
                            1)

    return X_reshaped, y_reshaped

def intersection_over_union(confusion_matrix):
    """ Intersection-over-union metric for image segmentation """

    tn, fp, fn, tp = confusion_matrix.ravel()
    iou = tp / (tp + fn + fp)

    return iou

if __name__ == "__main__":

    X, y = data_processing.load_data(year=2017)
    X_scaled = standardize(X)

    # crop images into patches
    X_crops, y_crops = image_processing.get_img_patches(X_scaled, y)
    X_train, X_val, y_train, y_val = train_test_split(X_crops, y_crops, test_size=0.2, shuffle=False)

    # data augmentation with horizontal and vertical flips
    train_gen = get_augmented(
        X_train, y_train, batch_size=2,
        data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
    ))

    # initilaize network
    model = custom_unet(
        input_shape=X_train[0].shape,
        filters=32,
        use_batch_norm=True,
        num_classes=1,
        dropout=0.3,
        num_layers=4
    )

    # compile and train
    callback_checkpoint = ModelCheckpoint(
        MODEL_PATH,
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
    )

    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=[iou]
    )

    # train model
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=[callback_checkpoint]
    )

    # results
    y_pred = model.predict(X_val)
    y_pred_binary = np.round_(y_pred, 0)

    cm = confusion_matrix(y_val.flatten(), y_pred_binary.flatten())

    print('Accuracy: %.2f' % accuracy_score(y_pred_binary.flatten(), y_val.flatten()))
    print('IoU: %.2f' % iou(cm))
    print('Confusion Matrix: \n',cm)
    print('Classification report:\n', classification_report(y_pred_binary.flatten(), y_val.flatten()))
