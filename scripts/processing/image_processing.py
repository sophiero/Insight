import numpy as np
import gdal
import osgeo.gdalnumeric as gdn
from PIL import Image
import matplotlib.pyplot as plt
from keras_unet.utils import get_patches


def img_to_array(input_file, dim_ordering="channels_last", dtype='float32':
    """
    Loads multi-spectral raster to numpy array using gdal.

    Parameters:
    input_file (tiff): multi-spectral raster
    dim_ordering (string): reorders dimensions, so that channels are last

    Returns:
    arr (ndarray): 2D array
    """

    file  = gdal.Open(input_file)
    bands = [file.GetRasterBand(i) for i in range(1, file.RasterCount + 1)]
    arr = np.array([gdn.BandReadAsArray(band) for band in bands])

    if dim_ordering=="channels_last":
        arr = np.transpose(arr, [1, 2, 0])  # Reorders dimensions, so that channels are last

    return arr

def labels_to_array(input_file):
    """ Loads geotif labels to numpy array """

    labels = Image.open(input_file)
    arr = np.array(labels)

    return arr


def get_img_patches(X, y):
    """
    Takes array of images and returns crops using sliding window method.
    If stride < size it will do overlapping.
    """

    X_crops = []
    y_crops = []

    CROP_SIZE = 512

    for i, x in enumerate(X):
        x_sample = get_patches(
            img_arr=x,
            size=CROP_SIZE,
            stride=CROP_SIZE)

        y_sample = get_patches(
          img_arr=y[i].reshape(HEIGHT,WIDTH,1),
          size=CROP_SIZE,
          stride=CROP_SIZE)

        X_crops.append(x_sample)
        y_crops.append(y_sample)

    X_crops = np.array(X_crops).reshape(X.shape[0] * X.shape[1],
                                        X.shape[2],
                                        X.shape[3],
                                        X.shape[4])

    y_crops = np.array(y_crops).reshape(y.shape[0] * y.shape[1],
                                        y.shape[2],
                                        y.shape[3],
                                        y.shape[4])

    return X_crops, y_crops


def plot_image(image, factor=1):
    """ Plots RGB image from multispectral bands """
    fig = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))

    if np.issubdtype(image.dtype, np.floating):
        plt.imshow(np.minimum(image * factor, 1))
    else:
        plt.imshow(image)
