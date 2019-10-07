import numpy as np
import gdal
import osgeo.gdalnumeric as gdn
from PIL import Image
import matplotlib.pyplot as plt

# Function to load multi-spectral raster to numpy array
# https://gis.stackexchange.com/questions/32995/fully-load-raster-into-a-numpy-array
def img_to_array(input_file, dim_ordering="channels_last", dtype='float32':
    file  = gdal.Open(input_file)
    bands = [file.GetRasterBand(i) for i in range(1, file.RasterCount + 1)]
    arr = np.array([gdn.BandReadAsArray(band) for band in bands])

    if dim_ordering=="channels_last":
        arr = np.transpose(arr, [1, 2, 0])  # Reorders dimensions, so that channels are last

    return arr

# Function to load geotif labels to numpy array
def labels_to_array(input_file):
    labels = Image.open(input_file)
    arr = np.array(labels)

    return arr

# Function for plotting RGB image from multispectral bands
def plot_image(image, factor=1):
    fig = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))

    if np.issubdtype(image.dtype, np.floating):
        plt.imshow(np.minimum(image * factor, 1))
    else:
        plt.imshow(image)
