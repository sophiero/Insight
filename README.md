# Corn Acerage Forecasting using Satellite Imagery
Insight AI project

[View Presentation Slides](https://docs.google.com/presentation/d/1GmVpUNw_DBMHZnufedEwjZ4IvqOwcysUB-lbLHi818U/edit?usp=sharing)

## Dataset Sources

Two datasets, Sentinel-2 satellite imagery and Cropland Data Layer, are used for this project.
The area of interest was Southeast Nebraska, an argricultural region of the U.S.


### 1. Sentinel-2 Satellite Imagery

Sentinel-2 is a open satellite imagery source, with global coverage recaptured every 5 days. The images are acquired at a high spatial resolution (10m), offering multi-spectral data with 13 bands in the visible, near infrared, and short wave infrared part of the spectrum.
The python library [sentinel-hub](https://github.com/sentinel-hub/sentinelhub-py) was used to download Sentinel-2 satellite imagery.

Below is a sample satellite image from the region of interest:

<img src="https://github.com/sophiero/Insight/blob/master/notebooks/figures/satellite_sample.png" width="250"/>

### 2. Cropland Data Layer

CropScape is a web service that offers crop-specific land cover data layer based on USDA agricultural reports.
Crop labels were downloaded by area of interest, year and crop from https://nassgeodata.gmu.edu/CropScape/.

Below are the corresponding crop labels for the sample image shown above:

<img src="https://github.com/sophiero/Insight/blob/master/notebooks/figures/labels_sample.png" width="250"/>

## Modeling Approaches

For remote sensing problems, there are two main approaches existing in literature. The result of both methods is essentially the same - given an input of a satellite image, the model should output a mask of the same size that assigns a label (corn/not corn) to every pixel in the image.

<p align="center">
  <img src="https://github.com/sophiero/Insight/blob/master/notebooks/figures/model_input_output.png" width="600" />
</p>

### 1. Single Pixel Classification

In the single pixel classification approach, the model learns to classify each pixel as corn/not corn (0/1). We use the 13 spectral bands of the satellite image as features, with each data instance represented as a pixel. This is a naive approach that considers only the context (spectral bands) to determine the corn crops from the satellite image. We train a logistic regression for binary classification of the pixels.

<p align="center">
  <img src="https://github.com/sophiero/Insight/blob/master/notebooks/figures/single_pixel_structure.png" width="500"/>
</p>

### 2. Image Segmentation

In the image segmentation approach, rather than single pixels, the input are patches of the image. Given this input, the model is able to learn both contextual and spatial information. A convolutional neural network called U-Net was developed for image segmentation tasks. The symmetric contracting and expanding paths of the model architecture allow the model to extract features and increase the resolution of the output. Using the [keras-unet](https://github.com/karolzak/keras-unet) package, we trained a custom U-Net model with 32 filters and 4 layers for the corn crop segmentation problem.

The figure below depicts the U-Net architecture from the original paper, <br />
[U-net: Convolutional networks for biomedical image segmentation](https://arxiv.org/abs/1505.04597):

<p align="center">
  <img src="https://github.com/sophiero/Insight/blob/master/notebooks/figures/unet.png" width="600" />
</p>

## Results

The two models were trained on July 2017 data, and tested on July 2018 data in the same region. The results show that the image segmentation approach performs better than the single pixel classification approach, with an 10% increase in interesection-over-union. Salt-and-pepper noise can be seen in the single pixel classification predictions, as a result of the model learning on a pixel level not capturing spatial information. The issue does not appear with the image segmentation approach, as the U-Net captures both the contextual and spatial information.

<img src="https://github.com/sophiero/Insight/blob/master/notebooks/figures/results.png" />
