# Corn Acerage Forecasting using Satellite Imagery
Insight AI project

## Dataset Sources

Two datasets, Sentinel-2 satellite imagery and Cropland Data Layer, are used for this project.
The area of interest was Southeast Nebraska, an argricultural region of the U.S.


### 1. Sentinel-2 Satellite Imagery

Sentinel-2 is a open satellite imagery source, with global coverage recaptured every 5 days. The images are acquired at a high spatial resolution (10m), offering multi-spectral data with 13 bands in the visible, near infrared, and short wave infrared part of the spectrum.
The python library sentinel-hub was used to download Sentinel-2 satellite imagery.

Below is a sample satellite image from the region of interest:

<img src="https://github.com/sophiero/Insight/blob/master/notebooks/figures/satellite_sample.png" width="250"/>


### 2. Cropland Data Layer

CropScape is a web service that offers crop-specific land cover data layer based on USDA agricultural reports.
Crop labels were downloaded by area of interest, year and crop from https://nassgeodata.gmu.edu/CropScape/.

Below are the corresponding crop labels for the sample image shown above:

<img src="https://github.com/sophiero/Insight/blob/master/notebooks/figures/labels_sample.png" width="250"/>


## Modeling Approaches

For remote sensing problems, there are two main approaches existing in literature. The result of both methods is the same - given an input of a satellite image, the model should output a mask of the same size that assigns a label (corn/not corn) to every pixel in the image.

### 1. Single Pixel Classification

In the single pixel classification approach, the model learns to classify each pixel as corn/not corn (0/1). We use the 13 spectral bands of the satellite image as features, with each data instance represented as a pixel. This is a naive approach that considers only the context (spectral bands) to determine the corn crops from the satellite image. The model does not capture spatial information. We train a logistic regression for binary classification of the pixels.

### 2. Image Segmentation

In the image segmentation approach, the input are patches of the satellite image rather than single pixels. Given this input, the model is able to learn both contextual and spatial information. A convolutional neural network called U-Net was developed specifically for image segmentation tasks. The symmetric contracting and expanding architecture of the model allows the model to extract features and increase the resolution of the output. We trained a custom U-Net model with 32 filters and 4 layers for the corn crop segmentation problem.

## Results
