# Corn Acerage Forecasting using Satellite Imagery
Insight AI project

## Dataset Sources

Two datasets, Sentinel-2 satellite imagery and Cropland Data Layer, are used for this project.
The area of interest for this project was Southeast Nebraska, an argricultural region  high fertility of the soils


### 1. Sentinel-2 Satellite Imagery

Sentinel-2 is a open satellite imagery source, with global coverage recaptured every 5 days. The images are acquired at a high spatial resolution (10m), offering multi-spectral data with 13 bands in the visible, near infrared, and short wave infrared part of the spectrum.

The python library sentinel-hub was used to download Sentinel-2 satellite imagery for this project.

![alt text]('https://github.com/sophiero/Insight/blob/master/notebooks/figures/satellite_sample.png')



### 2. Cropland Data Layer

CropScape is a web service that offers crop-specific land cover data layer based on USDA agricultural reports.
Crop labels were downloaded by area of interest, year and crop from https://nassgeodata.gmu.edu/CropScape/.

![alt text]('https://github.com/sophiero/Insight/blob/master/notebooks/figures/labels_sample.png')


## Model Approaches

1. Single Pixel Classification



2. Image Segmentation
