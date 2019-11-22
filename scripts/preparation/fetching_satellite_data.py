import numpy as np
from shapely.geometry import shape, Polygon
from sentinelhub import BBox, OsmSplitter read_data, CRS, DataSource
import gdal
import osgeo.gdalnumeric as gdn

# Sentinel-Hub configuration
INSTANCE_ID = '29ce4ad4-0be2-42db-ac61-144fbc856641'
LAYER = 'BANDS-S2-L1C'
DATA_FOLDER = '../data/raw/'

def get_bounding_boxes(geojson, crs=CRS.WGS84):
    """
    A tool that converts GeoJSON file to Shapely polygon format and splits the given area into smaller parts.
    Returns the lat/lon bounding box for each part.

    Parameters:
    geojson (GeoJSON): A GeoJSON file describing the area of interest
    crs (CRS): Coordinate reference system of GeoJSON file

    Returns:
    bounding_boxes (ndarray): Array of bounding boxes
    """

    # Loading geometry of area of interest (AOI)
    geojson = read_data(geojson)
    AOI = shape(geojson['features'][0]['geometry'])

    # Split the area into smaller bounding boxes which can be used for obtaining data by calling WMS/WCS requests.
    osm_splitter = OsmSplitter([AOI], crs, zoom_level=10)

    bounding_boxes = []
    for bbox in osm_splitter.get_bbox_list():
      bounding_boxes.append(list(bbox))
    bounding_boxes = np.array(bounding_boxes)

    return bounding_boxes

def request_data(AOI, time_range, cloud_cover=0.0):
    """
    Creates an instance of Sentinel Hub WMS (Web Map Service) request,
    which provides access to Senintel-2's unprocessed bands.

    Parameters:
    AOI (list): Lat/lon coordinates of bounding box representing area of interest
    time (str or(str, str)): time or time range for which to resutn the results (year-mond-date format)
    cloud_cover (float): maximum accepted cloud coverage of an image. Float between 0.0 and 1.0

    Returns:
    wms_bands_img (ndarray): Array of bounding boxes
    """

    AOI_bbox = BBox(bbox=AOI, crs=CRS.WGS84)

    wms_bands_request = WcsRequest(layer=LAYER,
                               bbox=AOI_bbox,
                               time=time_range, # acquisition date
                               maxcc=cloud_cover,
                               resx='10m', resy='10m',
                               image_format=MimeType.TIFF,
                               instance_id=INSTANCE_ID,
                               data_folder=DATA_FOLDER)

    wms_bands_img = wms_bands_request.get_data(save_data=True) # Save downloaded data to disk
    return wms_bands_img

def img_to_array(input_file, dim_ordering="channels_last", dtype='float32'):
    """
    Loads multi-spectral raster to numpy array using gdal.
    Based on: https://gis.stackexchange.com/questions/32995/fully-load-raster-into-a-numpy-array

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


def main():

    geo_file = '../data/york.geo.json'
    bboxes = get_bounding_boxes(geo_file, crs=CRS.WGS84)

    # fetching data using bounding boxes of geojson region
    time_range = ('2018-07-01', '2018-07-15')
    cloud_cover = 0.0

    for bbox in bboxes:
        satellite_imgs = request_data(bbox, time_range, cloud_cover)

if __name__== "__main__":
  main()
