import numpy as np
from shapely.geometry import shape, Polygon
from sentinelhub import BBox, OsmSplitter read_data, CRS, DataSource

# Sentinel-Hub configuration
INSTANCE_ID = '29ce4ad4-0be2-42db-ac61-144fbc856641'
LAYER = 'BANDS-S2-L1C'
DATA_FOLDER = './data/raw/'

def get_bounding_boxes(geo_json_file):

    # Loading geometry of area of interest (AOI)
    geo_json = read_data(geo_json_file)
    AOI = shape(geo_json['features'][0]['geometry'])

    ## Split the area into smaller bounding boxes which can be used for obtaining data by calling WMS/WCS requests.
    osm_splitter = OsmSplitter([AOI], CRS.WGS84, zoom_level=10)

    bounding_boxes = []
    for bbox in osm_splitter.get_bbox_list():
      bounding_boxes.append(list(bbox))
    bounding_boxes = np.array(bounding_boxes)

    return bounding_boxes

def request_data(AOI, time_range, cloud_cover=0.0):

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
