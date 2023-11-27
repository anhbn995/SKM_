import rasterio
import pyproj
import os
import math

def get_utm_from_wgs(lon, lat):
    """
    Use longitude, latitude of location for get EPSG code.

    Parameters
    ----------
    lon,lat :
        Longitude, latitude of location you want to get EPSG code

    Returns
    -------
    EPSG code of this location
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code1 = '326' + utm_band
    else:
        epsg_code1 = '327' + utm_band
    return epsg_code1

def get_bound_image(image_path):
    """
    Get image information from path.

    Parameters
    ----------
    image_path : string
        Path to image file - GeoTiff

    Returns
    -------
    long_min: min Longitude image
    lat_min: min Latitude image
    long_max: max Longitude image
    lat_max: max Latitude image
    transform: Affine instance
        Transform of image.
    """
    with rasterio.open(image_path, mode='r+') as src:
        transform = src.transform
        print(src.width, src.height)
        w, h = src.width, src.height
        projstr = src.crs.to_string()
        check_epsg = src.crs.is_epsg_code
        if check_epsg:
            epsg_code = src.crs.to_epsg()
        else:
            epsg_code = None
    X_res = transform[0]
    Y_res = transform[4]
    trans_X_min, trans_Y_min = transform[2], transform[5]
    trans_X_max, trans_Y_max = trans_X_min + X_res * w, trans_Y_min + Y_res * h
    if epsg_code:
        if epsg_code == 4326:
            long_min, lat_min, long_max, lat_max = trans_X_min, trans_Y_min, trans_X_max, trans_Y_max
        else:
            inproj = pyproj.Proj(init='epsg:{}'.format(epsg_code))
            outproj = pyproj.Proj(init='epsg:{}'.format(4326))
            long_min, lat_min = pyproj.transform(inproj, outproj, trans_X_min, trans_Y_min)
            long_max, lat_max = pyproj.transform(inproj, outproj, trans_X_max, trans_Y_max)
    else:
        inproj = pyproj.Proj(projstr)
        outproj = pyproj.Proj(init='epsg:{}'.format(4326))
        long_min, lat_min = pyproj.transform(inproj, outproj, trans_X_min, trans_Y_min)
        long_max, lat_max = pyproj.transform(inproj, outproj, trans_X_max, trans_Y_max)
    return long_min, lat_min, long_max, lat_max, transform, w, h


def gis_data_latlong_to_utm(long_min, lat_min, long_max, lat_max):
    output_epsg = get_utm_from_wgs(long_min, lat_min)
    inproj = pyproj.Proj(init='epsg:{}'.format(4326))
    outproj = pyproj.Proj(init='epsg:{}'.format(output_epsg))
    trans_X_min_out, trans_Y_min_out = pyproj.transform(inproj, outproj, long_min, lat_min)
    trans_X_max_out, trans_Y_max_out = pyproj.transform(inproj, outproj, long_max, lat_max)
    return trans_X_min_out, trans_Y_min_out, trans_X_max_out, trans_Y_max_out, output_epsg

def get_resolution_meter(image_path):
    long_min, lat_min, long_max, lat_max, transform, w, h = get_bound_image(image_path)
    trans_X_min_out, trans_Y_min_out, trans_X_max_out, trans_Y_max_out, output_epsg = gis_data_latlong_to_utm(long_min, lat_min, long_max, lat_max)
    x_meter = abs(trans_X_min_out - trans_X_max_out)/w
    y_meter = abs(trans_Y_min_out - trans_Y_max_out)/h
    return (math.sqrt(x_meter**2 + y_meter**2)/(math.sqrt(2)))
