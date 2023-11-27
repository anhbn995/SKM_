import os
import re
import rasterio
import numpy as np
from time import strftime, strptime
import geopandas as gp
import pandas as pd
from shapely.strtree import STRtree

def filter_result_with_bound_and_id(shape_path, boundary_path, image_id, polygon_path, point_path):
    """Filter predicted objects by boundary.
    shape_path: input shapefile
    boundary_path: boundary shapefiles
    image_id: image file name
    polygon_path: output polygon objects
    point_path: output point objects

    Returns:
        None
    """
    tree_shp = gp.read_file(shape_path)
    crs = tree_shp.crs
    if boundary_path:
        bound_shp = gp.read_file(boundary_path)
        bound_shp = bound_shp.to_crs(crs)
        bound_aoi_table = bound_shp.loc[bound_shp['IMG_ID'] == image_id]
        if len(bound_aoi_table) > 0:
            bound_aoi = bound_aoi_table.iloc[0].geometry
            bound_aoi = bound_aoi.buffer(-1.0)
        else:
            bound_aoi = None
    else:
        bound_aoi = None

    tree_polygon = [geom.geometry for index, geom in tree_shp.iterrows()]
    tree_point = [geom.geometry.centroid for index, geom in tree_shp.iterrows()]
    if bound_aoi:

        strtree_point = STRtree(tree_point)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(tree_point))

        list_point = strtree_point.query(bound_aoi)
        list_point_inside = [x for x in list_point if bound_aoi.contains(x)]

        index_point = [index_by_id[id(pt)] for pt in list_point_inside]
        tree_polygon_out = [tree_polygon[index] for index in index_point]
        tree_point_out = list_point_inside
    else:
        tree_polygon_out = tree_polygon
        tree_point_out = tree_point
    df_polygon = pd.DataFrame(tree_polygon_out, columns=['geometry'])
    gdf_polygon = gp.GeoDataFrame(df_polygon, geometry='geometry', crs=crs)
    df_point = pd.DataFrame(tree_point_out, columns=['geometry'])
    gdf_point = gp.GeoDataFrame(df_point, geometry='geometry', crs=crs)
    gdf_polygon.to_file(polygon_path)
    gdf_point.to_file(point_path)
    return True


def transform_geom_to_poly_px(polygon, geotransform):
    """
    Convert polygon from geographical coordinate to pixel coordinate.
    
    Parameters
    ----------
    polygon : geometry
        Polygon contain a grid.
    geotransform : Affine
        Image's information of transform.
    Returns
    -------
    poly_px : 
        Pixel Coordinate of each tree.
    """
    geo_polygon_bound = np.dstack(polygon.exterior.coords.xy)[0]
    top_left_x = geotransform[2]

    top_left_y = geotransform[5]
    x_res = geotransform[0]
    y_res = geotransform[4]
    poly = np.array(geo_polygon_bound)
    poly_px = (poly - np.array([top_left_x, top_left_y])) / np.array([x_res, y_res])
    return poly_px.round()


def transform_poly_px_to_geom(polygon, geotransform):
    """
    Convert polygon from pixel coordinate to geographical coordinate.

    Parameters
    ----------
    polygon : Polygon format pixel coordinate
        Polygon contain a grid.
    geotransform : Affine
        Image's information of transform.
    Returns
    -------
    poly_rs :
        Geographical coordinate of each tree.
    """
    top_left_x = geotransform[2]
    top_left_y = geotransform[5]
    x_res = geotransform[0]
    y_res = geotransform[4]
    poly = np.array(polygon)
    poly_rs = poly * np.array([x_res, y_res]) + np.array([top_left_x, top_left_y])
    return poly_rs


def convert_window_to_polygon(xoff, yoff, xcount, ycount):
    """
    Create bounding boxes polygon by window.

    Parameters
    ----------
    xoff, yoff: Int
        Top-left pixel X,Y coordinate
    xcount, ycount: int
        With and height of window.
    Returns
    -------
        Polygon from window.
    """
    return [[xoff, yoff],
            [xoff + xcount, yoff],
            [xoff + xcount, yoff + ycount],
            [xoff, yoff + ycount],
            [xoff, yoff]]

def get_time_from_path(path):
    """
    Get date information from name file.
    
    Parameters
    ----------
    path : string
        Path of file, especially date imformation (yyyymmddhhmm) have to be included in file name.
        ex: "../Z01_202004041154_RGB.geojson" -> '202004041154' is yyyymmddhhmm
    Returns
    -------
    time_struct : string
        Date information will be written to file csv
    """

    # if os.path.isdir(path):
    #     raise Exception("\nIt is a directory")
    # if os.path.isfile(path):
    _, tail_file_name = os.path.split(path)
    time_str = re.findall(r'_(\d{12})', tail_file_name)
    time_struct = 'None'
    if len(time_str) != 0:
        time_struct_time = strptime(time_str[0], "%Y%m%d%H%M")
        time_struct = strftime("%Y-%m-%d %H:%M", time_struct_time)
    return time_struct
    # raise Exception("It is a special file (socket, FIFO, device file)")


def get_image_information(image_path):
    """
    Get information of image eg: geotransform, crs

    Parameters
    ----------
    image_path : string
        Path of image
    Returns
    -------
    geo_transform : Affine instance
        Affine transformation mapping the pixel space to geographic space
    crs : tuple
        The coordinate reference system.
    """
    with rasterio.open(image_path) as src:
        geo_transform = src.transform
        crs = src.crs
    return geo_transform, crs


