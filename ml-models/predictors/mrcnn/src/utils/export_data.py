import rasterio
from shapely.geometry import Polygon
import numpy as np
from shapely.strtree import STRtree
import pandas as pd
import geopandas as gp


def transform_poly_px_to_geom(polygon, geotransform):
    top_left_x = geotransform[2]
    top_left_y = geotransform[5]
    x_res = geotransform[0]
    y_res = geotransform[4]
    poly = np.array(polygon)
    poly_rs = poly * np.array([x_res, y_res]) + \
        np.array([top_left_x, top_left_y])
    return poly_rs


def get_bound(image_path):
    """get Aoi bound from AOI care, if none, return image bound"""
    with rasterio.open(image_path) as src:
        transform = src.transform
        w, h = src.width, src.height
        proj_str = (src.crs.to_string())
    bound_image = ((0, 0), (w, 0), (w, h), (0, h), (0, 0))
    bound_aoi = Polygon(transform_poly_px_to_geom(bound_image, transform))
    return bound_aoi


def export_predict_result_to_file(polygon_result_all, score_result_all, bound_aoi, transform, proj_str, output_path):
    """Export result to shapefile.
    polygon_result_all: list polygon result after predict
    score_result_all: score for each polygon
    transform: transform get from image by rasterio
    projstr: project string get from image by rasterio
    output_shape_file : path to output shape file
    """

    list_geo_polygon = [Polygon(transform_poly_px_to_geom(
        polygon, transform)) for polygon in polygon_result_all]
    tree_polygon = [geom for geom in list_geo_polygon]
    tree_point = [geom.centroid for geom in list_geo_polygon]
    strtree_point = STRtree(tree_point)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(tree_point))

    list_point = strtree_point.query(bound_aoi)
    list_point_inside = [x for x in list_point if bound_aoi.contains(x)]

    index_point = [index_by_id[id(pt)] for pt in list_point_inside]
    tree_polygon_rs = [tree_polygon[index] for index in index_point]
    tree_score_rs = [score_result_all[index] for index in index_point]
    tree_id = list(range(len(tree_score_rs)))
    data_tree = list(zip(tree_polygon_rs, tree_score_rs, tree_id))

    df_polygon = pd.DataFrame(data_tree, columns=['geometry', 'score', "FID"])
    gdf_polygon = gp.GeoDataFrame(
        df_polygon, geometry='geometry', crs=proj_str)

    gdf_polygon.to_file(output_path)
    return True
