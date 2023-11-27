import geojson
from osgeo import gdal, ogr, osr
from pyproj import Proj, transform
import multiprocessing
import pandas as pd
import geopandas as gp
import rasterio
from shapely.strtree import STRtree
from ml_lib.utils import transform_poly_px_to_geom
from shapely.geometry import Polygon

core_of_computer = multiprocessing.cpu_count()


def export_geojson_str(geo_polygons, geo_transform, projection_str):
    features = []
    list_geopolygon = list_polygon_to_list_geopolygon(geo_polygons, geo_transform)
    list_geopolygon = transformToLatLong(list_geopolygon, projection_str)
    for geo_polygon in list_geopolygon:
        # geo_polygon = np.array(geo_polygon).tolist()
        polygon = geojson.Polygon([geo_polygon])
        feature = geojson.Feature(geometry=polygon)
        features.append(feature)
    return geojson.dumps(geojson.FeatureCollection(features))


def list_polygon_to_list_geopolygon(list_polygon, geotransform):
    list_geopolygon = []
    for polygon in list_polygon:
        geopolygon = polygon_to_geopolygon(polygon, geotransform)
        list_geopolygon.append(geopolygon)
    return list_geopolygon


def polygon_to_geopolygon(polygon, geotransform):
    temp_geopolygon = []
    for point in polygon:
        geopoint = point_to_geopoint(point, geotransform)
        temp_geopolygon.append(geopoint)
    geopolygon = tuple(temp_geopolygon)
    return geopolygon


def point_to_geopoint(point, geotransform):
    topleftX = geotransform[0]
    topleftY = geotransform[3]
    XRes = geotransform[1]
    YRes = geotransform[5]
    geopoint = (topleftX + point[0] * XRes, topleftY + point[1] * YRes)
    return geopoint


def transformToLatLong(list_geopolygon, epsr):
    new_list_geopolygon = []
    inProj = Proj(init='epsg:%s' % (epsr))
    outProj = Proj(init='epsg:4326')
    for geopolygon in list_geopolygon:
        new_geopolygon = []
        try:
            for point in geopolygon:
                # print('point ', point)
                newPoint = transform(inProj, outProj, point[0], point[1])
                new_geopolygon.append(newPoint)
            new_list_geopolygon.append(new_geopolygon)
        except Exception as e:
            print(e)
    return new_list_geopolygon


def exportResult2(list_polygon, geotransform, src_projection, outputFileName, driverName, epsr=None):
    list_geopolygon = list_polygon_to_list_geopolygon(list_polygon, geotransform)
    driver = ogr.GetDriverByName(driverName)
    data_source = driver.CreateDataSource(outputFileName)
    tar_projection = osr.SpatialReference()
    tar_projection.ImportFromEPSG(4326)
    outLayer = data_source.CreateLayer("Result", tar_projection, ogr.wkbPolygon)
    wgs84_trasformation = osr.CoordinateTransformation(src_projection, tar_projection)

    featureDefn = outLayer.GetLayerDefn()
    for geopolygon in list_geopolygon:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in geopolygon:
            ring.AddPoint_2D(point[0], point[1])
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        polygon.Transform(wgs84_trasformation)

        # center_point = polygon.Centroid()

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(polygon)
        outLayer.CreateFeature(outFeature)
    ###############################################################################
    # destroy the feature
    outFeature = None
    # destroy the feature
    outLayer = None
    # Close DataSources
    data_source = None


def exportResult3(list_polygon, geotransform, src_projection, outputFileName, driverName, epsr=None):
    driver = ogr.GetDriverByName(driverName)
    data_source = driver.CreateDataSource(outputFileName)
    tar_projection = osr.SpatialReference()
    tar_projection.ImportFromEPSG(4326)
    outLayer = data_source.CreateLayer("Result", tar_projection, ogr.wkbPolygon)
    wgs84_trasformation = osr.CoordinateTransformation(src_projection, tar_projection)
    featureDefn = outLayer.GetLayerDefn()

    for geopolygon in list_polygon:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in geopolygon.exterior.coords:
            ring.AddPoint(point[0], point[1])
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        polygon.Transform(wgs84_trasformation)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(polygon)
        outLayer.CreateFeature(outFeature)
    ###############################################################################
    # destroy the feature
    outFeature = None
    # destroy the feature
    outLayer = None
    # Close DataSources
    data_source = None

def export_predict_result_to_file(polygon_result_all, score_result_all, bound_aoi, transform, proj_str, output_path):
    """Export result to shapefile.
    polygon_result_all: list polygon result after predict
    score_result_all: score for each polygon
    transform: transform get from image by rasterio
    projstr: project string get from image by rasterio
    output_shape_file : path to output shape file
    """

    list_geo_polygon = [Polygon(transform_poly_px_to_geom(polygon, transform)) for polygon in polygon_result_all]
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

    df_polygon = pd.DataFrame(data_tree, columns=['geometry', 'score',"FID"])
    gdf_polygon = gp.GeoDataFrame(df_polygon, geometry='geometry', crs=proj_str)


    gdf_polygon.to_file(output_path, driver='GeoJSON')
    return True

def get_bound(image_path):
    """get Aoi bound from AOI care, if none, return image bound"""
    with rasterio.open(image_path) as src:
        transform = src.transform
        w, h = src.width, src.height
        proj_str = (src.crs.to_string())
    bound_image = ((0, 0), (w, 0), (w, h), (0, h), (0, 0))
        # if don't have bound aoi then predict all image
    bound_aoi = Polygon(transform_poly_px_to_geom(bound_image, transform))
    return bound_aoi