from osgeo import ogr, osr


def exportResult2(list_polygon, geotransform, src_projection, outputFileName, driverName, epsr=None):
    list_geopolygon = list_polygon_to_list_geopolygon(
        list_polygon, geotransform)
    driver = ogr.GetDriverByName(driverName)
    data_source = driver.CreateDataSource(outputFileName)
    tar_projection = osr.SpatialReference()
    tar_projection.ImportFromEPSG(4326)
    outLayer = data_source.CreateLayer(
        "Result", tar_projection, ogr.wkbPolygon)
    wgs84_trasformation = osr.CoordinateTransformation(
        src_projection, tar_projection)

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
