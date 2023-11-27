from osgeo import gdal, ogr, osr
import numpy as np
import shapefile as shp
import glob, os
import sys
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import time
from shapely.geometry import Polygon
from osgeo import ogr
import shapely.wkt
from shapely.geometry.multipolygon import MultiPolygon
# from shapely.ops import transform
from tqdm import *
import rasterio

core = multiprocessing.cpu_count()
import shapefile as shp
def load_shapefile(shape_path):
    # union_poly = ogr.Geometry(ogr.wkbPolygon)
    sf=shp.Reader(shape_path)
    my_shapes=sf.shapes()
    my_shapes_list = list(map(lambda shape: shape.points, my_shapes))
    # print(my_shapes_list)
    # union_poly = ogr.Geometry(ogr.wkbMultiPolygon)

    # make the union of polygons
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shape_path, 0)
    layer_in = dataSource.GetLayer()
    srs = layer_in.GetSpatialRef()
    geom_type = layer_in.GetGeomType()
    epsr1 =  srs.GetAttrValue('AUTHORITY',1)
    geom_type = layer_in.GetGeomType()
    layer_defn = layer_in.GetLayerDefn()
    list_polygon = []
    for feature in layer_in:
        geom =feature.GetGeometryRef()
        if geom is not None:
            if (geom.GetGeometryName()) == "POLYGON":
                # print(True)
    #         union_poly.AddGeometry(geom)
    #     # print(geom) 
    # result = union_poly.UnionCascaded()
    # print(result)
                shape_obj = (shapely.wkt.loads(geom.ExportToWkt()))
                polygon = (tuple(shape_obj.exterior.coords))
                # geo2d = ogr.Geometry(ogr.wkbPolygon)
                # geo2d.AddGeometry(polygon)
                # list_polygon.append(polygon)
                # poly_rs = []
                # ring = ogr.Geometry(ogr.wkbLinearRing)
                # for point in polygon:
                #     ring.AddPoint(point[0],point[1])
                # poly = ogr.Geometry(ogr.wkbPolygon)
                # print(ring)
                # poly.AddGeometry(ring)
                # union_poly.AddGeometry(poly) 
                poly_rs = []
                for point in polygon:
                    poly_rs.append((point[0],point[1]))
                poly_rs = tuple(poly_rs) 
                if len(poly_rs)>1:  
                    list_polygon.append(Polygon(poly_rs))
            elif (geom.GetGeometryName()) == "MULTIPOLYGON":
                for geo1 in geom:
                    if geo1 is not None:
                        shape_obj = (shapely.wkt.loads(geo1.ExportToWkt()))
                        polygon1 = (tuple(shape_obj.exterior.coords))
                        # geo2d = ogr.Geometry(ogr.wkbPolygon)
                        # geo2d.AddGeometry(polygon)
                        # list_polygon.append(polygon)
                        # poly_rs = []
                        # ring = ogr.Geometry(ogr.wkbLinearRing)
                        # for point in polygon1:
                        #     ring.AddPoint(point[0],point[1])
                        #     # print(point[0],point[1])
                        # # print(ring)
                        # poly = ogr.Geometry(ogr.wkbPolygon)
                        # poly.AddGeometry(ring)
                        # union_poly.AddGeometry(poly)    
                        poly_rs = []
                        for point in polygon1:
                            poly_rs.append((point[0],point[1]))
                        poly_rs = tuple(poly_rs)
                        if len(poly_rs)>1:
                            list_polygon.append(Polygon(poly_rs))
    mul = MultiPolygon(list_polygon)
    # print(mul)
    # result = union_poly.UnionCascaded()
    for geom in mul:
        if not(geom.is_valid):
            # print(tuple(geom.exterior.coords))
            print(1)

    result = MultiPolygon([geom if geom.is_valid else geom.buffer(0) for geom in mul])
    return result,epsr1,srs,geom_type

def polygon_to_geopolygon(polygon, geotransform):
    topleftX = geotransform[2]
    topleftY = geotransform[5]
    XRes = geotransform[0]
    YRes = geotransform[4]
    poly = np.array(polygon)
    poly_rs = poly*np.array([XRes,YRes])+np.array([topleftX,topleftY])
    return poly_rs

def load_image_geom(image_path,epsr1):
    # dataset_image = gdal.Open(image_path)
    # w,h = dataset_image.RasterXSize,dataset_image.RasterYSize
    with rasterio.open(image_path) as src:
        geotransform1 = src.transform
        w,h = src.width,src.height
        projstr = (src.crs.to_string())
        epsg_code = src.crs.to_epsg()
    polygon = ((0,0),(w,0),(w,h),(0,h),(0,0))
    # geotransform = dataset_image.GetGeoTransform()
    geopolygon = polygon_to_geopolygon(polygon,geotransform1)
    # polygon = Polygon(geopolygon)
    # project = pyproj.Transformer.from_proj(
    # pyproj.Proj(projstr), # source coordinate system
    # pyproj.Proj(init='epsg:%s'%(epsr1)))
    # geopolygon1 = transform(project.transform, polygon)
    # proj = osr.SpatialReference(wkt=dataset_image.GetProjection())
    # epsr2 = (proj.GetAttrValue('AUTHORITY',1))
    # # epsr1=32643
    # # epsr2=32643
    # inProj = Proj(projstr)
    # inProj = Proj(init='epsg:%s'%(epsg_code))
    # outProj = Proj(init='epsg:%s'%(epsr1))
    list_point = []
    # transformer = Transformer.from_proj(inProj, outProj, always_xy = True)
    for point in geopolygon:
        long1,lat = point[0],point[1]
        # x,y = transform(inProj,outProj,long1,lat)
        # x,y = transformer.transform(long1,lat)
        x,y = long1,lat
        list_point.append((x,y))
    # print(tuple(list_point))
    # return geopolygon1
    return Polygon(tuple(list_point))

def create_list_id(path,shape_name):
    list_id = []
    os.chdir(path)
    len_id = len(shape_name)
    for file in glob.glob("*.tif"):
        # list_id.append(file[:-4])
        if file[0:len_id]==shape_name:
            list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def create_list_id_shape(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.shp"):
        list_id.append(file[:-4])
        # if file[0:6]=='SX9192':
            # list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def crop_shape(shape_name):
    shape_path=os.path.join(shape_dir,shape_name+'.shp')
    result,epsr1,srs,geom_type = load_shapefile(shape_path)
    list_id = create_list_id(foder_image,shape_name)
    if not os.path.exists(os.path.join(parent,foder_name+'_shape')):
        os.makedirs(os.path.join(parent,foder_name+'_shape'))
    path_shape_crop = os.path.join(parent,foder_name+'_shape')
    for id_image in list_id:
        print(10)
        result2 = load_image_geom(os.path.join(foder_image,id_image+'.tif'),epsr1)
        # a = result.intersection(result2) 
        a = [(geom.intersection(result2)) if geom.is_valid else (geom.buffer(0).intersection(result2)) for geom in result]
        b = [aa for aa in a if aa.geom_type == 'Polygon' ]
        # print(a)
        # for geom in result:
        #     if geom.is_valid:
        #         try:
        #             print(geom.intersection(result2).geom_type)
        #         except Exception as e:
        #             print('fail:', e)
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds_out = driver.CreateDataSource(os.path.join(path_shape_crop,id_image+'.shp'))
        layer_out = ds_out.CreateLayer("Building area", srs = srs, geom_type = geom_type)
        layer_defn = layer_out.GetLayerDefn()
        for polygon in b:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for point in polygon.exterior.coords:
                # print(point)
                ring.AddPoint(point[0], point[1])
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            outFeature = ogr.Feature(layer_defn)
            outFeature.SetGeometry(poly)
            layer_out.CreateFeature(outFeature)
        outFeature = None
        # destroy the feature
        outLayer = None
        # Close DataSources
        data_source = None
    # # print(a,len(a))
# # print(result3)
def crop_shape_pool(id_image,epsr1,result,path_shape_crop
                    ,shape_path,foder_image):
    result2 = load_image_geom(os.path.join(foder_image,id_image+'.tif'),epsr1)
    dataset = gdal.Open(os.path.join(foder_image,id_image+'.tif'))

    srs = osr.SpatialReference(dataset.GetProjectionRef())

    a = [(geom.intersection(result2)) if geom.is_valid else \
             (geom.buffer(0).intersection(result2)) for geom in result]
    b = [aa for aa in a if aa.geom_type == 'Polygon' ]

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds_out = driver.CreateDataSource(os.path.join(path_shape_crop,id_image+'.shp'))
    layer_out = ds_out.CreateLayer("Building area", srs = srs, geom_type = ogr.wkbPolygon)
    layer_defn = layer_out.GetLayerDefn()
    for polygon in b:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in polygon.exterior.coords:
            # print(point)
            ring.AddPoint(point[0], point[1])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        outFeature = ogr.Feature(layer_defn)
        outFeature.SetGeometry(poly)
        layer_out.CreateFeature(outFeature)
    outFeature = None
    # destroy the feature
    outLayer = None
    # Close DataSources
    data_source = None
    return True
def crop_shape2(foder_image,shape_name,shape_dir):
    shape_path=os.path.join(shape_dir,shape_name+'.shp')
    print(shape_path)
    result,epsr1,srs,geom_type = load_shapefile(shape_path)
    list_id = create_list_id(foder_image,shape_name)
    # print(list_id)
    foder_name = os.path.basename(foder_image)
#  size_crop = int(sys.argv[3])
    parent = os.path.dirname(foder_image)
    if not os.path.exists(os.path.join(parent,foder_name+'_shape')):
        os.makedirs(os.path.join(parent,foder_name+'_shape'))
    path_shape_crop = os.path.join(parent,foder_name+'_shape')
    p_cropshape = Pool(processes=core)
    pool_result = p_cropshape.imap_unordered(partial(crop_shape_pool,epsr1=epsr1,result=result,path_shape_crop=path_shape_crop,shape_path=shape_path,foder_image=foder_image), list_id)
    with tqdm(total=len(list_id)) as pbar:
        for i,_ in tqdm(enumerate(pool_result)):
            pbar.update()
    p_cropshape.close()
    p_cropshape.join()
def main_crop_shape(foder_image,shape_dir):
    x1 = time.time()
    list_shape = create_list_id_shape(shape_dir)
    for shape_name in list_shape:
        crop_shape2(foder_image,shape_name,shape_dir)
    print(time.time() - x1, "second")
    foder_name = os.path.basename(foder_image)
#  size_crop = int(sys.argv[3])
    parent = os.path.dirname(foder_image)
    path_shape_crop = os.path.join(parent,foder_name+'_shape')
    return path_shape_crop
if __name__ == "__main__":
    x1 = time.time()
    foder_image = os.path.abspath(sys.argv[1])
    shape_dir = os.path.abspath(sys.argv[2])
    list_shape = create_list_id_shape(shape_dir)
    for shape_name in list_shape:
        crop_shape2(foder_image,shape_name,shape_dir)
    print(time.time() - x1, "second")
