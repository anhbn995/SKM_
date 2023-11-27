import rasterio
import cv2
import uuid
from osgeo import gdal, ogr, osr
import os
import json


from pathlib import Path
import numpy as np
import utils.ogr2ogr as ogr2ogr
import geopy.distance
from math import ceil

GDAL_OGR_TYPE_MAPPER = {
    gdal.GDT_Byte: ogr.OFTInteger,
    gdal.GDT_UInt16: ogr.OFTInteger,
    gdal.GDT_Int16: ogr.OFTInteger,
    gdal.GDT_UInt32: ogr.OFTInteger,
    gdal.GDT_Int32: ogr.OFTInteger,
    gdal.GDT_Float32: ogr.OFTReal,
    gdal.GDT_Float64: ogr.OFTReal,
    gdal.GDT_CInt16: ogr.OFTInteger,
    gdal.GDT_CInt32: ogr.OFTInteger,
    gdal.GDT_CFloat32: ogr.OFTReal,
    gdal.GDT_CFloat64: ogr.OFTReal
}


def vectorize(input_tif, output_geojson, selected_values, meta, epsilon, band):
    labels = meta.get('labels')
    image_resolution = meta.get('image_resolution')
    model_name = meta.get('model_name')
    image_source = meta.get('image_source')
    created_at = meta.get('created_at')
    with rasterio.open(input_tif, 'r', driver='GTiff') as src:
        img = src.read(band)
        crs = dict(src.crs)
        transform = src.transform
        w, h = src.width, src.height
        num_bands = src.count
        transform = src.transform
    img[img > 0] = img[img > 0] + 1
    dst = gdal.Open(input_tif)
    gt = dst.GetGeoTransform()
    coords_origin = (gt[3], gt[0])
    coords_right = (gt[3], gt[0] + gt[1])
    pixel_size = geopy.distance.vincenty(coords_origin, coords_right).km * 1000

    if epsilon > 0:
        ele_value = round(epsilon / abs(pixel_size))
        ele_value = 1 if ele_value == 0 else ele_value
        kernel2 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ele_value, ele_value))
        kernel3 = cv2.getStructuringElement(
            cv2.MORPH_RECT, (ele_value, ele_value))
        ele_close = ceil(ele_value / 2) + 1
        kernel2_closing = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ele_close, ele_close))
        kernel3_closing = cv2.getStructuringElement(
            cv2.MORPH_RECT, (ele_close, ele_close))

        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3_closing)
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2_closing)

        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel2)

        image = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel3)
    else:
        image = img
    mask_ = 1 * (image > 1)

    for value in selected_values:
        mask_ = mask_ + (1 * (image == value + 1))
    mask_bool = mask_ > 1
    masked = 1 * mask_bool
    mask_final = masked * image

    mask_int = np.array(mask_final).astype('uint8')
    mask_int[mask_int > 0] = mask_int[mask_int > 0] - 1
    folder = Path(input_tif).parent
    str_id = uuid.uuid4().hex
    tmp_tif_path = '{}/{}.tif'.format(folder, str_id)
    tmp_geojson_path = '{}/{}.geojson'.format(folder, str_id)

    result = rasterio.open(tmp_tif_path, 'w', driver='GTiff',
                           height=h, width=w,
                           count=num_bands, dtype='uint8',
                           crs=crs,
                           transform=transform,
                           compress='lzw')
    result.write(mask_int, 1)
    result.close()

    polygonize(tmp_tif_path, tmp_geojson_path)
    ogr2ogr.main(["", "-f", "geojson", '-t_srs', 'epsg:4326',
                  output_geojson, tmp_geojson_path])
    os.remove(tmp_tif_path)
    os.remove(tmp_geojson_path)
    with open(output_geojson) as outfile:
        data = json.load(outfile)

    def find(arr, el):
        print(arr, el)
        for e in arr:
            if int(e['value']) == int(el):
                return e

    selected_labels = []
    if labels:
        for value in selected_values:
            selected_labels.append(find(labels, value))

    with open(output_geojson, 'w') as outfile:
        data['labels'] = selected_labels
        data['image_resolution'] = image_resolution
        data['model_name'] = model_name
        data['image_source'] = image_source
        data['created_at'] = created_at
        json.dump(data, outfile)

    return {
        'labels': selected_labels,
        'image_resolution': image_resolution,
        'model_name': model_name,
        'image_source': image_source,
        'created_at': created_at
    }


def polygonize(img, shp_path):
    ds = gdal.Open(img)
    prj = ds.GetProjection()
    srcband = ds.GetRasterBand(1)
    dst_layername = "Shape"
    drv = ogr.GetDriverByName("GeoJSON")
    dst_ds = drv.CreateDataSource(shp_path)
    srs = osr.SpatialReference(wkt=prj)

    dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)
    raster_field = ogr.FieldDefn(
        'raster_val', GDAL_OGR_TYPE_MAPPER[srcband.DataType])
    raster_field_1 = ogr.FieldDefn(
        'label', GDAL_OGR_TYPE_MAPPER[srcband.DataType])
    dst_layer.CreateField(raster_field)
    dst_layer.CreateField(raster_field_1)
    gdal.Polygonize(srcband, srcband, dst_layer, 0, [], callback=None)
    del img, ds, srcband, dst_ds, dst_layer
