import rasterio
from tqdm import tqdm
import numpy as np
from skimage.segmentation import slic
from osgeo import gdal, osr, ogr
from .utils import get_quantile_schema

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
def slic_image(input_image, out_fp_predict, qt_scheme, rgb=[4,3,2], numSegments=12000, crop_size =512):
    
    with rasterio.open(input_image) as src:
        h,w = src.height,src.width
        source_crs = src.crs
        source_transform = src.transform
        image_data = src.read(rgb)
        # image_data = src.read()
        dtype = src.dtypes[0]
        num_band = 3
        msk = (src.read_masks(1)/255).astype(np.uint16)

    # mask = np.moveaxis(msk, 0, -1)
    rgb_image = np.moveaxis(image_data, 0, -1)

    list_weight = list(range(0, w, crop_size))
    list_hight = list(range(0, h, crop_size))

    output_ds = np.zeros_like(image_data[0]).astype(np.uint16)
    with tqdm(total=len(list_hight)*len(list_weight)) as pbar:
        for start_h_org in list_hight:
            for start_w_org in list_weight:
                h_crop_start = start_h_org 
                w_crop_start = start_w_org 

                size_w_crop = min(crop_size , w - start_w_org )
                size_h_crop = min(crop_size , h - start_h_org )
                new_num_seg = round(size_h_crop/crop_size * size_w_crop/crop_size * numSegments)

                cut_image = rgb_image[h_crop_start:h_crop_start+size_h_crop, w_crop_start:w_crop_start+size_w_crop]
                new_image = np.zeros((cut_image.shape[0], cut_image.shape[1], num_band), dtype=np.uint8)

                for i in range(num_band):
                    band = cut_image[...,i]
                    if str(dtype) == 'uint8':
                        band_nor = band.astype(int)
                    else:
                        band_qt = qt_scheme[rgb[i] - 1]
                        # band_qt = qt_scheme[i - 1]
                        band_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (0, 255)).astype(int)
                    new_image[..., i] = band_nor

                segments = slic(new_image, n_segments=new_num_seg, sigma=3, compactness=3, convert2lab=True)
                output_ds[ h_crop_start:h_crop_start+size_h_crop, w_crop_start:w_crop_start+size_w_crop] = segments
                pbar.update()

    
    output_ds = (output_ds + 1) * msk
    new_dataset = rasterio.open(out_fp_predict, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, 
                                crs=source_crs,
                                transform=source_transform,
                                nodata=0,
                                dtype='uint16',
                                compress='lzw') 
    new_dataset.write(output_ds, 1)
    new_dataset.close()
    return out_fp_predict

def polygonize(img, shp_path):
    ds = gdal.Open(img)
    prj = ds.GetProjection()
    srcband = ds.GetRasterBand(1)
    dst_layername = "Shape"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(shp_path)
    srs = osr.SpatialReference(wkt=prj)

    dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)
    # raster_field = ogr.FieldDefn('raster_val', GDAL_OGR_TYPE_MAPPER[srcband.DataType])
    raster_field_1 = ogr.FieldDefn('label', GDAL_OGR_TYPE_MAPPER[srcband.DataType])
    # dst_layer.CreateField(raster_field)
    dst_layer.CreateField(raster_field_1)
    gdal.Polygonize(srcband, srcband, dst_layer, 0, [], callback=None)
    del ds, srcband, dst_ds, dst_layer

def main(img_path):
    return True
if __name__ == "__main__":
    img_path = '/home/boom/data/data/Linh/label_new_mongolia/LC08_L2SP_133026_20211031_20211109_02_T1/LC08_L2SP_133026_20211031_20211109_02_T1.tif'
    out_slic_path = '/home/boom/boom/SLIC_DBSCAN_Superpixels/examples/slic_image/LC08_L2SP_133026_20211031_20211109_02_T1.tif'
    out_slic_shp = '/home/boom/boom/SLIC_DBSCAN_Superpixels/examples/shape/LC08_L2SP_133026_20211031_20211109_02_T1.shp'

    qt_scheme = get_quantile_schema(img_path)
    slic_path = slic_image(img_path, out_slic_path, qt_scheme)
    polygonize(slic_path, out_slic_shp)