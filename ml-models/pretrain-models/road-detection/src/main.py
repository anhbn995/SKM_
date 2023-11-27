import params
from segmen_model.script.road_detection import main


def reproject_image(src_path, dst_path, dst_crs='EPSG:4326'):
    from osgeo import gdal
    import os
    import rasterio
    with rasterio.open(src_path) as ds:
        nodata = ds.nodata or 0
    temp_path = dst_path.replace('.tif', 'temp.tif')
    option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
    gdal.Translate(temp_path, src_path, options=option)
    option = gdal.WarpOptions(gdal.ParseCommandLine(
        "-t_srs {} -dstnodata {}".format(dst_crs, nodata)))
    gdal.Warp(dst_path, temp_path, options=option)
    os.remove(temp_path)
    return True


if __name__ == '__main__':
    print('Start running model')
    input_path = params.INPUT_PATH
    ouput_path = params.OUTPUT_PATH
    num_class = 1
    img_size = 512
    numband = 3
    confidence = 0.95
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/road-detection/v1/weights/model.h5'
    reproject_image_path = f'{params.TMP_PATH}/reproject_image.tif'
    reproject_image(input_path, reproject_image_path)
    main(reproject_image_path, ouput_path, img_size,
         numband, model_path, num_class, confidence)
