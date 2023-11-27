import params
from utils.on_training_proccessing import on_training_proccessing
import rasterio
from osgeo import gdal
from pyproj import Proj, transform, Transformer
from processing import PreprocessingUnet
from train import TrainerUnet
import json
import numpy as np
def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)
def stretch_image(input, output):
    ds = gdal.Open(input, gdal.GA_ReadOnly)
    bcount = ds.RasterCount
    rows = ds.RasterXSize
    cols = ds.RasterYSize

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output, rows, cols, bcount, gdal.GDT_Byte)
    # sets same geotransform as input
    outdata.SetGeoTransform(ds.GetGeoTransform())
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input

    from rio_tiler.io import COGReader
    with COGReader(input) as cog:
        qt_scheme = []

        stats = cog.stats(pmax=98, pmin=2)
        for _, value in stats.items():
            try:
                qt_scheme.append({
                    'p2': value['pc'][0],
                    'p98': value['pc'][1],
                })
            except:
                qt_scheme.append({
                    'p2': value['percentiles'][0],
                    'p98': value['percentiles'][1],
                })

    for i in range(bcount):
        band = np.array(ds.GetRasterBand(i + 1).ReadAsArray())
        band_qt = qt_scheme[i]
        band = np.interp(
            band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(int)
        outdata.GetRasterBand(i + 1).WriteArray(band)
        outdata.GetRasterBand(i + 1).SetNoDataValue(0)
        band = None
    outdata.FlushCache()  # saves to disk!!
    outdata = None
    # close dataset
    ds = None
def compute_bound(img_path):
    from osgeo import gdal
    from osgeo.gdalconst import GA_ReadOnly

    data = gdal.Open(img_path, GA_ReadOnly)
    inProj = Proj(data.GetProjection())
    outProj = Proj('epsg:4326')
    del data
    with rasterio.open(img_path) as ds:
        minx = ds.bounds.left
        miny = ds.bounds.top
        maxx = ds.bounds.right
        maxy = ds.bounds.bottom
    del ds
    point_list = [[minx, miny], [minx, maxy], [
        maxx, maxy], [maxx, miny], [minx, miny]]
    tar_point_list = []
    for _point in point_list:
        _x, _y = Transformer.from_crs(
            inProj.crs, outProj.crs, always_xy=True).transform(_point[0], _point[1])
        tar_point_list.append([_x, _y])
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [tar_point_list]
                }
            }
        ]
    }
def geom_to_file(geom, path):
    import json
    with open(path, 'w') as file:
        file.write(json.dumps(geom))
    return True
def get_mask_label(file_path):
    with rasterio.open(file_path) as ds:
        image_tags = ds.tags()
    keys = list(image_tags.keys())
    keys = list(filter(lambda el: 'LABEL_' in el, keys))
    labels = [None] * len(keys)

    datasource = gdal.Open(file_path)
    band1 = datasource.GetRasterBand(1)
    ct = band1.GetColorTable()

    for label_value in keys:
        idx = int(label_value.replace('LABEL_', ''))
        r, g, b, _ = ct.GetColorEntry(idx)
        labels[idx-1] = {
            'value': idx,
            'name': image_tags.get(label_value),
            'color': rgb2hex(r, g, b)
        }
    return labels

if __name__ == '__main__':
    print("runnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
    image_path = params.IMAGE_PATH
    mask_path = params.MASK_PATH
    output_dir = params.OUTPUT_DIR
    meta_path = f'{output_dir}/model.json'
    model_path = f'{output_dir}/model.h5'
    tmp_path = params.TMP_PATH
    bound_path = '{}/bound.geojson'.format(tmp_path)
    task_id = params.TASK_ID
    with rasterio.open(image_path) as src:
        numbands = src.count
        dtype = src.dtypes[0]
    print('aaaaaaaaaaaaaaaaaaaaaaaaaa',dtype)
    kwargs = {
        'numbands': numbands,
        'trainer_size': params.TRAINER_SIZE,
        'labels': get_mask_label(mask_path),
        'optimizer': params.OPTIMIZER,
        'loss': params.LOSS,
        'metrics': json.loads(params.METRICS),
        'batch_size': None,
        'epochs': params.EPOCHS,
        'patience_early': None,
        'factor': None,
        'patience_reduce': None,
        'min_lr': None,
        'input_type': params.INPUT_TYPE,
        'n_filters': params.N_FILTERS,
        'dropout': params.DROPOUT,
        'batchnorm': params.BATCHNORM,
        'task_id': task_id
    }    
    print(dtype)
    if dtype != 'uint8':
        new_image_path = '{}/image_uint8.tif'.format(tmp_path)
        stretch_image(image_path, new_image_path)
        image_path = new_image_path        
    on_training_proccessing(0.1, task_id)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    geom_to_file(compute_bound(image_path), bound_path)
    unet_data_preparation = PreprocessingUnet(
        bound_path, mask, tmp_path, output_dir, image_path, "unet", **kwargs)
    unet_data_preparation.preprocess_run()
    on_training_proccessing(0.3, task_id)
    unet_trainer = TrainerUnet(model_path, tmp_path, "unet", **kwargs)
    model_config = unet_trainer.training()
    unet_trainer = None
    kwargs.update(model_config)
    metadata = {
        "param": kwargs,
        "dtype": dtype,
    }
    with open(meta_path, 'w') as file:
        json.dump(metadata, file)
