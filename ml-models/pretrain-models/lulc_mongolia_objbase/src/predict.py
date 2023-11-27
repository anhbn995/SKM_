import geopandas as gpd
from tqdm import tqdm
from itertools import chain
from tqdm import tqdm
import rasterio
import rasterio.mask
import numpy as np
import cv2
from keras.models import load_model
from rasterio.features import rasterize
from osgeo import ogr
from .utils import get_quantile_schema

def predict_to_shp(model, slic_shp, image_path, qt_scheme, batch_size = 500, predict_size = 8):

    df_slic = gpd.read_file(slic_shp)
    bound_shp = df_slic['geometry']
    labels = []
    out_image = []

    markers = list(range(0, len(df_slic.index), batch_size))

    if markers[len(markers)-1] != len(df_slic.index):
        markers[len(markers)-1] = len(df_slic.index)

    src = rasterio.open(image_path)
    # num_band = src.count
    num_band = 7
    dtype = src.dtypes[0]

    with tqdm(total=len(markers) -1) as pbar:
        for i in range(len(markers)-1):
            image_batch = []
            for j in range(markers[i],markers[i+1]):
                geo = bound_shp[j]
                out_image, out_transform = rasterio.mask.mask(src, [geo], crop=True)
                out_image = np.moveaxis(out_image, 0, -1)[...,:num_band]
                if out_image.shape[0] > predict_size and out_image.shape[1] > predict_size:
                    out_image = np.resize(out_image, (predict_size, predict_size, num_band))
                elif out_image.shape[0] > predict_size or out_image.shape[1] > predict_size:
                    pad_size = max(out_image.shape[0], out_image.shape[1])
                    pad_arr = np.zeros((pad_size, pad_size, num_band))
                    pad_arr[:out_image.shape[0],:out_image.shape[1],:] = out_image
                    out_image = np.resize(pad_arr, (predict_size, predict_size, num_band))
                else:
                    pad_arr = np.zeros((predict_size, predict_size, num_band))
                    pad_arr[:out_image.shape[0],:out_image.shape[1],:] = out_image
                    out_image = pad_arr

                new_image = np.zeros((out_image.shape[0], out_image.shape[1], num_band), dtype=np.uint8)
                for i in range(num_band):
                    band = out_image[..., i]
                    if str(dtype) == 'uint8':
                        band_nor = band.astype(int)
                    else:
                        band_qt = qt_scheme[i]
                        band_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (0, 255)).astype(int)
                    new_image[..., i] = band_nor

                x = np.expand_dims(new_image, axis=0)/255
                image_batch.append(x)

            images = np.vstack(image_batch)
            classes = model.predict(images, batch_size=batch_size)
            label_predicts = classes[:,None].argmax(-1) + 1
            labels.extend(list(chain.from_iterable(label_predicts.tolist())))
            pbar.update()

    df_slic['label'] = labels
    # df_slic.to_file(out_path)
    return df_slic

def df_to_tif(src_img, df_shp, out_tif):
    with rasterio.open(src_img) as src:
        h,w = src.height,src.width
        source_transform = src.transform
        src_crs = src.crs

    rasterize_rivernet = rasterize(
        [(shape, value) for shape, value in zip(df_shp['geometry'], df_shp['label'])],
        out_shape=(h,w),
        transform=source_transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8)


    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for i in range(5):
        img = cv2.morphologyEx(rasterize_rivernet, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    with rasterio.open(
        out_tif, 'w',
        driver='GTiff',
        dtype=rasterio.uint8,
        count=1,
        width=w,
        height=h,
        crs = src_crs,
        nodata=0,
        transform=source_transform
    ) as dst:
        dst.write(img, indexes=1)
        dst.write_colormap(
                        1, {
                            1: (237, 2, 42),
                            2: (255, 220, 92),
                            3:(167, 210, 130),
                            4:(200, 200, 200),
                            5:(238, 207, 168),
                            6:(52, 130, 33),
                            7:(26, 91, 171),
                            8:(216, 227, 219)
                            })

if __name__ == "__main__":

    model = load_model('/home/boom/boom/SLIC_DBSCAN_Superpixels/examples/models/LC08_L2SP_131027_20210830_20210909_02_T1.h5', compile=False)
    slic_shp_img = '/home/boom/boom/SLIC_DBSCAN_Superpixels/examples/shape/LC08_L2SP_131027_20210830_20210909_02_T1.shp'
    image_path = '/home/boom/data/data/Linh/label_new_mongolia/LC08_L2SP_131027_20210830_20210909_02_T1/LC08_L2SP_131027_20210830_20210909_02_T1.tif'
    out_shp = '/home/boom/boom/SLIC_DBSCAN_Superpixels/examples/shape/landsat_result.shp'
    out_tif = '/home/boom/boom/SLIC_DBSCAN_Superpixels/examples/shape/landsat_result.tif'

    df = predict_to_shp(model, slic_shp_img, image_path, out_shp)
    df_to_tif(image_path, out_shp, out_tif) 
    print('Done')