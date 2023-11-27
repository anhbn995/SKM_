import numpy as np
import rasterio
import argparse
from rasterio.windows import Window
import threading
from tqdm import tqdm
import concurrent.futures
import warnings
from rio_tiler.io import COGReader
from rasterio import Affine
import cv2
import tensorflow as tf
from rasterio.enums import ColorInterp
# tf.keras.backend.set_image_data_format('channels_first')
warnings.filterwarnings("ignore")
import os

import tensorflow as tf
import time
import os
from glob import glob
import argparse
import builtins
import numpy as np

def get_center_pixel(dataset):
    """This function return the pixel coordinates of the raster center
    """
    width, height = dataset.width, dataset.height
    # We calculate the middle of raster
    x_pixel_med = width // 2
    y_pixel_med = height // 2
    return (x_pixel_med, y_pixel_med)

def create_list_id(dir_name):
    list_id = []
    for file in os.listdir(dir_name):
        # if file.endswith(".tif") and file.startswith("COG"):
        if file.endswith(".tif"):
            list_id.append(os.path.join(dir_name,file))
    return list_id

def get_quantile_schema(input_path):
    qt_scheme = []
    try:
        with COGReader(input_path) as cog:
            stats = cog.stats()
            for _, value in stats.items():
                qt_scheme.append({
                    'p2': value['percentiles'][0],
                    'p98': value['percentiles'][1],
                })
    except:
        with COGReader(input_path) as cog:
            stats = cog.statistics()
            for _, value in stats.items():
                qt_scheme.append({
                    'p2': value['percentile_2'],
                    'p98': value['percentile_98'],
                })
    return qt_scheme


class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self,
                 scale,
                 model_path,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None
                 ):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None

        self.model = tf.keras.models.load_model(model_path)
    def pre_process(self, img):
        img = np.expand_dims(img,axis =0)
        # img = np.transpose(img, (2, 0, 1))
        return img

    def process(self, img):
        # model inference
        output = self.model.predict(np.array(img))
        return output

    def post_process(self, output):
        # remove extra pad
        return output[0]

    def enhance(self, input_path, output_path, outscale=None, alpha_upsampler='realesrgan'):
        size = 256
        qt_scheme = get_quantile_schema(input_path)
        with rasterio.open(input_path) as raster:
            meta = raster.meta
            meta.update({'count': 3, "dtype": "uint8","nodata":0})
            height, width = raster.height, raster.width
            input_size = size
            # padding = 10
            # stride_size= input_size-2*padding
            stride_size = input_size - 32
            # stride_size = input_size - input_size // 4
            padding = int((input_size - stride_size) / 2)

            list_coordinates = []
            for start_y in range(0, height, stride_size):
                for start_x in range(0, width, stride_size):
                    x_off = start_x if start_x == 0 else start_x - padding
                    y_off = start_y if start_y == 0 else start_y - padding

                    end_x = min(start_x + stride_size + padding, width)
                    end_y = min(start_y + stride_size + padding, height)

                    x_count = end_x - x_off
                    y_count = end_y - y_off
                    list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))
            # t = raster.transform
            # transform_new = Affine(t.a / outscale, t.b, t.c, t.d, t.e / outscale, t.f)  # <== division
            pivot = get_center_pixel(raster)
            transform_new = raster.transform*Affine.rotation(0, pivot) * Affine.scale(1/outscale)
            height_new = raster.height * outscale  # <== multiplication
            width_new = raster.width * outscale
            meta.update({'transform': transform_new, 'height': height_new, 'width': width_new})
            with rasterio.open(output_path, 'w+', **meta, compress='lzw', BIGTIFF='YES') as r:
                r.colorinterp = [ColorInterp.blue, ColorInterp.green, ColorInterp.red]
                read_lock = threading.Lock()
                write_lock = threading.Lock()

                def process(coordinates):
                    x_off, y_off, x_count, y_count, start_x, start_y = coordinates
                    read_wd = Window(x_off, y_off, x_count, y_count)

                    with read_lock:
                        values = raster.read(window=read_wd)

                        nodata = np.all(values == 0, axis=0)
                        nodata = nodata.astype(np.uint8)
                        # values = values_pr[[2,1,0]]

                    if raster.profile["dtype"] == "uint8":
                        image_detect = values[[0,1,2]]
                    else:
                        datas = []
                        for chain_i in range(3):
                            band_qt = qt_scheme[chain_i]
                            band = values[chain_i]
                            cut_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(int)
                            datas.append(cut_nor)
                        image_detect = np.array(datas)
                    image_detect = image_detect[[2,1,0]]
                    image_detect = image_detect.transpose(1, 2, 0)
                    img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                    nodata_temp = np.ones((input_size, input_size),dtype = np.uint8)
                    mask = np.pad(np.ones((stride_size* outscale, stride_size* outscale), dtype=np.uint8),
                                  ((padding* outscale, padding* outscale), (padding* outscale, padding* outscale)))
                    shape = (stride_size, stride_size)
                    if y_count < input_size or x_count < input_size:
                        img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                        nodata_temp = np.ones((input_size, input_size))
                        mask = np.zeros((input_size* outscale, input_size* outscale), dtype=np.uint8)
                        if start_x == 0 and start_y == 0:
                            img_temp[(input_size - y_count):input_size,
                            (input_size - x_count):input_size] = image_detect

                            nodata_temp[(input_size - y_count):input_size,
                            (input_size - x_count):input_size] = nodata

                            mask[(input_size - y_count)* outscale:input_size* outscale - padding* outscale,
                            (input_size - x_count)* outscale:(input_size - padding)* outscale] = 1
                            shape = (y_count - padding, x_count - padding)
                        elif start_x == 0:
                            img_temp[0:y_count, (input_size - x_count):input_size] = image_detect
                            nodata_temp[0:y_count, (input_size - x_count):input_size] = nodata
                            if y_count == input_size:
                                mask[padding* outscale:y_count* outscale - padding* outscale, (input_size - x_count)* outscale:input_size* outscale - padding* outscale] = 1
                                shape = (y_count - 2 * padding, x_count - padding)
                            else:
                                mask[padding* outscale:y_count* outscale, (input_size - x_count)* outscale:input_size* outscale - padding* outscale] = 1
                                shape = (y_count - padding, x_count - padding)
                        elif start_y == 0:
                            img_temp[(input_size - y_count):input_size, 0:x_count] = image_detect
                            nodata_temp[(input_size - y_count):input_size, 0:x_count] = nodata
                            if x_count == input_size:
                                mask[(input_size - y_count)* outscale:input_size* outscale - padding* outscale, padding* outscale:x_count* outscale - padding* outscale] = 1
                                shape = (y_count - padding, x_count - 2 * padding)
                            else:
                                mask[(input_size - y_count)* outscale:input_size* outscale - padding* outscale, padding* outscale:x_count* outscale] = 1
                                shape = (y_count - padding, x_count - padding)
                        else:
                            img_temp[0:y_count, 0:x_count] = image_detect
                            nodata_temp[0:y_count, 0:x_count] = nodata
                            mask[padding* outscale:y_count* outscale, padding* outscale:x_count* outscale] = 1
                            shape = (y_count - padding, x_count - padding)
                        nodata = nodata_temp
                        # nodata = nodata.astype(bool)
                        image_detect = img_temp
                    mask = (mask != 0)

                    if np.count_nonzero(image_detect) > 0:
                        if len(np.unique(image_detect)) <= 2:
                            pass
                        else:
                            h_input, w_input = image_detect.shape[0:2]
                            output_img = self.pre_process(image_detect/255.0)
                            # output_img = self.model(output_img)
                            output_img = self.process(output_img)
                            output_img = self.post_process(output_img)
                            output_img = np.clip(output_img,1/255,1)
                            # output_img = np.transpose(output_img, (1, 2, 0))
                            output = (output_img * 255.0).round().astype(np.uint8)
                            output[output==0]=1
                            # print(output.shape)
                            output = cv2.resize(output, (int(w_input * outscale), int(h_input * outscale),),
                                                interpolation=cv2.INTER_LANCZOS4)
                            nodata_ds = (cv2.resize(nodata, (int(w_input * outscale), int(h_input * outscale)), interpolation=cv2.INTER_NEAREST) > 0).astype(bool)
                            # output[:, :, 3][nodata_ds]=0
                            output[:, :, 2][nodata_ds]=0
                            output[:, :, 1][nodata_ds]=0
                            output[:, :, 0][nodata_ds]=0
                            # nir = (output[:, :, 3][mask]).reshape((shape[0] * outscale, shape[1] * outscale))
                            blue = (output[:, :, 2][mask]).reshape((shape[0] * outscale, shape[1] * outscale))
                            green = (output[:, :, 1][mask]).reshape((shape[0] * outscale, shape[1] * outscale))
                            red = (output[:, :, 0][mask]).reshape((shape[0] * outscale, shape[1] * outscale))


                            with write_lock:
                                r.write(red, indexes=1,
                                        window=Window(start_x * outscale, start_y * outscale, shape[1] * outscale,
                                                      shape[0] * outscale), )
                                r.write(green, indexes=2,
                                        window=Window(start_x * outscale, start_y * outscale, shape[1] * outscale,
                                                      shape[0] * outscale))
                                r.write(blue, indexes=3,
                                        window=Window(start_x * outscale, start_y * outscale, shape[1] * outscale,
                                                      shape[0] * outscale))
                                # r.write(nir, indexes=4,
                                #         window=Window(start_x * outscale, start_y * outscale, shape[1] * outscale,
                                #                       shape[0] * outscale))

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))

def run_sr(model_path,img_path,out_path,scale=8):
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None)

    upsampler.enhance(img_path, out_path, outscale=scale)

if __name__ == "__main__":
    img_path = r"/home/skymap/data_2/Anh_workspace/GAN_X8/HaNoi_AOI2_Sen2.tif"
    out_path = r"/home/skymap/data_2/Anh_workspace/GAN_X8/HaNoi_AOI2_Sen2_SR_nm.tif"
    model_path = r"/media/skymap/big_data/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_100k/models/net_g_260000.pth"

    size = 256
    upsampler = RealESRGANer(
        
        scale=8,
        model_path=model_path,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None)
    # # folder_image_path = r"Z:\new_pif_saudi\skysat"
    # # list_image=create_list_id(folder_image_path)
    # # for img_path in tqdm(list_image):
    # #     img_name = os.path.basename(img_path)
    # #     out_path = os.path.join(folder_image_path,"SKM_SR_{}".format(img_name))
    upsampler.enhance(img_path, out_path, outscale=8)
