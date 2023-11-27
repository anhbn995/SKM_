# -*- coding: utf-8 -*-
# from data import create_dataset
# from util.util import mkdir
import rasterio
from tqdm import tqdm
from rasterio.windows import Window
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from geopandas import GeoDataFrame
import rasterio
import numpy as np
import tensorflow as tf
import os
input_size = 480
crop_size = 400



def write_window_many_chanel(output_ds, arr_c, window_draw_pre):
    s_h, e_h ,s_w, e_w, sw_w, sw_h, size_w_crop, size_h_crop = window_draw_pre 
    output_ds.write(arr_c[s_h:e_h,s_w:e_w],window = Window(sw_w, sw_h, size_w_crop, size_h_crop), indexes = 1)


def tensor2im(input_image, imtype=np.uint8, normalize=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        # print('a'*100)
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            # print('b'*100)
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # print(image_numpy.shape,'t'*100)
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            # print('c'*100)
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # print(image_numpy.shape)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if normalize:
            # print('d'*100)
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = image_numpy.transpose(2,0,1)[0]
    return image_numpy.astype(imtype)


def read_window_and_index_result(h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w, tmp_img_size_model, src_img, num_band_train):
    """
        Trả về img de predict vs kich thước model
        Và vị trí để có thể ghi mask vào trong đúng vị trí ảnh
    """
    if h_crop_start < 0 and w_crop_start < 0:
        # continue
        h_crop_start = 0
        w_crop_start = 0
        size_h_crop = crop_size + padding
        size_w_crop = crop_size + padding
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        tmp_img_size_model[:, padding:, padding:] = img_window_crop
        window_draw_pre = [padding, crop_size + padding, padding, crop_size + padding, start_w_org, start_h_org, crop_size, crop_size]

    # truong hop h = 0 va w != 0
    elif h_crop_start < 0:
        h_crop_start = 0
        size_h_crop = crop_size + padding
        size_w_crop = min(crop_size + 2*padding, w - start_w_org + padding)
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        if size_w_crop == w - start_w_org + padding:
            end_c_index_w =  size_w_crop
            tmp_img_size_model[:,padding:,:end_c_index_w] = img_window_crop
        else:
            end_c_index_w = crop_size + padding
            tmp_img_size_model[:, padding:,:] = img_window_crop
        window_draw_pre = [padding, crop_size + padding ,padding, end_c_index_w, start_w_org, start_h_org,  min(crop_size, w - start_w_org), crop_size]

    # Truong hop w = 0, h!=0 
    elif w_crop_start < 0:
        w_crop_start = 0
        size_w_crop = crop_size + padding
        size_h_crop = min(crop_size + 2*padding, h - start_h_org + padding)
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        if size_h_crop == h - start_h_org + padding:
            end_c_index_h =  size_h_crop
            tmp_img_size_model[:,:end_c_index_h,padding:] = img_window_crop
        else:
            end_c_index_h = crop_size + padding
            tmp_img_size_model[:,:, padding:] = img_window_crop
        window_draw_pre = [padding, end_c_index_h, padding, crop_size + padding, start_w_org, start_h_org, crop_size, min(crop_size, h - start_h_org)]
        
    # Truong hop ca 2 deu khac khong
    else:
        size_w_crop = min(crop_size +2*padding, w - start_w_org + padding)
        size_h_crop = min(crop_size +2*padding, h - start_h_org + padding)
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        # print(img_window_crop.shape, size_w_crop, size_h_crop)
        if size_w_crop < (crop_size + 2*padding) and size_h_crop < (crop_size + 2*padding):
            # print(img_window_crop.shape, size_w_crop, size_h_crop)
            end_c_index_h = size_h_crop
            end_c_index_w = size_w_crop
            tmp_img_size_model[:,:end_c_index_h,:   end_c_index_w] = img_window_crop
        elif size_w_crop < (crop_size + 2*padding):
            end_c_index_h = crop_size + padding
            end_c_index_w = size_w_crop
            tmp_img_size_model[:,:,:end_c_index_w] = img_window_crop
        elif size_h_crop < (crop_size + 2*padding):
            end_c_index_w = crop_size + padding
            end_c_index_h = size_h_crop
            tmp_img_size_model[:,:end_c_index_h,:] = img_window_crop
        else:
            end_c_index_w = crop_size + padding
            end_c_index_h = crop_size + padding
            tmp_img_size_model[:,:,:] = img_window_crop
        window_draw_pre = [padding, end_c_index_h, padding, end_c_index_w, start_w_org, start_h_org, min(crop_size, w - start_w_org), min(crop_size, h - start_h_org)]    
    # print(h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w)
    return tmp_img_size_model, window_draw_pre


def predict_win(model, A_img, B_img):
    # print(A_img.shape)
    A = A_img.transpose(1,2,0)
    B = B_img.transpose(1,2,0)
    img = np.concatenate((A,B),axis=-1)
    y_pred = model.predict(img[np.newaxis,...]/255.)[0]
    y_pred = (y_pred[0,...,0] > 0.5).astype(np.uint8)
    return  y_pred


def predict_big(input_1,input_2,model_path,tmppath,output):
    model = tf.keras.models.load_model(model_path)
    output_dir = os.path.dirname(output)
    os.makedirs(output_dir, exist_ok=True)
    print('p'*100, output_dir)

    image1_path = input_1
    image2_path = input_2
    outputFileName = output
    output_name = os.path.basename(output).replace(".geojson", ".tif")
    output_file_tmp_tif = os.path.join(tmppath,output_name)
    print(image1_path, '\n', image2_path)

    # num_band_train = opt.input_nc
    num_band_train = 3

    with rasterio.open(image1_path) as src_imgA:
        h,w = src_imgA.height,src_imgA.width
        source_crs = src_imgA.crs
        source_transform = src_imgA.transform
    
    with rasterio.open(output_file_tmp_tif, 'w', driver='GTiff',
                                    height = h, width = w,
                                    count=1, dtype='uint8',
                                    crs=source_crs,
                                    transform=source_transform,
                                    nodata=0,
                                    compress='lzw') as output_ds:
            output_ds = np.empty((1,h,w))
    
    padding = int((input_size - crop_size)/2)
    list_weight = list(range(0, w, crop_size))
    list_hight = list(range(0, h, crop_size))

    with rasterio.open(output_file_tmp_tif,"r+") as output_ds:
        with tqdm(total=len(list_hight)*len(list_weight)) as pbar:
            with rasterio.open(image1_path) as src_imgA:
                with rasterio.open(image2_path) as src_imgB:
                    for start_h_org in list_hight:
                        for start_w_org in list_weight:
                            # vi tri bat dau
                            h_crop_start = start_h_org - padding
                            w_crop_start = start_w_org - padding
                            # kich thuoc
                            
                            tmp_A_size_model = np.zeros((num_band_train, input_size,input_size))
                            tmp_A_size_model, window_draw_pre = read_window_and_index_result(h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w, tmp_A_size_model, src_imgA, num_band_train)

                            tmp_B_size_model = np.zeros((num_band_train, input_size,input_size))
                            tmp_B_size_model, window_draw_pre = read_window_and_index_result(h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w, tmp_B_size_model, src_imgB, num_band_train)

                            mask_predict_win = predict_win(model, tmp_A_size_model.astype('uint8'), tmp_B_size_model.astype('uint8'))
                            write_window_many_chanel(output_ds, mask_predict_win, window_draw_pre)
                            # img_predict = predict_win(model, tmp_B_size_model, tmp_B_size_model)
                            # print(img_predict.shape)

                            # write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, end_c_index_w, 
                            #                                 start_w_org, start_h_org, min(crop_size, w - start_w_org), min(crop_size, h - start_h_org))
                            pbar.update()

    with rasterio.open(output_file_tmp_tif, 'r+') as dst:
        dst.nodata = 0
    with rasterio.open(output_file_tmp_tif) as src:
        data = src.read(1, masked=True)
        shape_gen = ((shape(s), v) for s, v in shapes(data, transform=src.transform))
        gdf = GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=src.crs)
        gdf = gdf.to_crs(4326)
        gdf.to_file(output, driver = "GeoJSON")

if __name__ == '__main__':
    predict_big(
        r"/home/skymap/big_data/change_api/test/A/A1.tif",
        r"/home/skymap/big_data/change_api/test/B/A1.tif",
        r"/home/skymap/big_data/change_api/checkpoints/Dubai_change_CDF0_ver512/192_F1_1_0.61344_net_F.pth",
        r"/home/skymap/big_data/change_api/tmp",
        r"/home/skymap/big_data/change_api/tmp/a1.geojson"

    )