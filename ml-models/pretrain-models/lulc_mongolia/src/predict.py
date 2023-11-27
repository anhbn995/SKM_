import profile
import gdal
import json

from numpy import dtype, percentile
from .utils import *
import ast

def predict_building(img_url, cnn_model, out_put, threshold = 0.5):
    num_band = 7
    input_size = 256
    INPUT_SIZE = 256
    crop_size = 200
    batch_size = 1
    
    dataset1 = gdal.Open(img_url)
    index_band =[35,42]
    percentile_band = ast.literal_eval(dataset1.GetMetadata().get('percentile'))
    values = dataset1.ReadAsArray()[index_band[0]:index_band[1]]

    h,w = values.shape[1:3]

    new_image = np.zeros((h, w, num_band), dtype=np.uint8)
    for i in range(index_band[0], index_band[1]):
        band = values[i-index_band[0]]
        band_qt = percentile_band[i]
        new_band = np.interp(band, (band_qt.get('min'), band_qt.get('max')), (1, 255)).astype(np.uint8)
        new_image[...,i-index_band[0]] = new_band
    new_image = np.moveaxis(new_image, -1, 0)

    padding = int((input_size - crop_size)/2)
    # print("pading",padding)
    padded_org_im = []
    cut_imgs = []
    new_w = w + 2*padding
    new_h = h + 2*padding
    cut_w = list(range(padding, new_w - padding, crop_size))
    cut_h = list(range(padding, new_h - padding, crop_size))

    list_hight = []
    list_weight = []
    for i in cut_h:
        if i < new_h - padding - crop_size:
            list_hight.append(i)
    list_hight.append(new_h-crop_size-padding)

    for i in cut_w:
        if i < new_w - crop_size - padding:
            list_weight.append(i)
    list_weight.append(new_w-crop_size-padding)

    img_coords = []
    for i in list_weight:
        for j in list_hight:
            img_coords.append([i, j])
    
    for i in range(num_band):
        band = np.pad(new_image[i], padding, mode='reflect')
        padded_org_im.append(band)

    new_image = np.array(padded_org_im).swapaxes(0,1).swapaxes(1,2)
    del padded_org_im

    def get_im_by_coord(org_im, start_x, start_y,num_band):
        startx = start_x-padding
        endx = start_x+crop_size+padding
        starty = start_y - padding
        endy = start_y+crop_size+padding
        result=[]
        img = org_im[starty:endy, startx:endx]
        img = img.swapaxes(2,1).swapaxes(1,0)
        for chan_i in range(num_band):
            result.append(cv2.resize(img[chan_i],(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC))
        return np.array(result).swapaxes(0,1).swapaxes(1,2)

    for i in range(len(img_coords)):
        im = get_im_by_coord(
            new_image, img_coords[i][0], img_coords[i][1],num_band)
        cut_imgs.append(im)

    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)

    y_pred = []
    for i in range(len(a)-1):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]]).astype(np.float32)/255.0
        y_batch = cnn_model.predict(x_batch)
        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w)).astype(np.float16)
    for i in range(len(cut_imgs)):
        true_mask = y_pred[i].reshape((INPUT_SIZE,INPUT_SIZE))
        true_mask = (true_mask>0.5).astype(np.uint8)
        true_mask = (cv2.resize(true_mask,(input_size, input_size), interpolation = cv2.INTER_CUBIC)>0.5).astype(np.uint8)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size] = true_mask[padding:padding+crop_size, padding:padding+crop_size]

    del cut_imgs
    big_mask = (big_mask > 0.5).astype(np.uint8)*255
    with rasterio.open(img_url) as src:
        transform1 = src.transform
        w,h = src.width,src.height
        crs = src.crs

    new_dataset = rasterio.open(out_put, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw',
                                nodata=0)
    new_dataset.write(big_mask,1)
    new_dataset.close()

    return out_put

def predict_df(df_all_predict, model, crop_size, shape_win):
    predictions = model.predict(df_all_predict, batch_size=1000, verbose=1)
    predictions = np.transpose(predictions)
    predictions = np.reshape(predictions, (-1, shape_win[0], shape_win[1]))
    print('shape of predictions', predictions.shape)
    return np.array([np.argmax(predictions, axis=0).astype('uint8')])

def predict_nobuilding(out_fp_predict, fp_img_stack, crop_size, model):
    with rasterio.open(fp_img_stack) as src:
        h,w = src.height,src.width
        source_crs = src.crs
        source_transform = src.transform
        
    with rasterio.open(out_fp_predict, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, 
                                crs=source_crs,
                                transform=source_transform,
                                nodata=0,
                                dtype=rasterio.uint8,
                                compress='lzw') as output_ds:
        output_ds = np.empty((1,h,w))
        
        
    input_size = crop_size    
    padding = int((input_size - crop_size)/2)
    list_weight = list(range(0, w, crop_size))
    list_hight = list(range(0, h, crop_size))

    with rasterio.open(out_fp_predict,"r+") as output_ds:
        with tqdm(total=len(list_hight)*len(list_weight)) as pbar:
            for start_h_org in list_hight:
                for start_w_org in list_weight:
                    h_crop_start = start_h_org - padding
                    w_crop_start = start_w_org - padding
                    if h_crop_start < 0 and w_crop_start < 0:
                        h_crop_start = 0
                        w_crop_start = 0
                        size_h_crop = crop_size + padding
                        size_w_crop = crop_size + padding
                        
                        df_all_predict, shape_win =create_df_from_window_oke(fp_img_stack, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
                        img_predict = predict_df(df_all_predict, model, crop_size, shape_win) + 1
                        write_window_many_chanel(output_ds, img_predict, padding, crop_size + padding, padding, crop_size + padding, 
                                                                        start_w_org, start_h_org, crop_size, crop_size)
                    elif h_crop_start < 0:
                        h_crop_start = 0
                        size_h_crop = crop_size + padding
                        size_w_crop = min(crop_size + 2*padding, w - start_w_org + padding)
                        if size_w_crop == w - start_w_org + padding:
                            end_c_index_w =  size_w_crop
                        else:
                            end_c_index_w = crop_size + padding

                        df_all_predict, shape_win =create_df_from_window_oke(fp_img_stack, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
                        img_predict = predict_df(df_all_predict, model, crop_size, shape_win) + 1
                        write_window_many_chanel(output_ds, img_predict, padding, crop_size + padding ,padding, end_c_index_w, 
                                                    start_w_org, start_h_org,  min(crop_size, w - start_w_org), crop_size)
                    elif w_crop_start < 0:
                        w_crop_start = 0
                        size_w_crop = crop_size + padding
                        size_h_crop = min(crop_size + 2*padding, h - start_h_org + padding)
                        
                        if size_h_crop == h - start_h_org + padding:
                            end_c_index_h =  size_h_crop
                        else:
                            end_c_index_h = crop_size + padding

                        df_all_predict, shape_win =create_df_from_window_oke(fp_img_stack, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
                        img_predict = predict_df(df_all_predict, model, crop_size, shape_win) + 1
                        write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, crop_size + padding, 
                                                    start_w_org, start_h_org, crop_size, min(crop_size, h - start_h_org))
                    else:
                        size_w_crop = min(crop_size +2*padding, w - start_w_org + padding)
                        size_h_crop = min(crop_size +2*padding, h - start_h_org + padding)
                        if size_w_crop < (crop_size + 2*padding) and size_h_crop < (crop_size + 2*padding):
                            end_c_index_h = size_h_crop
                            end_c_index_w = size_w_crop
                            
                        elif size_w_crop < (crop_size + 2*padding):
                            end_c_index_h = crop_size + padding
                            end_c_index_w = size_w_crop
                            
                        elif size_h_crop < (crop_size + 2*padding):
                            end_c_index_w = crop_size + padding
                            end_c_index_h = size_h_crop
                            
                        else:
                            end_c_index_w = crop_size + padding
                            end_c_index_h = crop_size + padding
                            
                        df_all_predict, shape_win =create_df_from_window_oke(fp_img_stack, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
                        img_predict = predict_df(df_all_predict, model, crop_size, shape_win) + 1 
                        write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, end_c_index_w, 
                                                    start_w_org, start_h_org, min(crop_size, w - start_w_org), min(crop_size, h - start_h_org))
                    pbar.update()