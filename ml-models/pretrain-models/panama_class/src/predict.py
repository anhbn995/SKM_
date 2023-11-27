import cv2
import numpy as np

from tqdm import tqdm
import rasterio
from .utils import dilation_obj, remove_small_items

def get_im_by_coord(org_im, start_x, start_y,num_band, padding, crop_size, input_size):
    startx = start_x-padding
    endx = start_x+crop_size+padding
    starty = start_y - padding
    endy = start_y+crop_size+padding
    result=[]
    final_result = []
    img = org_im[starty:endy, startx:endx]
    img = img.swapaxes(2,1).swapaxes(1,0)
    try:
        
        for chan_i in range(num_band):
            result.append(cv2.resize(img[chan_i],(input_size, input_size), interpolation = cv2.INTER_AREA))
        final_result = np.array(result).swapaxes(0,1).swapaxes(1,2)
    except:
        for chan_i in range(num_band):
            result.append(np.resize(img[chan_i],(input_size, input_size)))
        final_result = np.array(result).swapaxes(0,1).swapaxes(1,2)

    return final_result

def get_img_coords(w, h, padding, crop_size):
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
    return img_coords

def padded_for_org_img(values, num_band, padding):
    padded_org_im = []
    for i in range(num_band):
        band = np.pad(values[i], padding, mode='reflect')
        padded_org_im.append(band)

    values = np.array(padded_org_im).swapaxes(0,1).swapaxes(1,2)
    # print(values.shape)
    del padded_org_im
    return values

def predict(model, values, img_coords, num_band, h, w, padding, crop_size, 
            input_size, batch_size, thresh_hold, choose_stage):
    cut_imgs = []
    for i in range(len(img_coords)):
        im = get_im_by_coord(values, img_coords[i][0], img_coords[i][1],
                            num_band,padding, crop_size, input_size)
        cut_imgs.append(im)
    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)

    y_pred = []
    for i in tqdm(range(len(a)-1)):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]])
        y_batch = model.predict(x_batch)
        if len(model.outputs)>1:
            y_batch = y_batch[choose_stage]
        mutilabel = False
        if y_batch.shape[-1]>=2:
            mutilabel = True
            y_batch = np.argmax(y_batch, axis=-1)
        # print(np.unique(y_batch), y_batch.shape)
            
        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w)).astype(np.float16)
    for i in range(len(cut_imgs)):
        true_mask = y_pred[i].reshape((input_size,input_size))
        if not mutilabel:
            true_mask = (true_mask>thresh_hold).astype(np.uint8)
            true_mask = (cv2.resize(true_mask,(input_size, input_size), interpolation = cv2.INTER_CUBIC)>thresh_hold).astype(np.uint8)
            # true_mask = true_mask.astype(np.float16)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size] = true_mask[padding:padding+crop_size, padding:padding+crop_size]
    del cut_imgs
    return big_mask

def inference(model, image_path, img_size=512, crop_size=400, num_band=8,
            batch_size=1, thresh_hold=0.05, dil=False, rm_small= False, choose_stage=None):
  
    with rasterio.open(image_path) as src:
        h,w = src.height,src.width
        dataset = src.read()
        values = dataset/255
    padding = int((img_size - crop_size)/2)
    img_coords = get_img_coords(w, h, padding, crop_size)
    values = padded_for_org_img(values, num_band, padding)
    big_mask = predict(model, values, img_coords, num_band, h, w, padding, crop_size, 
                        img_size, batch_size, thresh_hold, choose_stage)
    print('unique:', np.unique(big_mask))
    if dil:
        big_mask = dilation_obj(big_mask)
    
    if rm_small:
        big_mask = remove_small_items(big_mask, threshhlod_rm_holes=256,
                                        threshhold_rm_obj=100)
    
    return big_mask

def predict_pixel_model(model, image_path, crop_size=100, num_band=8):

    with rasterio.open(image_path) as src:
        h,w = src.height,src.width
        num_band = src.count
        img_data = np.transpose(src.read(), (1,2,0))

    list_weight = list(range(0, w, crop_size))
    list_hight = list(range(0, h, crop_size))
    final_result = np.zeros((h,w))

    with tqdm(total=len(list_hight)*len(list_weight)) as pbar:
        for start_h_org in list_hight:
            for start_w_org in list_weight:
                h_crop_start = start_h_org 
                w_crop_start = start_w_org 

                size_w_crop = min(crop_size , w - start_w_org )
                size_h_crop = min(crop_size , h - start_h_org )
                cut_image = img_data[h_crop_start:h_crop_start+size_h_crop, w_crop_start:w_crop_start+size_w_crop]
                data_img = np.reshape(cut_image, (cut_image.shape[0]*cut_image.shape[1],num_band))

                resultss = model.predict(data_img)
                img_result = np.reshape(resultss, (cut_image.shape[0],cut_image.shape[1]))
                final_result[h_crop_start:h_crop_start+size_h_crop, w_crop_start:w_crop_start+size_w_crop] = img_result
                pbar.update()
    return final_result
