import cv2
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf

from tqdm import tqdm
from rasterio.windows import Window
from mrcnn.config import Config
from mrcnn import model as modellib
tf.compat.v1.enable_eager_execution()
from shapely.geometry import Polygon


class InferenceConfig(Config):
    """Config for predict tree counting model"""
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    IMAGE_MAX_DIM = 512+64
    IMAGE_MIN_DIM = 512+64
    DETECTION_MAX_INSTANCES = 100
    MAX_GT_INSTANCES = 60

    MASK_SHAPE = [28, 28]

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    NAME = "tree_counting_model"
    BACKBONE = "resnet101"

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.85


def write_window_many_chanel(output_ds, arr_c, window_draw_pre):
    s_h, e_h ,s_w, e_w, sw_w, sw_h, size_w_crop, size_h_crop = window_draw_pre 
    output_ds.write(arr_c[s_h:e_h,s_w:e_w],window = Window(sw_w, sw_h, size_w_crop, size_h_crop), indexes = 1)



def read_window_and_index_result(crop_size, h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w, tmp_img_size_model, src_img, num_band_train):
    """
        Trả về img de predict vs kich thước model
        Và vị trí để có thể ghi mask vào trong đúng vị trí ảnh
    """
    if h_crop_start < 0 and w_crop_start < 0:
        h_crop_start = 0
        w_crop_start = 0
        size_h_crop = crop_size + padding
        size_w_crop = crop_size + padding
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        tmp_img_size_model[:, padding:, padding:] = img_window_crop
        window_draw_pre = [padding, crop_size + padding, padding, crop_size + padding, start_w_org, start_h_org, crop_size, crop_size]

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
    
    else:
        size_w_crop = min(crop_size +2*padding, w - start_w_org + padding)
        size_h_crop = min(crop_size +2*padding, h - start_h_org + padding)
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        if size_w_crop < (crop_size + 2*padding) and size_h_crop < (crop_size + 2*padding):
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
    return tmp_img_size_model, window_draw_pre


def predict_single(model, image, numband, model_size, batch_size):
    image_detect = image[0:numband].swapaxes(0, 1).swapaxes(1, 2)
    # print('sai'*100,image_detect.shape)
    predictions = model.detect([image_detect], verbose=0)
    # print('ka'*100,predictions)
    p = predictions[0]
    # get box and  score and convert box to opencv contour fomat
    boxes = p['rois']
    N = boxes.shape[0]
    list_contour_temp = []
    list_score_temp = []
    list_label_temp = []
    
    for i in range(N):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        score = p["scores"][i]
        label = p["class_ids"][i]
        contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
        contour = contour.reshape(-1, 1, 2)
        try:
            if cv2.contourArea(contour) > 10:
                if (contour.max() < (model_size - 3)) and (contour.min() > 3):
                    list_contour_temp.append(contour)
                    list_score_temp.append(score)
                    list_label_temp.append(label)
        except Exception:
            pass
    
    return list_contour_temp, list_score_temp, list_label_temp
        
    
def predict_lager(fp_img, model_tree, model_size, crop_size, batch_size):
    num_band_train = 3
    with rasterio.open(fp_img) as src:
        h,w = src.height,src.width
        source_crs = src.crs
        source_transform = src.transform
    
    padding = int((model_size - crop_size)/2)
    list_weight = list(range(0, w, crop_size))
    list_hight = list(range(0, h, crop_size))
    
    list_contours = []
    list_scores = []
    list_labels = []
    
    with tqdm(total=len(list_hight)*len(list_weight)) as pbar:
        with rasterio.open(fp_img) as src:
            for start_h_org in list_hight:
                for start_w_org in list_weight:
                    # vi tri bat dau
                    h_crop_start = start_h_org - padding
                    w_crop_start = start_w_org - padding
                    
                    # kich thuoc
                    tmp_img_model = np.zeros((num_band_train, model_size,model_size))
                    tmp_img_model, _ = read_window_and_index_result(crop_size, h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w, tmp_img_model, src, num_band_train)
                    list_contour_temp, list_score_temp, list_label_temp = predict_single(model_tree, tmp_img_model, num_band_train, model_size, batch_size)
                    temp_contour = []
                    for contour in list_contour_temp:
                        tmp_poly_window = contour.reshape(-1, 2)
                        tmp_poly = tmp_poly_window + np.array([w_crop_start, h_crop_start])
                        con_rs = tmp_poly.reshape(-1, 1, 2)
                        temp_contour.append(con_rs)
                    list_contours.extend(temp_contour)
                    list_scores.extend(list_score_temp)
                    list_labels.extend(list_label_temp)
                    pbar.update()
    list_polygons = [list_contours[i].reshape(-1, 2) for i in range(len(list_contours))]
    return list_polygons, list_scores, list_labels, source_transform, source_crs
                    

def NMS_polygons(list_polygons, list_scores, list_labels, iou_threshold=0.2):
    list_shapely_polygons = [Polygon(polygon) for polygon in list_polygons]
    list_bound = [np.array(polygon.bounds) for polygon in list_shapely_polygons]
    indexes = tf.image.non_max_suppression(np.array(list_bound), np.array(list_scores), len(list_scores),iou_threshold=iou_threshold)
    result_polygons = [list_polygons[idx] for idx in indexes]
    result_scores = [list_scores[idx] for idx in indexes]
    result_labels = [list_labels[idx] for idx in indexes]
    return result_polygons, result_scores, result_labels


def transform_poly_px_to_geom(polygon, geotransform):
    """
    Convert polygon from pixel coordinate to geographical coordinate.

    Parameters
    ----------
    polygon : Polygon format pixel coordinate
        Polygon contain a grid.
    geotransform : Affine
        Image's information of transform.
    Returns
    -------
    poly_rs :
        Geographical coordinate of each tree.
    """
    top_left_x = geotransform[2]
    top_left_y = geotransform[5]
    x_res = geotransform[0]
    y_res = geotransform[4]
    poly = np.array(polygon)
    poly_rs = poly * np.array([x_res, y_res]) + np.array([top_left_x, top_left_y])
    return poly_rs


def export_result_to_shp(result_polygons_nms, score_result_nms, label_result_nms, transform, source_crs, output_path):
    list_geo_polygon = [Polygon(transform_poly_px_to_geom(polygon, transform)) for polygon in result_polygons_nms]
    tree_id = list(range(len(score_result_nms)))
    data_tree = list(zip(list_geo_polygon, score_result_nms, label_result_nms, tree_id))
    df_polygon = pd.DataFrame(data_tree, columns=['geometry','score','label',"FID"])
    gdf_polygon = gpd.GeoDataFrame(df_polygon, geometry='geometry', crs=source_crs)
    gdf_polygon.to_file(output_path)
    

def main(fp_img, fp_model, model_size, crop_sizes, fp_out_shape):
    config = InferenceConfig()
    model_tree = modellib.MaskRCNN(
        mode="inference", model_dir='', config=config)
    model_tree.load_weights(fp_model, by_name=True)
    
    source_crs = None
    source_transform = None
    list_polygons_all = []
    list_scores_all = []
    list_labels_all = []
    for crop_size in crop_sizes:
        list_polygons, list_scores, list_labels, source_transform, source_crs = predict_lager(fp_img, model_tree, model_size, crop_size, config.BATCH_SIZE)
        list_polygons_all+=list_polygons
        list_scores_all += list_scores
        list_labels_all += list_labels
        
        
    result_polygons_nms, score_result_nms, label_result_nms = NMS_polygons(list_polygons_all, list_scores_all, list_labels_all, iou_threshold=0.2)
    export_result_to_shp(result_polygons_nms, score_result_nms, label_result_nms, source_transform, source_crs, fp_out_shape)
    
if __name__=="__main__":
    fp_img = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/DaNang_23_07_2009_LowAccuracy.tif"
    fp_model = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/data_faster-rnn/image_crop_box/tmp/gen_TauBien_Original_3band_cut_512_stride_200_time_20230709_001006/images_data/Model/taubien20230709T0103/mask_rcnn_taubien_0193.h5"
    model_size = 512
    crop_size = [512]
    # output_type_bbox = 'no_bbox'
    fp_out_shape = r'/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/RS_Tmp/DaNang_23_07_2009_LowAccuracy_512_ok.shp'
    main(fp_img, fp_model, model_size, crop_size, fp_out_shape)
    