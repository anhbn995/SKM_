import os
import numpy as np
from tqdm import *
import cv2
import geopandas as gp
import pandas as pd
import rasterio
from rasterio.windows import Window
from mrcnn.config import Config
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from utils.utils import transform_poly_px_to_geom, convert_window_to_polygon
import tensorflow as tf
from get_image_resolution_meter import get_resolution_meter
from tensorflow.compat.v1.keras.backend import set_session

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# set_session(tf.compat.v1.Session(config=config))
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# MODEL_SIZE = int(round(0.3*256/get_resolution_meter(r"/media/skymap/Learnning/public/tmp_Nam/buildingtesst/test2.tif")/64)*64)
MODEL_SIZE = 512
MODEL_DIR=""
NUM_BAND=3
overlap_size = int(MODEL_SIZE*3/4)
print(overlap_size, MODEL_SIZE)

class InferenceConfig(Config):
    """Config for predict tree counting model"""
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    IMAGE_MAX_DIM = 512+64
    IMAGE_MIN_DIM = 512+64
    DETECTION_MAX_INSTANCES = 500
    MAX_GT_INSTANCES = 60

    MASK_SHAPE = [28, 28]

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    NAME = "tree_counting_model"
    BACKBONE = "resnet50"

    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.85

def read_image_by_window(dataset_image, x_off, y_off, x_count, y_count, start_x, start_y, ):
    """ 
    This function to read image window by coordinates
    """
    num_band = NUM_BAND
    input_size = MODEL_SIZE

    image_detect = dataset_image.read(window=Window(x_off, y_off, x_count, y_count))[0:num_band].swapaxes(0, 1).swapaxes(1, 2)
    if image_detect.shape[0] < input_size or image_detect.shape[1] < input_size:
        img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
        if start_x == 0 and start_y == 0:
            img_temp[(input_size - image_detect.shape[0]):input_size, (input_size - image_detect.shape[1]):input_size] = image_detect
        elif start_x == 0:
            img_temp[0:image_detect.shape[0], (input_size - image_detect.shape[1]):input_size] = image_detect
        elif start_y == 0:
            img_temp[(input_size - image_detect.shape[0]):input_size, 0:image_detect.shape[1]] = image_detect
        else:
            img_temp[0:image_detect.shape[0], 0:image_detect.shape[1]] = image_detect
        image_detect = img_temp
    print('sai'*1000, image_detect.shape)
    return image_detect.astype(np.uint8)

def gen_list_slide_windows(h, w, input_size, stride_size):
    """ 
    This function to gen all window coordinates for predict big image
    """
    list_coordinates = []
    padding = int((input_size - stride_size) / 2)
    new_w = w + 2 * padding
    new_h = h + 2 * padding
    cut_w = list(range(padding, new_w - padding, stride_size))
    cut_h = list(range(padding, new_h - padding, stride_size))
    list_height = []
    list_weight = []
    # print(w, h)
    for i in cut_h:
        list_height.append(i)

    for i in cut_w:
        list_weight.append(i)
    for i in range(len(list_height)):
        top_left_y = list_height[i]
        for j in range(len(list_weight)):
            top_left_x = list_weight[j]
            start_x = top_left_x - padding
            end_x = min(top_left_x + stride_size + padding, new_w - padding)
            start_y = top_left_y - padding
            end_y = min(top_left_y + stride_size + padding, new_h - padding)
            if start_x == 0:
                x_off = start_x
            else:
                x_off = start_x - padding
            if start_y == 0:
                y_off = start_y
            else:
                y_off = start_y - padding
            x_count = end_x - padding - x_off
            y_count = end_y - padding - y_off
            list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))
    return list_coordinates

def predict_small_image(dataset_image, model_path):
    """
        predict for image small than model input size
    """
    num_band = NUM_BAND
    from mrcnn import model2 as modellib
    w, h = dataset_image.width, dataset_image.height
    image = dataset_image.read()[0:num_band].swapaxes(0, 1).swapaxes(1, 2).astype(np.uint8)
    image_res = 0.02
    config = InferenceConfig()
    config.IMAGE_MAX_DIM = (round(max(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64
    config.IMAGE_MIN_DIM = (round(min(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config)

    print("inside predict api")
    print(model_path)
    model.load_weights(model_path, by_name=True)
    predictions = model.detect([image] * config.BATCH_SIZE, verbose=1)
    p = predictions[0]
    boxes = p['rois']
    scores = p['scores']
    N = boxes.shape[0]
    list_contours = []
    list_score = []
    for i in range(N):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
        contour = contour.reshape(-1, 1, 2)
        try:
            if cv2.contourArea(contour) > 100:
                list_contours.append(contour)
                list_score.append(scores[i])
        except Exception:
            pass
    predictions = None
    p = None
    model = None
    return list_contours, list_score

def predict_big_image(dataset_image, model_path, bound_aoi, out_type, verbose):
    """
    Function to predict big image with stride window.

    dataset_image: dataset open by rasterio
    model_path: Mask-RCNN h5 weight
    num_band: num channel config for image
    input_size: input size image to push to model
    stride_size: stride window size
    bound_aoi: shapely polygon area AIO care in this image predict
    """
    from mrcnn import model as modellib
    tf.compat.v1.enable_eager_execution()
    # get transform and image hw for calculator
    w, h = dataset_image.width, dataset_image.height
    transform = dataset_image.transform
    # config model predict
    config = InferenceConfig()
    # config.display()
    input_size = MODEL_SIZE
    stride_size = overlap_size
    # padding size for each stride window
    padding = int((input_size - stride_size) / 2)
    # call model
    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config)
    # load weight
    if verbose:
        print("Loaded model")
    model.load_weights(model_path, by_name=True)
    # create list for save result when predict
    return_contour = []
    return_score = []
    return_label = []
    # Calculator all window location before predict
    list_window_coordinates = gen_list_slide_windows(h, w, input_size, stride_size)
    if verbose:
        print("Predicting ...")
    with tqdm(total=len(list_window_coordinates), disable = not(verbose)) as p_bar:
        for window_coordinate in list_window_coordinates:
            # get each coordinates in list window coordinates
            x_off, y_off, x_count, y_count, start_x, start_y = window_coordinate
            # get image window by coordinate
            image_detect = read_image_by_window(dataset_image, x_off, y_off, x_count, y_count, start_x, start_y)

            # calculator bound polygon of window for check intersect with AOI care
            polygon_bound = convert_window_to_polygon(x_off, y_off, x_count, y_count)
            geo_polygon = Polygon(transform_poly_px_to_geom(polygon_bound, transform))
            check_inter = geo_polygon.intersects(bound_aoi)
            # if image not no data and intersect with AIO care then push to predict
            if np.count_nonzero(image_detect) > 0 and check_inter:
                predictions = model.detect([image_detect] * config.BATCH_SIZE, verbose=0)
                p = predictions[0]
                ##########################################################
                # get box and  score and convert box to opencv contour fomat
                boxes = p['rois']
#                 print(boxes)
                N = boxes.shape[0]
                list_temp = []
                list_score_temp = []
                list_label_temp = []
                for i in range(N):
                    if not np.any(boxes[i]):
                        continue
                    y1, x1, y2, x2 = boxes[i]
                    score = p["scores"][i]
                    label = p["class_ids"][i]
                    if out_type=="bbox":
                        contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
                        contour = contour.reshape(-1, 1, 2)
                    else:
                        true_mask_result = p['masks'][:, :, i].astype(np.uint8)
                        contours, hierarchy = cv2.findContours(true_mask_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if len(contours)>0:
                            contour = contours[0]
                            _, radius_f = cv2.minEnclosingCircle(contour)
                            for cnt in contours:
                                _, radius = cv2.minEnclosingCircle(cnt)
                                if radius>radius_f:
                                    radius_f = radius
                                    contour = cnt
                    try:
                        if cv2.contourArea(contour) > 10:
                            if (contour.max() < (input_size - 3)) and (contour.min() > 3):
                                list_temp.append(contour)
                                list_score_temp.append(score)
                                list_label_temp.append(label)
#                             elif (contour.max() < (input_size - padding)) and (contour.min() > padding):
#                                 list_temp.append(contour)
#                                 list_score_temp.append(score)
                    except Exception:
                        pass
                #########################################################
                # change polygon from window image predict coords to big image coords
                temp_contour = []
                for contour in list_temp:
                    tmp_poly_window = contour.reshape(-1, 2)
                    tmp_poly = tmp_poly_window + np.array([start_x - padding, start_y - padding])
                    con_rs = tmp_poly.reshape(-1, 1, 2)
                    temp_contour.append(con_rs)
                return_contour.extend(temp_contour)
                return_score.extend(list_score_temp)
                return_label.extend(list_label_temp)
            p_bar.update()
            # FOR LOOP ALL WINDOW
    predictions = None
    p = None
    model = None
    list_contours = return_contour
    list_scores = return_score
    list_labels = return_label
    return list_contours, list_scores, list_labels

def get_bound(image_path, bound_path, id_image):
    """get Aoi bound from AOI care, if none, return image bound"""
    with rasterio.open(image_path) as src:
        transform = src.transform
        w, h = src.width, src.height
        proj_str = (src.crs.to_string())
    bound_image = ((0, 0), (w, 0), (w, h), (0, h), (0, 0))
    try:
        petak_id = id_image.split('_')[-2]
    except:
        petak_id = ""
    if bound_path:
        bound_shp = gp.read_file(bound_path)
        bound_shp = bound_shp.to_crs(proj_str)
        bound_aoi_table = bound_shp.loc[bound_shp[FIELDS_NAME] == petak_id]
        # If have AOI go to predict
        if len(bound_aoi_table) > 0:
            # get AIO geometry
            bound_aoi = bound_aoi_table.iloc[0].geometry
            bound_aoi = bound_aoi.buffer(-1)
        else:
            bound_aoi = Polygon(transform_poly_px_to_geom(bound_image, transform))
    else:
        # if don't have bound aoi then predict all image
        bound_aoi = Polygon(transform_poly_px_to_geom(bound_image, transform))
    return bound_aoi


def export_predict_result_to_file(polygon_result_all, score_result_all, label_result_all, bound_aoi, transform, proj_str, out_format, output_path):
    list_geo_polygon = [Polygon(transform_poly_px_to_geom(polygon, transform)) for polygon in polygon_result_all]
    df_polygon = pd.DataFrame(list_geo_polygon, columns=['geometry'])
    gdf_polygon = gp.GeoDataFrame(df_polygon, geometry='geometry', crs=proj_str)
    gdf_polygon.to_file(output_path)
    



def export_predict_result_to_file_2(polygon_result_all, score_result_all, label_result_all, bound_aoi, transform, proj_str, out_format, image_id, output_path):
    list_geo_polygon = [Polygon(transform_poly_px_to_geom(polygon, transform)) for polygon in polygon_result_all]
    print('helo'*100,list_geo_polygon)
    tree_polygon = [geom for geom in list_geo_polygon]
    tree_point = [geom.centroid for geom in list_geo_polygon]
    strtree_point = STRtree(tree_point)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(tree_point))

    list_point = strtree_point.query(bound_aoi)
    print('helo'*100,list_point)
    list_point_inside = [x for x in list_point if bound_aoi.contains(x)]

    index_point = [index_by_id[id(pt)] for pt in list_point_inside]
    tree_polygon_rs = [tree_polygon[index] for index in index_point]
    tree_score_rs = [score_result_all[index] for index in index_point]
    tree_label_rs = [label_result_all[index] for index in index_point]
    tree_id = list(range(len(tree_score_rs)))
    data_tree = list(zip(tree_polygon_rs, tree_score_rs, tree_label_rs, tree_id))

    df_polygon = pd.DataFrame(data_tree, columns=['geometry','score','label',"FID"])
    gdf_polygon = gp.GeoDataFrame(df_polygon, geometry='geometry', crs=proj_str)

    if out_format == "shp":
        gdf_polygon.to_file(output_path)
    return True


def nms_result(list_polygons, list_scores, list_labels, iou_threshold=0.2):
    list_shapely_polygons = [Polygon(polygon) for polygon in list_polygons]
    list_bound = [np.array(polygon.bounds) for polygon in list_shapely_polygons]
    indexes = tf.image.non_max_suppression(np.array(list_bound), np.array(list_scores), len(list_scores),iou_threshold=iou_threshold)
    result_polygons = [list_polygons[idx] for idx in indexes]
    result_scores = [list_scores[idx] for idx in indexes]
    result_labels = [list_labels[idx] for idx in indexes]
    return result_polygons,result_scores,result_labels

def predict_main(image_path, model_path, output_path, bound_path=None,out_type="bbox",verbose=1):
    """Predict image and write result to shape file out put.
    image_path: input image tiff file
    model_path: h5 weight model Mask-RCNN
    output_path: output shape file path
    bound_path: Path to shape file contain AOI care.
    """
    out_format ="shp"
    # Open data set for predict step ( Read by Window)  
    with rasterio.open(image_path) as dataset_image:
        # Read image information
        transform = dataset_image.transform
        w, h = dataset_image.width, dataset_image.height
        proj_str = (dataset_image.crs.to_string())
        # Get id image
        image_name = os.path.basename(image_path)
        image_id = os.path.splitext(image_name)[0]
        # Get AOI by image ID
        bound_aoi = get_bound(image_path, bound_path, image_id)
        # Config image model and stride size
        input_size = MODEL_SIZE

        if h <= input_size or w <= input_size:
            list_contours, list_scores = predict_small_image(dataset_image, model_path)
        else:
            list_contours, list_scores, list_labels = predict_big_image(dataset_image, model_path, bound_aoi, out_type, verbose)
            
    list_polygons = [list_contours[i].reshape(-1, 2) for i in range(len(list_contours))]
    if verbose:
        print("Start Non-Maximum Suppression Tree ...")
    polygon_result_nms, score_result_nms, label_result_nms = nms_result(list_polygons, list_scores, list_labels)
    if verbose:
        print("Exporting result ...")
    export_predict_result_to_file(polygon_result_nms, score_result_nms, label_result_nms, bound_aoi, transform, proj_str, out_format, output_path)

if __name__ == '__main__':
    image_path = r"/home/skm/SKM16/Tmp/TreeCounting/Test_EOF/data_training/mongkos part 3_transparent_mosaic_group1_7.tif"
    model_path = r"/home/skm/SKM16/Tmp/TreeCounting/Test_EOF/Model/tree_mask_rcnn_ducanh20230213T1713/mask_rcnn_tree_mask_rcnn_ducanh_0261.h5"
    output_path = r"/home/skm/SKM16/Tmp/TreeCounting/Test_EOF/data_training/test222.shp"
    predict_main(image_path, model_path, output_path, bound_path=None, out_type="bbox",verbose=1)

