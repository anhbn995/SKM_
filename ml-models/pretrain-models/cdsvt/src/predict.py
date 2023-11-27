import os



import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import *
import json
import glob,os
import pytesseract
import matplotlib.pyplot as plt
import shutil
from shapely.strtree import STRtree
from pathlib import Path

# from processing.mrcnn.config import Config
# from mrcnn.config import Config
from shapely.geometry import Polygon
# from processing.mrcnn.config import Config
# # from mrcnn.config import Config
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from PIL import ImageEnhance
from PIL import Image
import argparse
from PIL import Image
from tqdm import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model

import concurrent.futures
from tqdm import tqdm
import threading

import geopandas as gp


'''
    - input :đường dẫn file pdf 
    - input2 : thu muc out anh crop
    - out: list duong dan den anh da dc crop
'''

# ham lay list anh jpg
'''
    - input 
        - input_flie_path : đường dẫn file pdf
        - out_dir : thư mục lưu anh crop

'''
def get_path_jpg(input_file_path,out_dir):
    list_path_jpg = []
    name = os.path.basename(input_file_path)
    name1 = name.split()
    name2 = name1[0].split('.')[0]
    print(input_file_path)
    images = convert_from_path(input_file_path,dpi=280)
    out = out_dir
    # os.makedirs(out)
    for i, image in enumerate(images):
        out_file = out +f"/{name2}_{i}.PNG"
        # print(out_file)
        image = np.array(image)
        image = get_square_box_from_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.astype(np.uint8)
        
        img = Image.fromarray(image)
        img.save(out_file, "PNG")
        list_path_jpg.append(out_file)
    
    return list_path_jpg

def preprocess(img, factor: int):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(img).enhance(factor)
    if gray.std() < 30:
        enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
    return np.array(enhancer)

def group_h_lines(h_lines, thin_thresh):
    new_h_lines = []
    while len(h_lines) > 0:
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
  
        lines = [line for line in h_lines if thresh[1] -
                 thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
        h_lines = [line for line in h_lines if thresh[1] - thin_thresh >
                   line[0][1] or line[0][1] > thresh[1] + thin_thresh]
    
        x = []
        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
   
        x_min, x_max = min(x) - int(10*thin_thresh), max(x) + int(10*thin_thresh)
    
        new_h_lines.append([x_min, thresh[1], x_max, thresh[1]])
    return new_h_lines

def group_v_lines(v_lines, thin_thresh):
    new_v_lines = []
    while len(v_lines) > 0:
        thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
        lines = [line for line in v_lines if thresh[0] -
                 thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
        v_lines = [line for line in v_lines if thresh[0] - thin_thresh >
                   line[0][0] or line[0][0] > thresh[0] + thin_thresh]
        y = []
        for line in lines:
            y.append(line[0][1])
            y.append(line[0][3])
        y_min, y_max = min(y) - int(4*thin_thresh), max(y) + int(4*thin_thresh)
        new_v_lines.append([thresh[0], y_min, thresh[0], y_max])
    return new_v_lines

def get_bottom_right(right_points, bottom_points, points):

    for right in right_points:
       
        for bottom in bottom_points:
            
            if [right[0], bottom[1]] in points:
             
                return right[0], bottom[1]
       
    return None, None

def euclidian_distance(point1, point2):
    # Calcuates the euclidian distance between the point1 and point2
    #used to calculate the length of the four sides of the square 
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance

def order_corner_points(corners):
    # The points obtained from contours may not be in order because of the skewness  of the image, or
    # because of the camera angle. This function returns a list of corners in the right order 
    sort_corners = [(corner[0][0], corner[0][1]) for corner in corners]
    sort_corners = [list(ele) for ele in sort_corners]
    x, y = [], []

    for i in range(len(sort_corners[:])):
        x.append(sort_corners[i][0])
        y.append(sort_corners[i][1])

    centroid = [sum(x) / len(x), sum(y) / len(y)]

    for _, item in enumerate(sort_corners):
        if item[0] < centroid[0]:
            if item[1] < centroid[1]:
                top_left = item
            else:
                bottom_left = item
        elif item[0] > centroid[0]:
            if item[1] < centroid[1]:
                top_right = item
            else:
                bottom_right = item

    ordered_corners = [top_left, top_right, bottom_right, bottom_left]

    return np.array(ordered_corners, dtype="float32")

def image_preprocessing(image, corners):
    # This function undertakes all the preprocessing of the image and return  
    ordered_corners = order_corner_points(corners)
    # print("ordered corners: ", ordered_corners[0])
    top_left, top_right, bottom_right, bottom_left = ordered_corners
    top_left = top_left+np.array([-35.,-35.])
    top_right = top_right+np.array([35.,-35.])
    bottom_right = bottom_right+np.array([35.,35.])
    bottom_left = bottom_left+np.array([-35.,35.])
    ordered_corners[0] = top_left
    ordered_corners[1] = top_right
    ordered_corners[2] = bottom_right
    ordered_corners[3] = bottom_left
    # print("ordered corners 2:",ordered_corners)
    # Determine the widths and heights  ( Top and bottom ) of the image and find the max of them for transform 

    width1 = euclidian_distance(bottom_right, bottom_left)
    width2 = euclidian_distance(top_right, top_left)

    height1 = euclidian_distance(top_right, bottom_right)
    height2 = euclidian_distance(top_left, bottom_left)

    width = max(int(width1), int(width2)) 
    height = max(int(height1), int(height2)) 
    

    # To find the matrix for warp perspective function we need dimensions and matrix parameters
    dimensions = np.array([[0, 0], [width, 0], [width, height],
                           [0, height]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))

    #Now, chances are, you may want to return your image into a specific size. If not, you may ignore the following line
    # transformed_image = cv2.resize(transformed_image, (252, 252), interpolation=cv2.INTER_AREA)

    return transformed_image ,ordered_corners


def get_bottom_right(right_points, bottom_points, points):
    for right in right_points:
        for bottom in bottom_points:
            if [right[0], bottom[1]] in points:
                return right[0], bottom[1]
    return None, None

def get_square_box_from_image(image):
    # This function returns the top-down view of the puzzle in grayscale.
    # 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    corners = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = corners[0] if len(corners) == 2 else corners[1]
    corners = sorted(corners, key=cv2.contourArea, reverse=True)
    for corner in corners:
        length = cv2.arcLength(corner, True)
        approx = cv2.approxPolyDP(corner, 0.015 * length, True)
        # print(approx)
      

        puzzle_image,box= image_preprocessing(image, approx)
        break
    return puzzle_image
###################################################################33


"""
    
    step 1: u2net : get tọa do ô chứa msssv, điểm 
        - input : list d dan anh
        - input_2 : thu muc out json
        - output : json : tuong ung vs moi anh

"""

# warnings.filterwarnings("ignore")
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# set_session(tf.compat.v1.Session(config=config))
num_bands = 3
size = 256
def predict_edge(model, path_image, path_predict, size=256):

    
    img = Image.open(path_image)
    width, height = img.size
    print(img.size)
    input_size = size
    stride_size = input_size - input_size // 4
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

    num_bands = 3 
    image_data = np.array(img)

    with Image.new('L', (width, height)) as result_img:
        result_data = np.array(result_img)

        read_lock = threading.Lock()
        write_lock = threading.Lock()

        def process(coordinates):
            x_off, y_off, x_count, y_count, start_x, start_y = coordinates
            read_wd = (x_off, y_off, x_off + x_count, y_off + y_count)
            with read_lock:
                values = image_data[y_off:y_off + y_count, x_off:x_off + x_count, :num_bands]

            if image_data.dtype == 'uint8':
                image_detect = values.astype(int)
        

            img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
            mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8), ((padding, padding), (padding, padding)))
            shape = (stride_size, stride_size)

            if y_count < input_size or x_count < input_size:
                img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                mask = np.zeros((input_size, input_size), dtype=np.uint8)

                if start_x == 0 and start_y == 0:
                    img_temp[(input_size - y_count):input_size, (input_size - x_count):input_size] = image_detect
                    mask[(input_size - y_count):input_size - padding, (input_size - x_count):input_size - padding] = 1
                    shape = (y_count - padding, x_count - padding)
                elif start_x == 0:
                    img_temp[0:y_count, (input_size - x_count):input_size] = image_detect
                    if y_count == input_size:
                        mask[padding:y_count - padding, (input_size - x_count):input_size - padding] = 1
                        shape = (y_count - 2 * padding, x_count - padding)
                    else:
                        mask[padding:y_count, (input_size - x_count):input_size - padding] = 1
                        shape = (y_count - padding, x_count - padding)
                elif start_y == 0:
                    img_temp[(input_size - y_count):input_size, 0:x_count] = image_detect
                    if x_count == input_size:
                        mask[(input_size - y_count):input_size - padding, padding:x_count - padding] = 1
                        shape = (y_count - padding, x_count - 2 * padding)
                    else:
                        mask[(input_size - y_count):input_size - padding, padding:x_count] = 1
                        shape = (y_count - padding, x_count - padding)
                else:
                    img_temp[0:y_count, 0:x_count] = image_detect
                    mask[padding:y_count, padding:x_count] = 1
                    shape = (y_count - padding, x_count - padding)

                image_detect = img_temp

            mask = (mask != 0)

            if np.count_nonzero(image_detect) > 0:
                if len(np.unique(image_detect)) <= 2:
                    pass
                else:
                    y_pred = model.predict(image_detect[np.newaxis, ...] / 255.)

                    # Chuyển đổi thành numpy array
                    y_pred = np.array(y_pred)

                    # Áp dreeshold
                    y_pred = (y_pred[0, 0, ..., 0] > 0.4).astype(np.uint8)

                    y = y_pred[mask].reshape(shape)
         

                    with write_lock:
                        result_data[start_y:start_y + shape[0], start_x:start_x + shape[1]] = y

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))

        result_img.putdata(result_data.flatten())
        result_img.save(path_predict)

        return path_predict , path_image

def sort_list_to_row(cell):
    thresh = sorted(cell, key=lambda x: [x[1],x[0]])
    # thresh = sorted(cell, key=lambda x: x[0])
    epsilon = 45
    min_x = thresh[0][1]
    list_row = []
    list_tmp = []
    for i in range(len(thresh)):
        box_x = thresh[i]
        if abs(box_x[1] - min_x) < epsilon:
            list_tmp.append(box_x)
        else:
            list_row.append(list_tmp)
            list_tmp = []
            list_tmp.append(box_x)
            min_x = box_x[1]
        if i ==len(thresh)-1:
            list_row.append(list_tmp)
    return list_row
def sort_list_to_col(list_row):
    list_result = []
    for row in list_row:
        thresh = sorted(row, key=lambda x: x[0])
        list_result.append(thresh)

    return list_result

def get_coordinates_all(input_path,input_mask):
    image = Image.open(input_path)
    image = np.array(image)
    
    # print(image.shape)
    re = np.zeros_like(image)
    re = image.astype(np.uint8)
    bgr_image = cv2.convertScaleAbs(re)

    mask = np.array(Image.open(input_mask))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # print(mask.shape)
    w,h = mask.shape[0], mask.shape[1]

    masked_image = cv2.bitwise_and( image,  image, mask=mask)
    masked_image = masked_image.astype(np.uint8)
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(len(contours))

    cell = []
    toa_do_x = []
    width_ = []
    for contour in contours:

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            if h<30 or h > 300:
                continue
    
        # cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            else:

                cell.append([x,y,w,h])
                width_.append(w)
                toa_do_x.append(x)
    for box in cell:
        x,y,w,h = box
        cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    # import matplotlib.pyplot as plt
    # imgplot = plt.imshow(bgr_image)
    # plt.show()
    list_sort_raw = sort_list_to_row(cell)
    list_sort_col = sort_list_to_col(list_sort_raw)
    return list_sort_col,input_path

def get_infomation(list_sort_col,col_select,input_path,dir_out_json):
    #col select [0:6]

    col_mssv = col_select[1]
    col_diem = col_select[2]
    col_stt = col_select[0]
    ID = 1
    dict_box = {
        ID :{

            'MSSV' :[],
            'DIEM': [],
            'STT' : [],
        }

        }
    for i in range(len(list_sort_col)):
        # print(i)
        # print(list_sort_col[i])
        # print(len(list_sort_col[i]) )
        if i == 0:
            continue
        if len(list_sort_col[i])-1 < col_diem:
            ID = i

            if ID in dict_box:
                dict_box[ID]['MSSV'].append(None)
                dict_box[ID]['DIEM'].append(None)
                dict_box[ID]['STT'].append(None)
            else:
                # Khởi tạo ID trong dict_box nếu chưa tồn tại
                dict_box[ID] = {
                    'MSSV': [None],
                    'DIEM': [None],
                    'STT': [None]
                }
          
        
        
        else:
            ID = i

            if ID in dict_box:
                dict_box[ID]['MSSV'].append(list_sort_col[i][col_mssv])
                dict_box[ID]['DIEM'].append(list_sort_col[i][col_diem])
                dict_box[ID]['STT'].append(list_sort_col[i][col_stt])
            else:
                # Khởi tạo ID trong dict_box nếu chưa tồn tại
                dict_box[ID] = {
                    'MSSV': [list_sort_col[i][col_mssv]],
                    'DIEM': [list_sort_col[i][col_diem]],
                    'STT': [list_sort_col[i][col_stt]]


    }
    
    # print(dict_box)
    outjson = os.path.join(dir_out_json,os.path.basename(input_path).replace('.PNG','.json'))
    

    with open(outjson, "w") as json_file:
        json.dump(dict_box, json_file)
    return outjson
##################################################################################


'''
   step2 : maskcnn:
        - input : list d dan anh crop
        - input 2: out json
        - out_put : json: toa do box tuong ung vs moi anh


'''

MODEL_SIZE = 512
MODEL_DIR=""
NUM_BAND=3
overlap_size = int(MODEL_SIZE*3/4)




def read_image_by_window(dataset_image, x_off, y_off, x_count, y_count, start_x, start_y ):
    """ 
    This function to read image window by coordinates
    """
    input_size = MODEL_SIZE
  
    dataset_image = dataset_image.transpose(1,2,0)
    # print('gggggggggg',dataset_image.shape)
    image = Image.fromarray(dataset_image)

    # Crop và chuyển đổi kích thước
    cropped_image = image.crop((x_off, y_off, x_off + x_count, y_off + y_count)).resize((x_count, y_count))

    # Chuyển đổi thành mảng numpy
    image_array = np.array(cropped_image)
    image_array = image_array.transpose(2,0,1)
    # print('kkkkkkkkkkkkkkkkkk',image_array.shape)
    # Swap axes
    image_detect = image_array.swapaxes(0, 1).swapaxes(1, 2)

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


def predict_big_image(dataset_image, model_path,config, out_type, verbose):
    w,h = dataset_image.shape[2],dataset_image.shape[1]
    
    input_size = MODEL_SIZE
    stride_size = overlap_size
    # padding size for each stride window
    padding = int((input_size - stride_size) / 2)
    # call model
    # model = modellib.MaskRCNN(
    #     mode="inference", model_dir=MODEL_DIR, config=config)
    # load weight
    if verbose:
        print("Loaded model")
    # model.load_weights(model_path, by_name=True)
    # model = model_path
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
            # if image not no data and intersect with AIO care then push to predict
            if np.count_nonzero(image_detect) > 0:
                predictions = model_path.detect([image_detect] * config.BATCH_SIZE, verbose=0)
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
                    # print(con_rs)
                    # print(type(con_rs))
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
    # Read image
    image = Image.open(image_path)
    w, h = image.size
    
    bound_image = ((0, 0), (w, 0), (w, h), (0, h), (0, 0))
    bound_aoi = Polygon(bound_image)
        
    return bound_aoi

def export_predict_result_to_file(polygon_result_all, bound_aoi,  out_dir_json,fp_name_output_path):

    list_geo_polygon = [Polygon(polygon) for polygon in polygon_result_all]
   
    tree_polygon = [geom for geom in list_geo_polygon]

    tree_point = [geom.centroid for geom in list_geo_polygon]
    # t1 = time.time() - t0
    strtree_point = STRtree(tree_point)
    
    index_by_id = dict((id(pt), i) for i, pt in enumerate(tree_point))
   
    list_point = strtree_point.query(bound_aoi)
    # list_point = [p for p in tree_point if p.intersects(bound_aoi)]

    # print(list_point)
    list_point_inside = [x for x in list_point if bound_aoi.contains(x)]
    # # list_point_inside = [x for x in list_point if bound_aoi.contains(x) and isinstance(x,Polygon) == True]

    index_point = [index_by_id[id(pt)] for pt in list_point_inside]
    tree_polygon_rs = [tree_polygon[index] for index in index_point]
   
    list_coordinates = [list(x.exterior.coords) for x in tree_polygon_rs ]
    # print(list_coordinates[0:5])
    thresh = sorted(list_coordinates, key=lambda x: [x[0][0],x[0][1]])

    dict_obj = {
        'BOX_OB' :{}
        
    }
    id_box = 0
    for x in thresh:
        dict_obj['BOX_OB'][id_box] = x
        id_box += 1
    outjson = os.path.join(out_dir_json , fp_name_output_path.replace('.PNG','.json'))
    with open(outjson, "w") as json_file:
        json.dump(dict_obj, json_file)           

   
    return outjson

def nms_result(list_polygons, list_scores, list_labels, iou_threshold=0.2):
    list_shapely_polygons = [Polygon(polygon) for polygon in list_polygons]
    list_bound = [np.array(polygon.bounds) for polygon in list_shapely_polygons]
    indexes = tf.image.non_max_suppression(np.array(list_bound), np.array(list_scores), len(list_scores),iou_threshold=iou_threshold)
    result_polygons = [list_polygons[idx] for idx in indexes]
    result_scores = [list_scores[idx] for idx in indexes]
    result_labels = [list_labels[idx] for idx in indexes]
    return result_polygons,result_scores,result_labels

def predict_location_number(image_path, model_path,out_dir_json ,config, bound_path=None,out_type="bbox",verbose=1 ):
 
    out_dir_json = out_dir_json
    fp_name_output_path = os.path.basename(image_path)
    # Open data set for predict step ( Read by Window)  
    dataset_image = np.array(Image.open(image_path))
    # print(dataset_image.shape)
    dataset_image = dataset_image.transpose(2,0,1)
    # print(dataset_image.shape)
    w, h = dataset_image.shape[2],dataset_image.shape[1]

    image_name = os.path.basename(image_path)
    image_id = os.path.splitext(image_name)[0]
    # Get AOI by image ID
    bound_aoi = get_bound(image_path, bound_path, image_id)
    # print("z"*100,bound_aoi)
    # Config image model and stride size
    input_size = MODEL_SIZE


    list_contours, list_scores, list_labels = predict_big_image(dataset_image, model_path,config, out_type, verbose)
            
    list_polygons = [list_contours[i].reshape(-1, 2) for i in range(len(list_contours))]

    if verbose:
        print("Start Non-Maximum Suppression Tree ...")
    polygon_result_nms, score_result_nms, label_result_nms = nms_result(list_polygons, list_scores, list_labels)
    if verbose:
        print("Exporting result ...")
    path_json = export_predict_result_to_file(polygon_result_nms, bound_aoi, out_dir_json,fp_name_output_path)
    return path_json

###############################################
"""
    step3 : classifiy
        - input1: out json step2
        - input 2: list anh step1
        - out put: list box va class tuong ung json
    
"""

def model_classify(model_path):

    net = VGG16(weights=None, include_top=False, input_shape=(128,128,3))
    n_classes = 12
    top_model = net.layers[-5].output
    top_model = Flatten(name="flatten")(top_model)
    # top_model = Dense(4096, activation='relu',name="fc1")(top_model)
    top_model = Dense(1024, activation='relu',name="fc2")(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
        
        # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=net.input, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.01), metrics=['accuracy'])
    
    model = load_model(model_path)

    return model



def classify(input_image,model):
    """
        - box [x,y,w,h]
    """
    img_crop1 = np.zeros((128,128,3),dtype = np.uint8)
    x = (img_crop1.shape[1] - input_image.shape[1])//2
    y = (img_crop1.shape[0] - input_image.shape[0])//2
    # print(x,y)
    img_crop1[y:y+input_image.shape[0],x:x+input_image.shape[1],0:3] = input_image[:,:,0:3]

    
    image = cv2.cvtColor(img_crop1, cv2.COLOR_BGR2RGB)
    image = image/255.0
    pre = model.predict(image[np.newaxis,:,:,:])
    pre =  np.argmax(pre, axis=-1)

    return pre 
def get_xy_w_h(data):
    top_left = data[0]
    top_right = data[1]
    bottom_right = data[2]
    # bottom_left = data[0][3]
    w = abs(int(top_right[0] - top_left[0]))
    h = abs(int(bottom_right[1]-top_right[1]))
    x = int(top_left[0])
    y = int(top_left[1])
    # center = (x+ w//2,y+ h//2)
    return x,y,w,h
def run_classify(json_path,image_path,model,out_dir_class):
    img = cv2.imread(image_path)
    name = os.path.basename(image_path)
    name1 = name.split()
    name2 = name1[0].split('.')[0]
    with open(json_path, 'r') as file:
        box_number = json.load(file)
    id_x = 0
    dict_ ={
        id_x :{
            'TOADO':[],
            'DIEM' :[]
        }
        }

    for k in range(len(box_number['BOX_OB'])):
        x1,y1,w1,h1 = get_xy_w_h(box_number['BOX_OB'][str(k)])
              
        img_crop = img[y1:y1+h1,x1:x1+w1]
        id_x = k
        if img_crop.shape[0] >128 or img_crop.shape[1] > 128:
            dict_[id_x] = {
                    'TOADO':[[x1,y1,w1,h1]],
                    'DIEM':[['NULL']]
                    }
            continue

        pre = classify(img_crop,model)
        
        # print(type(id_x))
        if id_x in dict_:
            dict_[id_x]['TOADO'].append([x1,y1,w1,h1])
            dict_[id_x]['DIEM'].append([int(pre[0])])
        else:
            dict_[id_x] = {
                    'TOADO':[[x1,y1,w1,h1]],
                    'DIEM':[[int(pre[0])]]
                    }
    # print(out_dir_class
    # outjson = os.path.join(out_dir_class ,f'{name2}.json')
    outjson = out_dir_class + f'{name2}.json'
    # print(outjson)
    with open(outjson, "w") as json_file:
        json.dump(dict_, json_file)   
    return outjson
#############################
  
def get_xy_w_h(data):
    top_left = data[0]
    top_right = data[1]
    bottom_right = data[2]
    w = abs(int(top_right[0] - top_left[0]))
    h = abs(int(bottom_right[1]-top_right[1]))
    x = int(top_left[0])
    y = int(top_left[1])
    return x,y,w,h

def check_point_in_rectangle(x, y, w, h, x1, y1):
    # Kiểm tra xem điểm A có nằm trong đa giác hay không
    if x <= x1 <= x + w and y <= y1 <= y + h:
        return True
    else:
        return False
def convert_xywh_to_polygon(data):
    x,y,w,h = data
    rectangle = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    return rectangle   

def get_diem(list_coordinates,box_number):
    unknow = None
    diem_ = []
   
    if len(list_coordinates) ==0:
        return unknow
    if len(list_coordinates) ==1:
        for k in range(len(box_number)):
        
            if list_coordinates[0][0] in box_number[str(k)]['TOADO'][0] and list_coordinates[0][1] in box_number[str(k)]['TOADO'][0]:
                diem = box_number[str(k)]['DIEM'][0][0]
              
                if diem in [0,1,2,3,4,5,6,7,8,9]:
                    return str(diem)
                else:
                    return unknow
  
    if len(list_coordinates) == 2:
     
        for k in range(len(box_number)):
         
            if (list_coordinates[0][0] in box_number[str(k)]['TOADO'][0] and list_coordinates[0][1] in box_number[str(k)]['TOADO'][0]):
                diem = box_number[str(k)]['DIEM'][0][0]
                diem_.append(diem)
            if (list_coordinates[1][0] in box_number[str(k)]['TOADO'][0] and list_coordinates[1][1] in box_number[str(k)]['TOADO'][0]):
                diem = box_number[str(k)]['DIEM'][0][0]
              
                diem_.append(diem)
  
        if diem_[0] in [0,1,2,3,4,5,6,7,8,9] and diem_[1] in [0,1,2,3,4,5,6,7,8,9]:

            diem_l = f'{diem_[0]}.{diem_[1]}'
            
            return diem_l
        else:
            return unknow
    if len(list_coordinates) == 3:
        for k in range(len(box_number)):
            if list_coordinates[0][0] in box_number[str(k)]['TOADO'][0] and list_coordinates[0][1] in box_number[str(k)]['TOADO'][0]:
                diem = box_number[str(k)]['DIEM'][0][0]
                diem_.append(diem)
            if list_coordinates[1][0] in box_number[str(k)]['TOADO'][0] and list_coordinates[1][1] in box_number[str(k)]['TOADO'][0]:
                diem = box_number[str(k)]['DIEM'][0][0]
                diem_.append(diem)

        if diem_[0] ==0 and diem_[1] == 0:
            diem = '0'
            return diem
        if diem_[0] == 0 and diem_[1] == 1:
            diem = '1'
            return diem
        if diem_[0] == 1 and diem_[1] ==0:
            diem = '10'
            return diem
        else:
            return unknow
    else:
        return unknow

def remove_folder(path_folder):
    relative_path = path_folder
    absolute_path = Path(relative_path).resolve()
    shutil.rmtree(absolute_path)

def convert_x1y1x2y2_to_polygon(data):
    x,y,w,h = get_xy_w_h(data)
    return x,y,w,h 

def mk_dir(path_image):
    if not os.path.exists(path_image):
        
        os.makedirs(path_image)

    return path_image

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def convert_xywh_to_polygon(data):
    x,y,w,h = data
    rectangle = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    return rectangle

def predict_main(file_path,out_path,model_detect_edge,model_detect_location_number,model_class,config,job_id):
    input_file_pdf = file_path

    size = 256

    print(os.path.dirname(input_file_pdf))

    out_dir_crop = os.path.join(os.path.dirname(input_file_pdf) + f'/STEP1_{job_id}','CROP')

    out_dir_edge = os.path.join(os.path.dirname(input_file_pdf) + f'/STEP2_{job_id}','edge/')
    out_dir_json_edge = os.path.join(os.path.dirname(input_file_pdf) + f'/STEP2_{job_id}' + '/edge','json_edge/')
    out_dir_json_detect_obj = os.path.join(os.path.dirname(input_file_pdf) + f'/STEP2_{job_id}','obj/')
    out_dir_json_class = os.path.join(os.path.dirname(input_file_pdf) + f'/STEP2_{job_id}','class/')
   
    out_dir_crop = mk_dir(out_dir_crop)
    out_dir_edge = mk_dir(out_dir_edge)
    out_dir_json_edge = mk_dir(out_dir_json_edge)
    out_dir_json_detect_obj = mk_dir(out_dir_json_detect_obj)
    out_dir_json_class = mk_dir(out_dir_json_class)

    root_dir_crop = os.path.join(os.path.dirname(input_file_pdf) , f'STEP1_{job_id}')
    root_dir_edge =  os.path.join(os.path.dirname(input_file_pdf) ,f'STEP2_{job_id}')

    # final_dir = os.path.join(os.path.dirname(input_file_pdf) ,'FINAL_FOLDER') 

    final_dir = os.path.join(out_path ,'FINAL_FOLDER') 
    final_dir = mk_dir(final_dir)

    out_result = os.path.join(os.path.dirname(input_file_pdf) ,f'OUT_EXCEL_{job_id}/')
    out_result = mk_dir(out_result)
    

    list_path_crop_predict = get_path_jpg(input_file_pdf,out_dir_crop)

    col_select = [0,1,4]
    list_mssv =[]
    list_diem = []
    list_stt = []
    list_json = []
    page = 1
    dict_all = {
        page : None
    }
    id_page =1
    for fp_img in tqdm(list_path_crop_predict):
        page = id_page
        output_path = os.path.join(out_dir_edge,os.path.basename(fp_img))
        path_mask,path_image = predict_edge(model_detect_edge, fp_img, output_path, size=size)    
        list_to_col,input_path = get_coordinates_all(path_image,path_mask)
        out_json = get_infomation(list_to_col,col_select,input_path,out_dir_json_edge)

        path_json_detect_obj = predict_location_number(fp_img, model_detect_location_number,out_dir_json_detect_obj,config, bound_path=None, out_type="bbox",verbose=1)
        path_json_detect_class = run_classify(path_json_detect_obj,path_image,model_class,out_dir_json_class)
    
        img = cv2.imread(path_image)
        
        final = final_dir + f'/Final_json_{job_id}.json'
        
        with open(out_json, 'r') as file:
            box_col_row = json.load(file)
        with open(path_json_detect_class, 'r') as file:
            box_number = json.load(file)
 
        if len(box_number) <= 1:
            if page in dict_all:

                dict_all[page] = None
            else:
                print(page)
                dict_all[page] = None
            id_page +=1
       
            continue
                
        for j in range(len(box_col_row)):
            list_box = []

            j+=1
            
            if (box_col_row[str(j)]['DIEM'][0] == None or box_col_row[str(j)]['MSSV'][0] == None or box_col_row[str(j)]['STT'][0]) == True:
                list_box.append(None)
                list_mssv.append(None)
                list_stt.append(None)
                continue
         
            mssv_x,mssv_y,mssv_w,mssv_h = (box_col_row[str(j)]['MSSV'])[0]
        

            img_crop_mssv = img[mssv_y:mssv_y+mssv_h,mssv_x  : mssv_x+mssv_w]
            img_crop_mssv = get_grayscale(img_crop_mssv)
            img_crop_mssv = thresholding(img_crop_mssv)
            mssv = pytesseract.image_to_string(img_crop_mssv, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            numbers = ''.join(filter(str.isdigit, mssv))
            list_mssv.append(numbers)

            stt_x,stt_y,stt_w,stt_h = (box_col_row[str(j)]['STT'])[0]
            img_crop_stt = img[stt_y:stt_y+stt_h, stt_x+30 : stt_x+stt_w]
            img_crop_stt = get_grayscale(img_crop_stt)
            img_crop_stt = thresholding(img_crop_stt)
            stt =  pytesseract.image_to_string(img_crop_stt, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            numbers_stt = ''.join(filter(str.isdigit, stt))

            list_stt.append(numbers_stt)
            diem = []
            list_box = []

            bound_aoi = convert_xywh_to_polygon((box_col_row[str(j)]['DIEM'])[0])
            list_geo_polygon = [ convert_xywh_to_polygon(box_number[str(k)]['TOADO'][0]) for  k in range(len(box_number))]
            tree_polygon = [geom for geom in list_geo_polygon]
            tree_point = [geom.centroid for geom in list_geo_polygon]
            strtree_point = STRtree(tree_point)
            index_by_id = dict(((pt.x,pt.y), z) for z, pt in enumerate(tree_point))
            list_point = strtree_point.query(bound_aoi)
            list_point_inside = [x for x in list_point if bound_aoi.contains(x)]
            index_point = [index_by_id[(pt.x,pt.y)] for pt in list_point_inside]
            tree_polygon_rs = [tree_polygon[index] for index in index_point]
            list_coordinates = [list(x.exterior.coords) for x in tree_polygon_rs ]
            list_coordinates = [convert_x1y1x2y2_to_polygon(x) for x in list_coordinates]
            list_coordinates = sorted(list_coordinates, key=lambda x: x[0])
            diem = get_diem(list_coordinates,box_number)
            list_diem.append(diem)

        data1 = list(zip(list_stt, list_mssv,list_diem))


        df = pd.DataFrame(data1, columns=['STT','MSSV', 'DIEM'])
        df.set_index(np.arange(1, len(df) + 1), inplace=True)

        out_json = os.path.join(out_result,os.path.basename(path_image).replace('.PNG','.json'))
        print(f'.......GHI FILE {os.path.basename(out_json)}........')
        # t2 = time.time() - t1 - t0-t4
        json_file = df.to_json(out_json,orient='records')
        # t3 = time.time() - t2 -t1-t0 -t4
        list_mssv = []
        list_diem = []
        list_stt  = []
        list_json.append([page,out_json])
        id_page+=1
   
    for page,fp_js in list_json:
        # print(fp_js)
      
        with open(fp_js,'r') as f:
            data = json.load(f)
        if page in dict_all:
            print(page)
            # dict_all[page].append(data)
            dict_all[page] = data
        else:
            print(page)
            dict_all[page] = data              
      
    with open(final, "w") as json_file:
        json.dump(dict_all, json_file)



  
    remove_folder(out_result)
    remove_folder(out_dir_json_edge)
    remove_folder(out_dir_json_class)

    remove_folder(out_dir_json_detect_obj)
    remove_folder(out_dir_edge)
    remove_folder(root_dir_crop)
    
    remove_folder(root_dir_edge)

    return final
      




    
