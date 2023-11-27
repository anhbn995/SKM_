import os
from tqdm import *
import geopandas as gpd
import gdal
import numpy as np
import cv2
import gdal
import osr
from utils.convert_datatype import list_contour_to_list_polygon
from utils.export_data import exportResult2 as exportResult2
import mrcnn.model as modellib
from mrcnn.config import Config
import osr
from glob import glob
from params import TMP_PATH

MODEL_DIR = TMP_PATH

image_size = 512
from utils.valid_vector import remove_invalid_gdf

class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    IMAGE_MAX_DIM = image_size + 64
    IMAGE_MIN_DIM = image_size + 64
    DETECTION_MAX_INSTANCES = 200
    MAX_GT_INSTANCES = 200

    MASK_SHAPE = [28, 28]
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    NAME = "crowdai-mapping-challenge"

    """/home/skm/SKM/WORK/Catalyst_Treecouting"""
    DETECTION_NMS_THRESHOLD = 0.4
    DETECTION_MIN_CONFIDENCE = 0.6
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    IMAGE_CHANNEL_COUNT = 3


def resize(image_path, path_create):
    image_id = os.path.basename(image_path)
    output = os.path.join(path_create, image_id)
    resolution_origin = 0.05
    resolution_destination = 0.3
    size_destination = resolution_origin / resolution_destination * 100
    options_list = [
        f'-outsize {size_destination}% {size_destination}%',
        '-of GTiff',
        '-r cubic',
        '-ot Byte'
    ]
    options_string = " ".join(options_list)

    gdal.Translate(output,
                   image_path,
                   options=options_string)
    return output


def main_predict(fp_in_img_path, model_path, dir_tmp, out_path):
    img_path = os.path.join(dir_tmp, os.path.basename(fp_in_img_path))
    resize(fp_in_img_path, img_path)
    if not os.path.exists(dir_tmp):
        os.makedirs(dir_tmp)
    list_size_crop = [400, 412]
    for size_crop in list_size_crop:
        out_path_tmp = os.path.join(dir_tmp, str(size_crop) + '.shp')
        predict_large_img(img_path, model_path, size_crop, out_path_tmp)
    remove_intersection(dir_tmp, out_path)


def check_intersec(contour, input_size, overlapsize):
    padding = int((input_size - overlapsize) / 2)
    cnt1 = np.array(
        [[padding, padding], [padding, input_size - padding], [input_size - padding, input_size - padding],
         [input_size - padding, padding]])
    contour1 = np.array(cnt1.reshape(-1, 1, 2), dtype=np.int32)
    img1 = np.zeros((input_size, input_size)).astype(np.uint8)
    img1 = cv2.fillConvexPoly(img1, contour1, 255)
    img = np.zeros((input_size, input_size)).astype(np.uint8)
    img = cv2.fillConvexPoly(img, contour, 255)
    img_result = cv2.bitwise_and(img1, img)
    contours_rs, hierarchy = cv2.findContours(img_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        if cv2.contourArea(contours_rs[0]) / float(cv2.contourArea(contour)) > 0.50:
            return True
        else:
            return False
    except Exception:
        return False


def predict_large_img(image_path, model_path, crop_size, output_path, option=None):
    dataset_image = gdal.Open(image_path)
    w, h = dataset_image.RasterXSize, dataset_image.RasterYSize
    # model_path = PRETRAINED_MODEL_PATH
    num_band = dataset_image.RasterCount
    num_band = 3
    input_size = 512
    # crop_size = 400#image_size-100
    if h <= input_size or w <= input_size:
        image = dataset_image.ReadAsArray()[0:num_band].swapaxes(0, 1).swapaxes(1, 2).astype(np.uint8)
        image_res = 0.3
        config = InferenceConfig()
        config.IMAGE_MAX_DIM = (round(max(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64
        config.IMAGE_MIN_DIM = (round(min(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64
        # config = InferenceConfig()
        config.display()

        model = modellib.MaskRCNN(
            mode="inference", model_dir=MODEL_DIR, config=config)

        # model_path = PRETRAINED_MODEL_PATH

        # or if you want to use the latest trained model, you can use :
        # model_path = model.find_last()[1
        print("inside predict api")
        print(model_path)
        model.load_weights(model_path, by_name=True, exclude='conv1')

        # # Run Prediction on a single Image (and visualize results)

        # In[37]:
        class_names = ['BG', 'building']
        predictions = model.detect([image] * config.BATCH_SIZE, verbose=1)
        p = predictions[0]
        boxes = p['rois']
        N = boxes.shape[0]
        list_contours = []
        for i in range(N):
            if not np.any(boxes[i]):
                continue
            y1, x1, y2, x2 = boxes[i]
            contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
            contour = contour.reshape(-1, 1, 2)
            try:
                if cv2.contourArea(contour) > 100:
                    list_contours.append(contour)
            except Exception:
                pass
        return_contour = list_contours
        predictions = None
        p = None
        model = None
    else:
        # base = dataset_base.ReadAsArray().swapaxes(0,1).swapaxes(1,2)
        # image = dataset_image.ReadAsArray().swapaxes(0,1).swapaxes(1,2)
        # padded_org_im = np.concatenate((base,image),axis=-1)

        config = InferenceConfig()
        config.display()

        model = modellib.MaskRCNN(
            mode="inference", model_dir=MODEL_DIR, config=config)

        # model_path = PRETRAINED_MODEL_PATH

        # or if you want to use the latest trained model, you can use :
        # model_path = model.find_last()[1]

        model.load_weights(model_path, by_name=True)
        class_names = ['BG', 'change']
        return_contour = []
        padding = int((input_size - crop_size) / 2)
        new_w = w + 2 * padding
        new_h = h + 2 * padding
        cut_w = list(range(padding, new_w - padding, crop_size))
        cut_h = list(range(padding, new_h - padding, crop_size))
        list_hight = []
        list_weight = []
        print(w, h)
        for i in cut_h:
            list_hight.append(i)

        for i in cut_w:
            list_weight.append(i)

        with tqdm(total=len(list_hight) * len(list_weight)) as pbar:
            for i in range(len(list_hight)):
                start_y = list_hight[i]
                #            hight_tiles_down = list_hight[i+1]
                for j in range(len(list_weight)):
                    start_x = list_weight[j]
                    startx = start_x - padding
                    endx = min(start_x + crop_size + padding, new_w - padding)
                    starty = start_y - padding
                    endy = min(start_y + crop_size + padding, new_h - padding)
                    if startx == 0:
                        xoff = startx
                    else:
                        xoff = startx - padding
                    if starty == 0:
                        yoff = starty
                    else:
                        yoff = starty - padding
                    xcount = endx - padding - xoff
                    ycount = endy - padding - yoff
                    # inage_detect = padded_org_im[starty:endy,startx:endx]
                    # print(xoff, yoff, xcount, ycount)
                    # base =  dataset_base.ReadAsArray(xoff, yoff, xcount, ycount).swapaxes(0,1).swapaxes(1,2)

                    if num_band == 1:
                        inage_detect = dataset_image.ReadAsArray(xoff, yoff, xcount, ycount)
                        inage_detect = inage_detect[..., np.newaxis]
                        # print(inage_detect.shape, xoff, yoff, xcount, ycount)
                    else:
                        inage_detect = dataset_image.ReadAsArray(xoff, yoff, xcount, ycount)[0:num_band].swapaxes(0,
                                                                                                                  1).swapaxes(
                            1, 2)
                        # print(inage_detect.shape, xoff, yoff, xcount, ycount)
                    # inage_detect = np.concatenate((base,image),axis=-1)
                    if inage_detect.shape[0] < input_size or inage_detect.shape[1] < input_size:
                        img_temp = np.zeros((input_size, input_size, inage_detect.shape[2]))
                        if (startx == 0 and starty == 0):
                            img_temp[(input_size - inage_detect.shape[0]):input_size,
                            (input_size - inage_detect.shape[1]):input_size] = inage_detect
                        elif startx == 0:
                            img_temp[0:inage_detect.shape[0],
                            (input_size - inage_detect.shape[1]):input_size] = inage_detect
                        elif starty == 0:
                            img_temp[(input_size - inage_detect.shape[0]):input_size,
                            0:inage_detect.shape[1]] = inage_detect
                        else:
                            img_temp[0:inage_detect.shape[0], 0:inage_detect.shape[1]] = inage_detect
                        # inage_detect = img_temp
                        # img_temp[0:inage_detect.shape[0],0:inage_detect.shape[1]]=inage_detect
                        inage_detect = img_temp
                    if np.count_nonzero(inage_detect) > 0:
                        predictions = model.detect([inage_detect] * config.BATCH_SIZE, verbose=0)
                        p = predictions[0]
                        # print(p['masks'].shape)

                        ##########################################################
                        # boxes = p['rois']
                        # N = boxes.shape[0]
                        # list_temp=[]
                        # for i in range(N):
                        #     if not np.any(boxes[i]):
                        #         continue
                        #     y1, x1, y2, x2 = boxes[i]
                        #     contour = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2],[x1,y1]])
                        #     contour = contour.reshape(-1, 1, 2)
                        #     try:
                        #         if cv2.contourArea(contour) > 10:
                        #             if (contour.max()<(input_size-padding)) and (contour.min()>padding):
                        #             # print(1)
                        #                 list_temp.append(contour)
                        #             elif (contour.max()<(input_size-5)) and (contour.min()>5) and check_intersec(contour, input_size, crop_size):
                        #                 list_temp.append(contour)
                        #     except Exception:
                        #         pass
                        ##########################################################
                        # print(p['masks'].shape)
                        list_temp = []
                        for i in range(p['masks'].shape[2]):
                            mask = p['masks'][:, :, i].astype(np.uint8)
                            # boxes = p['rois']
                            # print(boxes,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                            # from matplotlib import pyplot as plt
                            # plt.imshow(mask)
                            # plt.show()
                            contours, hierarchy = cv2.findContours(
                                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            try:
                                if cv2.contourArea(contours[0]) > 10:
                                    if (contours[0].max() < (input_size - padding)) and (
                                            contours[0].min() > padding):
                                        # print(1)
                                        list_temp.append(contours[0])
                                    elif (contours[0].max() < (input_size - 5)) and (
                                            contours[0].min() > 5) and check_intersec(contours[0], input_size,
                                                                                      crop_size):
                                        list_temp.append(contours[0])
                            except Exception:
                                pass

                        #########################################################
                        temp_contour = []
                        for contour in list_temp:
                            anh = contour.reshape(-1, 2)
                            anh2 = anh + np.array([startx - padding, starty - padding])
                            con_rs = anh2.reshape(-1, 1, 2)
                            temp_contour.append(con_rs)
                        # print(temp_contour)
                        return_contour.extend(temp_contour)
                    pbar.update()
            predictions = None
            p = None
            model = None
            list_contours = return_contour
    # driverName = "GeoJson"
    driverName = "ESRI Shapefile"
    outputFileName = output_path
    geotransform = dataset_image.GetGeoTransform()
    projection = osr.SpatialReference(dataset_image.GetProjectionRef())
    print(projection)
    polygons_result = list_contour_to_list_polygon(list_contours)
    exportResult2(polygons_result, geotransform, projection, outputFileName, driverName)
    return len(polygons_result)


def get_list_name_file(path_dir, name_file='*.tif'):
    list_file_dir = []
    for file_ in glob(os.path.join(path_dir, name_file)):
        # head, tail = os.path.split(file_)
        list_file_dir.append(file_)
    return list_file_dir


def remove_invalid_geom(gdf):
    invalid_geom = ~gdf['geometry'].is_valid
    gdf.loc[invalid_geom, 'geometry'] = gdf.loc[invalid_geom, 'geometry'].buffer(0)

    return remove_invalid_gdf(gdf)


def remove_intersection(dir_shp, out_file):
    list_path_file = get_list_name_file(dir_shp, '*.shp')

    df1 = gpd.read_file(list_path_file[0])
    df1 = remove_invalid_geom(df1)
    name_index_df1 = df1.columns[0]
    print(name_index_df1)

    df2 = gpd.read_file(list_path_file[1])
    df2 = remove_invalid_geom(df2)
    name_index_df2 = df2.columns[0]
    # tim giao chi lay db gom idx df1, idx df2, va area

    res_intersection = gpd.overlay(df1, df2, how='intersection', keep_geom_type=False)
    res_intersection['area_intersec'] = res_intersection.to_crs({'init': 'epsg:3857'}).area
    name_index_res_intersection1 = res_intersection.columns[0]
    name_index_res_intersection2 = res_intersection.columns[1]
    res_intersection = res_intersection[[name_index_res_intersection1, name_index_res_intersection2, 'area_intersec']]

    df1['area1'] = df1.to_crs({'init': 'epsg:3857'}).area
    df2['area2'] = df2.to_crs({'init': 'epsg:3857'}).area


    df1 = df1[df1['area1'] > 10]
    df2 = df2[df2['area2'] > 10]

    dftmp1 = df1[[name_index_df1, 'area1']]
    dftmp2 = df2[[name_index_df2, 'area2']]

    # doi ten cot cho giong nhau
    dftmp1 = dftmp1.rename(columns={name_index_df1:name_index_res_intersection1})
    dftmp2 = dftmp2.rename(columns={name_index_df2:name_index_res_intersection2})
    res_intersection = res_intersection.merge(dftmp1, on=name_index_res_intersection1)
    res_intersection = res_intersection.merge(dftmp2, on=name_index_res_intersection2)

    res_intersection['IOU'] = res_intersection['area_intersec']/(res_intersection['area1'] + res_intersection['area2'])
    IOU = res_intersection[res_intersection['IOU'] > 0.2]
    df1 = df1.loc[~df1[name_index_df1].isin(IOU[name_index_res_intersection1])]
    df1 = df1.append(df2)
    del df1["area1"], df1["area2"]
    df1.to_file(out_file)


# def nms_shp(dir_shp, out_file):
#     list_path_file = get_list_name_file(dir_shp, '*.shp')
#     df1 = gpd.read_file(list_path_file[0])
#     df2 = gpd.read_file(list_path_file[1])
#     df = df1.append(df2)
#     tmp_file = os.path.join(dir_shp, '_tmp.shp')
#     df.to_file(tmp_file)
#     df = gpd.read_file(tmp_file)
#     df_bound = df.bounds
#     df_bound = df_bound.to_numpy()
#     score = np.random.rand(df_bound.shape[0])
#     result = tf.image.non_max_suppression(df_bound.tolist(), score.tolist(), 1000000, iou_threshold=0.4)
#     with tf.Session() as sess:
#         out = sess.run([result])
#         indexes = out[0]
#     rs_df = df.iloc[indexes]
#     rs_df.to_file(out_file)


def resize(image_path, output):
    resolution_origin = 0.15
    resolution_destination = 0.3
    size_destination = resolution_origin / resolution_destination * 100
    options_list = [
        f'-outsize {size_destination}% {size_destination}%',
        '-of GTiff',
        '-r cubic',
        '-ot Byte'
    ]
    options_string = " ".join(options_list)
    print(output)
    gdal.Translate(output,
                   image_path,
                   options=options_string)
    # return output
