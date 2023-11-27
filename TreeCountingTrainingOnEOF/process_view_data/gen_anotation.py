import cv2
import json
import glob, os
import datetime
import numpy as np
import shapefile as shp

from tqdm import *
from osgeo import gdal


def list_array_to_contour(list_array):
    contour = np.asarray(list_array,dtype=np.float32)
    contour_rs = contour.reshape(len(contour),1,2)
    return contour_rs


def list_list_array_to_list_contour(list_list_array):
    list_contour = []
    for list_array in list_list_array:
        contour = list_array_to_contour(list_array)
        list_contour.append(contour)
    return list_contour


def create_list_id(dir_img):
    return [os.path.basename(fp).replace('.tif','')  for fp in glob.glob(os.path.join(dir_img,'*.tif'))]


def create_list_annotation(fn_image, dir_shp_clip, dir_img_clip):
    "đọc shapefile"
    shape_path = os.path.join(dir_shp_clip, fn_image + '.shp')
    sf=shp.Reader(shape_path)
    my_shapes=sf.shapes()
    my_shapes_list = list(map(lambda shape: shape.points, my_shapes))
    
    "đọc ảnh"
    filename = os.path.join(dir_img_clip, fn_image +'.tif')
    dataset = gdal.Open(filename)
    list_list_point_convert = []
    for shapes in my_shapes_list:
        list_point=[]
        for point in shapes:
            x,y = point[0],point[1]
            list_point.append((x,y))
        list_list_point_convert.append(list_point)
        
    "chuyen sang toa do pixel"
    transformmer = dataset.GetGeoTransform()
    xOrigin = transformmer[0]
    yOrigin = transformmer[3]
    pixelWidth = transformmer[1]
    pixelHeight = -transformmer[5]

    list_list_point=[]
    for points_list in list_list_point_convert:
        lis_poly=[]
        for point in points_list:
            col = int((point[0] - xOrigin) / pixelWidth)
            row = int((yOrigin - point[1] ) / pixelHeight)
            lis_poly.append([col,row])
        lis_poly = np.asarray(lis_poly,dtype = np.int)
        list_list_point.append(lis_poly)
    list_cnt = list_list_array_to_list_contour(list_list_point)
    width, height = dataset.RasterXSize, dataset.RasterYSize
    return height, width, list_cnt


def main_gen_anotation(dir_img_clip, dir_shp_clip):
    parent = os.path.dirname(dir_img_clip)
    _final_object = {}
    _final_object["info"]= {
                            "contributor": "crowdAI.org",
                            "about": "Dataset for crowdAI Mapping Challenge",
                            "date_created": datetime.datetime.utcnow().isoformat(' '),
                            "description": "crowdAI mapping-challenge dataset",
                            "url": "https://www.crowdai.org/challenges/mapping-challenge",
                            "version": "1.0",
                            "year": 2018
                            }
                        
    _final_object["categories"]=[
                    {
                        "id": 100,
                        "name": "building",
                        "supercategory": "building"
                    }
                ]
    date_captured=datetime.datetime.utcnow().isoformat(' ')
    license_id=1
    coco_url=""
    flickr_url=""
    _images = []
    _annotations = []
    _list_image = create_list_id(dir_img_clip)
    _image_id = 0
    _annotation_id = 0
    
    with tqdm(total=len(_list_image)) as pbar:
        for _image_name in _list_image:
            pbar.update()
            _image_id = _image_id + 1
            _file_name = _image_name + '.tif'
            _height, _width, list_annotation = create_list_annotation(_image_name, dir_shp_clip, dir_img_clip)
            image_info = {
                        "id": _image_id,
                        "file_name": _file_name,
                        "width": _width,
                        "height": _height,
                        "date_captured": date_captured,
                        "license": license_id,
                        "coco_url": coco_url,
                        "flickr_url": flickr_url
                        }
            _images.append(image_info)
            for annotation in list_annotation:
                _annotation_id = _annotation_id + 1
                _area = cv2.contourArea(annotation)
                x,y,w,h = cv2.boundingRect(annotation)
                _bbox = [y,x,h,w]
                _segmentation = [list(annotation.astype(np.float64).reshape(-1))]
                annotation_info = {
                    "id": _annotation_id,
                    "image_id": _image_id,
                    "category_id": 100,
                    "iscrowd": 0,
                    "area": _area,
                    "bbox": _bbox,
                    "segmentation": _segmentation,
                    "width": _width,
                    "height": _height,
                }
                _annotations.append(annotation_info)
    _final_object["images"]=_images
    _final_object["annotations"]=_annotations
    fp = open(os.path.join(parent,'annotation.json'), "w")
    print("Writing JSON...")
    fp.write(json.dumps(_final_object))

if __name__ == "__main__":
    dir_img_clip = r'E:\TMP_XOA\mongkos_std\test\data_23_06_2020\tmp\val\images'
    dir_shp_clip = r'E:\TMP_XOA\mongkos_std\test\data_23_06_2020\shp_clip'
    main_gen_anotation(dir_img_clip,dir_shp_clip)
