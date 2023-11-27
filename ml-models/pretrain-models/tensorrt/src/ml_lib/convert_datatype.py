import numpy as np
import multiprocessing

core_of_computer = multiprocessing.cpu_count()


def contour_to_polygon(contour):
    list_point = []
    for point in contour:
        [x, y] = point[0]
        point_in_polygon = (x, y)
        list_point.append(point_in_polygon)
    for point in contour:
        [x, y] = point[0]
        point_in_polygon = (x, y)
        list_point.append(point_in_polygon)
        break
    poly = tuple(list_point)
    return poly


def list_contour_to_list_polygon(list_contour):
    list_polygon = []
    for contour in list_contour:
        poly = contour_to_polygon(contour)
        list_polygon.append(poly)
    return list_polygon


def polygon_to_contour(polygon):
    list_point_in_contours = []
    for point in polygon:
        point_in_contours = [list(point)]
        list_point_in_contours.append(point_in_contours)
    list_point_in_contours = np.asarray(list_point_in_contours, dtype=np.int32)
    return list_point_in_contours


def list_polygon_to_list_contour(list_polygon):
    list_contour = []
    for poly in list_polygon:
        contour = polygon_to_contour(poly)
        list_contour.append(contour)
    return list_contour


def contour_to_list_array(contour):
    contour = contour.reshape(len(contour), 2)
    contour_rs = list(contour)
    return contour_rs


def list_contour_to_list_list_array(list_contour):
    list_temp = []
    for contour in list_contour:
        list_array = contour_to_list_array(contour)
        list_temp.append(list_array)
    return list_temp


def list_array_to_contour(list_array):
    contour = np.asarray(list_array, dtype=np.float32)
    contour_rs = contour.reshape(len(contour), 1, 2)
    return contour_rs


def list_list_array_to_list_contour(list_list_array):
    list_contour = []
    for list_array in list_list_array:
        contour = list_array_to_contour(list_array)
        list_contour.append(contour)
    return list_contour


def convert_numpy_to_list(arr):
    list_list = []
    for i in arr:
        temp = i.tolist()
        list_list.append(temp)
    return list_list
