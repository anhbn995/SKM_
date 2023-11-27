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
