import numpy as np
from osgeo import gdal, ogr, osr
import cv2
import rasterio
from ml_lib.convert_datatype import list_contour_to_list_polygon, contour_to_polygon
from ml_lib.export_data import list_polygon_to_list_geopolygon, polygon_to_geopolygon, transformToLatLong
from shapely.geometry import Polygon
import fiona
from fiona.crs import from_epsg
from shapely import geometry

from tensorrtserver.api import ProtocolType, ServerHealthContext, ServerStatusContext, InferContext
from params import TENSORRT_SERVER_URL
from ml_lib.export_data import exportResult3 as exportResult3
from utils.image import get_info_image


def rm_polygon_err(list_polygon):
    list_poly_good = []
    for polygon in list_polygon:
        if len(polygon) >= 3:
            list_poly_good.append(polygon)
    return list_poly_good


def post_proccessing(list_polygon, geotransform):

    # data
    list_contour_not_holes = list_polygon[0]
    list_list_contour_parent = list_polygon[1]

    # # # xử lý không có lỗ
    list_polygon_not_holes = []
    list_poly_not_holes = list_contour_to_list_polygon(list_contour_not_holes)
    list_poly_not_holes = rm_polygon_err(list_poly_not_holes)
    list_geopolygon_not_holes = list_polygon_to_list_geopolygon(
        list_poly_not_holes, geotransform)

    for geopolygon in list_geopolygon_not_holes:
        geopolygon_not_holes = list(geopolygon)
        myPoly = geometry.Polygon(geopolygon_not_holes)
        list_polygon_not_holes.append(myPoly)

    # xử lý có lỗ
    list_polygon_have_holes = []
    for list_contour_parent in list_list_contour_parent:
        # những thằng là cha
        contour_parents = list_contour_parent[0]
        poly_parents = contour_to_polygon(contour_parents)
        geopolygon_parent = polygon_to_geopolygon(poly_parents, geotransform)

        geopolygon_parent = list(geopolygon_parent)
        # print(geopolygon_parent)
        # những thăng là con
        list_contour_child = np.delete(list_contour_parent, 0)
        list_contour_child = rm_polygon_err(list_contour_child)
        list_poly_child = list_contour_to_list_polygon(list_contour_child)
        list_geopolygon_child = list_polygon_to_list_geopolygon(
            list_poly_child, geotransform)

        # tung geopolygon đươc cho vao 1 list
        geopolygon_child_list = []
        for geopolygon_child in list_geopolygon_child:
            geopolygon_child_list.append(list(geopolygon_child))
        # print(geopolygon_child_list)
        geopolygon_child_list = list(list_geopolygon_child)
        myPoly = geometry.Polygon(geopolygon_parent, geopolygon_child_list)
        list_polygon_have_holes.append(myPoly)

    list_polygon = list_polygon_not_holes + list_polygon_have_holes
    return list_polygon


def predict(
    img_url,
    infer_ctx,
    on_processing=None,
    num_band=3,
    input_size=512,
    crop_size=512*3//4,
    batch_size=1,
    threshold=0.5
):
    dataset1 = gdal.Open(img_url)
    values = dataset1.ReadAsArray()[0:num_band].astype(np.uint8)
    print("values", len(values))
    h, w = values.shape[1:3]
    padding = int((input_size - crop_size)/2)
    print("pading", padding)
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
        print(values[i], padding)
        band = np.pad(values[i], padding, mode='reflect')
        padded_org_im.append(band)
    # values = np.pad(values, padding, mode='reflect',axis=-1)

    values = np.array(padded_org_im).swapaxes(0, 1).swapaxes(1, 2)
    print(values.shape)
    del padded_org_im

    def get_im_by_coord(org_im, start_x, start_y, num_band):
        startx = start_x-padding
        endx = start_x+crop_size+padding
        starty = start_y - padding
        endy = start_y+crop_size+padding

        result = []
        img = org_im[starty:endy, startx:endx]
        img = img.swapaxes(2, 1).swapaxes(1, 0)
        for chan_i in range(num_band):
            result.append(cv2.resize(
                img[chan_i], (input_size, input_size), interpolation=cv2.INTER_CUBIC))
        return np.array(result).swapaxes(0, 1).swapaxes(1, 2)

    for i in range(len(img_coords)):
        im = get_im_by_coord(
            values, img_coords[i][0], img_coords[i][1], num_band)
        # print(im.shape)
        # im2 = preprocess_grayscale(im)
        # im2 = normalize_4band_esri(im, num_band)
        # im2 = normalize_255(im)
        # im2 = normalize_bandcut(im, num_band)
        cut_imgs.append(im)
    # print(len(cut_imgs))

    a = list(range(0, len(cut_imgs), batch_size))

    a.append(len(cut_imgs))

    y_pred = []
    for i in range(len(a)-1):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]]).astype(np.float32)/255.0
        print(i, len(a)-1)
        process_percent = (i+1)/(len(a)-1)
        if(on_processing):
            on_processing(process_percent)

        trtis_result = infer_ctx.run(
            {
                'input_1': (*x_batch,)
            },
            {
                'sigmoid/Sigmoid': InferContext.ResultFormat.RAW
            },
            len(x_batch)
        )

        y_batch = trtis_result['sigmoid/Sigmoid']

        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w)).astype(np.float16)
    for i in range(len(cut_imgs)):
        true_mask = y_pred[i].reshape((input_size, input_size))
        true_mask = (true_mask > 0.5).astype(np.uint8)
        true_mask = (cv2.resize(true_mask, (input_size, input_size),
                     interpolation=cv2.INTER_CUBIC) > 0.5).astype(np.uint8)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        # print(start_x-padding, start_x-padding+crop_size, start_y-padding, start_y -
        #         padding+crop_size)
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                 padding+crop_size] = true_mask[padding:padding+crop_size, padding:padding+crop_size]

    del cut_imgs
    big_mask = (big_mask > 0.5).astype(np.uint8)
    return big_mask*255


def unique(list1):
    x = np.array(list1)
    return np.unique(x)


def infer(
    image_path,
    output_path,
    on_processing=None,
    url=TENSORRT_SERVER_URL,
    protocol='http',
    http_headers=None,
    model_name='builtup',
    model_version=None,
    async_set=False,
    driver="GeoJSON"
):
    """
    Parameters:
        image_path: Path to input image.
        output_path: Path to where result is saved.
        url: Server's url.
        protocol: Protocol ("http"/"grpc") used to communicate with inference service. Default is "http".
        http_headers: HTTP headers to add to inference server requests. Format is -H"Header:Value".
        model_name: The name of selected model.
        model_version: The version of the model to use for inference,
            or None to indicate that the latest (i.e. highest version number) version should be used.
        async_set: Use asynchronous inference API.
        driver: Output's driver, e.i. GeoJSON, ESRI Shapefile, ...; default is GeoJSON.

    Returns:
        None
    """

    protocol = ProtocolType.from_str(protocol)

    # Create a health context, get the ready and live state of server.
    health_ctx = ServerHealthContext(url, protocol,
                                     http_headers=http_headers, verbose=False)

    # Create a status context and get server status
    status_ctx = ServerStatusContext(url, protocol, model_name,
                                     http_headers=http_headers, verbose=False)

    # Create the inference context for the model.
    infer_ctx = InferContext(url, protocol, model_name, model_version,
                             http_headers=http_headers, verbose=False)

    mask_base = predict(
        image_path,
        infer_ctx,
        on_processing=on_processing
    )

    h, w, crs, transform = get_info_image(image_path)

    new_dataset = rasterio.open(output_path, 'w', driver='GTiff',
                                height=h, width=w,
                                count=1, dtype='uint8',
                                crs=crs,
                                transform=transform,
                                nodata=0,
                                compress='lzw')
    new_dataset.write(mask_base, 1)


if __name__ == '__main__':
    infer(
        r"/home/boom/geoai/geoai/data/7/ffeb9e244b1b4a0899711f41163b6720/6caabe2a9cb04e6c9fa88dac7cd95400.tif",
        r"./test.GeoJSON",
    )
