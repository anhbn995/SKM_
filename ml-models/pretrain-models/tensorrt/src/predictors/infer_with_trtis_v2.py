import model.model3 as modellib
import os
import sys
import argparse
import numpy as np

# import coco  # a slightly modified version

from tqdm import tqdm
import cv2
import gdal
import osr
import scipy
from ml_lib.convert_datatype import list_contour_to_list_polygon
from ml_lib.export_data import exportResult2 as exportResult2

from tensorrtserver.api import ProtocolType, ServerHealthContext, ServerStatusContext, InferContext
import model.config as config
from params import TENSORRT_SERVER_URL

ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


class BuildingConfig(config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    IMAGE_MAX_DIM = 512 + 64
    IMAGE_MIN_DIM = 512 + 64
    DETECTION_MAX_INSTANCES = 500
    MAX_GT_INSTANCES = 500
    NAME = "crowdai-mapping-challenge"
    BACKBONE = "resnet101"
    ROI_POSITIVE_RATIO = 0.66


class TreecountConfig(config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    IMAGE_MAX_DIM = 256 + 64
    IMAGE_MIN_DIM = 256 + 64
    DETECTION_MAX_INSTANCES = 200
    MAX_GT_INSTANCES = 100

    MASK_SHAPE = [28, 28]
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    NAME = "crowdai-mapping-challenge"

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.7


class DeadtreeConfig(config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    IMAGE_MAX_DIM = 512 + 64
    IMAGE_MIN_DIM = 512 + 64
    DETECTION_MAX_INSTANCES = 15
    MAX_GT_INSTANCES = 100

    MASK_SHAPE = [28, 28]
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    NAME = "crowdai-mapping-challenge"

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.9


def infer(
    image_path,
    output_path,
    on_processing=None,
    url=TENSORRT_SERVER_URL,
    protocol='http',
    http_headers=None,
    model_name='bingbf-v2',
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
    # config = InferenceConfig()

    # Create a health context, get the ready and live state of server.
    health_ctx = ServerHealthContext(url, protocol,
                                     http_headers=http_headers, verbose=False)
    # Create a status context and get server status
    status_ctx = ServerStatusContext(url, protocol, model_name,
                                     http_headers=http_headers, verbose=False)

    # Create the inference context for the model.
    infer_ctx = InferContext(url, protocol, model_name, model_version,
                             http_headers=http_headers, verbose=False)

    # Read input image and get its meta-data
    dataset_image = gdal.Open(image_path)

    num_band = 3
    input_size = 512
    crop_size = input_size-100

    if model_name == 'bingbf-v2' or model_name == 'trees_uav' or model_name == 'palm_uav':
        model = modellib.MaskRCNN(
            mode="inference",
            model_dir='./',
            config=BuildingConfig()
        )
    elif model_name == 'dead_trees_uav':
        model = modellib.MaskRCNN(
            mode="inference",
            model_dir='./',
            config=TreecountConfig()
        )
    else:
        model = modellib.MaskRCNN(
            mode="inference",
            model_dir='./',
            config=DeadtreeConfig()
        )

    if model_name == 'bingbf-v2':

        list_contours = model.detect_with_trtis_v2(
            dataset_image=dataset_image,
            num_band=num_band,
            input_size=input_size,
            crop_size=crop_size,
            infer_ctx=infer_ctx,
            async_set=async_set,
            verbose=0,
            on_processing=on_processing
        )
    else:
        list_contours = model.detect_with_trtis_trees(
            dataset_image=dataset_image,
            num_band=num_band,
            input_size=input_size,
            crop_size=crop_size,
            infer_ctx=infer_ctx,
            async_set=async_set,
            verbose=0,
            on_processing=on_processing
        )

    # Save result

    driverName = "GeoJson"
    # driverName = "ESRI Shapefile"
    geotransform = dataset_image.GetGeoTransform()
    projection = osr.SpatialReference(dataset_image.GetProjectionRef())
    polygons_result = list_contour_to_list_polygon(list_contours)

    exportResult2(polygons_result, geotransform,
                  projection, output_path, driverName)


def main():
    parser = argparse.ArgumentParser(
        description='Inference module with TensorRT Inference Server.')
    parser.add_argument(
        '-i',
        '--img',
        type=str,
        help='Path to input image.',
        required=True
    )
    parser.add_argument(
        '-o',
        '--out',
        type=str,
        help='Path to where result is saved.',
        required=True
    )
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        help='Inference server URL. Default is 192.168.1.110:8000.',
        required=False,
        default='192.168.1.110:8000',
    )
    parser.add_argument(
        '-p',
        '--protocol',
        type=str,
        help='Protocol ("http"/"grpc") used to communicate with inference service. Default is "http".',
        default='http'
    )
    parser.add_argument(
        '-H',
        dest='http_headers',
        metavar="HTTP_HEADER",
        required=False,
        action='append',
        help='HTTP headers to add to inference server requests. ' +
        'Format is -H"Header:Value".'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='The name of selected model; default is bingbf-v2',
        default='bingbf-v2'
    )
    parser.add_argument(
        '--model_version',
        type=int,
        help='The version of the model to use for inference,\
            or None to indicate that the latest (i.e. highest version number) version should be used.',
        default=None
    )
    parser.add_argument(
        '-d',
        '--driver',
        help='Output\'s driver, e.i. GeoJSON, ESRI Shapefile, ...; default is GeoJSON.',
        default="GeoJSON"
    )
    parser.add_argument(
        '-a',
        '--async',
        dest="async_set",
        action="store_true",
        required=False,
        default=False,
        help='Use asynchronous inference API.'
    )

    args = parser.parse_args()
    image_path = args.img
    output_path = args.out
    url = args.url
    protocol = args.protocol
    http_headers = args.http_headers
    model_name = args.model_name
    model_version = args.model_version
    driver = args.driver
    async_set = args.async_set

    infer(
        image_path,
        output_path,
        url,
        protocol,
        http_headers,
        model_name,
        model_version,
        async_set,
        driver
    )


if __name__ == '__main__':
    from datetime import datetime
    start = datetime.now()
    main()
    print('Time in total:', datetime.now() - start)
