import params
import json
import rasterio
from predict import MrcnnPredictor
from processing import PostprocessingMrcnn
from utils.mrcnn import gen_mean_pixcels
from utils.image import resize_with_resolution_in_meter
if __name__ == '__main__':
    print(params.MODEL_META)
    meta = json.loads(params.MODEL_META)
    input_path = params.INPUT_PATH
    tmp_path = params.TMP_PATH
    model_path = params.MODEL_PATH
    output_path = params.OUTPUT_PATH
    with rasterio.open(input_path) as src:
        numbands = src.count
    mean_pixels = gen_mean_pixcels(meta.get(
        'numbands_predict') or numbands).tolist()
    resize = meta.get('resize')
    if resize:
        resized_image = f'{tmp_path}/resized.tif'
        resize_with_resolution_in_meter(input_path, resized_image, resize)
        input_path = resized_image
    kwargs = {
        'IMAGE_CHANNEL_COUNT': numbands,
        'trainer_size': meta.get('trainer_size'),
        'predict_size': params.PREDICT_SIZE,
        'DETECTION_MAX_INSTANCES': params.DETECTION_MAX_INSTANCES,
        'DETECTION_NMS_THRESHOLD': params.DETECTION_NMS_THRESHOLD,
        'IMAGE_MAX_DIM': params.IMAGE_MAX_DIM,
        'IMAGE_MIN_DIM': params.IMAGE_MIN_DIM,
        'DETECTION_MIN_CONFIDENCE': params.DETECTION_MIN_CONFIDENCE,
        **meta,
        'MEAN_PIXEL': mean_pixels[0:numbands],
    }
    mrcnn_predictor = MrcnnPredictor(
        'inference', input_path, model_path, **kwargs)
    list_contours, list_score = mrcnn_predictor.predict()
    mrcnn_predictor = None

    mrcnn_postprocessor = PostprocessingMrcnn(
        list_contours, list_score, output_path, input_path)
    mrcnn_postprocessor.postprocess_run()
