import params
import utils.ogr2ogr as ogr2ogr
import rasterio
from utils.mrcnn import gen_mean_pixcels
from processing import PreprocessingMrcnn
from train import MrcnnTrainer
import json
from utils.on_training_proccessing import on_training_proccessing
if __name__ == '__main__':
    bound_path = params.BOUND_PATH
    annotation_path = params.ANNOTATION_PATH
    image_path = params.IMAGE_PATH
    output_dir = params.OUTPUT_DIR
    meta_path = f'{output_dir}/model.json'
    tmp_path = params.TMP_PATH
    reprojected_bound_path = '{}/reprojected_bound.geojson'.format(tmp_path)
    reprojected_annotation_path = '{}/reprojected_annotation.geojson'.format(
        tmp_path)
    task_id = params.TASK_ID
    with rasterio.open(image_path) as src:
        crs = dict(src.crs).get('init')
        numbands = src.count
        dtype = src.dtypes[0]
    ogr2ogr.main(["", "-f", "geojson", '-t_srs', crs,
                  reprojected_bound_path, bound_path])
    ogr2ogr.main(["", "-f", "geojson", '-t_srs', crs,
                  reprojected_annotation_path, annotation_path])
    mean_pixels = gen_mean_pixcels(numbands).tolist()
    on_training_proccessing(0.1, task_id)
    kwargs = {
        'trainer_size': params.TRAINER_SIZE,
        'MEAN_PIXEL': mean_pixels,
        'IMAGE_RESIZE_MODE': params.IMAGE_RESIZE_MODE,
        'IMAGE_CHANNEL_COUNT': numbands,
        'RPN_ANCHOR_SCALES': tuple(json.loads(params.RPN_ANCHOR_SCALES)),
        'STEPS_PER_EPOCH': params.STEPS_PER_EPOCH,
        'VALIDATION_STEPS': params.VALIDATION_STEPS,
        'MAX_GT_INSTANCES': params.MAX_GT_INSTANCES,
        'ROI_POSITIVE_RATIO': params.ROI_POSITIVE_RATIO,
        'RPN_NMS_THRESHOLD': params.RPN_NMS_THRESHOLD,
        'epoch1': params.EPOCH1,
        'epoch2': params.EPOCH2,
        'epoch3': params.EPOCH3,
        'task_id': task_id
    }
    on_training_proccessing(0.3, task_id)
    mrcnn_data_preparation = PreprocessingMrcnn(
        image_path, reprojected_bound_path, reprojected_annotation_path, tmp_path, **kwargs)
    data_dir = mrcnn_data_preparation.preproces_run()
    mrcnn_trainer = MrcnnTrainer('training', output_dir, data_dir, **kwargs)
    model_config = mrcnn_trainer.run()
    on_training_proccessing(0.8, task_id)
    kwargs.update(model_config)
    metadata = {
        "param": kwargs,
        "dtype": dtype
    }
    with open(meta_path, 'w') as file:
        json.dump(metadata, file)
    mrcnn_trainer = None
