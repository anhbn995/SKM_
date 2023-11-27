import params
import rasterio
from utils.image import stretch_image
from utils.unet_data_preparation import shp2mask
import utils.ogr2ogr as ogr2ogr
import json
from train import TrainerUnet
from processing import PreprocessingUnet
from utils.on_training_proccessing import on_training_proccessing
if __name__ == "__main__":
    bound_path = params.BOUND_PATH
    annotation_path = params.ANNOTATION_PATH
    image_path = params.IMAGE_PATH
    output_dir = params.OUTPUT_DIR
    meta_path = f'{output_dir}/model.json'
    model_path = f'{output_dir}/model.h5'
    tmp_path = params.TMP_PATH
    reprojected_bound_path = '{}/reprojected_bound.geojson'.format(tmp_path)
    reprojected_annotation_path = '{}/reprojected_annotation.geojson'.format(
        tmp_path)
    labels = json.loads(params.LABELS)
    input_type = params.INPUT_TYPE
    unet_type = params.TRAINER_TYPE
    task_id = params.TASK_ID
    on_training_proccessing(0.1, task_id)
    with rasterio.open(image_path) as src:
        crs = dict(src.crs).get('init')
        numbands = src.count
        dtype = src.dtypes[0]
        tr = src.transform
        w, h = src.width, src.height
    kwargs = {
        'numbands': numbands,
        'trainer_size': params.TRAINER_SIZE,
        'labels': labels,
        'optimizer': params.OPTIMIZER,
        'loss': params.LOSS,
        'metrics': json.loads(params.METRICS),
        'batch_size': None,
        'epochs': params.EPOCHS,
        'patience_early': None,
        'factor': None,
        'patience_reduce': None,
        'min_lr': None,
        'input_type': input_type,
        'n_filters': params.N_FILTERS,
        'dropout': params.DROPOUT,
        'batchnorm': params.BATCHNORM,
        'task_id': task_id
    }
    ogr2ogr.main(["", "-f", "geojson", '-t_srs', crs,
                  reprojected_bound_path, bound_path])

    def norm_input(input_type, image_path):
        if input_type == 'vector':
            with rasterio.open(image_path) as src:
                crs = dict(src.crs).get('init')
            ogr2ogr.main(["", "-f", "geojson", '-t_srs', crs,
                          reprojected_annotation_path, annotation_path])
            label = list(map(lambda el: el.get('value'), labels))
            attribute = 'label'
            mask = shp2mask(reprojected_annotation_path,
                            reprojected_bound_path, attribute, label, h, w, tr)
        else:
            mask = json.loads(params.MASK)

        if dtype != 'uint8':
            new_image_path = f'{tmp_path}/image_uint8.tif'
            stretch_image(image_path, new_image_path)
            image_stretch_path = new_image_path
        else:
            image_stretch_path = image_path
        return mask, image_stretch_path
    on_training_proccessing(0.3, task_id)
    mask, image_stretch_path = norm_input(input_type, image_path)
    unet_data_preparation = PreprocessingUnet(
        reprojected_bound_path, mask, tmp_path, output_dir, image_stretch_path, unet_type, **kwargs)
    temp_mask_path = unet_data_preparation.preprocess_run()
    unet_trainer = TrainerUnet(model_path, tmp_path, unet_type, **kwargs)
    model_config = unet_trainer.training()
    unet_trainer = None
    on_training_proccessing(0.8, task_id)
    kwargs.update(model_config)
    metadata = {
        "param": kwargs,
        "dtype": dtype,
        "temp_mask_path": temp_mask_path
    }
    with open(meta_path, 'w') as file:
        json.dump(metadata, file)
