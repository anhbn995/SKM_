import params
from preprocessing import transform_image_annotation_2_dataset
from training import train
import rasterio
import json
if __name__ == '__main__':
    bound_path = params.BOUND_PATH
    annotation_path = params.ANNOTATION_PATH
    image_path = params.IMAGE_PATH
    output_dir = params.OUTPUT_DIR
    output_path = f'{output_dir}/model.h5'
    meta_path = f'{output_dir}/model.json'
    tmp_path = params.TMP_PATH
    epoch = params.EPOCHS
    transform_image_annotation_2_dataset(
        image_path, bound_path, annotation_path, tmp_path)
    train(tmp_path, output_path, nepoch=epoch)
    with rasterio.open(image_path) as src:
        numbands = src.count
        dtype = src.dtypes[0]
    metadata = {
        'dtype': dtype,
        'param': {
            'numbands': numbands,
            'labels': [{"value": 2, "name": "boundary"}],
            'trainer_size': 480,
        }
    }
    with open(meta_path, 'w') as file:
        json.dump(metadata, file)
