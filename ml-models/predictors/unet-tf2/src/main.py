from params import INPUT_PATH,MODEL_PATH,OUTPUT_PATH,INDEX_OUTPUTS
from predictor import BasePredictor
import json
import keras
import os
from pathlib import Path
import shutil
os.environ['PROJ_LIB'] = r'E:\Programs\anaconda3\envs\gis\Library\share\proj'
if __name__ == '__main__':
    input_path = INPUT_PATH
    model_path = MODEL_PATH
    output_path = OUTPUT_PATH
    output_dir = '/'.join(output_path.split('/')[0:-1])
    index_outputs = json.loads(INDEX_OUTPUTS) if INDEX_OUTPUTS else None
    model = keras.models.load_model(model_path, compile=False)
    base_model = BasePredictor(input_path, model_path, output_dir)
    base_model.predict(index_outputs)
    shutil.copy(f'{output_dir}/result_0.tif', output_path)