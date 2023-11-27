import params
import keras
from predict import predict_farm, raster_to_vector
import geopandas as gpd
if __name__ == '__main__':
    print('Start running model')
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/farm-boundary/v1/weights/model.h5'
    input_path = params.INPUT_PATH
    output_path =params.OUTPUT_PATH
    model = keras.models.load_model(model_path)
    result_path = f'{params.TMP_PATH}/result.tif'
    vector_path = f'{params.TMP_PATH}/vector.geojson'
    predict_farm(model, input_path, result_path)
    raster_to_vector(result_path, vector_path)
    gdf = gpd.read_file(vector_path)
    gdf.to_file(output_path)
    