import params
import keras
from predict_big_image import predict_farm
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from geopandas import GeoDataFrame
import rasterio
if __name__ == '__main__':
    print('Start running model')
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/polyhouse/v1/weights/model.h5'
    input_path = params.INPUT_PATH
    output_path =params.OUTPUT_PATH
    model = keras.models.load_model(model_path)
    result_path = f'{params.TMP_PATH}/result.tif'
    # vector_path = f'{params.TMP_PATH}/vector.geojson'
    predict_farm(model, input_path, result_path)
    with rasterio.open(result_path, 'r+') as dst:
        dst.nodata = 0
    with rasterio.open(result_path) as src:
        data = src.read(1, masked=True)
        shape_gen = ((shape(s), v) for s, v in shapes(data, transform=src.transform))
        gdf = GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=src.crs)
        gdf = gdf.to_crs(4326)
        gdf.to_file(output_path, driver = "GeoJSON")

    # raster_to_vector(result_path, vector_path)
    # gdf = gpd.read_file(vector_path)
    # gdf.to_file(output_path)
    