import params
import os
from export_shape_file_height import export_shp_height
import rasterio
import geopandas as gp
if __name__ == '__main__':
    print('Start running model')
    input_path_1 = params.INPUT_PATH_1
    input_path_2 = params.INPUT_PATH_2
    output_path =params.OUTPUT_PATH
    tmp_path = params.TMP_PATH
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    with rasterio.open(input_path_1, mode='r+') as src:
        projstr = src.crs.to_string()
    building_shp = gp.read_file(input_path_2)
    base_name = os.path.basename(input_path_2)
    building_shp = building_shp.to_crs(projstr)
    input_path_2_tmp = os.path.join(tmp_path,base_name)
    if base_name.endswith(".geojson"):
        building_shp.to_file(input_path_2_tmp,driver = "GeoJSON")
    else:
        building_shp.to_file(input_path_2_tmp)
    export_shp_height(input_path_1, input_path_2_tmp, output_path)

    

    