from enum import Enum

import params
import predictors.infer_with_trtis as buildup_area
import predictors.infer_with_trtis_v2 as building_footprint
import predictors.infer_with_trtis_v3 as change_detection
import geopandas as gpd


class ModelType(Enum):
    BUILDING_FOOTPRINT = 'building footprint'
    BUILDUP_AREA = 'buildup_area'
    ROAD_DETECTION = 'road_detection'
    WATER_DETECTION = 'water_detection'
    CHANGE_DETECTION = 'change_detection'
    TREE_COUNTING = 'tree_counting'
    PALM_TREE_COUNTING = 'palm_tree_counting'
    AGRICULTURE = 'agriculture'
    DEFORESTATION = 'deforestation'
    TREES_2M_UAV = 'trees_2m_uav'
    TREES_40D_UAV = 'trees_40d_uav'
    DEAD_TREES_UAV = 'dead_trees_uav'


if __name__ == '__main__':

    input_path = params.INPUT_PATH
    output_path = params.OUTPUT_PATH
    predictor = params.MODEL_TYPE
    output_type = params.OUTPUT_TYPE
    if output_type == 'vector':
        output_path = output_path.replace('shp', 'geojson')
    if predictor == ModelType.BUILDING_FOOTPRINT.value:
        building_footprint.infer(
            input_path, output_path)
    elif predictor == ModelType.BUILDUP_AREA.value:
        buildup_area.infer(
            input_path, output_path)
    elif predictor == ModelType.ROAD_DETECTION.value:
        buildup_area.infer(
            input_path, output_path, model_name='road_bing')
    elif predictor == ModelType.TREE_COUNTING.value:
        building_footprint.infer(
            input_path, output_path, model_name='trees_uav')
    elif predictor == ModelType.PALM_TREE_COUNTING.value:
        building_footprint.infer(
            input_path, output_path, model_name='palm_uav')
    elif predictor == ModelType.TREES_2M_UAV.value:
        building_footprint.infer(
            input_path, output_path, model_name='trees_2m_uav')
    elif predictor == ModelType.TREES_40D_UAV.value:
        building_footprint.infer(
            input_path, output_path, model_name='trees_40d_uav')
    elif predictor == ModelType.DEAD_TREES_UAV.value:
        building_footprint.infer(
            input_path, output_path, model_name='dead_trees_uav')
    elif predictor == ModelType.WATER_DETECTION.value:
        buildup_area.infer(
            input_path, output_path, model_name='water_bing')
    elif predictor == ModelType.CHANGE_DETECTION.value:
        change_detection.infer(
            input_path, output_path, model_name='deforest')
    elif predictor == ModelType.DEFORESTATION.value:
        change_detection.infer(
            input_path, output_path, model_name='deforest_s50m')
    if output_type == 'vector':
        gdf = gpd.read_file(output_path)
        gdf.to_file(params.OUTPUT_PATH)
