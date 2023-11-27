import os
import torch
from samgeo import SamGeo
import rasterio
import params
import geopandas as gpd
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def run_sam_point(model_path, img_path, input_prompt_shp, outputVector, dir_tmp):
    # check shp point
    df_prompt = gpd.read_file(input_prompt_shp)
    if df_prompt.geom_type.unique()[0] == 'Point' and 'id' in df_prompt.columns:
        os.makedirs(os.path.dirname(outputVector), exist_ok=True)
        os.makedirs(dir_tmp, exist_ok=True)
        
        sam = SamGeo(
                checkpoint=model_path,
                model_type="vit_h", # vit_l, vit_b , vit_h
                automatic=False,
                device=device,
                sam_kwargs=None,
            )
        sam.set_image(img_path)
        df_point = df_prompt['geometry']
        # Convert each geometry element to a pair of lists
        point_lists = []
        for geometry in df_point:
            point_lists.append([geometry.coords[0]])
        point_labels = df_prompt['id'].to_list()
        
        print(len(point_labels))
        print(len(point_labels), point_labels)
        point_labels =  np.array(point_labels)
        point_lists = np.array(point_lists).squeeze(1)
        
        epsg_code = "EPSG:" + str(df_prompt.crs.to_epsg())
        outputRaster_tmp = os.path.join(dir_tmp, "out_put.tif")
        sam.predict(point_lists, point_labels=point_labels, point_crs=epsg_code, output=outputRaster_tmp)
        sam.raster_to_vector(outputRaster_tmp, outputVector)#, simplify_tolerance=0.00001)
    else:
        print('shapefile dont have "id" field or Point geometry')
    
if __name__=="__main__":
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/sam_point/v1/weights/sam_vit_h_4b8939.pth'
    input_path1 = params.INPUT_PATH_1
    input_path2 = params.INPUT_PATH_2
    output_path = params.OUTPUT_PATH
    tmp_dir = params.TMP_PATH
    run_sam_point(model_path, input_path1, input_path2, output_path, tmp_dir)
