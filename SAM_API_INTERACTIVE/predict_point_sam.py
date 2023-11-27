import os
import torch
import numpy as np
import geopandas as gpd
from samgeo import SamGeo
import fiona

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def sam_point_main(model_sam, img_path, input_point_prompt_shp, out_vector_path, dir_tmp):
    df_prompt = gpd.read_file(input_point_prompt_shp)
    if df_prompt.geom_type.unique()[0] == 'Point' and 'id' in df_prompt.columns:
        os.makedirs(os.path.dirname(out_vector_path), exist_ok=True)
        os.makedirs(dir_tmp, exist_ok=True)
        model_sam.set_image(img_path)
        df_point = df_prompt['geometry']
        point_lists = []
        for geometry in df_point:
            point_lists.append([geometry.coords[0]])
        point_labels = df_prompt['id'].to_list()
        
        # print(len(point_labels))
        # print(len(point_labels), point_labels)
        
        point_labels =  np.array(point_labels)
        point_lists = np.array(point_lists).squeeze(1)
        
        # point_labels =  point_labels
        # point_lists = point_lists

        epsg_code = "EPSG:" + str(df_prompt.crs.to_epsg())

        print(len(point_lists), point_labels)
        print(len(point_labels), point_labels)
        
        outputRaster_tmp = os.path.join(dir_tmp, os.path.basename(img_path))
        model_sam.predict(point_lists, point_labels=point_labels, point_crs=epsg_code, output=outputRaster_tmp)
        model_sam.raster_to_vector(outputRaster_tmp, out_vector_path)
    else:
        print('shapefile dont have "id" field or Point geometry')
    
    
if __name__=="__main__":
    model_path = r"/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/weights/sam_vit_h_4b8939.pth"
    input_path1 = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM_API_INTERACTIVE/upload/fe0965c46aea4ad68f3dd97c71bf51fb.tif'
    input_path2 = r'/home/skm/SKM16/Tmp/ass.shp'
    output_path = r'/home/skm/SKM16/Tmp/zxz.shp'
    tmp_dir = r'/home/skm/SKM16/Tmp/z'
    model_sam = SamGeo(
        checkpoint=model_path,
        model_type="vit_h", # vit_l, vit_b , vit_h
        automatic=False,
        device=device,
        sam_kwargs=None,
    )
    sam_point_main(model_sam, input_path1, input_path2, output_path, tmp_dir)
