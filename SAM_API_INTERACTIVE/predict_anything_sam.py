import torch
import os,sys
import rasterio
from samgeo import SamGeo

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def main_sam_anything(model_sam, img_path, out_vector_path, dir_tmp, size_img):

    os.makedirs(os.path.dirname(out_vector_path), exist_ok=True)
    os.makedirs(dir_tmp, exist_ok=True)
    with rasterio.open(img_path) as src:
        width = src.width
        height = src.height
        
    outputRaster_tmp = os.path.join(dir_tmp, os.path.basename(img_path))
    
    if width*height > 1500*1500:
        model_sam.generate(source=img_path, output=outputRaster_tmp, batch=True, erosion_kernel=(3,3))
    else:
        model_sam.generate(source=img_path, output=outputRaster_tmp)
    model_sam.raster_to_vector(outputRaster_tmp, out_vector_path)
    return True

    

if __name__=="__main__":
    model_path = "ROOT_DATA_FOLDER/pretrain-models/sam_anything/v1/weights/sam_vit_h_4b8939.pth"

