import torch
import os,sys
import rasterio
# from samgeo import SamGeo

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main_sam_anything(model_sam, img_path, out_vector_path, dir_tmp, size_img):
    # try:
    os.makedirs(os.path.dirname(out_vector_path), exist_ok=True)
    os.makedirs(dir_tmp, exist_ok=True)
    with rasterio.open(img_path) as src:
        width = src.width
        height = src.height
        
    outputRaster_tmp = os.path.join(dir_tmp, os.path.basename(img_path))
    
    if width*height > size_img*size_img:
        return f'image size over {size_img} * {size_img}'
    else:
        model_sam.generate(source=img_path, output=outputRaster_tmp)
        model_sam.raster_to_vector(outputRaster_tmp, out_vector_path)
        return 'Done'
    # except:
    #     return False
    
