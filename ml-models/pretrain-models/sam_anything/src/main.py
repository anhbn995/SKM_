import torch
import os,sys
from samgeo import SamGeo
import rasterio
import params

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def run_sam_anything(model_path, img_path, outputVector, dir_tmp):
    os.makedirs(os.path.dirname(outputVector), exist_ok=True)
    if not os.path.exists(dir_tmp):
        os.makedirs(dir_tmp)
    sam = SamGeo(
            checkpoint=model_path,
            model_type="vit_h", # vit_l, vit_b , vit_h
            automatic=True,
            device=device,
            sam_kwargs=None,
            )
    with rasterio.open(img_path) as src:
        width = src.width
        height = src.height
    outputRaster = os.path.join(dir_tmp, "out_put.tif")
    if width*height > 1500*1500:
        sam.generate(source=img_path, output=outputRaster,batch=True, erosion_kernel=(3,3))
    else:
        sam.generate(source=img_path, output=outputRaster)
    sam.raster_to_vector(outputRaster, outputVector, simplify_tolerance=0)

if __name__=="__main__":
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/sam_anything/v1/weights/sam_vit_h_4b8939.pth'
    input_path = params.INPUT_PATH
    output_path = params.OUTPUT_PATH
    tmp_dir = params.TMP_PATH
    run_sam_anything(model_path, input_path, output_path, tmp_dir)
