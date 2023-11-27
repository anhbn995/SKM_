import os
import cv2
import rasterio
import rasterio.windows
import numpy as np


import torch
from PIL import Image
from samgeo.text_sam import LangSAM
from tqdm import tqdm
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from huggingface_hub import constants

device_sam = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GoundingDinoBuildSam(LangSAM):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def build_groundingdino(self):
        super().build_groundingdino()
        return self.groundingdino
        

class LangSAM_update(LangSAM):
    def __init__(self, model_build_groundingdino, model_build_sam):
        """Initialize the LangSAM instance.

        Args:
            model_type (str, optional): The model type. It can be one of the following: vit_h, vit_l, vit_b.
                Defaults to 'vit_h'. See https://bit.ly/3VrpxUh for more details.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.groundingdino = model_build_groundingdino
        self.sam = model_build_sam

        self.source = None
        self.image = None
        self.masks = None
        self.boxes = None
        self.phrases = None
        self.logits = None
        self.prediction = None
    

    def predict(
        self,
        image,
        text_prompt,
        box_threshold,
        text_threshold,
        fp_img,
        output=None,
        mask_multiplier=255,
        dtype=np.uint8,
        save_args={},
        return_results=False,
        **kwargs,
    ):
        """
        Run both GroundingDINO and SAM model prediction.

        Parameters:
            image (Image): Input PIL Image.
            text_prompt (str): Text prompt for the model.
            box_threshold (float): Box threshold for the prediction.
            text_threshold (float): Text threshold for the prediction.
            output (str, optional): Output path for the prediction. Defaults to None.
            mask_multiplier (int, optional): Mask multiplier for the prediction. Defaults to 255.
            dtype (np.dtype, optional): Data type for the prediction. Defaults to np.uint8.
            save_args (dict, optional): Save arguments for the prediction. Defaults to {}.
            return_results (bool, optional): Whether to return the results. Defaults to False.

        Returns:
            tuple: Tuple containing masks, boxes, phrases, and logits.
        """
        self.source = fp_img
        image_np = image.transpose((1, 2, 0))
        image_pil = Image.fromarray(image_np[:, :, :3])
        
        self.image = image_pil

        boxes, logits, phrases = self.predict_dino(
            image_pil, text_prompt, box_threshold, text_threshold
        )
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)

        if boxes.nelement() == 0:  # No "object" instances found
            # print("No objects found in the image.")
            pass
        else:
            # Create an empty image to store the mask overlays
            mask_overlay = np.zeros_like(
                image_np[..., 0], dtype=dtype
            )  # Adjusted for single channel

            for i, (box, mask) in enumerate(zip(boxes, masks)):
                # Convert tensor to numpy array if necessary and ensure it contains integers
                if isinstance(mask, torch.Tensor):
                    mask = (
                        mask.cpu().numpy().astype(dtype)
                    )  # If mask is on GPU, use .cpu() before .numpy()
                mask_overlay += ((mask > 0) * (i + 1)).astype(
                    dtype
                )  # Assign a unique value for each mask

            # Normalize mask_overlay to be in [0, 255]
            mask_overlay = (
                mask_overlay > 0
            ) * mask_multiplier  # Binary mask in [0, 255]

            
        self.masks = masks
        self.boxes = boxes
        self.phrases = phrases
        self.logits = logits
        self.mask_overlay = mask_overlay
        
        if return_results:
            return masks, boxes, phrases, logits, mask_overlay


def find_window(width, height, range_win= [1000,1500]):
    min_x, max_x = range_win
    size_win = None
    for x in range(max_x + 1, min_x, -1):
        if width % x > 100 and height % x > 100 and height % x < x and width % x < x:
            size_win = x
            break
    return size_win

@torch.no_grad()
def read_image_in_windows_and_predict(out_fp, fp_img, model_sam, text_prompt, chose_overlap=False):
    with rasterio.open(fp_img) as src:
        width = src.width
        height = src.height
        meta = src.meta
        meta.update({'count':1})
        
        if width*height > 1500*1500:
            window_size = find_window(width, height)
            if chose_overlap:
                overlap = window_size//4
            else:
                overlap = 0
            with rasterio.open(out_fp, 'w', **meta) as dst:
                for col in tqdm(range(0, width, window_size - overlap)):
                    for row in tqdm(range(0, height, window_size - overlap)):
                        width_win = min(window_size, width - col)
                        height_win = min(window_size, height - row)
                        window = rasterio.windows.Window(col, row, width_win, height_win)
                        # print(col, row, width_win, height_win)
                        data = src.read(window=window)
                        if np.all(data == 0):
                            mask_overlay = np.zeros((width_win, height_win))
                        else:
                            try:
                                # print(data.shape)
                                _, _, _, _, mask_overlay= model_sam.predict(data, text_prompt, box_threshold=0.24, text_threshold=0.24, fp_img=fp_img,return_results=True)
                            except:
                                mask_overlay = np.zeros((width_win, height_win))
                        dst.write(mask_overlay, window=window, indexes=1)
        else:
            data = src.read()
            # print(data.shape)
            try:
                _, _, _, _, mask_overlay = model_sam.predict(data, text_prompt, box_threshold=0.24, text_threshold=0.24, fp_img=fp_img, return_results=True)
                with rasterio.open(out_fp, 'w', **meta) as dst:
                    dst.write(mask_overlay, indexes=1)
            except:
                pass
                

# def main_sam_text(sam, fp_image, text_prompt, tmp_dir, fp_out_shp):
#     overlab = 0
#     os.makedirs(tmp_dir, exist_ok=True)
#     fp_out_img_tmp = os.path.join(tmp_dir, os.path.basename(fp_image))
#     read_image_in_windows_and_predict(fp_out_img_tmp, fp_image, sam, text_prompt, overlab)
#     os.makedirs(os.path.dirname(fp_out_shp), exist_ok=True)
#     sam.raster_to_vector(fp_out_img_tmp, fp_out_shp)
    
def main_sam_text(model_build_groundingdino, model_build_sam, fp_image, text_prompt, tmp_dir, fp_out_shp):
    overlab = 0
    sam = LangSAM_update(model_build_groundingdino, model_build_sam)
    os.makedirs(tmp_dir, exist_ok=True)
    fp_out_img_tmp = os.path.join(tmp_dir, os.path.basename(fp_image))
    read_image_in_windows_and_predict(fp_out_img_tmp, fp_image, sam, text_prompt, overlab)
    try:
        os.makedirs(os.path.dirname(fp_out_shp), exist_ok=True)
        sam.raster_to_vector(fp_out_img_tmp, fp_out_shp)
        return 1
    except:
        return 0

if __name__=="__main__":
    model_path_sam = r'/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/weights/sam_vit_h_4b8939.pth'
    # input_path = r'/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/example-data/a_4326.tif'
    input_path = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM_API_INTERACTIVE/upload/London_c70aa95385a54292ba924c20f233904d_gg.tif'
    text_prompts = ['tree','car']
    output_path = r'/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/RS/qwqwqqq.shp'
    tmp_dir = r'/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/tmp/qwqqwqqww'
    
    model_diano = GoundingDinoBuildSam()
    model_diano_ = model_diano.build_groundingdino()
    
    # print(model_diano_)
    model_type = 'vit_h'
    sam_model = sam_model_registry[model_type](checkpoint=model_path_sam)
    sam_model.to(device=device_sam)
    sam_model = SamPredictor(sam_model)
    
    import time
    x = time.time()
    for text_prompt in text_prompts:
        main_sam_text(model_diano_, sam_model, input_path, text_prompt, tmp_dir, output_path)
    print((time.time()-x)/60)
