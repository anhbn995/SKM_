import os
import cv2
import rasterio
import rasterio.windows
import numpy as np
from PIL import Image
from typing import Tuple, List

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


from samgeo.text_sam import LangSAM
from tqdm import tqdm
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from huggingface_hub import constants

from groundingdino.util.misc import interpolate
from groundingdino.util.utils import get_phrases_from_posmap
from torchvision.transforms import Resize

device_sam = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def predict_batch(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    # print(f" Image shape: {image.shape}") # Image shape: torch.Size([num_batch, 3, 800, 1200])
    with torch.no_grad():        
        outputs = model(image, captions=[caption for _ in range(len(image))]) # <------- I use the same caption for all the images for my use-case
    # print(f'{outputs["pred_logits"].shape}') # torch.Size([num_batch, 900, 256]) 
    # print(f'{outputs["pred_boxes"].shape}') # torch.Size([num_batch, 900, 4])
    all_boxes = []
    all_logits = []
    all_phrases = []

    for id in range (len(outputs["pred_boxes"])):
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[id]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[id]  # prediction_boxes.shape = (nq, 4)

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)

        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]
        all_boxes.append(boxes)
        all_logits.append(logits.max(dim=1)[0])
        all_phrases.append(phrases)
    return all_boxes, all_logits, all_phrases

def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0] > 0.5
        )

    return rescaled_image, target

class Resize(object):
    def __init__(self, size):
        assert isinstance(size, (list, tuple))
        self.size = size

    def __call__(self, img, target=None):
        return resize(img, target, self.size)

from samgeo.common import *
from huggingface_hub import hf_hub_download
try:
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import box_ops
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

except ImportError:
    print("Installing GroundingDINO...")
    install_package("groundingdino-py")
    print("Please restart the kernel and run the notebook again.")

def load_model_hf(
    repo_id: str, filename: str, ckpt_config_filename: str, device: str = "cpu"
) -> torch.nn.Module:
    """
    Loads a model from HuggingFace Model Hub.

    Args:
        repo_id (str): Repository ID on HuggingFace Model Hub.
        filename (str): Name of the model file in the repository.
        ckpt_config_filename (str): Name of the config file for the model in the repository.
        device (str): Device to load the model onto. Default is 'cpu'.

    Returns:
        torch.nn.Module: The loaded model.
    """

    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename, force_filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    model.to(device)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, force_filename=filename)

    # cache_config_file = "/home/skymap/data/SAM demo/GroundingDINO_SwinT_OGC.py"
    # args = SLConfig.fromfile(cache_config_file)
    # model = build_model(args)
    # model.to(device)
    # cache_file = "/home/skymap/data/SAM demo/groundingdino_swint_ogc.pth"
    
    checkpoint = torch.load(cache_file, map_location="cuda:0")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model
class GoundingDinoBuildSam(LangSAM):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # def build_groundingdino(self):
    #     super().build_groundingdino()
    #     return self.groundingdino
    
    def build_groundingdino(self):
        """Build the GroundingDINO model."""
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        # ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        # ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        ckpt_filename = "groundingdino_swint_ogc.pth"
        ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
        self.groundingdino = load_model_hf(
            ckpt_repo_id, ckpt_filename, ckpt_config_filename, self.device
        )
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
    

    def transform_image(image: Image) -> torch.Tensor:
        """
        Transforms an image using standard transformations for image-based models.

        Args:
            image (Image): The PIL Image to be transformed.

        Returns:
            torch.Tensor: The transformed image as a tensor.
        """
        transform = T.Compose(
            [
                # T.RandomResize([1500], max_size=1500),
                Resize((800, 800)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # image_transformed, _ = transform(image, None)
        image_transformed = transform(image)
        return image_transformed

    def transform_image_batch(self, image: Image) -> torch.Tensor:
        """
        Transforms an image using standard transformations for image-based models.

        Args:
            image (Image): The PIL Image to be transformed.

        Returns:
            torch.Tensor: The transformed image as a tensor.
        """
        transform = T.Compose(
            [
                Resize((800, 800)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # image_transformed = transform(image)
        image_transformed, _ = transform(image, None)
        return image_transformed
    
    from groundingdino.util import box_ops
    def predict_dino(self, image, text_prompt, box_threshold, text_threshold):
        """
        Run the GroundingDINO model prediction.

        Args:
            image (Image): Input PIL Image.
            text_prompt (str): Text prompt for the model.
            box_threshold (float): Box threshold for the prediction.
            text_threshold (float): Text threshold for the prediction.

        Returns:
            tuple: Tuple containing boxes, logits, and phrases.
        """

        image_trans = self.transform_image(image)
        boxes, logits, phrases = self.predict(
            model=self.groundingdino,
            image=image_trans,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        W, H = image.size
        from groundingdino.util import box_ops
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        idx = np.array(np.intersect1d(np.where(boxes[:, 2] < W-50), np.where(boxes[:, 3] < H-50)))
        # raise ValueError(phrases, idx)
        new_boxes =  boxes[idx]        
        new_logits = logits[idx]
        new_phrases = phrases[:len(new_logits)]
        return new_boxes, new_logits, new_phrases
    
    def predict_dino_batch(self, image, text_prompt, box_threshold, text_threshold):
        """
        Run the GroundingDINO model prediction.

        Args:
            image (Image): Input PIL Image.
            text_prompt (str): Text prompt for the model.
            box_threshold (float): Box threshold for the prediction.
            text_threshold (float): Text threshold for the prediction.

        Returns:
            tuple: Tuple containing boxes, logits, and phrases.
        """

        # image_trans = transform_image(image)
        from groundingdino.util.inference import load_image
        # from groundingdino.util.inference import predict_batch
        image_trans = torch.stack([self.transform_image_batch(img) for img in image])
        boxes, logits, phrases = predict_batch(
            model=self.groundingdino,
            image=image_trans,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        #### Transform boxes to xyz coordinates ####
        #### bbox shape = (num_batch, num_box, 4)
        transform_boxes = []
        new_logits = []
        new_phrases = []
        from groundingdino.util import box_ops
        for idx in range (len(boxes)):

            W, H = image[idx].size
            boxes[idx] = box_ops.box_cxcywh_to_xyxy(boxes[idx]) * torch.Tensor([W, H, W, H])

        for idx in range (len(boxes)):
            choose = np.intersect1d(np.where(boxes[idx][:, 2] < W-80), np.where(boxes[idx][:, 3] < H-80)) 
            # choose = np.intersect1d(np.intersect1d(np.intersect1d(np.where(boxes[idx][:, 2] < W-50), np.where(boxes[idx][:, 3] < H-50)), np.where(boxes[idx][:, 0]>0)), np.where(boxes[idx][:, 1]>0))
            index = np.array(choose)

            new_b =  boxes[idx][index]
            new_l = logits[idx][index]
            new_p = phrases[idx][:len(new_l)]
            transform_boxes.append(new_b)
            new_logits.append(new_l)
            new_phrases.append(new_p)
        return transform_boxes, new_logits, new_phrases

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
        image_np = image
        image_pil = Image.fromarray(image_np[:, :, :3])
        
        self.image = image_pil

        boxes, logits, phrases = self.predict_dino(
            image_pil, text_prompt, box_threshold, text_threshold
        )
        # raise ValueError(boxes, logits, phrases)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            # raise ValueError("STOPPPING !!!:", masks.shape)
            masks = masks.squeeze(1)

        if boxes.nelement() == 0:
            # return_results = 0
            mask_overlay = np.zeros((image_pil.width, image_pil.height))
            # No "object" instances found
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

        self.mask_overlay = mask_overlay
        self.masks = masks
        self.boxes = boxes
        self.phrases = phrases
        self.logits = logits
        
        
        if return_results:
            return masks, boxes, phrases, logits, mask_overlay
        
    def prepare_image(self, image, transform, device):
        image = transform.apply_image(np.array(image))
        image = torch.as_tensor(image, device=device.device) 
        return image.permute(2, 0, 1).contiguous()

    def prepare_batched_inputs(self, list_img, boxes, sam_model=None) -> list:
        prepare_input = []
        from segment_anything.utils.transforms import ResizeLongestSide
        resize_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        batched_inputs = []
        for idx, im in enumerate(list_img):
            batched_inputs.append({
                'image': self.prepare_image(im, resize_transform, device=sam_model),
                'boxes': resize_transform.apply_boxes_torch(boxes[idx], np.array(im).shape[:2]).cuda(),
                'original_size': np.array(im).shape[:2]
            })
        return batched_inputs
        
    def predict_v2(
        self,
        image_np,
        text_prompt,
        box_threshold,
        text_threshold,
        fp_img,
        sam_model=None,
        output=None,
        mask_multiplier=255,
        dtype=np.uint8,
        save_args={},
        return_results=False,
        **kwargs,
    ):
        """
        Run both GroundingDINO and SAM model prediction.
        == Batching version ==
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
        image_pil = [Image.fromarray(im[:, :, :3]) for im in image_np]
        
        self.image = image_pil
        # generate list images
        boxes, logits, phrases = self.predict_dino_batch(
            image_pil, text_prompt, box_threshold, text_threshold
        )
        masks = torch.tensor([])
        temp_box = torch.tensor([[0.8755, 0.0536, 0.0539, 0.0472]])
        temp_logits = torch.tensor([0.5])
        temp_phrases = {'tree'}
        all_empty = True
        is_empty =[]
        for cnt in range (len(boxes)):
            if len(boxes[cnt])==0:
                boxes[cnt] = temp_box
                logits[cnt] = temp_logits
                phrases[cnt] = temp_phrases
                is_empty.append(cnt)
            else: all_empty=False

        if not all_empty:
            batched_output = sam_model(self.prepare_batched_inputs(list_img=image_pil, boxes=boxes, sam_model= sam_model), multimask_output=False)
            masks = [batched_output[i]['masks'].squeeze(1) for i in range(len(batched_output))]

        mask_all = []
        for id, (bbox, mak) in enumerate(zip(boxes, masks)):
            if id in is_empty: #if bbox.nelement() == 0:
                # return_results = 0
                mask_overlay = np.zeros((image_pil[id].width, image_pil[id].height))
                # No "object" instances found
                # print("No objects found in the image.")
                pass
            else:
                # Create an empty image to store the mask overlays
                mask_overlay = np.zeros_like(
                    image_np[id][..., 0], dtype=dtype
                )  # Adjusted for single channel

                for i, (box, mask) in enumerate(zip(bbox, mak)):
                    # Convert tensor to numpy array if necessary and ens ure it contains integers
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
            mask_all.append(mask_overlay)


        self.mask_overlay = mask_all
        self.masks = masks
        self.boxes = boxes
        self.phrases = phrases
        self.logits = logits        
        
        if return_results:
            return masks, boxes, phrases, logits, mask_all

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
        
        if width*height > 1300*1300:
            window_size = find_window(width, height)
            if chose_overlap:
                overlap = (window_size)//5
            else:
                overlap = 0
            with rasterio.open(out_fp, 'w', **meta) as dst:
                for col in tqdm(range(0, width, window_size - overlap)):
                    for row in tqdm(range(0, height, window_size - overlap)):
                        width_win = min(window_size, width - col)
                        height_win = min(window_size, height - row)
                        window = rasterio.windows.Window(col, row, width_win, height_win)
                        # print(col, row, width_win, height_win)
                        data = src.read(window=window)[0:3].transpose((1, 2, 0))
                        # meta.update({'width':1495, 'height':1495})
                        # with rasterio.open("/home/skymap/data/SAM demo/test/test_london.tif", "w", **meta) as test:
                        #     test.write(data[:,:,0], window=window, indexes=1)
                        #     test.write(data[:,:,1], window=window, indexes=2)
                        #     test.write(data[:,:,2], window=window, indexes=3)
                        #     raise ValueError(meta)
                        if np.all(data == 0):
                            mask_overlay = np.zeros((width_win, height_win))
                        else:
                            # try:
                                # print(data.shape)
                            _, _, _, _, mask_overlay = model_sam.predict(data, text_prompt, box_threshold=0.2, text_threshold=0.2, fp_img=fp_img,return_results=True)
                            model_sam
                            mask_overlay
                            # except:
                                # mask_overlay = np.zeros((width_win, height_win))
                        # raise ValueError("data shp:", data.shape, "\n mask shp:", mask_overlay.shape)
                        dst.write(mask_overlay, window=window, indexes=1)
        else:
            data = src.read()[0:3].transpose((1, 2, 0))
            # print(data.shape)
            # try:
            _, _, _, _, mask_overlay = model_sam.predict(data, text_prompt, box_threshold=0.2, text_threshold=0.2, fp_img=fp_img, return_results=True)
            with rasterio.open(out_fp, 'w', **meta) as dst:
                dst.write(mask_overlay, indexes=1)
        ## Run morphology
            # except:
            #     pass

@torch.no_grad()           
def read_image_in_windows_and_predict_v2(out_fp, fp_img, model_sam, text_prompt, chose_overlap=False, batch_size=2, sam_model=None):
    with rasterio.open(fp_img) as src:
        width = src.width
        height = src.height
        meta = src.meta
        meta.update({'count':1})
        
        if width*height > 1500*1500:
            window_size = find_window(width, height)
            if chose_overlap:
                overlap = (window_size)//5
            else:
                overlap = 0
            with rasterio.open(out_fp, 'w', **meta) as dst:
                stack_win = []
                count = 0
                for col in tqdm(range(0, width, window_size - overlap)):
                    for row in tqdm(range(0, height, window_size - overlap)):
                        ###################################################################################
                        # If this is the last batch and the num windows is odd, then process single element
                        ###################################################################################
                        width_win = min(window_size, width - col)
                        height_win = min(window_size, height - row)
                        stack_win.append(rasterio.windows.Window(col, row, width_win, height_win))
                        count+=1
                        if count==batch_size:
                            count = 0
                            data = [src.read(window=win)[0:3].transpose((1, 2, 0)) for win in stack_win]
                            _, _, _, _, mask_overlay= model_sam.predict_v2(data, text_prompt, box_threshold=0.24, text_threshold=0.24, fp_img=fp_img,return_results=True, sam_model=sam_model)
                            model_sam
                            mask_overlay
                            for idx, mask in enumerate(mask_overlay):
                                dst.write(mask, window=stack_win[idx], indexes=1)
                            
                            del mask_overlay                            
                            stack_win = []
        else:
            data = src.read()[0:3].transpose((1, 2, 0))
            _, _, _, _, mask_overlay = model_sam.predict(data, text_prompt, box_threshold=0.24, text_threshold=0.24, fp_img=fp_img, return_results=True)
            with rasterio.open(out_fp, 'w', **meta) as dst:
                dst.write(mask_overlay, indexes=1)
   


import time

def main_sam_text(model_build_groundingdino, model_build_sam, fp_image, text_prompt, tmp_dir, fp_out_shp, batch_size, sam_model):
    overlab = 0
    sam = LangSAM_update(model_build_groundingdino, model_build_sam)
    os.makedirs(tmp_dir, exist_ok=True)
    fp_out_img_tmp = os.path.join(tmp_dir, os.path.basename(fp_image))
    overlab = True
    if batch_size == 1 or batch_size==None:
        read_image_in_windows_and_predict(fp_out_img_tmp, fp_image, sam, text_prompt, overlab)
    else:
        read_image_in_windows_and_predict_v2(fp_out_img_tmp, fp_image, sam, text_prompt, overlab, batch_size=batch_size, sam_model=sam_model)
    global x3
    x3 = time.time()
    print("DONE INFERING, SAVING TO SHAPEFILE!")
    try:
        os.makedirs(os.path.dirname(fp_out_shp), exist_ok=True)
        sam.raster_to_vector(fp_out_img_tmp, fp_out_shp)
        return 1
    except:
        return 0

if __name__=="__main__":
    # model_path_sam = r'/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/weights/sam_vit_h_4b8939.pth'
    # # input_path = r'/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/example-data/a_4326.tif'
    # input_path = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM_API_INTERACTIVE/upload/London_c70aa95385a54292ba924c20f233904d_gg.tif'
    # text_prompts = ['tree','car']
    # output_path = r'/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/RS/qwqwqqq.shp'
    # tmp_dir = r'/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/tmp/qwqqwqqww'
    
    ###########################################################
    ############ TESTING AREA #################################
    model_path_sam = r'/home/skymap/data/SAM demo/sam_vit_b_01ec64.pth'
    input_path = r'/home/skymap/data/SAM demo/test/London-7a12a3c8071d4b8d98d2887dbc969d32.tif' #London-7a12a3c8071d4b8d98d2887dbc969d32
    # text_prompts = ['house. building. polygon. rooftop. shape. resident. construction.']
    # text_prompts = [' cars. buses. vehicles. trucks. container.']
    text_prompts = ['car']
    output_path = r'/home/skymap/data/SAM demo/test/London-7a12a3c8071d4b8d98d2887dbc969d32_car_batch.shp'
    tmp_dir = r'/home/skymap/data/SAM demo/test/tmp'
    ###########################################################
    ###########################################################
    
    x1 = time.time()
    model_diano = GoundingDinoBuildSam()
    model_diano_ = model_diano.build_groundingdino()
    # sam_vit_h_4b8939
    # sam_vit_l_0b3195
    # sam_vit_b_01ec64
    model_type = 'vit_b'
    sam_model = sam_model_registry[model_type](checkpoint=model_path_sam)
    sam_model.to(device=device_sam)
    # sam_model = SamPredictor(sam_model)
    sam_predict = SamPredictor(sam_model)
    batch_size = 2

    global x3
    x3 = 0
    x2 = time.time()
    print("DONE LOADING!!!")
    for text_prompt in text_prompts:
        main_sam_text(model_diano_, sam_predict, input_path, text_prompt, tmp_dir, output_path, batch_size = batch_size, sam_model=sam_model)
    print("TEXT PROMPTS:", text_prompts)
    print("Load model estimate time:",(x2-x1), "seconds")
    print("Inference estimate time:",(x3-x2), "seconds")
    print("Total estimate time:",(time.time()-x1), "seconds")
