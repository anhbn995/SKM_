"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import numpy as np
import model.utils as utils
import math
import cv2
from tensorrtserver.api import InferContext
from tqdm import *
from tensorrtserver.api import *


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
          int(math.ceil(image_shape[1] / stride))]
         for stride in config.BACKBONE_STRIDES])


class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(
                a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(
                masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def check_intersec(self, contour0, input_size, overlapsize):
        padding = int((input_size - overlapsize) / 2)
        cnt1 = np.array(
            [[padding, padding], [padding, input_size - padding], [input_size - padding, input_size - padding],
             [input_size - padding, padding]])
        contour1 = np.array(cnt1.reshape(-1, 1, 2), dtype=np.int32)
        img1 = np.zeros((input_size, input_size)).astype(np.uint8)
        img1 = cv2.fillConvexPoly(img1, contour1, 255)
        img = np.zeros((input_size, input_size)).astype(np.uint8)
        img = cv2.fillConvexPoly(img, contour0, 255)
        img_result = cv2.bitwise_and(img1, img)
        contours_rs, hierarchy = cv2.findContours(
            img_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            if cv2.contourArea(contours_rs[0]) / float(cv2.contourArea(contour0)) > 0.50:
                return True
            else:
                return False
        except Exception:
            return False

    def preprocess_images(self, images, verbose):
        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(
            anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)

        input_image = molded_images.astype(np.float32)
        input_image_meta = image_metas.astype(np.float32)
        input_anchors = anchors.astype(np.float32)

        return input_image, input_image_meta, input_anchors, windows

    def postprocess(self, images, detections, mrcnn_mask, molded_images, windows):
        # Process detections
        pos_results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            pos_results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })

        return pos_results

    def detect_with_trtis(self, images, infer_ctx, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.
        infer_ctx: Inference context (for TensorRT Inference Server)

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(
            anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)

        input_image = molded_images.astype(np.float32)
        input_image_meta = image_metas.astype(np.float32)
        input_anchors = anchors.astype(np.float32)

        # Run object detection
        # detections, _, _, mrcnn_mask, _, _, _ =\
        #     self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        trtis_result = infer_ctx.run({
            'input_image': (*input_image,),
            'input_image_meta': (*input_image_meta,),
            'input_anchors': (*input_anchors,)
        },
            {
                'mrcnn_detection/Reshape_1': InferContext.ResultFormat.RAW,
                'mrcnn_mask/Reshape_1': InferContext.ResultFormat.RAW
        },
            self.config.BATCH_SIZE
        )

        detections = trtis_result['mrcnn_detection/Reshape_1']
        mrcnn_mask = trtis_result['mrcnn_mask/Reshape_1']

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def detect_with_trtis_v2(self, dataset_image, num_band, input_size, crop_size, infer_ctx, async_set, verbose=0,
                             on_processing=None):
        """Runs the detection pipeline.

        dataset_image: GDAL generated object.
        num_band: The number of bands of input images.
        input_size: The size of each input images.
        crop_size:
        infer_ctx: Inference context (for TensorRT Inference Server)
        async_set: Whether or not to use asynchronous requests.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        assert self.mode == "inference", "Create model in inference mode."
        # assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        w, h = dataset_image.RasterXSize, dataset_image.RasterYSize

        if h <= input_size or w <= input_size:
            image = dataset_image.ReadAsArray()[0:num_band].swapaxes(
                0, 1).swapaxes(1, 2).astype(np.uint8)
            image_res = 0.3

            self.config.IMAGE_MAX_DIM = (
                round(max(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64
            self.config.IMAGE_MIN_DIM = (
                round(min(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64

            images = [image] * self.config.BATCH_SIZE
            input_image, input_image_meta, input_anchors, windows = self.preprocess_images(
                images, verbose)

            trtis_result = infer_ctx.run({
                'input_image': (*input_image,),
                'input_image_meta': (*input_image_meta,),
                'input_anchors': (*input_anchors,)
            },
                {
                    'mrcnn_detection/Reshape_1': InferContext.ResultFormat.RAW,
                    'mrcnn_mask/Reshape_1': InferContext.ResultFormat.RAW
            },
                self.config.BATCH_SIZE
            )

            detections = trtis_result['mrcnn_detection/Reshape_1']
            mrcnn_mask = trtis_result['mrcnn_mask/Reshape_1']

            pos_results = self.postprocess(
                images, detections, mrcnn_mask, input_image, windows)

            p = pos_results[0]
            # print(p['masks'].shape)
            list_contours = []
            for i in range(p['masks'].shape[2]):
                mask = p['masks'][:, :, i].astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                try:
                    if cv2.contourArea(contours[0]) > 100:
                        list_contours.append(contours[0])
                except Exception:
                    pass
        else:

            return_contour = []
            padding = int((input_size - crop_size) / 2)
            new_w = w + 2 * padding
            new_h = h + 2 * padding
            cut_w = list(range(padding, new_w - padding, crop_size))
            cut_h = list(range(padding, new_h - padding, crop_size))
            list_hight = []
            list_weight = []
            # print(w,h)
            for i in cut_h:
                list_hight.append(i)

            for i in cut_w:
                list_weight.append(i)

            results = []
            request_ids = []

            with tqdm(total=len(list_hight) * len(list_weight)) as pbar:
                for i in range(len(list_hight)):
                    # on_processing((i + 1) / (2 * len(list_hight)))
                    start_y = list_hight[i]
                    for j in range(len(list_weight)):
                        start_x = list_weight[j]
                        startx = start_x - padding
                        endx = min(start_x + crop_size +
                                   padding, new_w - padding)
                        starty = start_y - padding
                        endy = min(start_y + crop_size +
                                   padding, new_h - padding)
                        if startx == 0:
                            xoff = startx
                        else:
                            xoff = startx - padding
                        if starty == 0:
                            yoff = starty
                        else:
                            yoff = starty - padding
                        xcount = endx - padding - xoff
                        ycount = endy - padding - yoff
                        # print(xoff, yoff, xcount, ycount)
                        image_detect = dataset_image.ReadAsArray(xoff, yoff, xcount, ycount)[0:num_band].swapaxes(0,
                                                                                                                  1).swapaxes(
                            1, 2)
                        if image_detect.shape[0] < input_size or image_detect.shape[1] < input_size:
                            img_temp = np.zeros(
                                (input_size, input_size, image_detect.shape[2]))
                            if (startx == 0 and starty == 0):
                                img_temp[(input_size - image_detect.shape[0]):input_size,
                                         (input_size - image_detect.shape[1]):input_size] = image_detect
                            elif startx == 0:
                                img_temp[0:image_detect.shape[0],
                                         (input_size - image_detect.shape[1]):input_size] = image_detect
                            elif starty == 0:
                                img_temp[(input_size - image_detect.shape[0]):input_size,
                                         0:image_detect.shape[1]] = image_detect
                            else:
                                img_temp[0:image_detect.shape[0],
                                         0:image_detect.shape[1]] = image_detect

                            image_detect = img_temp
                        if np.count_nonzero(image_detect) > 0:
                            images = [image_detect] * self.config.BATCH_SIZE
                            input_image, input_image_meta, input_anchors, windows = self.preprocess_images(images,
                                                                                                           verbose)

                            kwargs = {
                                'inputs': {
                                    'input_image': (*input_image,),
                                    'input_image_meta': (*input_image_meta,),
                                    'input_anchors': (*input_anchors,)
                                },
                                'outputs': {
                                    'mrcnn_detection/Reshape_1': InferContext.ResultFormat.RAW,
                                    'mrcnn_mask/Reshape_1': InferContext.ResultFormat.RAW
                                },
                                'batch_size': self.config.BATCH_SIZE
                            }

                            if not async_set:

                                results.append(infer_ctx.run(**kwargs))
                            else:
                                request_ids.append(
                                    infer_ctx.async_run(**kwargs))

                        pbar.update()

            # For async, retrieve results according to the send order
            if async_set:
                for request_id in tqdm(request_ids):
                    results.append(
                        infer_ctx.get_async_run_results(request_id, True))

            results_iter = iter(results)

            with tqdm(total=len(list_hight) * len(list_weight)) as pbar:
                for i in range(len(list_hight)):
                    # on_processing((i + 1) / (2 * len(list_hight)) + 0.5)
                    start_y = list_hight[i]
                    for j in range(len(list_weight)):

                        start_x = list_weight[j]
                        startx = start_x - padding
                        endx = min(start_x + crop_size +
                                   padding, new_w - padding)
                        starty = start_y - padding
                        endy = min(start_y + crop_size +
                                   padding, new_h - padding)
                        if startx == 0:
                            xoff = startx
                        else:
                            xoff = startx - padding
                        if starty == 0:
                            yoff = starty
                        else:
                            yoff = starty - padding
                        xcount = endx - padding - xoff
                        ycount = endy - padding - yoff
                        # print(xoff, yoff, xcount, ycount)
                        image_detect = dataset_image.ReadAsArray(xoff, yoff, xcount, ycount)[0:num_band].swapaxes(0,
                                                                                                                  1).swapaxes(
                            1, 2)
                        if image_detect.shape[0] < input_size or image_detect.shape[1] < input_size:
                            img_temp = np.zeros(
                                (input_size, input_size, image_detect.shape[2]))
                            if (startx == 0 and starty == 0):
                                img_temp[(input_size - image_detect.shape[0]):input_size,
                                         (input_size - image_detect.shape[1]):input_size] = image_detect
                            elif startx == 0:
                                img_temp[0:image_detect.shape[0],
                                         (input_size - image_detect.shape[1]):input_size] = image_detect
                            elif starty == 0:
                                img_temp[(input_size - image_detect.shape[0]):input_size,
                                         0:image_detect.shape[1]] = image_detect
                            else:
                                img_temp[0:image_detect.shape[0],
                                         0:image_detect.shape[1]] = image_detect

                            image_detect = img_temp
                        if np.count_nonzero(image_detect) > 0:
                            trtis_result = next(results_iter)
                            detections = trtis_result['mrcnn_detection/Reshape_1']
                            mrcnn_mask = trtis_result['mrcnn_mask/Reshape_1']

                            pos_results = self.postprocess(
                                images, detections, mrcnn_mask, input_image, windows)

                            p = pos_results[0]
                            # print(p['masks'].shape)
                            list_temp = []

                            for i in range(p['masks'].shape[2]):
                                mask = p['masks'][:, :, i].astype(np.uint8)
                                contours, _ = cv2.findContours(
                                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                try:
                                    if cv2.contourArea(contours[0]) > 10:
                                        if (contours[0].max() < (input_size - padding)) and (
                                                contours[0].min() > padding):
                                            list_temp.append(contours[0])
                                        elif (contours[0].max() < (input_size - 5)) and (
                                                contours[0].min() > 5) and self.check_intersec(contours[0], input_size,
                                                                                               crop_size):
                                            list_temp.append(contours[0])
                                except Exception as e:
                                    print(e)
                                    pass

                            temp_contour = []
                            for contour in list_temp:
                                anh = contour.reshape(-1, 2)
                                anh2 = anh + \
                                    np.array(
                                        [startx - padding, starty - padding])
                                con_rs = anh2.reshape(-1, 1, 2)
                                temp_contour.append(con_rs)
                            return_contour.extend(temp_contour)
                        pbar.update()
                list_contours = return_contour

        return list_contours

    def detect_with_trtis_trees(self, dataset_image, num_band, input_size, crop_size, infer_ctx, async_set, verbose=0,
                                on_processing=None):
        """Runs the detection pipeline.

        dataset_image: GDAL generated object.
        num_band: The number of bands of input images.
        input_size: The size of each input images.
        crop_size:
        infer_ctx: Inference context (for TensorRT Inference Server)
        async_set: Whether or not to use asynchronous requests.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        assert self.mode == "inference", "Create model in inference mode."
        # assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        w, h = dataset_image.RasterXSize, dataset_image.RasterYSize

        if h <= input_size or w <= input_size:
            image = dataset_image.ReadAsArray()[0:num_band].swapaxes(
                0, 1).swapaxes(1, 2).astype(np.uint8)
            image_res = 0.3

            self.config.IMAGE_MAX_DIM = (
                round(max(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64
            self.config.IMAGE_MIN_DIM = (
                round(min(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64

            images = [image] * self.config.BATCH_SIZE
            input_image, input_image_meta, input_anchors, windows = self.preprocess_images(
                images, verbose)

            trtis_result = infer_ctx.run({
                'input_image': (*input_image,),
                'input_image_meta': (*input_image_meta,),
                'input_anchors': (*input_anchors,)
            },
                {
                    'mrcnn_detection/Reshape_1': InferContext.ResultFormat.RAW,
                    'mrcnn_mask/Reshape_1': InferContext.ResultFormat.RAW
            },
                self.config.BATCH_SIZE
            )

            detections = trtis_result['mrcnn_detection/Reshape_1']
            mrcnn_mask = trtis_result['mrcnn_mask/Reshape_1']

            pos_results = self.postprocess(
                images, detections, mrcnn_mask, input_image, windows)

            p = pos_results[0]
            # print(p['masks'].shape)
            list_contours = []
            for i in range(p['masks'].shape[2]):
                mask = p['masks'][:, :, i].astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                try:
                    if cv2.contourArea(contours[0]) > 100:
                        list_contours.append(contours[0])
                except Exception:
                    pass
        else:

            return_contour = []
            padding = int((input_size - crop_size) / 2)
            new_w = w + 2 * padding
            new_h = h + 2 * padding
            cut_w = list(range(padding, new_w - padding, crop_size))
            cut_h = list(range(padding, new_h - padding, crop_size))
            list_hight = []
            list_weight = []
            # print(w,h)
            for i in cut_h:
                list_hight.append(i)

            for i in cut_w:
                list_weight.append(i)

            results = []
            request_ids = []

            with tqdm(total=len(list_hight) * len(list_weight)) as pbar:
                for i in range(len(list_hight)):
                    # on_processing((i + 1) / (2 * len(list_hight)))
                    start_y = list_hight[i]
                    for j in range(len(list_weight)):
                        start_x = list_weight[j]
                        startx = start_x - padding
                        endx = min(start_x + crop_size +
                                   padding, new_w - padding)
                        starty = start_y - padding
                        endy = min(start_y + crop_size +
                                   padding, new_h - padding)
                        if startx == 0:
                            xoff = startx
                        else:
                            xoff = startx - padding
                        if starty == 0:
                            yoff = starty
                        else:
                            yoff = starty - padding
                        xcount = endx - padding - xoff
                        ycount = endy - padding - yoff
                        # print(xoff, yoff, xcount, ycount)
                        image_detect = np.zeros((512, 512, 4))
                        try:
                            image_detect = dataset_image.ReadAsArray(xoff, yoff, xcount, ycount)[0:num_band].swapaxes(0,
                                                                                                                      1).swapaxes(1, 2)
                        except Exception as e:
                            print(e)
                        if image_detect.shape[0] < input_size or image_detect.shape[1] < input_size:
                            img_temp = np.zeros(
                                (input_size, input_size, image_detect.shape[2]))
                            if (startx == 0 and starty == 0):
                                img_temp[(input_size - image_detect.shape[0]):input_size,
                                         (input_size - image_detect.shape[1]):input_size] = image_detect
                            elif startx == 0:
                                img_temp[0:image_detect.shape[0],
                                         (input_size - image_detect.shape[1]):input_size] = image_detect
                            elif starty == 0:
                                img_temp[(input_size - image_detect.shape[0]):input_size,
                                         0:image_detect.shape[1]] = image_detect
                            else:
                                img_temp[0:image_detect.shape[0],
                                         0:image_detect.shape[1]] = image_detect

                            image_detect = img_temp
                        if np.count_nonzero(image_detect) > 0:
                            images = [image_detect] * self.config.BATCH_SIZE
                            input_image, input_image_meta, input_anchors, windows = self.preprocess_images(images,
                                                                                                           verbose)

                            kwargs = {
                                'inputs': {
                                    'input_image': (*input_image,),
                                    'input_image_meta': (*input_image_meta,),
                                    'input_anchors': (*input_anchors,)
                                },
                                'outputs': {
                                    'mrcnn_detection/Reshape_1': InferContext.ResultFormat.RAW,
                                    'mrcnn_mask/Reshape_1': InferContext.ResultFormat.RAW
                                },
                                'batch_size': self.config.BATCH_SIZE
                            }

                            if not async_set:

                                results.append(infer_ctx.run(**kwargs))
                            else:
                                request_ids.append(
                                    infer_ctx.async_run(**kwargs))

                        pbar.update()

            # For async, retrieve results according to the send order
            if async_set:
                for request_id in tqdm(request_ids):
                    results.append(
                        infer_ctx.get_async_run_results(request_id, True))

            results_iter = iter(results)

            with tqdm(total=len(list_hight) * len(list_weight)) as pbar:
                for i in range(len(list_hight)):
                    # on_processing((i + 1) / (2 * len(list_hight)) + 0.5)
                    start_y = list_hight[i]
                    for j in range(len(list_weight)):

                        start_x = list_weight[j]
                        startx = start_x - padding
                        endx = min(start_x + crop_size +
                                   padding, new_w - padding)
                        starty = start_y - padding
                        endy = min(start_y + crop_size +
                                   padding, new_h - padding)
                        if startx == 0:
                            xoff = startx
                        else:
                            xoff = startx - padding
                        if starty == 0:
                            yoff = starty
                        else:
                            yoff = starty - padding
                        xcount = endx - padding - xoff
                        ycount = endy - padding - yoff
                        # print(xoff, yoff, xcount, ycount)
                        image_detect = np.zeros((512, 512, 4))
                        try:
                            image_detect = dataset_image.ReadAsArray(xoff, yoff, xcount, ycount)[0:num_band].swapaxes(0,
                                                                                                                      1).swapaxes(1, 2)
                        except Exception as e:
                            print(e)
                        if image_detect.shape[0] < input_size or image_detect.shape[1] < input_size:
                            img_temp = np.zeros(
                                (input_size, input_size, image_detect.shape[2]))
                            if (startx == 0 and starty == 0):
                                img_temp[(input_size - image_detect.shape[0]):input_size,
                                         (input_size - image_detect.shape[1]):input_size] = image_detect
                            elif startx == 0:
                                img_temp[0:image_detect.shape[0],
                                         (input_size - image_detect.shape[1]):input_size] = image_detect
                            elif starty == 0:
                                img_temp[(input_size - image_detect.shape[0]):input_size,
                                         0:image_detect.shape[1]] = image_detect
                            else:
                                img_temp[0:image_detect.shape[0],
                                         0:image_detect.shape[1]] = image_detect

                            image_detect = img_temp
                        if np.count_nonzero(image_detect) > 0:
                            trtis_result = next(results_iter)
                            detections = trtis_result['mrcnn_detection/Reshape_1']
                            mrcnn_mask = trtis_result['mrcnn_mask/Reshape_1']

                            pos_results = self.postprocess(
                                images, detections, mrcnn_mask, input_image, windows)

                            p = pos_results[0]

                            boxes = p['rois']
                            N = boxes.shape[0]
                            list_temp = []

                            for i in range(N):
                                if not np.any(boxes[i]):
                                    continue
                                y1, x1, y2, x2 = boxes[i]
                                contour = np.array(
                                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
                                contour = contour.reshape(-1, 1, 2)
                                try:
                                    if cv2.contourArea(contour) > 10:
                                        if (contour.max() < (input_size-padding)) and (contour.min() > padding):
                                            # print(1)
                                            list_temp.append(contour)
                                        elif (contour.max() < (input_size-5)) and (contour.min() > 5) and self.check_intersec(contour, input_size, crop_size):
                                            list_temp.append(contour)
                                except Exception:
                                    pass

                            temp_contour = []
                            for contour in list_temp:
                                anh = contour.reshape(-1, 2)
                                anh2 = anh + \
                                    np.array(
                                        [startx - padding, starty - padding])
                                con_rs = anh2.reshape(-1, 1, 2)
                                temp_contour.append(con_rs)
                            return_contour.extend(temp_contour)
                        pbar.update()
                list_contours = return_contour

        return list_contours


############################################################
#  Data Formatting
############################################################


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +  # size=1
        list(active_class_ids)  # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)
