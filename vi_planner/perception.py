"""perception module
"""

from typing import Dict

import cv2
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Mask2FormerForUniversalSegmentation,
)

from .config.coco_sem_meta import COCO_CATEGORIES, get_class_for_id_mmdet
from .config.viplanner_sem_meta import VIPlannerSemMetaHandler

dataset_meta = []

for row in COCO_CATEGORIES:
    dataset_meta.append(row["name"])

coco_viplanner_cls_mapping = get_class_for_id_mmdet(dataset_meta)

viplanner_meta = VIPlannerSemMetaHandler()
viplanner_sem_class_color_map = viplanner_meta.class_color

coco_viplanner_color_mapping = {}
for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items():
    coco_viplanner_color_mapping[coco_id] = viplanner_meta.class_color[
        viplanner_cls_name
    ]


def create_coco_label_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for index, row in enumerate(COCO_CATEGORIES):
        colormap[index] = row["color"]
    return colormap


def create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.

    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap


def semantic_to_rgb(pred_semantic_map: np.ndarray) -> np.ndarray:
    panoptic_mask = np.zeros(
        (pred_semantic_map.shape[0], pred_semantic_map.shape[1], 3),
        dtype=np.uint8,
    )
    for curr_sem_class in np.unique(pred_semantic_map):
        try:
            panoptic_mask[pred_semantic_map == curr_sem_class] = (
                coco_viplanner_color_mapping[curr_sem_class]
            )
        except KeyError:
            panoptic_mask[pred_semantic_map == curr_sem_class] = (
                viplanner_sem_class_color_map["static"]
            )
    return panoptic_mask


def depth_to_rgb(depth_map):
    depth_min, depth_max = np.min(depth_map), np.max(depth_map)
    if depth_max > depth_min:  # Avoid division by zero
        normalized_depth_map = (
            (depth_map - depth_min) / (depth_max - depth_min) * 255.0
        )
    else:
        normalized_depth_map = np.zeros_like(depth_map)

    # Convert to uint8
    normalized_depth_map_uint8 = np.uint8(normalized_depth_map)

    # Apply JET colormap
    color_mapped_image = cv2.applyColorMap(
        normalized_depth_map_uint8, cv2.COLORMAP_JET
    )

    return color_mapped_image


class SemDepthPerception:

    def __init__(self):
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        #########################################################
        # Depth
        # DepthAnything
        # depth_model = "LiheYoung/depth-anything-large-hf"
        depth_model = "LiheYoung/depth-anything-small-hf"
        depth_transform = AutoImageProcessor.from_pretrained(
            depth_model,
            device=device,
        )
        depth_estimator = AutoModelForDepthEstimation.from_pretrained(
            depth_model,
            device_map=device,
        )
        #########################################################
        # Semantics
        # sem_model = "facebook/mask2former-swin-large-cityscapes-semantic"
        sem_model = "facebook/mask2former-swin-base-coco-panoptic"
        model_mask2former = (
            Mask2FormerForUniversalSegmentation.from_pretrained(
                sem_model,
                device_map=device,
            )
        )
        image_processor = AutoImageProcessor.from_pretrained(
            sem_model,
            device=device,
        )
        #########################################################
        self.device = device
        self.depth_transform = depth_transform
        self.depth_estimator = depth_estimator
        self.model_mask2former = model_mask2former
        self.image_processor = image_processor
        print("Done loading perception models")

    @torch.no_grad()
    def infer(self, rgb_img_pil) -> Dict:
        # np image
        img = np.array(rgb_img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Depth
        # depth_outputs = depth_estimator(rgb_img_pil)
        # depth = depth_outputs.depth
        # depth_np = depth_estimator.infer_pil(rgb_img_pil)  # as numpy
        depth_input_batch = self.depth_transform(
            images=rgb_img_pil, return_tensors="pt"
        )
        depth_input_batch["pixel_values"] = depth_input_batch[
            "pixel_values"
        ].to(device=self.device)
        depth_prediction = self.depth_estimator(
            **depth_input_batch
        ).predicted_depth

        depth_prediction = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Semantics
        semantics_inputs = self.image_processor(
            rgb_img_pil, return_tensors="pt"
        )
        semantics_inputs["pixel_values"] = semantics_inputs["pixel_values"].to(
            device=self.device
        )
        semantics_outputs = self.model_mask2former(**semantics_inputs)

        # (H, W)
        pred_semantic_map = (
            self.image_processor.post_process_semantic_segmentation(
                semantics_outputs, target_sizes=[rgb_img_pil.size[::-1]]
            )[0]
        )

        disparity_np = depth_prediction.cpu().numpy()

        max_disparity = 100.0
        min_disparity = 0.001
        disparity_np[disparity_np > max_disparity] = -1
        disparity_np[disparity_np <= min_disparity] = -1

        depth_np = 1.0 / disparity_np
        depth_np *= 25.0

        # Thresholding
        max_depth = 10.0  # Same as original implementation
        min_depth = 1 / max_disparity
        depth_np[depth_np > max_depth] = max_depth
        depth_np[depth_np < min_depth] = max_depth
        depth_np[~np.isfinite(depth_np)] = max_depth

        disparity_np = 1.0 / depth_np

        pred_semantic_map_np = pred_semantic_map.cpu().numpy()

        pred_semantic_map_np_rgb = semantic_to_rgb(pred_semantic_map_np)

        depth_np_rgb = depth_to_rgb(disparity_np)

        frame_data = {
            "rgb": img,
            "depth": depth_np,
            "depth_rgb": depth_np_rgb,
            "semantics": pred_semantic_map_np,
            "semantics_rgb": pred_semantic_map_np_rgb,
        }

        return frame_data
