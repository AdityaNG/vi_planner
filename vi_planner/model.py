import os
from typing import Dict, Tuple

import cv2
import gdown
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from .perception import SemDepthPerception
from .settings import VIPLANNER_CACHE_DIR
from .trajectory_plot import (
    estimate_intrinsics,
    overlay_image_by_semantics,
    plot_trajectories,
)
from .vip_inference import VIPlannerInference


class PlannerConfig:
    def __init__(self, model_dir, m2f_config):
        self.model_save = model_dir
        self.m2f_config_path = m2f_config


def download_model():
    """
    Check if VIPLANNER_CACHE_DIR has the model and yaml, if not, download
    """

    cache_dir = VIPLANNER_CACHE_DIR
    model_path = os.path.join(cache_dir, "model.pt")
    yaml_path = os.path.join(cache_dir, "model.yaml")

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Download files if they don't exist
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        gdown.download(
            "https://drive.google.com/uc?id=1PY7XBkyIGESjdh1cMSiJgwwaIT0WaxIc",
            model_path,
            quiet=False,
        )

    if not os.path.exists(yaml_path):
        print("Downloading model config...")
        gdown.download(
            "https://drive.google.com/uc?id=1r1yhNQAJnjpn9-xpAQWGaQedwma5zokr",
            yaml_path,
            quiet=False,
        )


class VIPlanner:

    def __init__(
        self,
        fov_x: float = 100,
        fov_y: float = 100,
        height: int = 256,
        width: int = 512,
        offsets: Tuple[float, float, float] = (0, 1.0, 0),
        rotation: Tuple[float, float, float] = (0, 0, 0),
    ):
        download_model()
        self.model_dir = VIPLANNER_CACHE_DIR
        self.intrinsic_matrix = estimate_intrinsics(
            fov_x=fov_x,
            fov_y=fov_y,
            height=height,
            width=width,
        )
        self.extrinsic_matrix = np.eye(4)

        self.extrinsic_matrix[:3, 3] = offsets
        yaw, pitch, roll = rotation
        rotation_mat = R.from_euler(
            "ZYX", [yaw, pitch, roll], degrees=True
        ).as_matrix()
        self.extrinsic_matrix[:3, :3] = rotation_mat

        cfg = PlannerConfig(self.model_dir, "")

        self.planner = VIPlannerInference(cfg)
        self.perception = SemDepthPerception()

    @torch.no_grad()
    def run(
        self,
        image: np.ndarray,
        goal: Tuple[float, float, float],
    ):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        frame_pil = Image.fromarray(frame_rgb)

        perception_outputs = self.perception.infer(frame_pil)
        depth_viz = perception_outputs["depth_rgb"]
        sem_viz = perception_outputs["semantics_rgb"]

        # Create depth image
        depth = perception_outputs["depth"]

        goal_tensor = torch.tensor([goal], dtype=torch.float32)
        # Run planner
        if self.planner.train_cfg.sem or self.planner.train_cfg.rgb:
            trajectory, fear = self.planner.plan(depth, sem_viz, goal_tensor)
        else:
            trajectory, fear = self.planner.plan_depth(depth, goal_tensor)

        trajectory_cam = trajectory[:, [1, 2, 0]]
        trajectory_cam[:, 0] *= -1
        trajectory_cam[:, 1] *= -1

        model_output = dict(
            trajectory=trajectory_cam,
            depth=depth_viz,
            sem=sem_viz,
            fear=fear,
        )
        return model_output

    def visualize(
        self,
        image: np.ndarray,
        model_output: Dict,
    ):

        trajectory_cam = model_output["trajectory"]
        depth_viz = model_output["depth"]
        sem_viz = model_output["sem"]
        fear = model_output["fear"]
        # Visualize results
        vis_frame = image.copy()

        trajectory_cam[:, 1] = 0.0
        plot_trajectories(
            frame_img=vis_frame,
            trajectories=[trajectory_cam],
            intrinsic_matrix=self.intrinsic_matrix,
            extrinsic_matrix=self.extrinsic_matrix,
            line=True,
            track=False,
            draw_grid=True,
            interpolation_samples=0,
            grid_range_img=(2, 8),
        )

        vis_frame = overlay_image_by_semantics(
            vis_frame,
            image,
            sem_viz,
            [
                (0, 255, 0),
                (255, 128, 0),
                # (0, 0, 255,),
            ],
        )

        # Add fear value text to visualization
        fear_text = f"Fear: {float(fear):.2f}"
        cv2.putText(
            vis_frame,
            fear_text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        bottom = np.hstack((depth_viz, sem_viz))
        bottom = cv2.resize(bottom, (0, 0), fx=0.5, fy=0.5)
        vis_frame = np.vstack((vis_frame, bottom))

        return vis_frame
