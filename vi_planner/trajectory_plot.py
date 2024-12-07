"""

trajectory plotting utils: https://github.com/AdityaNG/general-navigation/blob/306fd4eed07a54b0fbc5b6df0ecd1dc78f8ba497/general_navigation/models/model_utils.py  # noqa

"""

import hashlib
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union, cast

import cv2
import numpy as np


def estimate_intrinsics(
    fov_x: float,  # degrees
    fov_y: float,  # degrees
    height: int,  # pixels
    width: int,  # pixels
) -> np.ndarray:
    """
    The intrinsic matrix can be extimated from the FOV and image dimensions

    :param fov_x: FOV on x axis in degrees
    :type fov_x: float
    :param fov_y: FOV on y axis in degrees
    :type fov_y: float
    :param height: Height in pixels
    :type height: int
    :param width: Width in pixels
    :type width: int
    :returns: (3,3) intrinsic matrix
    """
    fov_x = np.deg2rad(fov_x)
    fov_y = np.deg2rad(fov_y)

    if fov_x == 0.0 or fov_y == 0.0:
        raise ZeroDivisionError("fov can't be zero")

    c_x = width / 2.0
    c_y = height / 2.0
    f_x = c_x / np.tan(fov_x / 2.0)
    f_y = c_y / np.tan(fov_y / 2.0)

    intrinsic_matrix = np.array(
        [
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1],
        ],
        dtype=np.float16,
    )

    return intrinsic_matrix


def apply_transform(
    points_3D: np.ndarray,
    transform_matrix: np.ndarray,
) -> np.ndarray:
    """
    Takes an (N,3) list of 3D points
    transform_matrix is (4,4)
    Returns an (N,3) list of 2D points on the camera plane

    :param points_3D: (N,3) list of 3D points
    :type points_3D: np.ndarray
    :param transform_matrix: (4,4) matrix
    :type transform_matrix: np.ndarray
    :returns: (N,3) list of 2D points on the camera plane
    """

    # points_3D is (N, 3)
    # points_3D_homo is (N, 4)
    # extrinsic_matrix is (4, 4)
    points_3D_homo = np.array(
        [
            points_3D[:, 0],
            points_3D[:, 1],
            points_3D[:, 2],
            np.ones_like(points_3D[:, 0]),
        ]
    ).T

    # points_3D_homo_transformed is (N, 4)
    points_3D_homo_transformed = (transform_matrix @ points_3D_homo.T).T

    return points_3D_homo_transformed[:, :3]


def project_world_cam_to_image(
    points_3D_cam: np.ndarray, intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    Project 3D points in camera coordinates to 2D image coordinates.

    :param points_3D_cam: (N,3) list of 3D points in camera coordinates
    :type points_3D_cam: np.ndarray
    :param intrinsic_matrix: (3,3) intrinsic matrix
    :type intrinsic_matrix: np.ndarray
    :returns: (N,2) list of 2D points on the image plane
    """
    if len(intrinsic_matrix.shape) != 2 or intrinsic_matrix.shape != (3, 3):
        raise ValueError(
            "intrinsic_matrix expected shape (3, 3), "
            + f"got {intrinsic_matrix.shape}"
        )

    if len(points_3D_cam.shape) != 2 or points_3D_cam.shape[1] != 3:
        raise ValueError(
            "points_3D_cam expected shape (N, 3), "
            + f"got {points_3D_cam.shape}"
        )

    intrinsic_matrix_homo = np.eye(4)
    intrinsic_matrix_homo[:3, :3] = intrinsic_matrix

    points_3D_homo = np.array(
        [
            points_3D_cam[:, 0],
            points_3D_cam[:, 1],
            points_3D_cam[:, 2],
            np.ones_like(points_3D_cam[:, 0]),
        ]
    ).T
    points_2D_homo = (intrinsic_matrix_homo @ points_3D_homo.T).T
    points_2D = np.array(
        [
            points_2D_homo[:, 0] / points_2D_homo[:, 2],
            points_2D_homo[:, 1] / points_2D_homo[:, 2],
        ]
    ).T

    return points_2D


def project_world_to_image(
    points_3D: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    return_points_3D_transformed: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Takes an (N,3) list of 3D points
    intrinsic_matrix is (3,3)
    Returns an (N,3) list of 2D points on the camera plane

    :param points_3D: (N,3) list of 3D points
    :type points_3D: np.ndarray
    :param intrinsic_matrix: (3,3) intrinsics
    :type intrinsic_matrix: np.ndarray
    :param extrinsic_matrix: offsets to adjust the trajectory by
    :type extrinsic_matrix: np.ndarray
    :returns: (N,2) list of 2D points on the camera plane
    """
    if len(intrinsic_matrix.shape) != 2 or intrinsic_matrix.shape != (3, 3):
        raise ValueError(
            "intrinsic_matrix expected shape (3, 3), "
            + f"got {intrinsic_matrix.shape}"
        )
    # trajectory is (N, 3)
    # trajectory_3D_homo is (N, 4)
    # extrinsic_matrix is (4, 4)
    points_3D_homo = np.array(
        [
            points_3D[:, 0],
            points_3D[:, 1],
            points_3D[:, 2],
            np.ones_like(points_3D[:, 0]),
        ]
    ).T
    # intrinsics_homo is (3, 4)
    intrinsics_homo = np.hstack((intrinsic_matrix, np.zeros((3, 1))))

    # points_3D_transformed_homo is (N, 4)
    points_3D_transformed_homo = (extrinsic_matrix @ points_3D_homo.T).T

    # points_2D_homo is (N, 3)
    points_2D_homo = (intrinsics_homo @ points_3D_transformed_homo.T).T
    # points_2D is (N, 2)
    points_2D = np.array(
        [
            points_2D_homo[:, 0] / points_2D_homo[:, 2],
            points_2D_homo[:, 1] / points_2D_homo[:, 2],
        ]
    ).T

    if return_points_3D_transformed:
        points_3D_transformed = np.array(
            [
                points_3D_transformed_homo[:, 0],
                points_3D_transformed_homo[:, 1],
                points_3D_transformed_homo[:, 2],
            ]
        ).T
        return points_2D, points_3D_transformed

    return points_2D


def project_image_to_world(
    image: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    subsample: int = 1,
    mask: Optional[np.ndarray] = None,
    bounds: float = float("inf"),  # meters
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes depth and the image as input and produces 3D pointcloud with color occupancy
    https://github.com/AdityaNG/socc_plotter/blob/2fda52641d2353e56b4f8fd280e789105981ff1b/socc_plotter/socc.py#L10-L77  # noqa

    Args:
        image (np.ndarray): (HxWx3) uint8
        depth (np.ndarray): (HxW) float32
        intrinsics (Optional[np):ndarray]): 3x3
        subsample (int): to reduce the size of the pointcloud
        mask ( Optional[np.ndarray]): default to all ones mask
        bounds (float): bounds of the point cloud to clip to in meters

    Returns:
        Tuple[np.ndarray, np.ndarray]: points, colors
    """

    HEIGHT, WIDTH = depth.shape

    assert subsample >= 1 and isinstance(
        subsample, int
    ), "subsample must be a positive int"

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    points = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

    if mask is None:
        # default to full mask
        mask = np.ones((HEIGHT, WIDTH), dtype=bool)

    U, V = np.ix_(
        np.arange(HEIGHT), np.arange(WIDTH)
    )  # pylint: disable=unbalanced-tuple-unpacking
    Z = depth.copy()

    X = (V - cx) * Z / fx
    Y = (U - cy) * Z / fy

    points[:, :, 0] = X
    points[:, :, 1] = Y
    points[:, :, 2] = Z

    colors = image.copy()

    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    # subsample
    points = points[::subsample, :]
    colors = colors[::subsample, :]

    points = points.clip(-bounds, bounds)

    return (points, colors)


def create_transform(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> np.ndarray:
    """Creates a 4x4 transformation matrix.

    This function takes the following arguments:
        x (float): The x translation in meters.
        y (float): The y translation in meters.
        z (float): The z translation in meters.
        roll (float): The roll angle in degrees.
        pitch (float): The pitch angle in degrees.
        yaw (float): The yaw angle in degrees.

    Returns:
        A 4x4 numpy array representing the transformation matrix.
    """
    # Convert degrees to radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # Create individual rotation matrices
    R_yaw = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    R_pitch = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )

    R_roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )

    # Combine the rotation matrices
    rotation_matrix = R_yaw @ R_pitch @ R_roll

    # Construct the transformation matrix
    transformation_matrix = np.array(
        [
            [
                rotation_matrix[0, 0],
                rotation_matrix[0, 1],
                rotation_matrix[0, 2],
                x,
            ],
            [
                rotation_matrix[1, 0],
                rotation_matrix[1, 1],
                rotation_matrix[1, 2],
                y,
            ],
            [
                rotation_matrix[2, 0],
                rotation_matrix[2, 1],
                rotation_matrix[2, 2],
                z,
            ],
            [0, 0, 0, 1],
        ]
    )

    return transformation_matrix


def interpolate_trajectory_3D(
    trajectory: np.ndarray,
    samples: int = 0,
) -> np.ndarray:
    """
    Interpolates the trajectory (N, 3) to (M, 3)
    Where M = N*(S+1)+1

    :param trajectory: (N,3) numpy trajectory
    :type trajectory: np.ndarray
    :param samples: number of samples
    :type samples: int
    :returns: (M,3) interpolated numpy trajectory
    """
    assert trajectory.shape[0] > 0, "Empty trajectory"
    # Calculate the number of segments
    num_segments = trajectory.shape[0] - 1

    # Generate the interpolated trajectory
    interpolated_trajectory = np.zeros((num_segments * (samples + 1) + 1, 3))

    # Fill in the interpolated points
    for i in range(num_segments):
        start = trajectory[i]
        end = trajectory[i + 1]
        interpolated_trajectory[
            i * (samples + 1) : (i + 1) * (samples + 1)
        ] = np.linspace(start, end, samples + 2)[:-1]

    # Add the last point
    interpolated_trajectory[-1] = trajectory[-1]

    return interpolated_trajectory


def interpolate_trajectory(
    trajectory: np.ndarray,
    samples: int = 0,
) -> np.ndarray:
    """
    Interpolates the trajectory (N, 2) to (M, 2)
    Where M = N*(S+1)+1

    :param trajectory: (N,2) numpy trajectory
    :type trajectory: np.ndarray
    :param samples: number of samples
    :type samples: int
    :returns: (M,2) interpolated numpy trajectory
    """
    assert trajectory.shape[0] > 0, "Empty trajectory"
    # Calculate the number of segments
    num_segments = trajectory.shape[0] - 1

    # Generate the interpolated trajectory
    interpolated_trajectory = np.zeros((num_segments * (samples + 1) + 1, 2))

    # Fill in the interpolated points
    for i in range(num_segments):
        start = trajectory[i]
        end = trajectory[i + 1]
        interpolated_trajectory[
            i * (samples + 1) : (i + 1) * (samples + 1)
        ] = np.linspace(start, end, samples + 2)[:-1]

    # Add the last point
    interpolated_trajectory[-1] = trajectory[-1]

    return interpolated_trajectory


def get_trajectory_rectangles_coords_3d(
    Pi: np.ndarray, Pj: np.ndarray, width: float
) -> np.ndarray:
    # Pi = Pi.reshape(4, 1)
    # Pj = Pj.reshape(4, 1)
    x_i, y_i = Pi[0], Pi[2]
    x_j, y_j = Pj[0], Pj[2]
    points_2D = get_trajectory_rectangle_coords(x_i, y_i, x_j, y_j, width)
    points_3D_l = []
    for index in range(points_2D.shape[0]):
        # point_2D = points_2D[index]
        point_3D = Pi.copy()
        point_3D[0] = points_2D[index, 0]
        point_3D[2] = points_2D[index, 1]

        points_3D_l.append(point_3D)

    points_3D = np.array(points_3D_l)
    return points_3D


def get_trajectory_rectangle_coords(
    x_i: float, y_i: float, x_j: float, y_j: float, width: float
) -> np.ndarray:
    """
    Takes two adjacent points on the trajecotry and returns the corners of
    a rectange that encompass the two points.
    """
    Pi = np.array([x_i, y_i])
    Pj = np.array([x_j, y_j])
    height = np.linalg.norm(Pi - Pj)
    diagonal = (width**2 + height**2) ** 0.5
    D = diagonal / 2.0

    M = ((Pi + Pj) / 2.0).reshape((2,))
    theta = np.arctan2(Pi[1] - Pj[1], Pi[0] - Pj[0])
    theta += np.pi / 4.0
    points = np.array(
        [
            M
            + np.array(
                [
                    D * np.sin(theta + 0 * np.pi / 2.0),
                    D * np.cos(theta + 0 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 1 * np.pi / 2.0),
                    D * np.cos(theta + 1 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 2 * np.pi / 2.0),
                    D * np.cos(theta + 2 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 3 * np.pi / 2.0),
                    D * np.cos(theta + 3 * np.pi / 2.0),
                ]
            ),
        ]
    )
    return points


def get_camera_trajectory_cropped(
    trajectory: np.ndarray,
    offsets: Tuple[float, float, float] = (0.0, 5.0, 3.0),
) -> np.ndarray:
    """
    Coordinate Frames:
        3D:
            x: horizontal plane
            y: vertical into the ground
            z: depth into the camera
        2D Image:
            px: x axis
            py: y axis
        2D Trajectory:
            xt -> x
            yt -> z
            height -> y

      Plots a trajectory onto a given frame image.
    """
    # trajectory_3D is (N,3)
    trajectory_3D = np.array(
        [
            trajectory[:, 0],
            np.zeros_like(trajectory[:, 0]),
            trajectory[:, 1],
        ]
    ).T

    extrinsics = np.array(
        [
            [1, 0, 0, -offsets[0]],
            [0, 1, 0, -offsets[1]],
            [0, 0, 1, -offsets[2]],
            [0, 0, 0, 1],
        ]
    )

    trajectory_3D_transformed = apply_transform(trajectory_3D, extrinsics)

    behind_cam = trajectory_3D_transformed[:, 2] < 0.0

    trajectory_3D = trajectory_3D[~behind_cam]

    trajectory_cropped = trajectory[~behind_cam]

    return trajectory_cropped


def create_concentric_circles(radii, separation_distance):
    circles = []

    for radius in radii:
        # Calculate the number of points needed for this circle
        circumference = 2 * np.pi * radius
        num_points = int(np.ceil(circumference / separation_distance))

        # Generate evenly spaced angles
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        # Create x and z coordinates
        x = radius * np.cos(angles)
        z = radius * np.sin(angles)

        # Create y coordinates
        # (all zeros since the circles are in the x-z plane)
        y = np.zeros_like(x)

        # Combine into a single array of shape (N, 3)
        circle_points = np.column_stack((x, y, z))

        circles.append(circle_points)

    return circles


# Global in-memory cache
__conc_circles_memory_cache: Dict[str, np.ndarray] = {}


def get_cached_frame_img_circles(
    frame_img: np.ndarray,
    intrinsic_matrix: np.ndarray,
    grid_range_img: Tuple[int, int],
    average_height: float,
    method: str,
):
    global __conc_circles_memory_cache

    # Round intrinsic matrix to 3 decimals
    rounded_intrinsic_matrix = np.round(intrinsic_matrix, decimals=3)

    # Ensure grid_range_img is a tuple of ints
    grid_range_img_tuple = tuple(map(int, grid_range_img))

    # Round average height to 3 decimals
    rounded_average_height = round(average_height, 3)

    hash_args = (
        rounded_intrinsic_matrix.tobytes(),
        grid_range_img_tuple,
        rounded_average_height,
        method,
    )
    hash_val = hashlib.md5(str(hash_args).encode()).hexdigest()

    if hash_val in __conc_circles_memory_cache:
        return __conc_circles_memory_cache[hash_val]

    cache_dir = os.path.join(os.path.expanduser("~/.cache"), "plot_traj")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{hash_val}.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            frame_img_circles = pickle.load(f)

    else:
        intrinsic_matrix = np.array(intrinsic_matrix)
        frame_img_circles = np.zeros_like(frame_img)
        circles = create_concentric_circles(
            range(grid_range_img[0], grid_range_img[1]),
            1.0,
        )
        for circle in circles:
            circle[:, 1] = average_height
            plot_steering_trajectory(
                frame_img=frame_img_circles,
                trajectory=circle,
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=np.eye(4),
                color=(255, 255, 255),
                method=method,
                track=False,
                line=True,
                draw_grid=False,
                interpolation_samples=0,
                thickness=1,
            )
        with open(cache_file, "wb") as f:
            pickle.dump(frame_img_circles, f)

    __conc_circles_memory_cache[hash_val] = frame_img_circles
    return frame_img_circles


def plot_steering_trajectory(
    frame_img: np.ndarray,
    trajectory: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    method: str = "add_weighted",
    track: bool = False,
    line: bool = False,
    draw_grid: bool = False,
    grid_range_img: Tuple[int, int] = (1, 5),
    track_width: float = 1.5,
    interpolation_samples: int = 0,
    thickness: int = 2,
) -> np.ndarray:
    """
    Coordinate Frames:
        3D:
            x: horizontal plane
            y: vertical into the ground
            z: depth into the camera
        2D Image:
            px: x axis
            py: y axis
        2D Trajectory:
            xt -> x
            yt -> z
            height -> y

      Plots a trajectory onto a given frame image.
    """
    assert method in ("overlay", "mask", "add_weighted")

    h, w = frame_img.shape[:2]

    assert intrinsic_matrix.shape == (
        3,
        3,
    ), f"Incorrect shape: {intrinsic_matrix.shape}"

    assert extrinsic_matrix.shape == (
        4,
        4,
    ), f"Incorrect shape: {extrinsic_matrix.shape}"

    if trajectory.shape[1] == 2:
        # 2D input
        trajectory = interpolate_trajectory(
            trajectory, samples=interpolation_samples
        )

        # trajectory_3D is (N,3)
        trajectory_3D = np.array(
            [
                trajectory[:, 0],
                np.zeros_like(trajectory[:, 0]),
                trajectory[:, 1],
            ]
        ).T
    elif trajectory.shape[1] == 3:
        # 3D input
        # trajectory_3D is (N,3)
        trajectory_3D = interpolate_trajectory_3D(
            trajectory, samples=interpolation_samples
        )
    else:
        raise Exception(
            f"Trajectory must be 2D or 3D, got: {trajectory.shape}"
        )

    trajectory_3D_cam = apply_transform(trajectory_3D, extrinsic_matrix)
    trajectory_2D = project_world_cam_to_image(
        trajectory_3D_cam, intrinsic_matrix
    )
    trajectory_2D = cast(np.ndarray, trajectory_2D)

    # Filter out outliers
    trajectory_2D = trajectory_2D.astype(np.int64)

    rect_frame = np.zeros_like(frame_img)

    for point_index in range(1, trajectory_2D.shape[0]):
        px, py = trajectory_2D[point_index]
        px_p, py_p = trajectory_2D[point_index - 1]
        point_3D = trajectory_3D_cam[point_index]
        prev_point_3D = trajectory_3D_cam[point_index - 1]

        in_range = px_p in range(0, w) and py_p in range(0, h)
        in_range_c = px in range(0, w) and py in range(0, h)
        in_front = point_3D[2] > 0 and prev_point_3D[2] > 0
        only_one_in_front = point_3D[2] > 0 and prev_point_3D[2] <= 0
        only_one_in_range = in_range_c and not in_range

        # Convert to int after range checks
        px, py = trajectory_2D[point_index].astype(np.int16)
        px_p, py_p = trajectory_2D[point_index - 1].astype(np.int16)

        if track and in_range:
            rect_coords_3D = get_trajectory_rectangles_coords_3d(
                point_3D, prev_point_3D, track_width
            )
            behind_cam_poly = rect_coords_3D[:, 2] < 0.0
            if behind_cam_poly.sum() == 0:
                rect_coords = project_world_cam_to_image(
                    rect_coords_3D,
                    intrinsic_matrix,
                )
                rect_coords = cast(np.ndarray, rect_coords)
                rect_coords = rect_coords.astype(np.int32)
                rect_frame = cv2.fillPoly(
                    rect_frame, pts=[rect_coords], color=color  # type: ignore
                )

        if line and in_range and in_front:
            frame_img = cv2.line(
                frame_img, (px_p, py_p), (px, py), color, thickness * 2
            )
            rect_frame = cv2.line(
                rect_frame, (px_p, py_p), (px, py), color, thickness * 2
            )

        if line and only_one_in_range and in_front:
            frame_img = cv2.line(
                frame_img, (px_p, py_p), (px, py), color, thickness * 2
            )
            rect_frame = cv2.line(
                rect_frame, (px_p, py_p), (px, py), color, thickness * 2
            )

        if line and only_one_in_front and point_index < 2:
            py_p = h + py_p
            frame_img = cv2.line(
                frame_img, (px_p, py_p), (px, py), color, thickness * 2
            )
            rect_frame = cv2.line(
                rect_frame, (px_p, py_p), (px, py), color, thickness * 2
            )

        if in_front:
            cv2.circle(frame_img, (px, py), thickness, color, -1)

    if method == "mask":
        mask = np.logical_and(
            rect_frame[:, :, 0] == color[0],
            rect_frame[:, :, 1] == color[1],
            rect_frame[:, :, 2] == color[2],
        )
        frame_img[mask] = color
    elif method == "overlay":
        frame_img += (0.2 * rect_frame).astype(np.uint8)
    elif method == "add_weighted":
        cv2.addWeighted(frame_img, 1.0, rect_frame, 0.5, 0.0, frame_img)

    if draw_grid:
        average_height = np.mean(trajectory_3D_cam[:, 1])
        frame_img_circles = get_cached_frame_img_circles(
            frame_img,
            intrinsic_matrix,
            grid_range_img,
            average_height,
            method,
        )

        mask = np.logical_and(
            frame_img_circles[:, :, 0] == 255,
            frame_img_circles[:, :, 1] == 255,
            frame_img_circles[:, :, 2] == 255,
        )
        frame_img[mask] = color

    return frame_img


def plot_bev_trajectory(
    trajectory: np.ndarray,
    image_size: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    grid_range: float = 20,  # meters
    line: bool = False,
    draw_grid_bev: bool = False,
    traj_plot: Optional[np.ndarray] = None,  # image to write to
) -> np.ndarray:
    """
    Plots out a trajectory in BEV space
    3D Trajectory:
        x: horizontal plane
        y: vertical into the ground
        z: depth into the camera
    2D Trajectory:
        x: horizontal plane
        y: depth into the camera
    """
    WIDTH, HEIGHT = image_size

    if traj_plot is None:
        traj_plot = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    else:
        assert (
            traj_plot.shape[:2] == image_size[::-1]
        ), f"Incorrect shape: {traj_plot.shape}, expected {image_size[::-1]}"

    if trajectory.shape[1] == 2:
        # 2D input

        Z = trajectory[:, 1]
        X = trajectory[:, 0]
    elif trajectory.shape[1] == 3:
        # 3D input

        Z = trajectory[:, 2]
        X = trajectory[:, 0]
    else:
        raise Exception(
            f"Trajectory must be 2D or 3D, got: {trajectory.shape}"
        )

    X_min, X_max = -grid_range, grid_range
    Z_min, Z_max = -0.1 * grid_range, grid_range
    X = (X - X_min) / (X_max - X_min)
    Z = (Z - Z_min) / (Z_max - Z_min)

    for traj_index in range(0, X.shape[0]):
        u = int(round(np.clip((X[traj_index] * (WIDTH - 1)), -1, WIDTH + 1)))
        v = int(round(np.clip((Z[traj_index] * (HEIGHT - 1)), -1, HEIGHT + 1)))
        u_p = int(
            round(np.clip((X[traj_index - 1] * (WIDTH - 1)), -1, WIDTH + 1))
        )
        v_p = int(
            round(np.clip((Z[traj_index - 1] * (HEIGHT - 1)), -1, HEIGHT + 1))
        )

        if u not in range(WIDTH) or v not in range(HEIGHT):
            continue

        cv2.circle(traj_plot, (u, v), thickness, color, -1)

        if line and traj_index > 0:
            cv2.line(traj_plot, (u_p, v_p), (u, v), color, thickness)

    if draw_grid_bev:
        # Horizontal lines
        for i in range(round(X_min), round(X_max), 1):
            p = (i - X_min) / (X_max - X_min)
            u = int(round(np.clip((p * (WIDTH - 1)), -1, WIDTH + 1)))
            v = 0
            u_p = u
            v_p = HEIGHT

            cv2.line(
                traj_plot, (u_p, v_p), (u, v), color, max(1, thickness // 4)
            )

        # Vertical lines
        for i in range(round(Z_min), round(Z_max), 1):
            p = (i - Z_min) / (Z_max - Z_min)
            u = 0
            v = int(round(np.clip((p * (HEIGHT - 1)), -1, HEIGHT + 1)))
            u_p = WIDTH
            v_p = v

            cv2.line(
                traj_plot, (u_p, v_p), (u, v), color, max(1, thickness // 4)
            )

    px = (0 - X_min) / (X_max - X_min)
    py = (0 - Z_min) / (Z_max - Z_min)
    u = int(round(np.clip((px * (WIDTH - 1)), -1, WIDTH + 1)))
    v = int(round(np.clip((py * (HEIGHT - 1)), -1, HEIGHT + 1)))
    cv2.circle(traj_plot, (u, v), thickness * 3, color, -1)

    traj_plot = cast(np.ndarray, cv2.flip(traj_plot, 0))
    return traj_plot


def rotate_image(image: np.ndarray, angle: float):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
    )
    return result


STEERING_IMG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "media/steering.png")
)


def plot_carstate_frame(
    frame_img: np.ndarray,
    steering_gt: float = 0.0,
    steering_pred: float = 0.0,
    steering_img_path: str = STEERING_IMG_PATH,
) -> np.ndarray:

    frame_img_steering = np.zeros_like(frame_img)

    steering_dim = round(frame_img.shape[0] * 0.2)

    # Draw Steering GT
    assert os.path.isfile(
        steering_img_path
    ), f"File not found: {steering_img_path}"
    steering_img = cv2.imread(steering_img_path)
    steering_img = rotate_image(steering_img, round(steering_gt))
    x_steering_start = round(frame_img_steering.shape[0] * 0.1)
    y_steering_start = (
        (frame_img_steering.shape[1] // 2) - (steering_dim // 2) - steering_dim
    )
    frame_img_steering[
        x_steering_start : x_steering_start + steering_dim,
        y_steering_start : y_steering_start + steering_dim,
    ] = cv2.resize(steering_img, (steering_dim, steering_dim))
    frame_img = cv2.addWeighted(frame_img, 1.0, frame_img_steering, 2.5, 0.0)

    # Draw Steering Pred
    steering_img = cv2.imread(steering_img_path)
    steering_img = rotate_image(steering_img, round(steering_pred))
    x_steering_start = round(frame_img_steering.shape[0] * 0.1)
    y_steering_start = (
        (frame_img_steering.shape[1] // 2) - (steering_dim // 2) + steering_dim
    )
    frame_img_steering[
        x_steering_start : x_steering_start + steering_dim,
        y_steering_start : y_steering_start + steering_dim,
    ] = cv2.resize(steering_img, (steering_dim, steering_dim))
    frame_img = cv2.addWeighted(frame_img, 1.0, frame_img_steering, -2.5, 0.0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    color = (255, 0, 0)
    thickness = 1

    frame_img = cv2.putText(
        frame_img,
        "steering_gt: " + str(round(steering_gt, 2)),
        (10, 15),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    frame_img = cv2.putText(
        frame_img,
        "steer_pred: " + str(round(steering_pred, 2)),
        (10, 25),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return frame_img


###############################################################################
# Batch trajectory plotting
###############################################################################


def plot_trajectories_image(
    frame_img: np.ndarray,
    trajectories: List[np.ndarray],
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    colors: List[Tuple[int, int, int]] = [(0, 255, 0)],
    method: str = "add_weighted",
    track: bool = False,
    line: bool = False,
    draw_grid: bool = False,
    grid_range_img: Tuple[int, int] = (1, 5),
    track_width: float = 1.5,
    interpolation_samples: int = 0,
    thickness: int = 2,
) -> np.ndarray:
    assert len(trajectories) == len(
        colors
    ), "Each trajectory must have a corresponding color"

    for trajectory, color in zip(trajectories, colors):
        frame_img = plot_steering_trajectory(
            frame_img,
            trajectory,
            intrinsic_matrix,
            extrinsic_matrix,
            color=color,
            method=method,
            track=track,
            line=line,
            draw_grid=draw_grid,
            grid_range_img=grid_range_img,
            track_width=track_width,
            interpolation_samples=interpolation_samples,
            thickness=thickness,
        )
    return frame_img


def plot_trajectories_bev(
    trajectories: List[np.ndarray],
    image_size: Tuple[int, int],
    colors: List[Tuple[int, int, int]] = [(0, 255, 0)],
    thickness: int = 2,
    grid_range: float = 20,  # meters
    line: bool = False,
    draw_grid_bev: bool = False,
) -> np.ndarray:
    assert len(trajectories) == len(
        colors
    ), "Each trajectory must have a corresponding color"

    bev_img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    for trajectory, color in zip(trajectories, colors):
        bev_img = cast(np.ndarray, cv2.flip(bev_img, 0))
        bev_img = plot_bev_trajectory(
            trajectory,
            image_size,
            color=color,
            thickness=thickness,
            grid_range=grid_range,
            line=line,
            draw_grid_bev=draw_grid_bev,
            traj_plot=bev_img,
        )

    return bev_img


def plot_trajectories(
    frame_img: np.ndarray,
    trajectories: List[np.ndarray],
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    colors: List[Tuple[int, int, int]] = [(0, 255, 0)],
    method: str = "add_weighted",
    track: bool = False,
    line: bool = False,
    draw_grid: bool = False,
    draw_grid_bev: bool = False,
    track_width: float = 1.5,
    interpolation_samples: int = 0,
    thickness: int = 2,
    thickness_bev: int = 2,
    grid_range: float = 20,
    grid_range_img: Tuple[int, int] = (1, 5),
) -> Tuple[np.ndarray, np.ndarray]:
    image_trajectory = plot_trajectories_image(
        frame_img,
        trajectories,
        intrinsic_matrix,
        extrinsic_matrix,
        colors=colors,
        method=method,
        track=track,
        line=line,
        draw_grid=draw_grid,
        track_width=track_width,
        interpolation_samples=interpolation_samples,
        thickness=thickness,
        grid_range_img=grid_range_img,
    )

    bev_trajectory = plot_trajectories_bev(
        trajectories,
        (frame_img.shape[1], frame_img.shape[0]),
        colors=colors,
        thickness=thickness_bev,
        grid_range=grid_range,
        line=line,
        draw_grid_bev=draw_grid_bev,
    )

    return image_trajectory, bev_trajectory


# Semantics plotting


def overlay_image_by_semantics(
    image_base: np.ndarray,
    image_overlay: np.ndarray,
    semantics_img: np.ndarray,
    semantics_colors: List[Tuple[int, int, int]],
) -> np.ndarray:
    mask = np.zeros_like(
        image_base[:, :, 0],
        dtype=bool,
    )

    # Setting the mask based on the semantics_img
    for idx, color in enumerate(semantics_colors):
        # Create a mask for each semantic label's corresponding color
        color_mask = semantics_img == color

        # Convert this boolean mask to a 2D mask by checking all channels
        single_channel_color_mask = np.all(color_mask, axis=-1)

        # Add this mask to the general mask
        mask[single_channel_color_mask] = True

    # Apply the mask to image_overlay to image_base
    image_base[~mask] = image_overlay[~mask]

    return image_base
