import cv2
import sys
import torch
import argparse

import numpy as np
import open3d as o3d
from pathlib import Path


CAMERA_ANGLE_X = 0.6911112070083618

def generate_reference_pcd(image: torch.Tensor, num_points: int = 500_000, distance: float = 1.0) -> o3d.geometry.PointCloud:

    height, width = image.shape[:2]

    focal_length = 0.5 * width / np.tan(0.5 * CAMERA_ANGLE_X)

    fx, fy = float(focal_length), float(focal_length)
    right = distance * (width / (2.0 * fx))
    top = distance * (height / (2.0 * fy))

    xyz = np.random.uniform(
        low=[-right, -top, 0],
        high=[right, top, 0],
        size=(num_points, 3),
    )

    aabb_min = np.array([-1.5, -1.5, -1.5])
    aabb_max = np.array([1.5, 1.5, 1.5])
    xyz = (xyz - aabb_min) / (aabb_max - aabb_min)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    colors = np.random.random((num_points, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def create_mask(image: torch.Tensor, reference_pcd: o3d.geometry.PointCloud, edited_pcd: o3d.geometry.PointCloud) -> torch.Tensor:

    points_ref = np.asarray(reference_pcd.points)
    points_edit = np.asarray(edited_pcd.points)

    height, width = image.shape[:2]

    x_min, x_max = points_ref[:, 0].min(), points_ref[:, 0].max()
    y_min, y_max = points_ref[:, 1].min(), points_ref[:, 1].max()

    x_norm = (points_edit[:, 0] - x_min) / (x_max - x_min)
    y_norm = (points_edit[:, 1] - y_min) / (y_max - y_min)

    x_pix = (x_norm * (width - 1)).astype(np.int32)
    y_pix = ((1 - y_norm) * (height - 1)).astype(np.int32)

    inside = (x_pix >= 0) & (x_pix < width) & (y_pix >= 0) & (y_pix < height)
    x_pix, y_pix = x_pix[inside], y_pix[inside]

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_pix, x_pix] = 255

    kernel_size = 8
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return torch.from_numpy(mask.astype(bool)).to(image.device)
