from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import torch
import numpy as np
import open3d as o3d 
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.blender_dataparser import Blender, BlenderDataParserConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
import math
import imageio

camera_angle_x = 0.6911112070083618


@dataclass
class MirageDataParserConfig(BlenderDataParserConfig):
    """Config dla Mirage parsera"""
    _target: Type = field(default_factory=lambda: MirageDataParser)

    distance: float = 1.0
    """Distance from camera to image plane."""
    image_name: str = "image.png"
    """Name of the image file to load."""
    num_points: int = 500_000
    """Number of points to sample."""
    xyz_path: Optional[Path] = None
    """Optional path to a point cloud file to load instead of sampling."""


class MirageDataParser(Blender):

    def __init__(self, config: MirageDataParserConfig):
        super().__init__(config)

    def _generate_dataparser_outputs(self, split: None, visualization = False):
        num_pts = self.config.num_points
        image_path = self.config.data 
        image_filenames = [image_path]

        img = imageio.v2.imread(image_path)
        image_height, image_width = img.shape[:2]
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
        cx = image_width / 2.0
        cy = image_height / 2.0

        distance = self.config.distance
        c2w = np.eye(4, dtype=np.float32)
        c2w[2, 3] = distance 
       
        aabb = torch.tensor([[-1.5, -1.5, -1.5],
                     [ 1.5,  1.5,  1.5]], dtype=torch.float32)
        scene_box = SceneBox(aabb=aabb)
        aabb_min = aabb[0].cpu().numpy()
        aabb_max = aabb[1].cpu().numpy()
        
        fx = float(focal_length)
        fy = float(focal_length)  # assume square pixels; change if you have different fy
        right = distance * (image_width  / (2.0 * fx))
        top   = distance * (image_height / (2.0 * fy))

        xyz = np.random.uniform(
            low=[-right, -top, 0],   
            high=[right, top, 0],    
            size=(num_pts, 3)
        )

        xyz = (xyz - aabb_min) / (aabb_max - aabb_min)

        if self.config.xyz_path is not None:
            xyz = o3d.io.read_point_cloud(self.config.xyz_path).points
            xyz = np.asarray(xyz)
            print("Loaded xyz from ply:", self.config.xyz_path)
        
        colors = np.random.random((num_pts, 3))  

        if visualization:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            camera_frame.transform(c2w)
            o3d.visualization.draw_geometries([pcd, camera_frame])

        camera_to_world = torch.from_numpy(c2w[:3]).unsqueeze(0)

        metadata = {}
        if xyz is not None and len(xyz) > 0:
            points3D = torch.from_numpy(xyz.astype(np.float32)) * (self.config.scale_factor if hasattr(self.config, "scale_factor") else 1.0)
            points3D_rgb = torch.from_numpy((colors * 255).astype(np.uint8))

            metadata.update({
                "points3D_xyz": points3D,
                "points3D_rgb": points3D_rgb,
            })

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.config.scale_factor,
            metadata=metadata,
        )
        return dataparser_outputs
    