import numpy as np
import torch
import trimesh

from nerfstudio.viewer.viewer_elements import ViewerElement
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from viser import ViserServer
from viser._scene_handles import PointCloudHandle


class ViewerPointCloud(ViewerElement[bool]):
    """A point cloud in the viewer

    Args:
        name: The name of the point cloud
        visible: If the point cloud is visible
    """

    scene_handle: PointCloudHandle

    def __init__(self, name: str, aabb: SceneBox, points: np.ndarray, confidence: np.ndarray, visible: bool = True):
        self.aabb = aabb
        self.points = points
        self.confidence = confidence
        super().__init__(name, visible=visible)

    def update(self, points: np.ndarray, confidence: np.ndarray) -> None:
        """Update the point cloud with new points."""
        self.points = points
        self.confidence = confidence
        if self.viser_server is not None:
            self._create_scene_handle(self.viser_server)

    def _create_scene_handle(self, viser_server: ViserServer) -> None:
        points = self.points.reshape(-1, self.points.shape[-1])
        aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy()
        aabb_min, aabb_max = aabb[0] * VISER_NERFSTUDIO_SCALE_RATIO, aabb[1] * VISER_NERFSTUDIO_SCALE_RATIO
        points = points * (aabb_max - aabb_min) + aabb_min

        pcd = trimesh.PointCloud(points)

        color_coeffs = np.random.uniform(0.4, 1.0, size=(pcd.vertices.shape[0]))
        colors = np.tile((0, 255, 255), pcd.vertices.shape[0]).reshape(-1, 3) * color_coeffs[:, None]
        colors[:, 1] *= (1 - self.confidence)
        colors[:, 2] *= self.confidence

        self.scene_handle = viser_server.scene.add_point_cloud(
            f"/{self.name}",
            points=pcd.vertices,
            colors=colors,
            point_size=0.02,
            point_shape="circle"
        )

    def install(self, viser_server: ViserServer) -> None:
        self.viser_server = viser_server
        self._create_scene_handle(viser_server)



class ViewerSamplerPoints(ViewerElement[bool]):
    """Sample points from the sampler in the viewer.

    Args:
        name: The name of the sample points.
        aabb: The scene bounding box.
        sample_points: The sampled 3D points.
        visible: If the sample points are visible.
    """

    def __init__(self, name: str, aabb: SceneBox, sample_points: torch.Tensor, visible: bool = True):
        self.sample_points = sample_points
        self.aabb = aabb
        super().__init__(name, visible=visible)

    def update(self, sample_points: torch.Tensor) -> None:
        self.sample_points = sample_points
        if self.viser_server is not None:
            self._create_scene_handle(self.viser_server)

    def _create_scene_handle(self, viser_server: ViserServer):
        if self.sample_points is None or len(self.sample_points) == 0:
            return
            
        points_np = self.sample_points.detach().cpu().numpy()
        points = points_np.reshape(-1, points_np.shape[-1])
        aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy()
        aabb_min, aabb_max = aabb[0] * VISER_NERFSTUDIO_SCALE_RATIO, aabb[1] * VISER_NERFSTUDIO_SCALE_RATIO
        points = points * (aabb_max - aabb_min) + aabb_min
        viser_server.scene.add_point_cloud(
                "/sampler_points",
                points=points,
                colors=np.tile((255, 0, 255), (self.sample_points.shape[0], 1)),
                point_size=0.02,
                point_shape="circle",
                visible=True,
            )

    def install(self, viser_server: ViserServer) -> None:
        self.viser_server = viser_server
        self._create_scene_handle(viser_server)



class ViewerAABB(ViewerElement[bool]):
    """A bounding box in the viewer

    Args:
        name: The name of the aabb
        visible: If the aabb is visible
    """

    def __init__(self, name: str, aabb: SceneBox, visible: bool = True):
        self.aabb = aabb
        super().__init__(name, visible=visible)

    def _create_scene_handle(self, viser_server: ViserServer) -> None:
        aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy() * VISER_NERFSTUDIO_SCALE_RATIO
        mesh = trimesh.creation.box(tuple((aabb[1] - aabb[0])))
        viser_server.scene.add_mesh_simple(
            name=f"/{self.name}",
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=(0, 0, 0),
            wireframe=True,
            visible=False
        )

    def install(self, viser_server: ViserServer) -> None:
        self._create_scene_handle(viser_server)
