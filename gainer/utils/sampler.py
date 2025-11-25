import torch
from typing import Tuple, Optional, Callable
from jaxtyping import Float
from torch import Tensor

# Nerfstudio core imports
from nerfstudio.cameras.rays import RayBundle, Frustums, RaySamples
from nerfstudio.model_components.ray_samplers import Sampler


class PlaneSampler(Sampler):
    def __init__(self, plane_z: float = 0.0, background_color: str = "white", sampling_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.plane_z = plane_z
        self.background_color = background_color
        self.sampling_mask = sampling_mask

    def forward(
        self,
        ray_bundle: RayBundle,
    ) -> Tuple[RaySamples, Float[Tensor, "total_samples"]]:

        origins = ray_bundle.origins.contiguous()
        directions = ray_bundle.directions.contiguous()

        t = (self.plane_z - origins[:, 2]) / directions[:, 2]

        if self.sampling_mask is None:
            self.sampling_mask = torch.ones_like(ray_bundle.origins[:, 0], dtype=torch.bool)

        ray_indices = torch.nonzero(self.sampling_mask, as_tuple=False).view(-1)

        if ray_indices.numel() == 0:
            ray_indices = torch.zeros((1,), dtype=torch.long, device=origins.device)
            starts = torch.ones((1, 1), dtype=origins.dtype, device=origins.device)
            ends = starts + 5e-4
        else:
            starts = t[self.sampling_mask][..., None]
            ends = starts + 5e-4 

        selected_origins = origins[ray_indices]
        selected_dirs = directions[ray_indices]


        camera_indices = getattr(ray_bundle, "camera_indices", None)
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]


        pixel_area = getattr(ray_bundle, "pixel_area", None)
        if pixel_area is not None:
            pixel_area = pixel_area[ray_indices]


        expected = torch.tensor([0., 0., 1.], device='cuda:0')
        selected_origins = torch.ones_like(selected_origins) * expected[None, :]

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=selected_origins,
                directions=selected_dirs,
                starts=starts,
                ends=ends,
                pixel_area=pixel_area,
            ),
            camera_indices=camera_indices,
        )

        if hasattr(ray_bundle, "times") and ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[ray_indices]

        return ray_samples, ray_indices
    
    def update_sampling_mask(self, new_mask: torch.Tensor):
        self.sampling_mask = new_mask
