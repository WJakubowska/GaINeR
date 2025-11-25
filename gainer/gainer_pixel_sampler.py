"""
Code for sampling pixels.
"""

import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.utils.pixel_sampling_utils import divide_rays_per_image, erode_mask
from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig


@dataclass
class GainerPixelSamplerConfig(PixelSamplerConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: GainerPixelSampler)
    """Target class to instantiate."""


class GainerPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: GainerPixelSamplerConfig

    def __init__(self, config: GainerPixelSamplerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        # Stateful batching over the full set of pixel indices
        self._full_indices: Optional[Tensor] = None
        self._cursor: int = 0
        self._signature: Optional[tuple] = None

    def _ensure_full_indices(
        self,
        num_images: int,
        image_height: int,
        image_width: int,
        device: Union[torch.device, str],
    ) -> None:
        """Build the full ordered list of (image_idx, y, x) once per image spec.

        Rebuilds if the image spec (signature) changes.
        """
        sig = (num_images, image_height, image_width, str(device))
        if self._full_indices is not None and self._signature == sig:
            return
        img_range = torch.arange(num_images, device=device)
        y_range = torch.arange(image_height, device=device)
        x_range = torch.arange(image_width, device=device)
        grid = torch.meshgrid(img_range, y_range, x_range, indexing="ij")
        indices = torch.stack([g.reshape(-1) for g in grid], dim=-1).long()
        self._full_indices = indices
        self._signature = sig
        self._cursor = 0


    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            if self.config.rejection_sample_mask:
                indices = self.rejection_sample_mask(
                    mask=mask,
                    num_samples=batch_size,
                    num_images=num_images,
                    image_height=image_height,
                    image_width=image_width,
                    device=device,
                )
            else:
                nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
                indices = nonzero_indices[chosen_indices]
        else:
            # Stateful sequential batching over the entire image grid.
            # Build (or reuse) the full list of indices and emit the next chunk.
            self._ensure_full_indices(num_images, image_height, image_width, device)
            assert self._full_indices is not None
            total = self._full_indices.shape[0]
            # Number left in the current sweep
            remaining = total - self._cursor
            take = min(batch_size, remaining)
            start = self._cursor
            end = start + take
            indices = self._full_indices[start:end]
            # Advance cursor; if we reached the end, reset for the next sweep
            self._cursor = end
            if self._cursor >= total:
                self._cursor = 0

        return indices
    
    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            elif self.config.fisheye_crop_radius is not None:
                indices = self.sample_method_fisheye(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            else:
                indices = self.sample_method(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
        else:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            elif self.config.fisheye_crop_radius is not None:
                indices = self.sample_method_fisheye(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            else:
                indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch
