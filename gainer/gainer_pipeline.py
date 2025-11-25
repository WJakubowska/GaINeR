import cv2
import typing
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from pathlib import Path

from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Optional, Type
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.models.base_model import Model

@dataclass
class GainerPipelineConfig(VanillaPipelineConfig):
    """Configuration for the GainerPipeline."""

    _target: Type = field(default_factory=lambda: GainerPipeline)
    """target class to instantiate"""
    
    mask_path: Optional[Path] = None
    """Optional path to a mask image to be used during rendering"""


class GainerPipeline(VanillaPipeline):

    config: GainerPipelineConfig
    datamanager: VanillaDataManager

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        img = cv2.imread(config.datamanager.data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {config.datamanager.data}")

        # If image has an alpha channel, use it as a sampling mask (non-zero alpha = valid)
        if config.mask_path is not None and Path(config.mask_path).exists():
            mask_img = cv2.imread(config.mask_path, cv2.IMREAD_GRAYSCALE)
            mask_bool = (mask_img > 127).astype(np.bool_)
            mask_tensor = torch.from_numpy(mask_bool.flatten()).to('cuda') 
            sampling_mask = mask_tensor
        elif img.ndim == 3 and img.shape[2] == 4:
            sampling_mask = torch.from_numpy((img[:, :, 3] > 127).flatten()).to('cuda')
        else:
            sampling_mask = None

        image_size = int(img.shape[0] * img.shape[1])
        config.datamanager.train_num_rays_per_batch = image_size
        config.datamanager.eval_num_rays_per_batch = image_size
        config.model.eval_num_rays_per_chunk = image_size

        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata  # type: ignore
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]  # type: ignore
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]  # type: ignore
            seed_pts = (pts, pts_rgb)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
            sampling_mask=sampling_mask,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }

        means_size = None
        for key, value in state.items():
            if key == "_model.field.mlp_base.encoder.gauss_params.means":
                means_size = value.shape[0]
                break
        
        self.model.field.mlp_base.encoder.reinitialize_params(means_size)

        self.model.update_to_step(step)
        self.load_state_dict(state)
        self.model.field.mlp_base.encoder.knn.fit(self.model.field.mlp_base.encoder.means)
