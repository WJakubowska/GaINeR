"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from typing import Union
from pathlib import Path

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification

from gainer.dataparsers.dataparsers import GainerBlenderDataParserConfig
from gainer.gainer_trainer import GainerTrainerConfig
from gainer.gainer_model import GainerModelConfig
from gainer.knn.knn_algorithms import TorchKNNConfig, OptixKNNConfig
from gainer.utils.schedulers import ChainedSchedulerConfig
from gainer.gainer_pipeline import GainerPipelineConfig
from gainer.dataparsers.mirage_dataparser import MirageDataParserConfig
from gainer.gainer_pixel_sampler import GainerPixelSamplerConfig

class CustomVanillaDataManagerConfig(VanillaDataManagerConfig):
    dataparser: Union[GainerBlenderDataParserConfig, MirageDataParserConfig]


MAX_NUM_ITERATIONS = 30_000
NUM_POINTS = 500_000

gainer = MethodSpecification(
    config=GainerTrainerConfig(
        method_name="gainer",
        steps_per_eval_batch=500,
        steps_per_save=1000,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=True,
        use_grad_scaler=True,
        train_mask=False,
        pipeline=GainerPipelineConfig(
            datamanager=CustomVanillaDataManagerConfig(
                dataparser=MirageDataParserConfig(
                    data=Path("data/KODAK"),
                    image_name="kodim12.png",
                    distance=1.0,
                    num_points=NUM_POINTS,
                ),
                pixel_sampler=GainerPixelSamplerConfig(
                    rejection_sample_mask=False
                ),
            ),
            model=GainerModelConfig(
                knn_algorithm=OptixKNNConfig(
                    n_neighbours=16,
                    chi_squared_radius=0.1,
                ),
                densify=False,
                prune=True,
                unfreeze_means=False,
                near_plane=0.0,
                far_plane=1e3,
                background_color="black",
                disable_scene_contraction=True,
                cone_angle=0.0, 
                disable_viewer_points=True,
                ),
            mask_path = None
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, weight_decay=1e-06),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
            "log_covs": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Geometry-Aware Implicit Neural Representation for Image Editing",
)