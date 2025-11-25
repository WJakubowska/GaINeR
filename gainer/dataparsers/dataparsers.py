from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig, Blender

@dataclass
class GainerBlenderDataParserConfig(BlenderDataParserConfig):
    """Configuration for Gainer Blender data parser."""

    _target: Type = field(default_factory=lambda: GainerBlender)

class GainerBlender(Blender):
    """Gainer Blender data parser.

    This class extends the BlenderDataParser to handle GaINeR-specific data parsing.
    """

    def __init__(self, config: GainerBlenderDataParserConfig):

        config.ply_path ="sparse_pc.ply"
        super().__init__(config)
