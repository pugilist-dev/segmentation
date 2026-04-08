from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    pretrained_model: Path
    device: str
    data_dir: Path
    image_extension: str
    mask_output_dir: Path
    offset: int = 10

