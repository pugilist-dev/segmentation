import torch
from cellpose import models, core, io
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt # use in debug console 
from src.deep_learning.base import BaseSegmenter
import loguru as log

class CellposeSegmentor(BaseSegmenter):
    def __init__(self, config):
        """
        Initialize the Cellpose segmentor.
        
        This class is a wrapper around the Cellpose deep learning model for image segmentation.
        It inherits from BaseSegmenter and implements the segment method.
        """
        self.config = config
        
        if core.use_gpu() == False:
            raise ImportError("No GPU access")
        
        if not Path(self.config.DEEP_LEARNING_MODELS_DIR).exists():
            log.logger.warning("Pretrained model path does not exist, using default model.")
            self.config.DEEP_LEARNING_CONFIG["model"]["name"] = "cpsam"  # Default model if not specified # not sure why syntax is so cursed 
        
        if self.config.MODEL == 'cellpose': 
            self.model = models.CellposeModel(gpu = True, 
                                            pretrained_model=str(Path(self.config.DEEP_LEARNING_MODELS_DIR, self.config.DEEP_LEARNING_CONFIG["model"]["name"])), 
                                            device=torch.device(self.config.DEEP_LEARNING_CONFIG["device"]))

        else:
            pass # For future addition of models 

        log.logger.debug("Cellpose Segmentor initialized.")

    def save_masks(self, masks):
        if not Path(self.config.PROCESSED_DATA_DIR).exists():
            self.config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        for i, mask in enumerate(masks):
            mask_path = Path(self.config.PROCESSED_DATA_DIR, f"mask_{i}.png")
            cv2.imwrite(mask_path, mask)

    def segment(self, images):
        masks, _, _ =  self.model.eval(images,diameter=15,channels=[0, 0]) # test if pasing all the frames at once or one at a time is faster 
        
        # return np.array(masks).astype(bool).astype(np.uint8)*255 # binarize the masks for visual check
        
        return masks