"""
Base class for all traditional segmentation algorithms.
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseSegmenter(ABC):
    """Base class that all traditional segmentation algorithms should inherit from."""
    
    def __init__(self, config=None):
        """
        Initialize the segmenter.
        
        Args:
            config (dict, optional): Configuration parameters for the segmenter.
        """
        self.config = config or {}
        
    @abstractmethod
    def segment(self, images) -> np.ndarray:
        """
        Segment the input images.
        
        Args:
            List of images (numpy.ndarray with shape NUM IMAGES * HEIGHT * WIDTH * 3): Input images to segment.
            
        Returns:
            numpy.ndarray: Binary mask where 1 indicates the segmented region.
        """
        pass
    
    @abstractmethod
    def preprocess(self, images) -> np.ndarray: # get_composites shouuld be here 
        """
        Preprocess the input image before segmentation.
        
        Args:
            image (numpy.ndarray): Input image to preprocess.
            
        Returns:
            numpy.ndarray: Preprocessed image.
        """
        pass
    
    @abstractmethod
    def postprocess(self, masks=None, images=None) -> list[np.ndarray]:
        """
        Postprocess the segmentation mask. Extracts cropped cell images using the segmented masks.       

        Arguments:
            masks (np.ndarray): Array of segmented masks with shape (N, C, H, W).
            images (np.ndarray): Array of original images with shape (N, C, H, W).
        Returns:
            List[np.ndarray]: List of cropped cell images.
        """
        pass
  