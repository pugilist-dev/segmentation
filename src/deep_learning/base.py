"""
Base class for all traditional segmentation algorithms.
"""
from abc import ABC, abstractmethod
from scipy import ndimage as ndi

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
    def segment(self, images):
        """
        Segment the input images.
        
        Args:
            List of images (numpy.ndarray with shape NUM IMAGES * HEIGHT * WIDTH * 3): Input images to segment.
            
        Returns:
            numpy.ndarray: Binary mask where 1 indicates the segmented region.
        """
        pass
    
    def preprocess(self, images_dir): # get_composites shouuld be here 
        """
        Preprocess the input image before segmentation.
        
        Args:
            image (numpy.ndarray): Input image to preprocess.
            
        Returns:
            numpy.ndarray: Preprocessed image.
        """
        pass
    
    def postprocess(self, mask):
        """
        Postprocess the segmentation mask.
        
        Args:
            mask (numpy.ndarray): Segmentation mask to postprocess.
            
        Returns:
            numpy.ndarray: Postprocessed mask.
        """
        pass
  