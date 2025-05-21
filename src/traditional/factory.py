"""
Factory class for creating and managing traditional segmentation algorithms.
"""

import numpy as np
import traceback
from .otsu import OtsuSegmenter
from .adaptive_threshold import AdaptiveThresholdSegmenter


class TraditionalSegmenterFactory:
    """
    Factory class for creating and running traditional segmentation algorithms.
    
    This class allows you to create multiple segmenters and run them sequentially
    or in combination to produce final segmentation results.
    """
    
    def __init__(self, config=None):
        """
        Initialize the factory with the configuration.
        
        Args:
            config (dict, optional): Configuration for all segmentation methods.
                Should have keys for each method ('otsu', 'adaptive_threshold')
                with their respective configurations.
        """
        self.config = config or {}
        self.segmenters = {}
        
        # Initialize enabled segmenters
        self._init_segmenters()
    
    def _init_segmenters(self):
        """Initialize all enabled segmentation methods based on config."""
        # Otsu thresholding
        otsu_config = self.config.get('otsu', {})
        if otsu_config.get('enable', True):
            self.segmenters['otsu'] = OtsuSegmenter(otsu_config)
        
        # Adaptive thresholding
        adaptive_config = self.config.get('adaptive_threshold', {})
        if adaptive_config.get('enable', True):
            self.segmenters['adaptive_threshold'] = AdaptiveThresholdSegmenter(adaptive_config)
    
    def segment(self, image, method=None):
        """
        Segment the image using the specified method or all enabled methods.
        
        Args:
            image (numpy.ndarray): The input image to segment.
            method (str, optional): The specific method to use ('otsu' or 'adaptive_threshold'). 
                                    If None, all enabled methods are used.
                                     
        Returns:
            dict or numpy.ndarray: If method is None, returns a dictionary with all results.
                                   Otherwise, returns the mask from the specified method.
        """
        if method is not None:
            if method not in self.segmenters:
                raise ValueError(f"Method '{method}' not available or not enabled")
            try:
                return self.segmenters[method].segment(image)
            except Exception as e:
                print(f"Error segmenting with method '{method}': {e}")
                # Return a default mask on error
                if len(image.shape) == 3:
                    return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                else:
                    return np.zeros_like(image, dtype=np.uint8)
        
        # Run all enabled methods and collect results
        results = {}
        for name, segmenter in self.segmenters.items():
            try:
                mask = segmenter.segment(image)
                # Validate mask
                if mask is not None and mask.size > 0:
                    results[name] = mask
                else:
                    print(f"Warning: Method '{name}' returned an invalid mask")
            except Exception as e:
                print(f"Error with segmentation method '{name}': {e}")
                traceback.print_exc()
        
        return results
    
    def segment_combine(self, image, combine_method='vote'):
        """
        Segment using all enabled methods and combine the results.
        
        Args:
            image (numpy.ndarray): Input image to segment.
            combine_method (str): Method to combine results:
                - 'vote': Majority voting (default)
                - 'union': Union of all masks
                - 'intersection': Intersection of all masks
                
        Returns:
            numpy.ndarray: Combined binary mask.
        """
        try:
            results = self.segment(image)
            if not results:
                raise ValueError("No segmentation methods enabled or all methods failed")
            
            # Stack masks for processing
            masks = np.stack(list(results.values()), axis=0)
            
            if combine_method == 'vote':
                # Majority vote (more than half of the methods)
                threshold = masks.shape[0] / 2
                combined = np.sum(masks, axis=0) > threshold
                
            elif combine_method == 'union':
                # Union (any method detected)
                combined = np.any(masks, axis=0)
                
            elif combine_method == 'intersection':
                # Intersection (all methods must detect)
                combined = np.all(masks, axis=0)
                
            else:
                raise ValueError(f"Unknown combine method: {combine_method}")
            
            return combined.astype(np.uint8)
        except Exception as e:
            print(f"Error combining segmentation methods: {e}")
            # Return a default mask on error
            if len(image.shape) == 3:
                return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                return np.zeros_like(image, dtype=np.uint8) 