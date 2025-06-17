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
                # Log error without printing to console
                return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Run all enabled methods and collect results
        results = {}
        for name, segmenter in self.segmenters.items():
            try:
                mask = segmenter.segment(image)
                results[name] = mask
            except Exception as e:
                # Log error without printing to console
                results[name] = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        return results
    
    def segment_combine(self, segmentation_results, image, combine_method='vote'):
        """
        Combine segmentation results from multiple methods.
        
        Args:
            segmentation_results (dict): Results from segment() method
            image (numpy.ndarray): Original input image  
            combine_method (str): Method to combine results ('vote', 'union', 'intersection')
            
        Returns:
            numpy.ndarray: Combined segmentation mask
        """
        try:
            # Filter out invalid results
            valid_results = {}
            for name, mask in segmentation_results.items():
                if mask is not None and mask.size > 0:
                    # Validate mask dimensions
                    expected_shape = (image.shape[0], image.shape[1])
                    if mask.shape[:2] == expected_shape:
                        valid_results[name] = mask
                    # else: silently skip invalid masks
                # else: silently skip invalid masks
            
            if not valid_results:
                # Return empty mask if no valid results
                return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # Convert all masks to same format (binary)
            binary_masks = []
            for name, mask in valid_results.items():
                # Ensure mask is binary (0 and 1)
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
                if np.max(mask) > 1:
                    mask = (mask > 0).astype(np.uint8)
                binary_masks.append(mask)
            
            # Stack masks for combination
            stacked = np.stack(binary_masks, axis=2)
            
            if combine_method == 'vote':
                # Majority voting
                return (np.mean(stacked, axis=2) >= 0.5).astype(np.uint8)
            elif combine_method == 'union':
                # Union (any method detects foreground)
                return (np.max(stacked, axis=2)).astype(np.uint8)
            elif combine_method == 'intersection':
                # Intersection (all methods must agree)
                return (np.min(stacked, axis=2)).astype(np.uint8)
            else:
                # Default to majority voting
                return (np.mean(stacked, axis=2) >= 0.5).astype(np.uint8)
                
        except Exception as e:
            # Return empty mask as fallback
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) 