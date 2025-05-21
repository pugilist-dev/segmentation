"""
Otsu thresholding segmentation implementation.
"""

import cv2
import numpy as np
from .base import BaseSegmenter


class OtsuSegmenter(BaseSegmenter):
    """
    Segmentation using Otsu's thresholding method.
    
    Otsu's method automatically determines an optimal threshold value by minimizing
    intra-class intensity variance, or equivalently, by maximizing inter-class variance.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Otsu segmenter.
        
        Args:
            config (dict, optional): Configuration parameters.
                enable (bool): Whether to enable this segmentation method.
                min_threshold (int): Minimum threshold value (default: 25)
        """
        super().__init__(config)
        self.min_threshold = self.config.get('min_threshold', 25)

    def preprocess(self, image):
        """
        Preprocess the image for Otsu's thresholding.
        
        Args:
            image (numpy.ndarray): Input image to preprocess.
            
        Returns:
            numpy.ndarray: Preprocessed grayscale image.
        """
        try:
            # Convert to grayscale if the image is color
            if len(image.shape) == 3 and image.shape[2] > 1:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            # Ensure the image is in uint8 format (required for threshold)
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)
            
            # Normalize the image to 0-255 range
            gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            return gray
        except Exception as e:
            print(f"Error in Otsu preprocessing: {e}")
            # Return a simple grayscale conversion as fallback
            if len(image.shape) == 3 and image.shape[2] > 1:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image.astype(np.uint8)
        
    def segment(self, image):
        """
        Segment the input image using Otsu's thresholding.
        
        Args:
            image (numpy.ndarray): Input image to segment.
            
        Returns:
            numpy.ndarray: Binary mask where 1 indicates the segmented region.
        """
        # Preprocess the image (convert to grayscale and normalize)
        gray = self.preprocess(image)
        
        # Ensure the image is in uint8 format (required for threshold)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        try:
            # First find the Otsu threshold value without applying the threshold
            # Use regular THRESH_BINARY with THRESH_OTSU to get the threshold value
            otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Print debug info about the threshold
            print(f"Otsu threshold value: {otsu_threshold}")
            
            # Take the minimum of Otsu's threshold and the configured minimum value
            threshold_value = min(otsu_threshold, self.min_threshold)
            
            # Apply the threshold with the calculated value
            # Use THRESH_BINARY_INV to get bright objects as foreground
            _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
            
            # Debug visualization of the mask distribution
            print(f"Otsu raw mask unique values: {np.unique(mask)}")
            foreground_percent = np.sum(mask > 0) / mask.size * 100
            print(f"Otsu thresholding: foreground is {foreground_percent:.2f}% of the image")
            
            # Debug info about the mask patterns
            if mask.size < 10000:  # Only for reasonably small masks
                print("Sample of Otsu threshold mask:")
                print(mask[0:min(10, mask.shape[0]), 0:min(10, mask.shape[1])])
            
            # Check if thresholding was successful
            if mask is None:
                raise ValueError("Otsu thresholding failed to produce a mask")
            
            # Normalize mask to binary (0 and 1)
            mask = (mask / 255).astype(np.uint8)
            
            # Apply any post-processing
            mask = self.postprocess(mask)
            
            return mask
        except Exception as e:
            print(f"Error in Otsu segmentation: {e}")
            # Fallback to simple thresholding if Otsu fails
            try:
                ret, mask = cv2.threshold(gray, self.min_threshold, 255, cv2.THRESH_BINARY_INV)
                simple_mask = (mask / 255).astype(np.uint8)
                return self.postprocess(simple_mask)
            except Exception as e2:
                print(f"Simple thresholding also failed: {e2}")
                return np.zeros_like(gray, dtype=np.uint8) 