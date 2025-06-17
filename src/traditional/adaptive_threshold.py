"""
Adaptive thresholding segmentation implementation.
"""

import cv2
import numpy as np
from .base import BaseSegmenter


class AdaptiveThresholdSegmenter(BaseSegmenter):
    """
    Segmentation using adaptive thresholding.
    
    Adaptive thresholding calculates the threshold for small regions of the image,
    which gives better results for images with varying illumination.
    """
    
    def __init__(self, config=None):
        """
        Initialize the adaptive threshold segmenter.
        
        Args:
            config (dict, optional): Configuration parameters.
                enable (bool): Whether to enable this segmentation method.
                block_size (int): Size of the local neighborhood for threshold calculation.
                c (int): Constant subtracted from the mean or weighted sum.
                dilation_kernel_size (int): Size of the kernel for dilation operation.
        """
        super().__init__(config)
        self.block_size = self.config.get('block_size', 11)
        self.c = self.config.get('c', 2)
        self.dilation_kernel_size = self.config.get('dilation_kernel_size', 3)
    
    def preprocess(self, image):
        """
        Preprocess the image for adaptive thresholding.
        
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
            print(f"Error in adaptive thresholding preprocessing: {e}")
            # Return a simple grayscale conversion as fallback
            if len(image.shape) == 3 and image.shape[2] > 1:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image.astype(np.uint8)
        
    def postprocess(self, mask):
        """
        Postprocess the mask for adaptive thresholding.
        
        Args:
            mask (numpy.ndarray): Input mask to postprocess.
            
        Returns:
            numpy.ndarray: Postprocessed mask.
        """
        # Call the parent class's postprocess method
        return super().postprocess(mask)
        
    def segment(self, image):
        """
        Segment the input image using adaptive thresholding.
        
        Args:
            image (numpy.ndarray): Input image to segment.
            
        Returns:
            numpy.ndarray: Binary mask where 1 indicates the segmented region.
        """
        # Create a default mask in case of errors
        default_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Preprocess the image
        try:
            gray = self.preprocess(image)
            if gray is None:
                print("Gray scale conversion failed, using fallback")
                # Create a fallback grayscale image
                if len(image.shape) == 3 and image.shape[2] > 1:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                gray = gray.astype(np.uint8)
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return self.postprocess(default_mask)
        
        # Ensure the image is in uint8 format
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
            
        try:
            # Ensure block_size is odd
            block_size = self.block_size
            if block_size % 2 == 0:
                block_size += 1
            
            # Apply adaptive thresholding with BINARY_INV to match Otsu's convention
            mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,  # Use INV to match Otsu's convention
                block_size,
                self.c
            )
            
            # Calculate foreground percentage for validation
            foreground_percent = np.sum(mask > 0) / mask.size * 100
            
            # Check if thresholding was successful
            if mask is None:
                raise ValueError("Adaptive thresholding failed to produce a mask")
            
            # Normalize to binary (0 and 1) with BINARY_INV convention
            binary_mask = (mask / 255).astype(np.uint8)
            
            # Invert to match Otsu's convention (foreground=0, background=1)
            binary_mask = 1 - binary_mask
            
            # Apply dilation to connect broken object outlines
            kernel_size = self.dilation_kernel_size
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # Dilate the foreground (where mask is 0)
            inverted_for_dilate = 1 - binary_mask
            dilated_inverted = cv2.dilate(inverted_for_dilate, kernel, iterations=1)
            dilated_mask = 1 - dilated_inverted  # Convert back to original convention (foreground=0)
            
            # Get the min_hole_size from config
            min_hole_size = self.config.get('min_hole_size', 20)
            
            # For hole filling, we need to work with foreground=1, background=0
            # So invert the mask for processing
            inverted_for_holes = 1 - dilated_mask
            
            # Step 1: Use morphological closing with a larger kernel for initial hole filling
            closing_kernel_size = max(3, min(min_hole_size // 4, 7))
            closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
            closed_inverted = cv2.morphologyEx(inverted_for_holes, cv2.MORPH_CLOSE, closing_kernel)
            
            # Step 2: Fill all holes in each connected component (working on inverted mask)
            num_labels, labels = cv2.connectedComponents(closed_inverted)
            
            # Create a mask for the filled result
            filled_inverted = np.zeros_like(closed_inverted)
            hole_count = 0
            
            # Process each labeled component
            for label in range(1, num_labels):  # Skip background (0)
                # Create a binary mask for this component
                component = (labels == label).astype(np.uint8)
                
                # Find the component's contour
                contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create a new mask and fill the contour completely
                component_filled = np.zeros_like(component)
                for contour in contours:
                    cv2.drawContours(component_filled, [contour], 0, 1, -1)
                
                # Count holes that were filled
                holes_filled = np.sum(component_filled) - np.sum(component)
                if holes_filled > 0:
                    hole_count += 1
                
                # Add this filled component to our result
                filled_inverted = filled_inverted | component_filled
            
            # Convert back to original convention (foreground=0, background=1)
            filled_mask = 1 - filled_inverted
            
            # Safety checks: If the filled mask has too much (>95%) or too little (<5%) foreground,
            # it's likely something went wrong - revert to the initial binary mask
            filled_foreground = np.sum(filled_mask == 0)  # Count foreground pixels (value=0)
            foreground_percentage = filled_foreground / filled_mask.size
            if foreground_percentage > 0.95:
                filled_mask = binary_mask
            elif foreground_percentage < 0.05:
                filled_mask = binary_mask
                
            # Apply post-processing (watershed separation, etc.)
            processed_mask = self.postprocess(filled_mask)
            
            return processed_mask
        except Exception as e:
            # Fallback to simple thresholding
            try:
                ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                simple_mask = (mask / 255).astype(np.uint8)
                return self.postprocess(simple_mask)
            except Exception as e2:
                return self.postprocess(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)) 