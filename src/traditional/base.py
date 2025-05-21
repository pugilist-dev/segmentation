"""
Base class for all traditional segmentation algorithms.
"""

import cv2
import numpy as np
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
    def segment(self, image):
        """
        Segment the input image.
        
        Args:
            image (numpy.ndarray): Input image to segment.
            
        Returns:
            numpy.ndarray: Binary mask where 1 indicates the segmented region.
        """
        pass
    
    def preprocess(self, image):
        """
        Preprocess the input image before segmentation.
        
        Args:
            image (numpy.ndarray): Input image to preprocess.
            
        Returns:
            numpy.ndarray: Preprocessed image.
        """
        # Default preprocessing just converts to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] > 1:
            return np.mean(image, axis=2).astype(np.uint8)
        return image
    
    def postprocess(self, mask):
        """
        Postprocess the segmentation mask.
        
        Args:
            mask (numpy.ndarray): Segmentation mask to postprocess.
            
        Returns:
            numpy.ndarray: Postprocessed mask.
        """
        # Debug: Check initial mask
        print(f"Postprocess - Initial mask: unique values: {np.unique(mask)}, foreground: {np.sum(mask > 0)}/{mask.size} ({np.sum(mask > 0)/mask.size:.2%})")
        
        # Get post-processing parameters from config
        use_watershed_separation = self.config.get('use_watershed_separation', False)
        min_distance = self.config.get('watershed_min_distance', 10)
        min_object_size = self.config.get('min_object_size', 20)
        min_hole_size = self.config.get('min_hole_size', 20)
        
        # Apply watershed separation if enabled
        if use_watershed_separation:
            mask_before = mask.copy()
            mask = self._watershed_separation(
                mask,
                min_distance=min_distance,
                min_object_size=min_object_size,
                min_hole_size=min_hole_size
            )
            print(f"Postprocess - After watershed: unique values: {np.unique(mask)}, foreground: {np.sum(mask > 0)}/{mask.size} ({np.sum(mask > 0)/mask.size:.2%})")
        
        # Filter small objects if specified
        if min_object_size > 0 and not use_watershed_separation:
            mask_before = mask.copy()
            mask = self._remove_small_objects(mask, min_object_size)
            print(f"Postprocess - After small object removal: unique values: {np.unique(mask)}, foreground: {np.sum(mask > 0)}/{mask.size} ({np.sum(mask > 0)/mask.size:.2%})")
        
        # Fill small holes if specified
        if min_hole_size > 0 and not use_watershed_separation:
            mask_before = mask.copy()
            mask = self._fill_holes(mask, min_hole_size)
            print(f"Postprocess - After hole filling: unique values: {np.unique(mask)}, foreground: {np.sum(mask > 0)}/{mask.size} ({np.sum(mask > 0)/mask.size:.2%})")
        
        return mask
    
    def _watershed_separation(self, binary_mask, min_distance=10, min_object_size=20, min_hole_size=20):
        """
        Use watershed algorithm to separate touching objects in binary masks.
        
        Args:
            binary_mask: Binary mask where 1 indicates foreground
            min_distance: Minimum distance between local maxima
            min_object_size: Minimum size of objects to keep
            min_hole_size: Minimum size of holes to fill
            
        Returns:
            Processed binary mask with separated objects
        """
        try:
            # Keep a copy of the original mask to limit the final result
            original_mask = binary_mask.copy()
            
            # Ensure mask is binary (0 and 1)
            if np.max(binary_mask) > 1:
                binary_mask = binary_mask.astype(bool).astype(np.uint8)
            
            # Fill small holes first to avoid creating tiny objects
            if min_hole_size > 0:
                mask_before_hole_filling = binary_mask.copy()
                binary_mask = self._fill_holes(binary_mask, min_hole_size)
                print(f"Watershed - After hole filling: foreground: {np.sum(binary_mask > 0)}/{binary_mask.size} ({np.sum(binary_mask > 0)/binary_mask.size:.2%})")
            
            # If the mask is empty, just return it
            if np.sum(binary_mask) == 0:
                print("Watershed - Empty mask, returning as is")
                return binary_mask
            
            # Compute the distance transform
            distance = ndi.distance_transform_edt(binary_mask)
            print(f"Watershed - Distance transform: min={np.min(distance)}, max={np.max(distance)}, mean={np.mean(distance):.2f}")
            
            # Find local maxima in the distance transform
            local_max = self._find_local_maxima(distance, min_distance)
            num_maxima = np.sum(local_max)
            print(f"Watershed - Found {num_maxima} local maxima")
            
            # Label local maxima to create markers
            markers, num_features = ndi.label(local_max, structure=np.ones((3, 3)))
            print(f"Watershed - Labeled {num_features} distinct features/markers")
            
            # Create watershed line image (needed for watershed)
            binary_mask_uint8 = np.zeros_like(binary_mask, dtype=np.uint8)
            binary_mask_uint8[binary_mask > 0] = 255
            
            # Only proceed with watershed if we have more than one local maximum
            if num_features > 1:
                print("Watershed - Applying watershed with multiple markers")
                
                # Make sure background markers are present
                # Add background marker (0) for watershed
                bg_mask = ~binary_mask.astype(bool)
                markers[bg_mask] = 0
                
                # Watershed segmentation
                watershed_result = cv2.watershed(
                    cv2.cvtColor(binary_mask_uint8, cv2.COLOR_GRAY2BGR),
                    markers.astype(np.int32)
                )
                
                # Create a new binary mask (watershed creates -1 at borders)
                separated_mask = np.zeros_like(binary_mask, dtype=np.uint8)
                separated_mask[watershed_result > 0] = 1
                
                # Limit the result to only include pixels from the original mask
                separated_mask = separated_mask & original_mask
                
                # Count number of separated objects
                labeled_result, num_objects = ndi.label(separated_mask)
                print(f"Watershed - Separated into {num_objects} objects")
            else:
                print("Watershed - Only one marker found, using original mask")
                # If only one maximum, just use the original mask
                separated_mask = binary_mask
            
            # Remove small objects
            if min_object_size > 0:
                mask_before_removal = separated_mask.copy()
                separated_mask = self._remove_small_objects(separated_mask, min_object_size)
                print(f"Watershed - After small object removal: foreground: {np.sum(separated_mask > 0)}/{separated_mask.size} ({np.sum(separated_mask > 0)/separated_mask.size:.2%})")
            
            return separated_mask
        except Exception as e:
            print(f"Error in watershed separation: {e}")
            return binary_mask
    
    def _find_local_maxima(self, distance_transform, min_distance):
        """Find local maxima in the distance transform."""
        try:
            # Find the maximum value in the distance transform
            max_val = np.max(distance_transform)
            if max_val == 0:
                return np.zeros_like(distance_transform, dtype=bool)
            
            # Normalize the distance transform to range [0, 1]
            distance_norm = distance_transform / max_val
            
            # Find local maxima using peak_local_max
            from skimage.feature import peak_local_max
            
            # Use a lower min_distance for small objects
            adjusted_min_distance = min(min_distance, max(3, int(max_val/2)))
            
            # Find local maxima
            coordinates = peak_local_max(
                distance_norm, 
                min_distance=adjusted_min_distance,
                exclude_border=False
            )
            
            # Convert coordinates to mask
            local_max = np.zeros_like(distance_transform, dtype=bool)
            if coordinates.size > 0:
                local_max[tuple(coordinates.T)] = True
            
            return local_max
        except ImportError:
            # Fall back to simple maximum if scikit-image is not available
            print("Warning: scikit-image not available, using simple maximum")
            binary = distance_transform > 0.7 * np.max(distance_transform)
            return binary
        except Exception as e:
            print(f"Error finding local maxima: {e}")
            return np.zeros_like(distance_transform, dtype=bool)
    
    def _remove_small_objects(self, binary_mask, min_size):
        """Remove small objects from a binary mask."""
        # Get the original foreground to check later
        original_foreground = np.sum(binary_mask > 0)
        print(f"Original foreground pixels: {original_foreground}/{binary_mask.size} ({original_foreground/binary_mask.size:.2%})")
        
        # Label connected components
        labeled, num_features = ndi.label(binary_mask, structure=np.ones((3, 3)))
        
        # Calculate area of each component
        component_sizes = np.bincount(labeled.ravel())
        
        print(f"Remove small objects - component sizes: {component_sizes}, num components: {len(component_sizes)}")
        if len(component_sizes) > 0:
            print(f"Remove small objects - largest component: {np.max(component_sizes)}, smallest component: {np.min(component_sizes[1:] if len(component_sizes) > 1 else [0])}")
        
        # Set threshold for size filtering (component 0 is background)
        too_small = component_sizes < min_size
        too_small[0] = False  # Don't remove background
        
        # Check if we would remove all objects
        if np.all(too_small[1:]) and len(too_small) > 1:
            print(f"Warning: All components are smaller than min_size={min_size}. Keeping the largest component.")
            # Find the largest component and keep it
            largest_comp_idx = np.argmax(component_sizes[1:]) + 1  # +1 because we skip background
            too_small[largest_comp_idx] = False
        
        # Create a mask of components to keep
        filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        for i in range(len(component_sizes)):
            if i > 0 and not too_small[i]:  # Only foreground components (i > 0)
                filtered_mask[labeled == i] = 1
        
        # Make sure we're not accidentally including background as foreground
        if binary_mask.dtype == np.uint8:
            filtered_mask &= binary_mask
        
        # Verify we're maintaining a reasonable mask
        new_foreground = np.sum(filtered_mask > 0)
        print(f"New foreground pixels: {new_foreground}/{filtered_mask.size} ({new_foreground/filtered_mask.size:.2%})")
        
        # If the filtered mask is empty but the original had foreground, keep the original
        if new_foreground == 0 and original_foreground > 0:
            print("Warning: Filtered mask is empty but original had foreground. Keeping original mask.")
            return binary_mask
        
        return filtered_mask
    
    def _fill_holes(self, binary_mask, min_hole_size):
        """Fill small holes in a binary mask."""
        # Invert the mask to detect holes
        inverted_mask = 1 - binary_mask
        
        # Label connected components in the inverted mask
        labeled, num_features = ndi.label(inverted_mask, structure=np.ones((3, 3)))
        
        # Calculate area of each component
        component_sizes = np.bincount(labeled.ravel())
        
        # Identify small holes (component 0 is the background/exterior)
        small_holes = (component_sizes > 0) & (component_sizes < min_hole_size)
        small_holes[0] = False  # Don't fill the exterior
        
        # Fill small holes
        filled_mask = binary_mask.copy()
        for i in range(1, num_features + 1):
            if small_holes[i]:
                filled_mask[labeled == i] = 1
        
        return filled_mask 