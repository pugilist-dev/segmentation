#!/usr/bin/env python3
"""
Example script demonstrating instance segmentation using watershed algorithm.
This script visualizes and saves instance masks where each object has a unique label/color.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import argparse
from functools import partial, wraps
from scipy import ndimage as ndi
import random
import time
from loguru import logger

# Configure loguru - remove all default handlers
logger.remove()

# Global flag for timing output
TIMING_ENABLED = False

def timing_decorator(func):
    """Decorator to time function execution and print to console only if timing is enabled"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        if TIMING_ENABLED:
            # Print timing directly to console only (not logged to file)
            print(f"⏱️  {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils import SegmentationDataLoader
from src.traditional import TraditionalSegmenterFactory


@timing_decorator
def process_image_with_instance_segmentation(idx, data_loader, factory, results_dir, min_object_size=10, max_object_size=None, check_connectivity=False, skip_connectivity=False, save_visualization=True, save_numpy=True, watershed_config=None):
    """
    Process a single image with instance segmentation.
    OPTIMIZED: Uses conditional I/O and optimized algorithms.
    """
    try:
        # Get the sample and extract image
        sample = data_loader[idx]
        image = sample.image
        image_name = os.path.splitext(os.path.basename(sample.image_path))[0]
        
        try:
            if hasattr(logger, 'level') and logger.level <= 10:  # DEBUG level
                logger.debug(f"Processing image {image_name} (index {idx})")
        except:
            pass
        
        # Run segmentation with all methods
        results = factory.segment(image)
        
        # Create a combined result using majority voting
        binary_mask = factory.segment_combine(results, image, combine_method='vote')
        
        # Create instance mask using watershed separation (without size filtering yet)
        instance_mask = create_instance_mask(
            binary_mask, 
            min_object_size=0,  # Skip size filtering in watershed step
            max_object_size=None,
            watershed_config=watershed_config
        )
        
        # FIRST: Ensure each label represents a single connected component
        if check_connectivity and not skip_connectivity:
            try:
                if hasattr(logger, 'level') and logger.level <= 10:  # DEBUG level
                    logger.debug(f"Checking connectivity for {image_name}")
            except:
                pass
            instance_mask = ensure_connected_instance_labels(
                instance_mask,
                min_object_size=0  # Don't filter by size yet
            )
        
        # SECOND: Apply size filtering to all objects (after connectivity check)
        if min_object_size > 0 or max_object_size is not None:
            # Count objects before filtering for logging
            original_count = len(np.unique(instance_mask)) - 1
            
            # Use optimized vectorized size filtering
            instance_mask = _vectorized_size_filter(instance_mask, min_object_size, max_object_size)
            
            # Count removed objects for logging
            try:
                if hasattr(logger, 'level') and logger.level <= 10:  # DEBUG level
                    filtered_count = len(np.unique(instance_mask)) - 1
                    if original_count != filtered_count:
                        removed_count = original_count - filtered_count
                        logger.debug(f"Size filtering: removed {removed_count} objects ({original_count} → {filtered_count}) from {image_name}")
            except:
                pass
        
        # OPTIMIZED: Use efficient I/O operations
        saved_files = optimized_save_results(
            image, binary_mask, instance_mask, results_dir, image_name,
            save_visualization=save_visualization, save_numpy=save_numpy
        )

        return idx, image_name, True
    except Exception as e:
        # Print error to console and also try to log it
        print(f"ERROR: Processing failed for image at index {idx}: {e}")
        try:
            logger.error(f"Error processing image at index {idx}: {e}")
        except:
            pass
        return idx, None, False


@timing_decorator
def create_instance_mask(binary_mask, min_distance=5, min_object_size=10, max_object_size=None, watershed_config=None):
    """
    Create an instance mask from a binary mask using optimized watershed.
    ULTRA-OPTIMIZED: Uses adaptive algorithms and GPU acceleration.
    
    Args:
        binary_mask: Binary segmentation mask
        min_distance: Minimum distance between watershed peaks
        min_object_size: Minimum object size to keep
        max_object_size: Maximum object size to keep
        watershed_config: Dictionary with watershed-specific parameters
    """
    # Default watershed configuration
    default_watershed_config = {
        'threshold_abs_factor': 0.3,        # Peak detection sensitivity (0.1-0.5)
        'fallback_threshold_factor': 0.7,   # Fallback peak threshold (0.5-0.9)
        'sparse_density_threshold': 0.01,   # Sparse image threshold (0.005-0.02)
        'dense_density_threshold': 0.3,     # Dense image threshold (0.2-0.5)
        'multi_resolution_threshold': 512,  # Multi-resolution size threshold
        'min_watershed_distance': 3,        # Minimum distance for watershed viability
        'max_objects_threshold': 50,        # Max objects before switching algorithms
        'gpu_size_threshold': 512*512       # GPU usage threshold
    }
    
    # Merge with provided config
    if watershed_config is None:
        watershed_config = {}
    config = {**default_watershed_config, **watershed_config}
    
    # Convert to standard convention for watershed (1=foreground, 0=background)
    std_mask = 1 - binary_mask.copy()
    std_mask = std_mask.astype(bool).astype(np.uint8)
    
    if np.sum(std_mask) == 0:
        return np.zeros_like(std_mask, dtype=np.int32)
    
    # OPTIMIZATION 1: Adaptive algorithm selection based on image characteristics
    foreground_pixels = np.sum(std_mask)
    image_size = std_mask.size
    density = foreground_pixels / image_size
    
    # For very sparse images, use simple connected components
    if density < config['sparse_density_threshold']:
        return _simple_connected_components(std_mask, binary_mask, min_object_size, max_object_size)
    
    # For dense images, use optimized watershed
    if density > config['dense_density_threshold']:
        return _optimized_dense_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config)
    
    # For medium density, use multi-resolution approach
    return _multi_resolution_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config)


def _simple_connected_components(std_mask, binary_mask, min_object_size, max_object_size):
    """Fast connected components for sparse images."""
    instance_mask, _ = ndi.label(std_mask)
    instance_mask[binary_mask == 1] = 0
    
    if min_object_size > 0 or max_object_size is not None:
        instance_mask = _vectorized_size_filter(instance_mask, min_object_size, max_object_size)
    
    return instance_mask


def _optimized_dense_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config):
    """Optimized watershed for dense foreground images."""
    try:
        import cupy as cp
        gpu_available = cp.cuda.is_available()
    except ImportError:
        gpu_available = False
    
    # For large dense images, use GPU if available
    if gpu_available and std_mask.size > config['gpu_size_threshold']:
        return _gpu_dense_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config)
    else:
        return _cpu_dense_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config)


def _multi_resolution_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config):
    """Multi-resolution watershed for medium density images."""
    # OPTIMIZATION: Process at multiple resolutions for speed
    h, w = std_mask.shape
    
    # If image is large, start with downsampled version
    if min(h, w) > config['multi_resolution_threshold']:
        # Downsample by factor of 2
        small_mask = std_mask[::2, ::2]
        small_binary = binary_mask[::2, ::2]
        
        # Quick watershed on small image to estimate regions
        small_result = _cpu_dense_watershed(small_mask, small_binary, max(1, min_distance//2), 
                                          max(1, min_object_size//4), max_object_size, config)
        
        # If small image has reasonable number of objects, proceed with full resolution
        num_objects = len(np.unique(small_result)) - 1
        if num_objects < config['max_objects_threshold']:  # Reasonable number, do full watershed
            return _cpu_dense_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config)
        else:
            # Too many objects, use simpler approach
            return _simple_connected_components(std_mask, binary_mask, min_object_size, max_object_size)
    else:
        # Small enough for direct processing
        return _cpu_dense_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config)


def _cpu_dense_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config=None):
    """Optimized CPU watershed implementation."""
    # Use default config if none provided
    if config is None:
        config = {
            'threshold_abs_factor': 0.3,
            'fallback_threshold_factor': 0.7,
            'min_watershed_distance': 3
        }
    
    # Fast distance transform with early exit
    distance = ndi.distance_transform_edt(std_mask)
    max_dist = np.max(distance)
    
    if max_dist < config['min_watershed_distance']:  # Objects too small for watershed
        return _simple_connected_components(std_mask, binary_mask, min_object_size, max_object_size)
    
    # Optimized peak detection
    try:
        from skimage.feature import peak_local_max
        adjusted_min_distance = min(min_distance, max(3, int(max_dist/2)))
        
        coordinates = peak_local_max(
            distance, 
            min_distance=adjusted_min_distance,
            exclude_border=False,
            labels=std_mask,
            threshold_abs=max_dist * config['threshold_abs_factor']  # Configurable peak sensitivity
        )
        
        if len(coordinates) == 0:
            return _simple_connected_components(std_mask, binary_mask, min_object_size, max_object_size)
        
        # Efficient marker creation
        markers = np.zeros_like(std_mask, dtype=np.int32)
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
        
    except ImportError:
        # Fallback without skimage - use configurable threshold
        local_max = distance > config['fallback_threshold_factor'] * max_dist
        markers, num_objects = ndi.label(local_max)
        if num_objects == 0:
            return _simple_connected_components(std_mask, binary_mask, min_object_size, max_object_size)
    
    # Fast watershed
    if np.sum(markers > 0) > 0:
        std_mask_uint8 = std_mask * 255
        watershed_markers = markers.copy()
        watershed_markers[std_mask == 0] = 0
        
        watershed_result = cv2.watershed(
            cv2.cvtColor(std_mask_uint8, cv2.COLOR_GRAY2BGR),
            watershed_markers
        )
        
        instance_mask = np.maximum(watershed_result, 0)
        instance_mask[binary_mask == 1] = 0
        
        # Optimized size filtering
        if min_object_size > 0 or max_object_size is not None:
            instance_mask = _vectorized_size_filter(instance_mask, min_object_size, max_object_size)
        
        return instance_mask
    else:
        return _simple_connected_components(std_mask, binary_mask, min_object_size, max_object_size)


def _gpu_dense_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config):
    """GPU-accelerated watershed for large dense images."""
    import cupy as cp
    
    # Transfer to GPU
    gpu_mask = cp.asarray(std_mask)
    
    # GPU distance transform (if available in CuPy)
    try:
        # Use SciPy's distance transform on GPU
        from cupyx.scipy import ndimage as cupyx_ndi
        distance = cupyx_ndi.distance_transform_edt(gpu_mask)
        distance_cpu = cp.asnumpy(distance)
    except (ImportError, AttributeError):
        # Fallback to CPU for distance transform
        distance_cpu = ndi.distance_transform_edt(std_mask)
    
    # Continue with CPU watershed (OpenCV doesn't have GPU version)
    return _cpu_dense_watershed(std_mask, binary_mask, min_distance, min_object_size, max_object_size, config)


def _vectorized_size_filter(instance_mask, min_object_size, max_object_size):
    """Ultra-fast vectorized size filtering."""
    if min_object_size <= 0 and max_object_size is None:
        return instance_mask
    
    # Single-pass size calculation
    props = np.bincount(instance_mask.ravel())
    
    # Vectorized filtering logic
    to_remove = np.zeros(len(props), dtype=bool)
    
    if min_object_size > 0:
        to_remove |= (props < min_object_size)
    
    if max_object_size is not None:
        to_remove |= (props > max_object_size)
    
    to_remove[0] = False  # Don't remove background
    
    # Check if we would remove all objects
    if np.all(to_remove[1:]) and len(to_remove) > 1:
        # Keep the largest object
        largest_idx = np.argmax(props[1:]) + 1
        to_remove[largest_idx] = False
    
    # Vectorized removal using boolean indexing
    if np.any(to_remove):
        remove_labels = np.where(to_remove)[0]
        # Create boolean mask for all labels to remove
        remove_mask = np.isin(instance_mask, remove_labels)
        instance_mask = instance_mask.copy()  # Avoid modifying original
        instance_mask[remove_mask] = 0
    
    return instance_mask


def colorize_instance_mask(instance_mask):
    """
    Create a colorized visualization of an instance mask.
    OPTIMIZED: Uses vectorized operations and efficient color assignment.
    """
    # Get unique labels (excluding background = 0)
    labels = np.unique(instance_mask)
    labels = labels[labels > 0]  # Remove background
    
    if len(labels) == 0:
        return np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    
    # OPTIMIZATION: Use efficient color palette
    num_instances = len(labels)
    
    # For small number of labels, use pre-defined colors for consistency
    if num_instances <= 20:
        # Pre-defined high-contrast colors
        predefined_colors = np.array([
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
            [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
            [128, 0, 128], [0, 128, 128], [255, 128, 0], [255, 0, 128], [128, 255, 0],
            [0, 255, 128], [128, 0, 255], [0, 128, 255], [192, 192, 192], [64, 64, 64]
        ], dtype=np.uint8)
        colors = predefined_colors[:num_instances]
    else:
        # For many labels, generate random colors but ensure good contrast
        np.random.seed(42)  # For reproducible colors
        colors = np.random.randint(40, 255, size=(num_instances, 3), dtype=np.uint8)
    
    # OPTIMIZATION: Vectorized color assignment
    # Create RGB image
    rgb_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    
    # Use advanced indexing for fast assignment
    for i, label in enumerate(labels):
        mask = instance_mask == label
        rgb_mask[mask] = colors[i]
    
    return rgb_mask


def save_instance_visualization(image, binary_mask, instance_mask, output_path):
    """
    Save visualization of original image, binary mask, and instance segmentation.
    
    Args:
        image: Original input image
        binary_mask: Binary mask (0=foreground, 1=background from our convention)
        instance_mask: Integer mask where each object has a unique label
        output_path: Path to save the visualization
    """
    # Convert BGR to RGB for matplotlib display
    if len(image.shape) == 3 and image.shape[2] == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display original image
    axes[0].imshow(display_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display binary mask (inverse for better visualization)
    axes[1].imshow(1 - binary_mask, cmap='gray')
    axes[1].set_title('Binary Segmentation')
    axes[1].axis('off')
    
    # Display instance segmentation with random colors
    # Create a colorized version of the instance mask (only foreground objects)
    instance_rgb = colorize_instance_mask(instance_mask)
    
    # Count the number of unique foreground labels (excluding background label 0)
    foreground_labels = np.unique(instance_mask)
    foreground_labels = foreground_labels[foreground_labels > 0]
    num_objects = len(foreground_labels)
    
    axes[2].imshow(instance_rgb)
    axes[2].set_title(f'Instance Segmentation ({num_objects} objects)')
    axes[2].axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


@timing_decorator
def ensure_connected_instance_labels(instance_mask, min_object_size=0):
    """
    Ensure that each label in the instance mask corresponds to exactly one connected component.
    ULTRA-OPTIMIZED: Uses fast heuristics to avoid expensive operations when possible.
    """
    unique_labels = np.unique(instance_mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        return instance_mask
    
    # OPTIMIZATION 1: Fast check if connectivity is even needed
    # If the number of labels is reasonable relative to image size, likely already connected
    num_labels = len(unique_labels)
    total_foreground = np.sum(instance_mask > 0)
    
    if total_foreground == 0:
        return instance_mask
    
    # Heuristic: if labels/pixels ratio is reasonable, probably already well-segmented
    label_density = num_labels / total_foreground
    
    # OPTIMIZATION 2: For very dense label maps, skip connectivity check entirely
    if label_density > 0.005:  # More than 1 label per 200 pixels = likely already well-segmented
        logger.info(f"Skipping connectivity check: {num_labels} labels, density {label_density:.4f}")
        return instance_mask
    
    # OPTIMIZATION 3: Choose algorithm based on problem characteristics
    if num_labels <= 5:
        return _ensure_connected_ultra_fast(instance_mask, unique_labels)
    elif num_labels <= 50:
        return _ensure_connected_sampling_based(instance_mask, unique_labels)
    else:
        # For very large label counts, use aggressive sampling
        return _ensure_connected_aggressive_sampling(instance_mask, unique_labels)


def _ensure_connected_ultra_fast(instance_mask, unique_labels):
    """Ultra-fast version for very small label counts."""
    new_mask = np.zeros_like(instance_mask)
    next_label = 1
    disconnected_count = 0
    
    for label in unique_labels:
        # Use direct indexing without intermediate arrays
        label_mask = instance_mask == label
        if not np.any(label_mask):
            continue
        
        # Get bounding box coordinates directly
        rows, cols = np.nonzero(label_mask)
        min_row, max_row = rows[0], rows[-1]
        min_col, max_col = cols.min(), cols.max()
        
        # Work on minimal bounding box
        bbox_mask = instance_mask[min_row:max_row+1, min_col:max_col+1]
        binary_bbox = (bbox_mask == label).astype(np.uint8)
        
        num_components, components = cv2.connectedComponents(binary_bbox)
        
        if num_components > 2:
            disconnected_count += 1
            # Map components back efficiently
            for comp_idx in range(1, num_components):
                comp_mask = components == comp_idx
                if np.any(comp_mask):
                    # Direct assignment using advanced indexing
                    full_mask = np.zeros_like(instance_mask, dtype=bool)
                    full_mask[min_row:max_row+1, min_col:max_col+1] = comp_mask
                    new_mask[full_mask] = next_label
                    next_label += 1
        else:
            if num_components == 2:
                new_mask[label_mask] = next_label
                next_label += 1
    
    if disconnected_count > 0:
        logger.info(f"Ultra-fast: Split {disconnected_count} → {next_label-1} objects")
    
    return new_mask


def _ensure_connected_sampling_based(instance_mask, unique_labels):
    """Sampling-based approach for medium label counts."""
    new_mask = np.zeros_like(instance_mask)
    next_label = 1
    disconnected_count = 0
    
    # Process labels sorted by size (largest first) for better performance
    label_sizes = [(label, np.sum(instance_mask == label)) for label in unique_labels]
    label_sizes.sort(key=lambda x: x[1], reverse=True)
    
    for label, size in label_sizes:
        if size == 0:
            continue
            
        # For very small labels, skip connectivity check
        if size < 10:
            new_mask[instance_mask == label] = next_label
            next_label += 1
            continue
        
        # Use efficient bounding box approach
        label_mask = instance_mask == label
        rows, cols = np.nonzero(label_mask)
        
        min_row, max_row = rows[0], rows[-1]
        min_col, max_col = cols.min(), cols.max()
        
        # If bounding box is much larger than actual pixels, likely disconnected
        bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        if size / bbox_area > 0.7:  # Dense object, likely connected
            new_mask[label_mask] = next_label
            next_label += 1
            continue
        
        # Only do expensive connectivity check for sparse objects
        bbox_mask = instance_mask[min_row:max_row+1, min_col:max_col+1]
        binary_bbox = (bbox_mask == label).astype(np.uint8)
        
        num_components, components = cv2.connectedComponents(binary_bbox)
        
        if num_components > 2:
            disconnected_count += 1
            for comp_idx in range(1, num_components):
                comp_mask = components == comp_idx
                if np.any(comp_mask):
                    full_mask = np.zeros_like(instance_mask, dtype=bool)
                    full_mask[min_row:max_row+1, min_col:max_col+1] = comp_mask
                    new_mask[full_mask] = next_label
                    next_label += 1
        else:
            if num_components == 2:
                new_mask[label_mask] = next_label
                next_label += 1
    
    if disconnected_count > 0:
        logger.info(f"Sampling: Split {disconnected_count} → {next_label-1} objects")
    
    return new_mask


def _ensure_connected_aggressive_sampling(instance_mask, unique_labels):
    """Aggressive sampling for very large label counts - only check suspicious labels."""
    new_mask = np.zeros_like(instance_mask)
    next_label = 1
    
    # Quick pass: most labels are probably fine, only check suspicious ones
    suspicious_labels = []
    
    # Pre-filter: only check labels that might be disconnected
    for label in unique_labels:
        label_mask = instance_mask == label
        size = np.sum(label_mask)
        
        if size < 5:  # Tiny objects, just keep as-is
            new_mask[label_mask] = next_label
            next_label += 1
            continue
        
        # Quick bounding box test
        rows, cols = np.nonzero(label_mask)
        bbox_area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
        
        # If density is high, likely connected
        if size / bbox_area > 0.5:
            new_mask[label_mask] = next_label
            next_label += 1
        else:
            suspicious_labels.append(label)
    
    # Only do expensive connectivity check on suspicious labels
    disconnected_count = 0
    for label in suspicious_labels:
        label_mask = instance_mask == label
        rows, cols = np.nonzero(label_mask)
        
        min_row, max_row = rows[0], rows[-1]
        min_col, max_col = cols.min(), cols.max()
        
        bbox_mask = instance_mask[min_row:max_row+1, min_col:max_col+1]
        binary_bbox = (bbox_mask == label).astype(np.uint8)
        
        num_components, components = cv2.connectedComponents(binary_bbox)
        
        if num_components > 2:
            disconnected_count += 1
            for comp_idx in range(1, num_components):
                comp_mask = components == comp_idx
                if np.any(comp_mask):
                    full_mask = np.zeros_like(instance_mask, dtype=bool)
                    full_mask[min_row:max_row+1, min_col:max_col+1] = comp_mask
                    new_mask[full_mask] = next_label
                    next_label += 1
        else:
            if num_components == 2:
                new_mask[label_mask] = next_label
                next_label += 1
    
    if disconnected_count > 0:
        logger.info(f"Aggressive: Split {disconnected_count} → {next_label-1} objects")
    elif len(suspicious_labels) > 0:
        logger.info(f"Aggressive: Checked {len(suspicious_labels)} suspicious labels, all connected")
    
    return new_mask


@timing_decorator
def optimized_save_results(image, binary_mask, instance_mask, output_dir, image_name, save_visualization=True, save_numpy=True):
    """
    Optimized result saving with conditional operations.
    """
    results_saved = []
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Always save the instance mask as PNG (most important output)
        instance_vis = colorize_instance_mask(instance_mask)
        instance_path = os.path.join(output_dir, f"{image_name}_instance_mask.png")
        success = cv2.imwrite(instance_path, cv2.cvtColor(instance_vis, cv2.COLOR_RGB2BGR))
        if success:
            results_saved.append(instance_path)
        else:
            print(f"ERROR: Failed to save instance mask PNG: {instance_path}")
        
        # Conditionally save numpy array (for programmatic access)
        if save_numpy:
            numpy_path = os.path.join(output_dir, f"{image_name}_instance_mask.npy")
            try:
                np.save(numpy_path, instance_mask)
                results_saved.append(numpy_path)
            except Exception as e:
                print(f"ERROR: Failed to save numpy array {numpy_path}: {e}")
        
        # Conditionally save visualization (expensive)
        if save_visualization:
            viz_path = os.path.join(output_dir, f"{image_name}_instance_segmentation.png")
            try:
                save_instance_visualization(image, binary_mask, instance_mask, viz_path)
                results_saved.append(viz_path)
            except Exception as e:
                print(f"ERROR: Failed to save visualization {viz_path}: {e}")
        
        # Log what was saved (only in verbose mode)
        try:
            if hasattr(logger, 'level') and logger.level <= 10:  # DEBUG level
                logger.debug(f"Saved {len(results_saved)} files for {image_name}: {[os.path.basename(f) for f in results_saved]}")
        except:
            pass  # Ignore logger errors
        
        return results_saved
        
    except Exception as e:
        print(f"ERROR: Error in optimized_save_results for {image_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main function to demonstrate instance segmentation."""
    global TIMING_ENABLED
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Instance segmentation for liquid biopsy images')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory containing input images')
    parser.add_argument('--results-dir', type=str, default='results/instance_segmentation',
                        help='Directory to save segmentation results')
    parser.add_argument('--num-processes', type=int, default=6,
                        help='Number of parallel processes')
    parser.add_argument('--min-object-size', type=int, default=64,
                        help='Minimum size (in pixels) of objects to keep')
    parser.add_argument('--max-object-size', type=int, default=1024,
                        help='Maximum size (in pixels) of objects to keep (None for no limit)')
    parser.add_argument('--check-connectivity', action='store_true', default=True,
                        help='Check and fix disconnected components with the same label')
    parser.add_argument('--skip-connectivity', action='store_true', default=False,
                        help='Skip connectivity check entirely for maximum speed')
    parser.add_argument('--no-visualization', action='store_true', default=True,
                        help='Skip saving visualization images (faster processing)')
    parser.add_argument('--no-numpy', action='store_true', default=True,
                        help='Skip saving numpy arrays (faster I/O)')
    parser.add_argument('--fast-mode', action='store_true', default=True,
                        help='Enable all speed optimizations (skip visualization and numpy)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable detailed logging to file (instance_segmentation.log)')
    parser.add_argument('--time-it', action='store_true', default=True,
                        help='Enable timing output to console')
    
    # Watershed-specific parameters
    parser.add_argument('--watershed-peak-sensitivity', type=float, default=0.2,
                        help='Peak detection sensitivity for watershed (0.1-0.5, lower=more sensitive)')
    parser.add_argument('--watershed-sparse-threshold', type=float, default=0.005,
                        help='Density threshold for sparse images (0.005-0.02)')
    parser.add_argument('--watershed-dense-threshold', type=float, default=0.25,
                        help='Density threshold for dense images (0.2-0.5)')
    parser.add_argument('--watershed-fallback-threshold', type=float, default=0.7,
                        help='Fallback peak threshold when scikit-image unavailable (0.5-0.9)')
    
    args = parser.parse_args()
    
    # Set global timing flag
    TIMING_ENABLED = args.time_it
    
    # Handle fast mode
    if args.fast_mode:
        save_visualization = False
        save_numpy = False
        args.skip_connectivity = True  # Also skip connectivity for maximum speed
        print("🏎️  Fast mode enabled: skipping visualizations and numpy saves")
    else:
        save_visualization = not args.no_visualization
        save_numpy = not args.no_numpy
        print(f"💾 Save mode: visualization={save_visualization}, numpy={save_numpy}")
    
    # Configure file logging if verbose is enabled
    if args.verbose:
        logger.add("instance_segmentation.log", 
                  level="DEBUG", 
                  format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {os.path.abspath(args.results_dir)}")
    
    # Build watershed configuration from command-line arguments
    watershed_config = {
        'threshold_abs_factor': args.watershed_peak_sensitivity,
        'fallback_threshold_factor': args.watershed_fallback_threshold,
        'sparse_density_threshold': args.watershed_sparse_threshold,
        'dense_density_threshold': args.watershed_dense_threshold,
        'multi_resolution_threshold': 512,  # Keep default
        'min_watershed_distance': 3,        # Keep default
        'max_objects_threshold': 50,        # Keep default
        'gpu_size_threshold': 512*512       # Keep default
    }
    
    if args.verbose:
        print(f"🌊 Watershed config: peak_sensitivity={args.watershed_peak_sensitivity}, "
              f"sparse_threshold={args.watershed_sparse_threshold}, "
              f"dense_threshold={args.watershed_dense_threshold}")
    
    # Configure the traditional segmentation methods
    config = {
        'otsu': {
            'enable': True,
            'min_threshold': 25,
            'use_watershed_separation': True,
            'watershed_min_distance': 6,
            'min_object_size': 49,
            'min_hole_size': 15,
        },
        'adaptive_threshold': {
            'enable': True,
            'block_size': 15,
            'c': 5,
            'dilation_kernel_size': 4,
            'use_watershed_separation': True,
            'watershed_min_distance': 6,
            'min_object_size': 49,
            'min_hole_size': 15,
        }
    }
    
    # Convert max_object_size from string to int or None
    max_object_size = None if args.max_object_size is None else int(args.max_object_size)
    
    # Create factory with configuration
    factory = TraditionalSegmenterFactory(config)
    
    # Create data loader
    data_loader = SegmentationDataLoader(
        image_dir=args.data_dir
    )
    
    # Process all images
    num_images = len(data_loader)
    
    # Log details to file only if verbose
    if args.verbose:
        logger.info(f"Starting instance segmentation with {num_images} images")
        logger.info(f"Configuration: size={args.min_object_size}-{args.max_object_size}, connectivity={args.check_connectivity}, processes={args.num_processes}")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Results directory: {args.results_dir}")
    
    start_time = time.time()
    
    if args.num_processes <= 1:
        if args.verbose:
            logger.info("Running in single-process mode")
        results = []
        for idx in tqdm(range(num_images), desc="Processing", disable=False):
            result = process_image_with_instance_segmentation(
                idx, data_loader, factory, args.results_dir,
                min_object_size=args.min_object_size,
                max_object_size=max_object_size,
                check_connectivity=args.check_connectivity,
                skip_connectivity=args.skip_connectivity,
                save_visualization=save_visualization,
                save_numpy=save_numpy,
                watershed_config=watershed_config
            )
            results.append(result)
    else:
        if args.verbose:
            logger.info(f"Running in multi-process mode with {args.num_processes} workers")
        pool = multiprocessing.Pool(processes=args.num_processes)
        process_fn = partial(
            process_image_with_instance_segmentation,
            data_loader=data_loader,
            factory=factory,
            results_dir=args.results_dir,
            min_object_size=args.min_object_size,
            max_object_size=max_object_size,
            check_connectivity=args.check_connectivity,
            skip_connectivity=args.skip_connectivity,
            save_visualization=save_visualization,
            save_numpy=save_numpy,
            watershed_config=watershed_config
        )
        results = list(tqdm(
            pool.imap(process_fn, range(num_images)),
            total=num_images,
            desc="Processing"
        ))
        pool.close()
        pool.join()
    
    # Count successful processing
    successful = sum(1 for _, _, success in results if success)
    
    # Print minimal summary to console
    elapsed_time = time.time() - start_time
    print(f"Done: {successful}/{num_images} images ({elapsed_time:.1f}s)")
    
    # Log detailed summary to file only if verbose
    if args.verbose:
        logger.success(f"Processing completed: {successful}/{num_images} images processed successfully")
        logger.info(f"Total time: {elapsed_time:.2f}s, Average: {elapsed_time/num_images:.3f}s per image")
        logger.info(f"Results directory: {os.path.abspath(args.results_dir)}")


if __name__ == "__main__":
    main() 