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
from functools import partial
from scipy import ndimage as ndi
import random

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.utils import SegmentationDataLoader
from src.traditional import TraditionalSegmenterFactory


def process_image_with_instance_segmentation(idx, data_loader, factory, results_dir):
    """
    Process a single image with instance segmentation.
    
    Args:
        idx: Index of the image in the data loader
        data_loader: SegmentationDataLoader instance
        factory: TraditionalSegmenterFactory instance
        results_dir: Directory to save results
        
    Returns:
        Tuple of (idx, image_name, success)
    """
    try:
        # Get the sample and extract image
        sample = data_loader[idx]
        image = sample.image
        image_name = os.path.splitext(os.path.basename(sample.image_path))[0]
        
        # Run segmentation with all methods
        results = factory.segment(image)
        
        # Create a combined result using majority voting
        binary_mask = factory.segment_combine(image, combine_method='vote')
        
        # Create instance mask using watershed separation
        instance_mask = create_instance_mask(binary_mask)
        
        # Save visualization of the instance segmentation
        save_instance_visualization(
            image, 
            binary_mask,
            instance_mask, 
            os.path.join(results_dir, f"{image_name}_instance_segmentation.png")
        )
        
        # Save the instance segmentation mask (as a color image to preserve labels)
        instance_vis = colorize_instance_mask(instance_mask)
        cv2.imwrite(
            os.path.join(results_dir, f"{image_name}_instance_mask.png"),
            cv2.cvtColor(instance_vis, cv2.COLOR_RGB2BGR)
        )
        
        # Also save the raw instance mask with unique integer labels
        np.save(os.path.join(results_dir, f"{image_name}_instance_mask.npy"), instance_mask)
        
        return idx, image_name, True
    except Exception as e:
        print(f"Error processing image at index {idx}: {e}")
        import traceback
        traceback.print_exc()
        return idx, None, False


def create_instance_mask(binary_mask, min_distance=5, min_object_size=10):
    """
    Create an instance mask from a binary mask using watershed.
    
    Args:
        binary_mask: Binary mask where 0 indicates foreground (from our thresholding convention)
        min_distance: Minimum distance between local maxima
        min_object_size: Minimum size of objects to keep
        
    Returns:
        Instance mask where each object has a unique integer label and background is 0
    """
    # Convert to standard convention for watershed (1=foreground, 0=background)
    # Our thresholding methods use the opposite convention (0=foreground)
    std_mask = 1 - binary_mask.copy()
    
    # Ensure mask is binary
    std_mask = std_mask.astype(bool).astype(np.uint8)
    
    # If the mask is empty, just return zeros
    if np.sum(std_mask) == 0:
        return np.zeros_like(std_mask, dtype=np.int32)
    
    # Compute the distance transform
    distance = ndi.distance_transform_edt(std_mask)
    
    # Find local maxima in the distance transform
    # Use skimage for better control
    try:
        from skimage.feature import peak_local_max
        
        # Adjust min_distance for small objects
        adjusted_min_distance = min(min_distance, max(3, int(np.max(distance)/2)))
        
        # Find local maxima
        coordinates = peak_local_max(
            distance, 
            min_distance=adjusted_min_distance,
            exclude_border=False,
            labels=std_mask
        )
        
        # Create markers for watershed
        markers = np.zeros_like(std_mask, dtype=np.int32)
        for i, (x, y) in enumerate(coordinates, 1):
            markers[x, y] = i
            
    except ImportError:
        # Fallback method if skimage is not available
        # Find peaks using simple threshold and connected components
        local_max = distance > 0.7 * np.max(distance)
        markers, num_objects = ndi.label(local_max)
    
    # Apply watershed
    # First convert binary mask to uint8 format needed by watershed
    std_mask_uint8 = std_mask * 255
    
    # Ensure background is 0 and create BGR image for watershed
    if np.sum(markers > 0) > 0:  # Only proceed if we found markers
        # Add background marker
        watershed_markers = markers.copy()
        watershed_markers[std_mask == 0] = 0
        
        # Apply watershed
        watershed_result = cv2.watershed(
            cv2.cvtColor(std_mask_uint8, cv2.COLOR_GRAY2BGR),
            watershed_markers
        )
        
        # Watershed assigns -1 to boundaries, remap to ensure all labels are ≥ 0
        instance_mask = np.maximum(watershed_result, 0)
        
        # IMPORTANT: Ensure all background pixels (from original binary mask) are set to 0
        # This ensures only foreground objects have instance labels
        instance_mask[binary_mask == 1] = 0
        
        # Remove small objects
        if min_object_size > 0:
            props = np.bincount(instance_mask.ravel())
            to_remove = np.zeros_like(props, dtype=bool)
            
            # Skip background (index 0)
            for i in range(1, len(props)):
                if props[i] < min_object_size:
                    to_remove[i] = True
            
            # Create cleaned mask
            cleaned_mask = instance_mask.copy()
            for label in np.where(to_remove)[0]:
                cleaned_mask[instance_mask == label] = 0
                
            instance_mask = cleaned_mask
    else:
        # If no markers found, fall back to connected components
        instance_mask, _ = ndi.label(std_mask)
        # Ensure only foreground pixels from original mask have labels
        instance_mask[binary_mask == 1] = 0
    
    return instance_mask


def colorize_instance_mask(instance_mask):
    """
    Create a colorized visualization of an instance mask.
    
    Args:
        instance_mask: Integer mask where each object has a unique label
        
    Returns:
        RGB image with each instance colored differently
    """
    # Get unique labels (excluding background = 0)
    labels = np.unique(instance_mask)
    labels = labels[labels > 0]  # Remove background
    
    # Create colormap
    num_instances = len(labels)
    np.random.seed(42)  # For reproducible colors
    
    # Create random colors for each instance
    colors = np.random.randint(40, 255, size=(num_instances, 3), dtype=np.uint8)
    
    # Create RGB image
    rgb_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    
    # Assign colors to instances
    for i, label in enumerate(labels):
        rgb_mask[instance_mask == label] = colors[i]
    
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


def main():
    """Main function to demonstrate instance segmentation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Instance segmentation with watershed")
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count() - 1, 
                        help="Number of processes to use (default: number of CPU cores - 1)")
    parser.add_argument("--image-dir", type=str, default=os.path.join(project_root, 'data', 'raw'),
                        help="Directory containing images to segment")
    parser.add_argument("--results-dir", type=str, default=os.path.join(project_root, 'results', 'instance_segmentation'),
                        help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=10, 
                        help="Batch size for multiprocessing (default: 10 images per batch)")
    args = parser.parse_args()
    
    # Make sure we have at least one process
    num_processes = max(1, args.processes)
    
    # Configure directories
    data_dir = args.image_dir
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the data loader
    data_loader = SegmentationDataLoader(
        image_dir=data_dir,
        image_ext=['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
        recursive=False
    )
    
    total_images = len(data_loader)
    print(f"Found {total_images} images in {data_dir}")
    print(f"Using {num_processes} processes for segmentation")
    
    # Configure the traditional segmentation methods
    # Enable watershed for better instance segmentation
    config = {
        'otsu': {
            'enable': True,
            'min_threshold': 25,
            'use_watershed_separation': True,  # Enable watershed for instance segmentation
            'watershed_min_distance': 5,
            'min_object_size': 10,
            'min_hole_size': 10,
        },
        'adaptive_threshold': {
            'enable': True,
            'block_size': 15,
            'c': 5,
            'dilation_kernel_size': 4, 
            'use_watershed_separation': True,  # Enable watershed for instance segmentation
            'watershed_min_distance': 5,
            'min_object_size': 40,
            'min_hole_size': 32,
        }
    }
    
    # Create factory with configuration
    factory = TraditionalSegmenterFactory(config)
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_image_with_instance_segmentation, 
        data_loader=data_loader, 
        factory=factory, 
        results_dir=results_dir
    )
    
    if num_processes > 1:
        # Process all images using multiprocessing
        indices = list(range(total_images))
        
        # Create process pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process images in batches with progress bar
            results = []
            for result in tqdm(
                pool.imap_unordered(process_func, indices, chunksize=args.batch_size),
                total=total_images,
                desc="Processing images"
            ):
                results.append(result)
        
        # Count successful segmentations
        successful = sum(1 for _, _, success in results if success)
    else:
        # Single process mode for debugging
        results = []
        for idx in tqdm(range(total_images), desc="Processing images"):
            results.append(process_func(idx))
        successful = sum(1 for _, _, success in results if success)
    
    print(f"Instance segmentation completed. Successfully processed {successful}/{total_images} images.")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    # Ensure proper handling of multiprocessing on Windows
    multiprocessing.freeze_support()
    main() 