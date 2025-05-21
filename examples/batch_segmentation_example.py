#!/usr/bin/env python3
"""
Example script demonstrating batch segmentation using the data loader with multiprocessing.
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

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils import SegmentationDataLoader
from src.traditional import TraditionalSegmenterFactory


def process_image(idx, data_loader, factory, results_dir):
    """
    Process a single image with segmentation.
    
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
        combined_mask = factory.segment_combine(image, combine_method='vote')
        
        # Save visualization of the segmentation
        save_visualization(
            image, 
            results, 
            combined_mask, 
            os.path.join(results_dir, f"{image_name}_segmentation.png")
        )
        
        # Save the best segmentation mask
        mask_path = os.path.join(results_dir, f"{image_name}_mask.png")
        cv2.imwrite(mask_path, combined_mask * 255)
        
        return idx, image_name, True
    except Exception as e:
        return idx, None, False


def save_visualization(image, method_results, combined_mask, output_path):
    """
    Save visualization of original image and segmentation results.
    
    Args:
        image: Original input image
        method_results: Dictionary with results from individual methods
        combined_mask: Combined segmentation mask
        output_path: Path to save the visualization
    """
    # Convert BGR to RGB for matplotlib display
    if len(image.shape) == 3 and image.shape[2] == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image
    
    # Calculate number of subplots needed
    num_methods = len(method_results)
    total_plots = num_methods + 2  # +1 for original image, +1 for combined result
    
    # Determine grid layout
    rows = 2
    cols = (total_plots + 1) // rows  # Ceiling division
    
    # Create figure
    plt.figure(figsize=(cols * 4, rows * 4))
    
    # Display original image
    plt.subplot(rows, cols, 1)
    plt.imshow(display_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Display each method's results
    for i, (method_name, mask) in enumerate(method_results.items(), 2):
        plt.subplot(rows, cols, i)
        plt.imshow(mask, cmap='gray')
        plt.title(method_name)
        plt.axis('off')
    
    # Display combined result
    plt.subplot(rows, cols, total_plots)
    plt.imshow(combined_mask, cmap='gray')
    plt.title('Combined Result (Majority Vote)')
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Main function to demonstrate batch segmentation using data loader with multiprocessing."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Batch segmentation with multiprocessing")
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count() - 1, 
                        help="Number of processes to use (default: number of CPU cores - 1)")
    parser.add_argument("--image-dir", type=str, default=os.path.join(project_root, 'data', 'raw'),
                        help="Directory containing images to segment")
    parser.add_argument("--results-dir", type=str, default=os.path.join(project_root, 'results', 'visualizations'),
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
    config = {
        'otsu': {
            'enable': True,
            'min_threshold': 25,  # Higher minimum threshold for Otsu
            'use_watershed_separation': False,  # Use watershed to separate touching objects
            'watershed_min_distance': 5,  # Minimum distance between local maxima for watershed
            'min_object_size': 10,  # Remove objects smaller than this
            'min_hole_size': 10,  # Fill holes smaller than this
        },
        'adaptive_threshold': {
            'enable': True,
            'block_size': 15,
            'c': 5,  # Increased from 2 to 10 - higher value means more aggressive thresholding
            'dilation_kernel_size': 4,  # Kernel size for dilation operation to connect broken outlines
            'use_watershed_separation': False,  # Use watershed to separate touching objects
            'watershed_min_distance': 5,  # Minimum distance between local maxima for watershed
            'min_object_size': 40,  # Remove objects smaller than this
            'min_hole_size': 32,  # Fill holes smaller than this
        }
    }
    
    # Create factory with configuration
    factory = TraditionalSegmenterFactory(config)
    
    # Create a partial function with fixed arguments
    process_func = partial(process_image, data_loader=data_loader, factory=factory, results_dir=results_dir)
    
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
    print(f"Segmentation completed. Successfully processed {successful}/{total_images} images.")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    # Ensure proper handling of multiprocessing on Windows
    multiprocessing.freeze_support()
    main() 