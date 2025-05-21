#!/usr/bin/env python3
"""
Example script demonstrating how to create an annotated dataset with multiprocessing.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import multiprocessing

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils import (
    create_dataset_structure,
    import_images,
    generate_automatic_annotations
)
from src.traditional import TraditionalSegmenterFactory


def preprocess_image(image):
    """
    Apply preprocessing to an image.
    
    Args:
        image: Input image
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if color
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert back to color for consistency
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image


def main():
    """Main function to demonstrate dataset creation and annotation."""
    parser = argparse.ArgumentParser(description="Create an annotated dataset from a directory of images")
    parser.add_argument("--source", type=str, required=True, help="Source directory containing images")
    parser.add_argument("--dataset", type=str, default=os.path.join(project_root, "data"), 
                        help="Target dataset directory")
    parser.add_argument("--recursive", action="store_true", help="Search source directory recursively")
    parser.add_argument("--preprocess", action="store_true", help="Apply preprocessing to images")
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count() - 1,
                        help="Number of processes to use (default: number of CPU cores - 1)")
    args = parser.parse_args()
    
    # Make sure we have at least one process
    num_processes = max(1, args.processes)
    print(f"Using {num_processes} processes for parallel processing")
    
    # Create dataset structure
    print(f"Creating dataset structure in {args.dataset}")
    dirs = create_dataset_structure(args.dataset)
    
    # Import images
    print(f"Importing images from {args.source}")
    preprocess_fn = preprocess_image if args.preprocess else None
    imported_paths = import_images(
        source_dir=args.source,
        target_dir=dirs['raw'],
        recursive=args.recursive,
        copy=True,  # Always copy, don't move
        preprocess_fn=preprocess_fn
    )
    print(f"Imported {len(imported_paths)} images")
    
    # Configure traditional segmentation methods
    config = {
        'otsu': {
            'enable': True,
            'min_threshold': 25,  # Minimum threshold for Otsu
            'use_watershed_separation': True,  # Use watershed to separate touching objects
            'watershed_min_distance': 10,  # Minimum distance between local maxima for watershed
            'min_object_size': 50,  # Remove objects smaller than this
            'min_hole_size': 20,  # Fill holes smaller than this
        },
        'adaptive_threshold': {
            'enable': True,
            'block_size': 15,
            'c': 2,
            'dilation_kernel_size': 3,  # Kernel size for dilation operation to connect broken outlines
            'use_watershed_separation': True,  # Use watershed to separate touching objects
            'watershed_min_distance': 10,  # Minimum distance between local maxima for watershed
            'min_object_size': 50,  # Remove objects smaller than this
            'min_hole_size': 20,  # Fill holes smaller than this
        }
    }
    
    # Create segmentation factory
    factory = TraditionalSegmenterFactory(config)
    
    # Create a segmentation function that uses the factory
    def segment_image(image):
        # Use majority voting to combine multiple segmentation methods
        return factory.segment_combine(image, combine_method='vote')
    
    # Generate annotations using multiprocessing
    print("Generating annotations using traditional segmentation methods")
    annotation_paths = generate_automatic_annotations(
        image_dir=dirs['raw'],
        annotation_dir=dirs['annotations'],
        segmentation_fn=segment_image,
        n_processes=num_processes
    )
    print(f"Generated {len(annotation_paths)} annotation masks")
    
    # Generate a preview of a few samples
    if imported_paths:
        preview_dir = os.path.join(dirs['base'], "preview")
        os.makedirs(preview_dir, exist_ok=True)
        
        print("Generating preview images")
        
        # Select up to 5 random samples
        num_samples = min(5, len(imported_paths))
        import random
        sample_indices = random.sample(range(len(imported_paths)), num_samples)
        
        for idx in sample_indices:
            img_path = imported_paths[idx]
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Load image and corresponding mask
            img = cv2.imread(img_path)
            mask_path = os.path.join(dirs['annotations'], f"{base_name}_mask.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Create preview image with original and segmentation overlay
            if img is not None and mask is not None:
                # Convert to BGR for consistent color display
                if len(img.shape) < 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Create a mask overlay (red)
                overlay = img.copy()
                overlay[mask > 0] = [0, 0, 255]  # Red color for the segmentation
                
                # Blend original and overlay
                alpha = 0.5
                preview = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
                
                # Add border to make it easier to see the segmentation
                preview = cv2.copyMakeBorder(preview, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                
                # Save preview image
                preview_path = os.path.join(preview_dir, f"{base_name}_preview.png")
                cv2.imwrite(preview_path, preview)
        
        print(f"Preview images saved to {preview_dir}")
    
    print("Dataset creation completed successfully!")


if __name__ == "__main__":
    # Ensure proper handling of multiprocessing on Windows
    multiprocessing.freeze_support()
    main() 