#!/usr/bin/env python3
"""
Example script demonstrating the use of traditional segmentation methods.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.traditional import TraditionalSegmenterFactory


def load_example_image():
    """
    Load a sample image for testing.
    If no image is found, create a synthetic test image.
    
    Returns:
        numpy.ndarray: Sample image for segmentation testing
    """
    # Check if there are any sample images in the data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    
    # Try to find an image file
    image_file = None
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_file = os.path.join(data_dir, file)
                break
    
    if image_file:
        print(f"Using sample image: {image_file}")
        return cv2.imread(image_file)
    
    # Create a synthetic test image if no real images are available
    print("No sample images found. Creating synthetic test image...")
    
    # Create a synthetic cell-like image (500x500)
    image = np.zeros((500, 500), dtype=np.uint8)
    
    # Add background noise
    noise = np.random.normal(50, 10, (500, 500))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Add some cell-like structures (bright elliptical regions)
    for _ in range(20):
        # Random center position
        center_x = np.random.randint(50, 450)
        center_y = np.random.randint(50, 450)
        
        # Random size
        a = np.random.randint(10, 30)  # Semi-major axis
        b = np.random.randint(10, 30)  # Semi-minor axis
        
        # Random rotation
        angle = np.random.uniform(0, 2*np.pi)
        
        # Create a coordinate grid
        y, x = np.mgrid[0:500, 0:500]
        
        # Apply rotation transformation
        x_r = (x - center_x) * np.cos(angle) + (y - center_y) * np.sin(angle)
        y_r = -(x - center_x) * np.sin(angle) + (y - center_y) * np.cos(angle)
        
        # Create ellipse mask
        ellipse_mask = ((x_r**2) / (a**2) + (y_r**2) / (b**2)) <= 1
        
        # Set pixel values (bright cells)
        intensity = np.random.randint(180, 250)
        image[ellipse_mask] = intensity
    
    # Convert to 3-channel image
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    return image


def display_results(image, results):
    """
    Display the original image and segmentation results.
    
    Args:
        image (numpy.ndarray): Original input image
        results (dict): Dictionary of segmentation results
    """
    # Debug: print mask value information
    for method_name, mask in results.items():
        unique_values = np.unique(mask)
        foreground_pixels = np.sum(mask > 0)
        print(f"{method_name} mask - unique values: {unique_values}, foreground pixels: {foreground_pixels}/{mask.size} ({foreground_pixels/mask.size:.2%})")
    
    # Convert BGR to RGB for matplotlib display
    if len(image.shape) == 3 and image.shape[2] == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image
    
    # Set up the figure
    n_methods = len(results) + 1  # +1 for the original image
    fig, axes = plt.subplots(1, n_methods, figsize=(n_methods * 4, 4))
    
    # Display original image
    axes[0].imshow(display_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display segmentation results
    for i, (method_name, mask) in enumerate(results.items(), 1):
        # Ensure we see something even if the mask is very sparse
        if np.max(mask) == 0:
            axes[i].imshow(np.zeros_like(mask), cmap='gray')
            axes[i].set_title(f'{method_name} (Empty)')
        else:
            axes[i].imshow(mask, cmap='gray')
            axes[i].set_title(f'{method_name}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate traditional segmentation methods."""
    # Load sample image
    image = load_example_image()
    
    # Configure the traditional segmentation methods
    config = {
        'otsu': {
            'enable': True,
            'min_threshold': 25,  # Higher minimum threshold for Otsu
            'use_watershed_separation': True,  # Use watershed to separate touching objects
            'watershed_min_distance': 5,  # Minimum distance between local maxima for watershed
            'min_object_size': 10,  # Remove objects smaller than this
            'min_hole_size': 10,  # Fill holes smaller than this
        },
        'adaptive_threshold': {
            'enable': True,
            'block_size': 15,
            'c': 5,  # Increased from 2 to 10 - higher value means more aggressive thresholding
            'dilation_kernel_size': 4,  # Kernel size for dilation operation to connect broken outlines
            'use_watershed_separation': True,  # Use watershed to separate touching objects
            'watershed_min_distance': 5,  # Minimum distance between local maxima for watershed
            'min_object_size': 40,  # Remove objects smaller than this
            'min_hole_size': 32,  # Fill holes smaller than this
        }
    }
    
    # Create factory with configuration
    factory = TraditionalSegmenterFactory(config)
    
    # Run segmentation with all methods
    results = factory.segment(image)
    
    # Display individual method results
    display_results(image, results)
    
    # Combine results using different methods
    combined_vote = factory.segment_combine(image, combine_method='vote')
    combined_union = factory.segment_combine(image, combine_method='union')
    combined_intersection = factory.segment_combine(image, combine_method='intersection')
    
    # Display combined results
    combined_results = {
        'Majority Vote': combined_vote,
        'Union': combined_union,
        'Intersection': combined_intersection
    }
    display_results(image, combined_results)


if __name__ == "__main__":
    main() 