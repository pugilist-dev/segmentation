"""
Utilities for creating annotated segmentation datasets.
"""

import os
import cv2
import numpy as np
import multiprocessing
from functools import partial
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Callable, Tuple
from pathlib import Path


def create_dataset_structure(base_dir: str) -> Dict[str, str]:
    """
    Create the standard dataset directory structure.
    
    Args:
        base_dir: Base directory for the dataset
        
    Returns:
        Dictionary with paths to the created directories
    """
    # Create main directories
    dirs = {
        'base': base_dir,
        'raw': os.path.join(base_dir, 'raw'),
        'processed': os.path.join(base_dir, 'processed'),
        'annotations': os.path.join(base_dir, 'annotations'),
    }
    
    # Create all directories
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    
    return dirs


def import_images(
    source_dir: str,
    target_dir: str,
    extensions: List[str] = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
    recursive: bool = False,
    copy: bool = True,
    preprocess_fn: Optional[Callable] = None
) -> List[str]:
    """
    Import images from a source directory to the dataset.
    
    Args:
        source_dir: Source directory containing images
        target_dir: Target directory to copy/move images to
        extensions: List of file extensions to consider
        recursive: Whether to search the source directory recursively
        copy: If True, copy the files; otherwise, move them
        preprocess_fn: Optional function to preprocess images before saving
        
    Returns:
        List of paths to the imported images in the target directory
    """
    # Make sure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(source_dir):
        if not recursive and root != source_dir:
            continue
            
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    
    # Sort paths for deterministic behavior
    image_paths.sort()
    
    # Import images
    imported_paths = []
    for src_path in image_paths:
        # Get relative path if using recursive mode
        if recursive:
            rel_path = os.path.relpath(src_path, source_dir)
            dst_path = os.path.join(target_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        else:
            dst_path = os.path.join(target_dir, os.path.basename(src_path))
        
        # Apply preprocessing if provided
        if preprocess_fn:
            img = cv2.imread(src_path)
            if img is not None:
                processed_img = preprocess_fn(img)
                cv2.imwrite(dst_path, processed_img)
                imported_paths.append(dst_path)
        else:
            # Copy or move the file
            if copy:
                import shutil
                shutil.copy2(src_path, dst_path)
            else:
                import shutil
                shutil.move(src_path, dst_path)
            
            imported_paths.append(dst_path)
    
    return imported_paths


def _process_single_annotation(args: Tuple[str, str, Callable]) -> Optional[str]:
    """
    Process a single image for annotation.
    
    Args:
        args: Tuple of (image_path, annotation_dir, segmentation_fn)
        
    Returns:
        Path to the generated annotation mask or None if failed
    """
    img_path, annotation_dir, segmentation_fn = args
    
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return None
        
        # Generate mask using the provided segmentation function
        mask = segmentation_fn(img)
        
        # Save mask with same name as image but in annotation directory
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(annotation_dir, f"{base_name}_mask.png")
        
        # Ensure mask is binary with values 0 and 255
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        
        cv2.imwrite(mask_path, mask)
        return mask_path
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def generate_automatic_annotations(
    image_dir: str,
    annotation_dir: str,
    segmentation_fn: Callable,
    extensions: List[str] = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
    n_processes: int = None
) -> List[str]:
    """
    Generate automatic annotations for images using a segmentation function.
    
    Args:
        image_dir: Directory containing the images
        annotation_dir: Directory to save the annotations
        segmentation_fn: Function that takes an image and returns a binary mask
        extensions: List of file extensions to consider
        n_processes: Number of processes to use (default: number of CPU cores - 1)
        
    Returns:
        List of paths to the generated annotation masks
    """
    # Make sure annotation directory exists
    os.makedirs(annotation_dir, exist_ok=True)
    
    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    
    # Sort paths for deterministic behavior
    image_paths.sort()
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_processes = max(1, n_processes)
    
    print(f"Generating annotations using {n_processes} processes...")
    
    # Prepare arguments for multiprocessing
    process_args = [(img_path, annotation_dir, segmentation_fn) for img_path in image_paths]
    
    # Generate annotations using multiprocessing
    annotation_paths = []
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(_process_single_annotation, process_args, chunksize=10),
            total=len(image_paths),
            desc="Generating annotations"
        ))
        
        # Filter out None results (failed processing)
        annotation_paths = [path for path in results if path]
    
    return annotation_paths 