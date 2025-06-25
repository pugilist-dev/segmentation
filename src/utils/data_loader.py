"""
Data loader for segmentation image datasets.
"""

import os
import glob
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SegmentationSample:
    """Represents a single sample for segmentation."""
    image_path: str
    image: np.ndarray
    mask_path: Optional[str] = None
    mask: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class SegmentationDataLoader:
    """
    Data loader for segmentation tasks.
    
    This class loads images from a directory structure and can optionally
    load corresponding ground truth masks for training or evaluation.

    This class is also capable of combining grayscale scans into a 
    single RGB image and save it for future use. 
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        image_ext: Union[str, List[str]] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff'),
        mask_ext: Union[str, List[str]] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff'),
        image_preprocessing: Optional[Callable] = None,
        mask_preprocessing: Optional[Callable] = None,
        recursive: bool = False
    ):
        """
        Initialize the data loader.
        
        Args:
            image_dir: Directory containing the images
            mask_dir: Optional directory containing the mask/annotation images
            image_ext: File extensions to consider for images
            mask_ext: File extensions to consider for masks
            image_preprocessing: Optional function to preprocess images
            mask_preprocessing: Optional function to preprocess masks
            recursive: Whether to search directories recursively
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ext = [image_ext] if isinstance(image_ext, str) else image_ext
        self.mask_ext = [mask_ext] if isinstance(mask_ext, str) else mask_ext
        self.image_preprocessing = image_preprocessing
        self.mask_preprocessing = mask_preprocessing
        self.recursive = recursive
        
        # Find all image files
        self.image_files = self._find_files(image_dir, self.image_ext)
        
        # Map images to masks (if available)
        self.sample_pairs = self._pair_images_with_masks() if mask_dir else None
    
    def _find_files(self, directory: str, extensions: List[str]) -> List[str]:
        """Find all files with the given extensions in the directory."""
        pattern = os.path.join(directory, '**' if self.recursive else '', '*')
        files = []
        
        for ext in extensions:
            if not ext.startswith('.'):
                ext = f'.{ext}'
                
            glob_pattern = f"{pattern}{ext}"
            found_files = glob.glob(glob_pattern, recursive=self.recursive)
            files.extend(found_files)
        
        return sorted(files)
    
    def _pair_images_with_masks(self) -> Dict[str, str]:
        """
        Pair images with their corresponding masks.
        
        Returns:
            Dictionary mapping image paths to mask paths
        """
        pairs = {}
        mask_files = self._find_files(self.mask_dir, self.mask_ext)
        
        # Create a dictionary of mask files for faster lookup
        mask_dict = {}
        for mask_path in mask_files:
            basename = os.path.splitext(os.path.basename(mask_path))[0]
            mask_dict[basename] = mask_path
        
        # Match image files with mask files
        for image_path in self.image_files:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            if basename in mask_dict:
                pairs[image_path] = mask_dict[basename]
        
        return pairs
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load an image and apply preprocessing if specified."""
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        if self.image_preprocessing:
            image = self.image_preprocessing(image)
            
        return image
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load a mask and apply preprocessing if specified."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        if self.mask_preprocessing:
            mask = self.mask_preprocessing(mask)
            
        return mask
    
    def __len__(self) -> int:
        """Get the number of images in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> SegmentationSample:
        """Get a sample at the given index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        image_path = self.image_files[idx]
        image = self._load_image(image_path)
        
        mask_path = None
        mask = None
        
        if self.sample_pairs and image_path in self.sample_pairs:
            mask_path = self.sample_pairs[image_path]
            mask = self._load_mask(mask_path)
        
        return SegmentationSample(
            image_path=image_path,
            image=image,
            mask_path=mask_path,
            mask=mask
        )
    
    def get_batch(self, indices: List[int]) -> List[SegmentationSample]:
        """Get a batch of samples at the given indices."""
        return [self[idx] for idx in indices]
    
    def load_all(self) -> List[SegmentationSample]:
        """Load all images and masks in the dataset."""
        return [self[i] for i in range(len(self))]
    
    def get_image_paths(self) -> List[str]:
        """Get list of all image paths."""
        return self.image_files.copy()
    
    def get_sample_with_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:

        """
        Get image and mask for the given index.
        
        Raises:
            ValueError: If no mask is available for the image
        """
        sample = self[idx]
        if sample.mask is None:
            raise ValueError(f"No mask available for image at index {idx}")
        return sample.image, sample.mask 
    
    def load_slides(self, slides_path):
        """
        Load images from the specified directory, and return a list of images as numpy arrays.
        
        Args:
            slides_path (str): Path to the directory containing the slide images.

        Returns:
            np.ndarray (16 bit): Array of images loaded from the directory, each image is a numpy array.
        """
        image_files = sorted(os.listdir(slides_path)) # list index must match the order of scans 

        frames = []
        for image_file in image_files:
            image = cv2.imread(Path(slides_path, image_file), cv2.IMREAD_GRAYSCALE)
            image = (image*257).astype(np.uint16)  # Convert 8-bit to 16-bit
            frames.append(image)

        return np.array(frames, dtype=np.uint16)


    def compute_composite(self, dapi, ck, cd45, fitc):
        """
        COmbine DAPI, CK, CD45, and FITC channels into a single RGB composite image. Used by CellposeSegmentor.

        Args:
            dapi (np.ndarray): DAPI channel image.
            ck (np.ndarray): CK channel image.
            cd45 (np.ndarray): CD45 channel image.
            fitc (np.ndarray): FITC channel image.
        """

        dtype = dapi.dtype
        max_val = np.iinfo(dapi.dtype).max

        dapi = dapi.astype(np.float32)
        ck = ck.astype(np.float32)
        cd45 = cd45.astype(np.float32)
        fitc = fitc.astype(np.float32)

        rgb = np.zeros((dapi.shape[0], dapi.shape[1], 3), dtype='float')
        
        rgb[...,0] = ck+fitc
        rgb[...,1] = cd45+fitc
        rgb[...,2] = dapi.astype(np.float32)+fitc 
        rgb[rgb > max_val] = max_val # Clips overflow 

        rgb = rgb.astype(dtype)
        return rgb

    def get_composites(self, slides, offset, save_composites=False):
        """
        Create composite images from the provided slides.
        """
        frames=[]
        for i in range(offset): 
            image0 = slides[i]
            image1 = slides[i+offset]
            image2 = slides[i+2*offset]
            # skip Bright Field scan
            image3 = slides[i+3*offset] 
            frames.append(self.compute_composite(image0, image1, image2, image3)) 

            if save_composites:
                composite_path = Path(self.mask_dir, f"composite_{i}.png")
                cv2.imwrite(str(composite_path), frames[-1]) 

        return frames
 