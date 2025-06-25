import os
import sys
import loguru as log
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import multiprocessing

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.traditional import TraditionalSegmenterFactory
# from src.preprocessing import Preprocessor
import configs.default_config as config

from src.utils.data_loader import SegmentationDataLoader
from src.deep_learning.cellpose import CellposeSegmentor

def main():
    log.logger.info("Starting segmentation process...")

    data_loader = SegmentationDataLoader(
        image_dir=config.RAW_DATA_DIR,
        mask_dir=config.PROCESSED_DATA_DIR,
        image_ext='.jpg', # ['.png', '.jpg', '.jpeg'],
        mask_ext='.jpg', # ['.png', '.jpg', '.jpeg'],
        recursive=True
    )
    log.logger.debug("Segmentor Data Loader initialized.")

    cellposeSegmentor = CellposeSegmentor(config) # disgusting way to pass around configs but fast

from src.preprocessing import Preprocessor
from configs.default_config import TRADITIONAL_CONFIG, CELL_SEGMENTATION_CONFIG
from src.utils import SegmentationDataLoader
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation for liquid biopsy images')
    parser.add_argument('data_dir', type=str, default='data/raw',
                        help='Directory containing input images')
    parser.add_argument('--result_dir', type=str, default='results/instance_segmentation',
                        help='Directory to save segmentation results')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel processed')
    parser.add_argument('--min_object_size', type=int, 
                        help='Minimum size (in pixels) of objects to keep')
    parser.add_argument('--max_object_size', type=int,
                        help='Maximum size (in pixels) of objects to keep')
    parser.add_argument('--check_connectivity', action='store_true', default=True,
                        help='Check and fix disconnected components with the same label')
    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    return args

def main():
    # Configure the traditional segmentation methods
    traditional_config = TRADITIONAL_CONFIG
    # Override the default min and max object sizes
    if args.min_object_size is not None:
        traditional_config['otsu']['min_object_size'] = args.min_object_size
        traditional_config['adaptive_threshold']['min_object_size'] = args.min_object_size
    if args.max_object_size is not None:
        traditional_config['otsu']['max_object_size'] = args.max_object_size
        traditional_config['adaptive_threshold']['max_object_size'] = args.max_object_size

    # Create factory with configuration
    factory = TraditionalSegmenterFactory(traditional_config)

    # Create data loader
    data_loader = SegmentationDataLoader(
        image_dir=args.data_dir
    )

    # Process all images
    num_images = len(data_loader)
    logger.info(f"Processing {num_images} images with configuration: {traditional_config}")

    if args.workers <= 1:
        results = []
        for idx in tqdm(range(num_images)):
            result = process_image_with_instance_segmentation(
                idx, data_loader, factory, args.result_dir,
                min_object_size=args.min_object_size,
                max_object_size=args.max_object_size,
                check_connectivity=args.check_connectivity
            )
            results.append(result)
    else:
        pool = multiprocessing.Pool(processes=args.workers)
    
    log.logger.debug("Loading slides...")
    slides = data_loader.load_slides(config.RAW_DATA_DIR)

    log.logger.debug("Creating composits...")
    composite_images = data_loader.get_composites(slides, config.SLIDE_INDEX_OFFSET) # creaing composites should be preprocessing which is in segmentor 

    log.logger.debug("Running Segmentation...")
    binary_masks = cellposeSegmentor.segment(composite_images)

    log.logger.debug("Saving masks...")
    cellposeSegmentor.save_masks(binary_masks)

    log.logger.info("Segmentation process completed successfully.")

if __name__ == "__main__":
    args = parse_args()
    main(args) 