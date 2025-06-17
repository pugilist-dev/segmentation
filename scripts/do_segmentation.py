import os
import sys
import loguru as log

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
    
    log.logger.debug("Loading slides...")
    slides = data_loader.load_slides(config.RAW_DATA_DIR)

    log.logger.debug("Creating composits...")
    composite_images = data_loader.get_composites(slides, config.SLIDE_INDEX_OFFSET)

    log.logger.debug("Running Segmentation...")
    binary_masks = cellposeSegmentor.segment(composite_images)

    log.logger.debug("Saving masks...")
    cellposeSegmentor.save_masks(binary_masks)

    log.logger.info("Segmentation process completed successfully.")

if __name__ == "__main__":
    main() 