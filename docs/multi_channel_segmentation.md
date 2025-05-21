# Multi-Channel Segmentation for Liquid Biopsy Images

This document describes the multi-channel fluorescence image segmentation functionality in this repository.

## Overview

The multi-channel segmentation module provides tools to:

1. Load and preprocess multi-channel fluorescence microscope images
2. Segment the nuclear (DAPI) channel using adaptive thresholding
3. Segment other fluorescence channels
4. Identify objects in other channels with and without associated nuclear masks
5. Efficiently process 16-bit TIFF files common in microscopy
6. Provide visualization and evaluation tools

## Required Dependencies

- OpenCV (cv2)
- scikit-image
- NumPy
- SciPy
- Matplotlib (for visualization)
- tqdm (for progress bars)

## Key Components

### 1. MultiChannelImageLoader

Located in `src/preprocessing/image_loader.py`, this class handles:

- Loading multi-channel images from various file formats
- Normalizing images with different bit depths
- Grouping files that belong to the same multi-channel sample
- Processing entire directories of images

### 2. MultiChannelSegmentation

Located in `src/traditional/cell_segmentation.py`, this class provides:

- Nuclear segmentation using Sauvola thresholding
- Segmentation of other fluorescence channels 
- Separation of events with and without nuclear signals
- Parallel processing for efficiency
- Configurability through the config system

### 3. Example Usage

There is an example script in `examples/multi_channel_segmentation_example.py` that shows how to:

- Process single or multiple channel images
- Visualize the segmentation results
- Save results to disk

## Configuration Options

The default configuration is defined in `configs/default_config.py` under `CELL_SEGMENTATION_CONFIG`:

```python
CELL_SEGMENTATION_CONFIG = {
    'nuclear': {
        'sauvola_window': 7,        # Window size for Sauvola thresholding
        'sauvola_k': -0.055,        # K parameter for Sauvola thresholding
        'min_threshold': 25,        # Minimum threshold value
        'opening_size': 5,          # Morphological opening kernel size
        'fill_holes': True          # Whether to fill holes in masks
    },
    'other_channels': {
        'sauvola_window': 7,
        'sauvola_k': -0.05,
        'min_threshold': 20,
        'opening_size': 3,
        'fill_holes': True
    },
    'use_multiprocessing': True,    # Whether to use parallel processing
    'n_jobs': -1,                   # Number of processes (-1 for all available)
    'channel_names': ['dapi', 'channel1', 'channel2', 'channel3'],
    'save_intermediate_results': False
}
```

You can customize these parameters for your specific microscopy images.

## Algorithm Details

### Nuclear Channel Segmentation

1. Convert to 8-bit if necessary (for faster processing)
2. Apply Otsu's thresholding for initial filtering
3. Apply local adaptive Sauvola thresholding
4. Perform morphological opening to remove small artifacts
5. Fill holes in the masks
6. Label connected components

### Event Segmentation with Nuclear Seeds

1. Segment each fluorescence channel using similar thresholding
2. Label all events in each channel
3. For each event, check if it contains any nuclear signal
4. Group events into "with nuclei" and "without nuclei" categories

## Command Line Usage

```bash
python examples/multi_channel_segmentation_example.py --input dapi.tif channel1.tif channel2.tif --output results/
```

Or for testing with a single image:

```bash
python examples/multi_channel_segmentation_example.py --input dapi.tif --output results/
```

## API Usage Example

```python
from src.preprocessing.image_loader import MultiChannelImageLoader
from src.traditional.cell_segmentation import MultiChannelSegmentation
from configs.default_config import CELL_SEGMENTATION_CONFIG

# Create the segmentation objects
loader = MultiChannelImageLoader()
segmenter = MultiChannelSegmentation(CELL_SEGMENTATION_CONFIG)

# Load multiple channels
channels = [
    loader.load_single_image('path/to/dapi.tif'),
    loader.load_single_image('path/to/channel1.tif'),
    loader.load_single_image('path/to/channel2.tif'),
]

# Process all channels
results = segmenter.process_all_channels(channels)

# Access the results
nuclear_mask = results['nuclear']['mask']
channel1_with_nuclei = results['channel_1']['with_nuclei']
channel1_without_nuclei = results['channel_1']['without_nuclei']
```

## Performance Considerations

- For large images, processing can be memory-intensive
- 16-bit images are converted to 8-bit for faster processing
- Multiprocessing is enabled by default for processing multiple channels
- The bottleneck is typically the Sauvola thresholding, which is computationally expensive 