"""
Default configuration for liquid biopsy segmentation.
"""

import os
import multiprocessing

# Index jump from DAPI to CK, CK to CD45 and CD45 to fitc
SLIDE_INDEX_OFFSET = 10 # 10 for sample_data

# Data paths
DATA_ROOT = 'sample_data'
RAW_DATA_DIR = os.path.join(DATA_ROOT, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, 'processed')
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, 'annotations')

# Results paths
RESULTS_DIR = 'results'
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualizations')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

# Model paths
MODELS_DIR = 'models'
TRADITIONAL_MODELS_DIR = os.path.join(MODELS_DIR, 'traditional')
DEEP_LEARNING_MODELS_DIR = os.path.join(MODELS_DIR, 'deep_learning')
MODEL = 'cellpose'

# Traditional segmentation parameters
TRADITIONAL_CONFIG = {
    'otsu': {
        'enable': True,
        'min_threshold': 25,  # Minimum threshold value for Otsu (used as min(otsu_threshold, 25))
        'use_watershed_separation': True,  # Whether to use watershed separation for touching objects
        'watershed_min_distance': 10,  # Minimum distance between local maxima for watershed
        'min_object_size': 50,  # Remove objects smaller than this
        'min_hole_size': 20,  # Fill holes smaller than this
    },
    'adaptive_threshold': {
        'enable': True,
        'block_size': 11,
        'c': 2,
        'use_watershed_separation': True,  # Whether to use watershed separation for touching objects
        'watershed_min_distance': 10,  # Minimum distance between local maxima for watershed
        'min_object_size': 50,  # Remove objects smaller than this
        'min_hole_size': 20,  # Fill holes smaller than this
    },
}

# Multi-channel cell segmentation parameters
CELL_SEGMENTATION_CONFIG = {
    'nuclear': {
        'sauvola_window': 7,
        'sauvola_k': -0.055,
        'min_threshold': 25,
        'opening_size': 5,
        'fill_holes': True
    },
    'other_channels': {
        'sauvola_window': 7,
        'sauvola_k': -0.05,
        'min_threshold': 20,
        'opening_size': 3,
        'fill_holes': True
    },
    'use_multiprocessing': True,
    'n_jobs': max(1, multiprocessing.cpu_count() - 1),  # Leave one CPU free
    'channel_names': ['dapi', 'channel1', 'channel2', 'channel3'],
    'save_intermediate_results': False
}

# Deep learning parameters
DEEP_LEARNING_CONFIG = {
    # Model parameters
    'model': {
        'name': 'cellpose_model',  # Options: 'unet', 'deeplab', 'maskrcnn'
        'backbone': 'resnet34',
        'input_channels': 3,
        'num_classes': 1,  # 1 for binary segmentation
        'bilinear': True,
    },
    
    # Training parameters
    'training': {
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 10,  # Early stopping patience
        'validation_interval': 1,
    },
    
    # Data parameters
    'data': {
        'input_size': (256, 256),
        'use_augmentation': True,
        'validation_split': 0.2,
        'test_split': 0.1,
    },

    'device': "cuda"  # Options: "cuda", "cpu", "mps"
}

# Evaluation parameters
EVALUATION_CONFIG = {
    'metrics': ['dice', 'iou', 'precision', 'recall', 'f1_score', 'hausdorff'],
    'threshold': 0.5,  # Threshold for binary segmentation
    'save_visualizations': True,
}

# Preprocessing parameters
PREPROCESSING_CONFIG = {
    'normalize': True,
    'clahe': False,
    'denoise': True,
    'resize': (512, 512),
    'crop': None,  # Set to (width, height) to enable cropping
}

# Post-processing parameters
POST_PROCESSING_CONFIG = {
    'use_watershed_separation': True,  # Global setting for watershed separation
    'watershed_min_distance': 10,  # Minimum distance between local maxima for watershed
    'min_object_size': 50,  # Remove objects smaller than this
    'min_hole_size': 20,  # Fill holes smaller than this
} 