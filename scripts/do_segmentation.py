import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.traditional import TraditionalSegmenterFactory
from src.preprocessing import Preprocessor
from configs.default_config import TRADITIONAL_CONFIG, CELL_SEGMENTATION_CONFIG

def main():
    pass    
    

if __name__ == "__main__":
    main() 