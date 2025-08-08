import cv2
import os

def load_img(args):
    """
    Util function for each worker in a pool for loading in raw data images
    """
    folder, filename = args
    full_path = os.path.join(folder,filename)
    return cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)