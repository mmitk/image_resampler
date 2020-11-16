import numpy as np
import cv2
import os
import sys
import shutil
import time
from exceptions shapeException, inputException
import imblearn as imb

resample_strategies = imb

def resample_to_directory(resampler, src, target_directory, target_size):
    """Perform sampling on an image dataset using imbalanced-learn's resampler
    and rewrite to the target directory

    Parameters
    ----------

    resampler: object
        resampler from imblearn.under_sampling or imblearn.over_sampling
        implementing fit_resample. Can be accessed via imresample.resample_strategies.

    src: str or tuple
        if str absolute path to source directory containing images to be resampled
            images must be sorted into seperate folders designating the class
        if tuple then (images array, targets array) 
            images array is type ndarray with with dimension == 2
            targets array is type ndarray with dimension == 1

    target_directory: str
        absolute path of directory into which resampled resampled image set is written.
    
    target_size: tuple 
        tuple (width, height), where width is the desired width, and height is desired height

    """
    
    if isinstance(src,str): 
        images_arr, targets = load_directory(src)
    elif isinstance(src, tuple):
        #TODO check ndarray dimensions
        if not isinstance(src[0],np.ndarray):
            raise inputException('if passing tuple must be: (ndarray, ndarray)')
        images_arr = src[0]
        targets = src[1]
    
    

def resample_image_set(resampler, src, target_size = None):

def load_directory(directory_path, target_size = None):

def write_to_directory(image_array)