# Author: Michael Mitkov
# Licencse: MIT

import numpy as np
import cv2
import os
import pathlib
import sys
import shutil
import time
from .exceptions import shapeException, inputException
import imblearn as imb

oversampling_strategies = imb.over_sampling
undersampling_strategies = imb.under_sampling

def resample_to_directory(resampler, src, target_directory, target_size = (64,64)):
    """Load images from source directory then perform sampling on an image dataset using imbalanced-learn's resampler
    and rewrite to the target directory

    Parameters
    ----------
    resampler: object
        resampler from imblearn.under_sampling or imblearn.over_sampling
            implementing fit_resample. Can be accessed via imresample.resample_strategies.

    src: str 
        absolute path to source directory containing images to be resampled
            images must be sorted into seperate folders designating the class
        
    target_directory: str
        absolute path of directory into which resampled resampled image set is written.
    
    target_size: tuple 
        tuple (width, height), where width is the desired width, and height is desired height

    """
     

    images_arr, targets, class_encoding = load_directory(src, target_size)

    images_arr = images_arr.reshape(images_arr.shape[0],images_arr.shape[1]*images_arr.shape[2]*images_arr.shape[3])
    targets = targets.reshape(-1,1)

    X, y = resampler.fit_resample(images_arr, targets)

    X = X.reshape(X.shape[0], target_size[0], target_size[1], 3)
    y = y.reshape(1,-1)

    write_to_directory(X, y, target_directory, class_encoding)



def resample_image_set(resampler, images_array, targets_array, x_reshape = False, y_reshape = False):
    """Perform resampling on a given numpy array of images and targets (classes), return resampled array's

    Parameters
    ----------
    resampler: object
        resampler from imblearn.under_sampling or imblearn.over_sampling
        implementing fit_resample. Can be accessed via imresample.resample_strategies.
    
    images_array: {array-like, sparse matrix}, shape (n_samples, n_images)
            Data array.
    
    targets_array: array-like, shape (n_samples,)
            Target array.
    
    x_reshape: boolean, optional (default=False)
        dictates whether reshaping of images array from 4 dimensions to 2 dimensions should occur
        (for imblearn resampler images array must be <=2 dimensions)
    
    y_reshape: boolean, optional (default=False)
        dictates whether reshaping of targets array should take place
        shape must be (n_samples,)

     Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape (n_samples_new, n_images_new)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`
    """


    if x_reshape:
        images_array = images_array.reshape(images_array.shape[0], images_array.shape[1]*images_array.shape[2]*images_array.shape[3])
    if y_reshape:
        targets_array = targets_array.reshape(-1,1)

    return resampler.fit_resample(images_array, targets_array)


def load_directory(directory_path, target_size = (64,64), flatten = False):
    """Loads directory into two numpy ndarray's and class encoding. One ndarray will hold
    the loaded images, one will hold the corresponding classes. Class encoding is a dict
    which stores the true class names in correlation to the returned ones.

    Parameters
    ----------
    directory_path: str
        absolute path to directory containing class folder (one folder per class) where each class folder 
        contains images belonging to that class
    
    target_size: tuple, optional (default=(64,64))
        tuple (width, height), where width is the desired target width, and height is desired target height
        function will resize each loaded image to this size

    flatten: boolean, optional (default=False)
        dictates whether reshaping from 4 dimensional to 2 dimensional array should occur.
        Note: 
        2D array is required for resampling
    
    Returns
    -------
        X_resampled : {ndarray, sparse matrix}, shape (n_samples_new, n_images_new)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`
    """

    class_encoding = dict()
    images_list = list()
    targets_list = list()

    for count, folder in enumerate(os.listdir(directory_path)):
        if os.path.isdir(os.path.join(directory_path, folder)):
            class_encoding[count] = folder
    
    for key in class_encoding:
        class_directory = os.path.join(directory_path, class_encoding[key])
        for image in os.listdir(class_directory):
            print('.', end='')
            image = os.path.join(class_directory, image)
            loaded_image = cv2.imread(image)
            loaded_image = cv2.resize(loaded_image, target_size, interpolation=cv2.INTER_CUBIC)
            images_list.append(loaded_image)
            targets_list.append(key)
    
    print()
    images_arr, targets_arr, class_encoding = np.asarray(images_list), np.asarray(targets_list), class_encoding
    if flatten:
        images_arr = images_arr.reshape(images_arr.shape[0],images_arr.shape[1]*images_arr.shape[2]*images_arr.shape[3])
        targets_arr = targets_arr.reshape(-1,1)
        return images_arr, targets_arr, class_encoding

    return images_arr, targets_arr, class_encoding



def write_to_directory(image_array, target_array, target_directory, class_encoding = None, x_reshape = False, y_reshape = False, target_size = None, write_format='jpg'):
    """Write a given image and target array to a given directory, a new folder is created for each class if one does not exist.

    Parameters
    ----------
    resampler: object

        images_array: {array-like, sparse matrix}, shape (n_samples, n_images) or (n_samples, image_width, image_height, 3)
            Data array.
    
        targets_array: array-like, shape (n_samples,) or (n_samples, n_classes)
            Target array.
    
        target_directory: str
            absolute path of directory into which resampled resampled image set is written.
            Directory will be created if it does not exist

        class_encoding: dict or None
            if dict then should contain mapping of true class names to encoded class names
            if None then encoded class names from target_array will be used to name class folders
            Note:
            A folder will be created within the target directory for each unique class in target_array,
            unless they already exist (will not overwrite)

        x_reshape: boolean, optional (default=False)
            dictates whether reshaping of images array from 2 dimensions to 4 dimensions should occur
            If image array is of shape (n_samples, n_images) then this should be true.
    
        y_reshape: boolean, optional (default=False)
            dictates whether reshaping of targets array should take place
            If image array is of shape (n_samples,) then this should be true

        target_size: tuple 
            tuple (width, height), where width is the desired image width, and height is desired image height
        
    """
    if not os.path.isdir(target_directory):
        pathlib.Path(target_directory).mkdir(parents = True, exist_ok=True)

    if x_reshape:
        image_array = image_array.reshape(image_array.shape[0], target_size[0], target_size[1], 3)
    if y_reshape:
        target_array = target_array.reshape(-1,1)
    
    for count, img in enumerate(image_array):
        if class_encoding is not None:
            class_directory = os.path.join(target_directory, class_encoding[target_array[0][count]])
        else:
            class_directory = os.path.join(target_directory, str(target_array[0][count]))
        
        if not os.path.isdir(class_directory):
            os.mkdir(class_directory)
        
        image_path = os.path.join(class_directory, 'img{}.{}'.format(count,write_format))

        try:
            cv2.imwrite(image_path, img)
        except Exception as e:
            continue



        

