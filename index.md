## Image-resampler

This toolkit allows you to resample a directory containing an imbalanced image dataset using resampling strategies from
imbalanced-learn (https://imbalanced-learn.readthedocs.io/en/stable/index.html)

Leverages both imbalanced-learn and opencv-python to create a resampled balanced image dataset from an imbalanced image dataset to help with machine learning and deep learning.

This library was written by Michael Mitkov while performing research for Professor Farhad Pourkamali-Anaraki at University of Massachusetts Lowell.

### Usage
Imbalanced learn resamplers can be accessed via imblearn directly (ClusterCentroids strategy is used as example):
```python
from imblearn.under_sampling import ClusterCentroids
CondNN = ClusterCentroids(kwargs)
```
or via imresample:
```python
import imresample.resample
cc = imresample.resample.undersampling_strategies.ClusterCentroids()

adasyn = imresample.resample.oversampling_strategies.ADASYN()
```

To resample a given directory containing image and rewrite into a new directory:
```python
imresample.resample.resample_to_directory(resampler, 'absolute/path/to/source/directory', 'absolute/path/to/target/directory')
```
Example:
```python
import imresample.resample

adasyn = imresample.resample.oversampling_strategies.ADASYN() # create instance of resampler
imresample.resample.resample_to_directory(adasyn, 'absolute/path/to/source/directory', 'absolute/path/to/target/directory') #resample image dataset and write into given target directory
```
Note:
Source directory must have images into folders corresponding to their class/category. A folder for each class will be created
within target directory will be created if it does not already exist. Any existing folders/files in target directory WILL NOT
be overwritten.

### Documentation

### imresample.resample.resample_to_directory 
`def resample_to_directory(resampler, src, target_directory, target_size = (64,64))`
Load images from source directory then perform sampling on an image dataset using imbalanced-learn's resamplerand rewrite to the target directory  
  
__Parameters__  

    resampler: object  
        resampler from imblearn.under_sampling or imblearn.over_sampling  
        implementing fit_resample. Can be accessed via imresample.resample_strategies  

    src: str 
        absolute path to source directory containing images to be resampled  
        images must be sorted into seperate folders designating the class  
        
    target_directory: str
        absolute path of directory into which resampled resampled image set is written.  
    
    target_size: tuple 
        tuple (width, height), where width is the desired width, and height is desired height  
 __Returns__
        
        None  
          
### imresample.resample.resample_image_set 
`def resample_image_set(resampler, images_array, targets_array, x_reshape = False, y_reshape = False):`
Perform resampling on a given numpy array of images and targets (classes), return resampled arrays    
  
__Parameters__  

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
 __Returns__
        
        X_resampled : {ndarray, sparse matrix}, shape (n_samples_new, n_images_new)  
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)  
            The corresponding label of `X_resampled`  
  
### imresample.resample.load_directory
`def load_directory(directory_path, target_size = (64,64), flatten = False):`
Loads directory into two numpy ndarray's and class encoding. One ndarray will hold the loaded images, one will hold the corresponding classes. Class encoding is a dict which stores the true class names in correlation to the returned ones.  
  
__Parameters__  

   
    directory_path: str
        absolute path to directory containing class folder (one folder per class) where each class 
        folder contains images belonging to that class
    
    target_size: tuple, optional (default=(64,64))
        tuple (width, height), where width is the desired target width, and height is desired target 
        height function will resize each loaded image to this size

    flatten: boolean, optional (default=False)
        dictates whether reshaping from 4 dimensional to 2 dimensional array should occur.
        Note: 
        2D array is required for resampling
 __Returns__
        
        X_resampled : {ndarray, sparse matrix}, shape (n_samples_new, n_images_new)
            The array containing the resampled image data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`

   
  ### imresample.resample.write_to_directory
`def load_directory(directory_path, target_size = (64,64), flatten = False):`
Write a given image and target array to a given directory, a new folder is created for each class if one does not exist.   
  
__Parameters__  

   
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
            A folder will be created within the target directory for each unique class in                         
            target_array, unless they already exist (will not overwrite)

        x_reshape: boolean, optional (default=False)
            dictates whether reshaping of images array from 2 dimensions to 4 dimensions should occur
            If image array is of shape (n_samples, n_images) then this should be true.
    
        y_reshape: boolean, optional (default=False)
            dictates whether reshaping of targets array should take place
            If image array is of shape (n_samples,) then this should be true

        target_size: tuple 
            tuple (width, height), where width is the desired image width, and height is desired                 image height
 __Returns__
        
        None

    
