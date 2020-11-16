## Image-resampler

This toolkit allows you to resample a directory containing an imbalanced image dataset using resampling strategies from
imbalanced-learn (https://imbalanced-learn.readthedocs.io/en/stable/index.html "Imbalanced-learn Homepage")

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

__imresample.resample.resample_to_directory__  
`def resample_to_directory(resampler, src, target_directory, target_size = (64,64))`
Load images from source directory then perform sampling on an image dataset using imbalanced-learn's resamplerand rewrite to the target directory  
_Parameters_  

    resampler: object
        resampler from imblearn.under_sampling or imblearn.over_sampling
        implementing fit_resample. Can be accessed via imresample.resample_strategies

    src: str 
        absolute path to source directory containing images to be resampled
        images must be sorted into seperate folders designating the class
        
    target_directory: str
        absolute path of directory into which resampled resampled image set is written.
    
    target_size: tuple 
        tuple (width, height), where width is the desired width, and height is desired height```
