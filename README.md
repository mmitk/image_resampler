# Image Resample

This toolkit allows you to resample a directory containing an imbalanced image dataset using resampling strategies from
imbalanced-learn (https://imbalanced-learn.readthedocs.io/en/stable/index.html "Imbalanced-learn Homepage")

Leverages both imbalanced-learn and opencv-python to create a resampled balanced image dataset from an imbalanced image dataset to help with machine learning and deep learning.

# Usage:

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
