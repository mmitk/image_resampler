import pathlib
import os
import sys

import imresample

parent_dir = pathlib.Path(__file__).parent.parent.absolute()
datasets_directory = os.path.join(parent_dir, 'datasets')

def test_resample_categorical(resampler, directory_name):
    res_directory = os.path.join(datasets_directory, 'resample\\{}'.format(directory_name))
    imresample.resample.resample_to_directory(resampler, os.path.join(datasets_directory, 'source_test_categorical'), res_directory)