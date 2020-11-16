import test_binary
import test_categorical
import sys
import pathlib

import imresample.resample

def test_binary_all():
    ros = imresample.resample.oversampling_strategies.RandomOverSampler()
    test_binary.test_resample_binary(ros, 'ROS_BINARY')

    adsn = imresample.resample.oversampling_strategies.ADASYN()
    test_binary.test_resample_binary(adsn, 'ADASYN_BINARY')

    bdln = imresample.resample.oversampling_strategies.BorderlineSMOTE()
    test_binary.test_resample_binary(bdln, 'bdlnSMOTE_BINARY')

    sm = imresample.resample.oversampling_strategies.SMOTE()
    test_binary.test_resample_binary(sm, 'SMOTE_BINARY')

    cc = imresample.resample.undersampling_strategies.ClusterCentroids()
    test_binary.test_resample_binary(cc, 'ClustCent_BINARY')

    enn = imresample.resample.undersampling_strategies.EditedNearestNeighbours()
    test_binary.test_resample_binary(enn, 'ENN_BINARY')

    allknn = imresample.resample.undersampling_strategies.AllKNN()
    test_binary.test_resample_binary(allknn, 'ALLKNN_BINARY')

    iht = imresample.resample.undersampling_strategies.InstanceHardnessThreshold()
    test_binary.test_resample_binary(iht, 'IntsHardThresh_BINARY')

    nm = imresample.resample.undersampling_strategies.NearMiss()
    test_binary.test_resample_binary(nm, 'NearMiss_BINARY')

def test_categorical_all():
    ros = imresample.resample.oversampling_strategies.RandomOverSampler()
    test_categorical.test_resample_categorical(ros, 'ROS_categorical')

    adsn = imresample.resample.oversampling_strategies.ADASYN()
    test_categorical.test_resample_categorical(adsn, 'ADASYN_categorical')

    bdln = imresample.resample.oversampling_strategies.BorderlineSMOTE()
    test_categorical.test_resample_categorical(bdln, 'bdlnSMOTE_categorical')

    kms = imresample.resample.oversampling_strategies.KMeansSMOTE()
    test_categorical.test_resample_categorical(kms, 'KMEANSSMOTE_categorical')

    sm = imresample.resample.oversampling_strategies.SMOTE()
    test_categorical.test_resample_categorical(sm, 'SMOTE_categorical')

    cc = imresample.resample.undersampling_strategies.ClusterCentroids()
    test_categorical.test_resample_categorical(cc, 'ClustCent_categorical')

    enn = imresample.resample.undersampling_strategies.EditedNearestNeighbours()
    test_categorical.test_resample_categorical(enn, 'ENN_categorical')

    allknn = imresample.resample.undersampling_strategies.AllKNN()
    test_categorical.test_resample_categorical(allknn, 'ALLKNN_categorical')

    iht = imresample.resample.undersampling_strategies.InstanceHardnessThreshold()
    test_categorical.test_resample_categorical(iht, 'IntsHardThresh_categorical')

    nm = imresample.resample.undersampling_strategies.NearMiss()
    test_categorical.test_resample_categorical(nm, 'NearMiss_categorical')


if __name__ == '__main__':
    test_binary_all()
    test_categorical_all()