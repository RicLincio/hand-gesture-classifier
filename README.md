# Hand pose classification
Hand pose classification from RGBD image representation with deep learning models.

Training set: RGBD frames obtained from "2.5D" sensor.
Test set: stereo frames converted to RGBD representaiton through disparity map computation.
Classes:
1. 'hi'
2. 'fist'
3. 'ok'

The task is carried out in matlab and python (underway).
The original dataset 'dataset_hand.mat' is missing, because too large (exceeding 100 MB).
The code is here presented with the preprocessed training data in 'data' folder,
which is the same used in the python section.

The chosen model is based on CNNs.