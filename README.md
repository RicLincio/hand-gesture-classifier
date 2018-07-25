# hand-gesture-classifier
Hand pose detection.

Data is acquired using RGBD sensors for the training set and stereo images for the test set.
A CNN model is trained on RGBD images: then from stereo images the disparity is computed and a RGBD object is created.
There are 3 different classes representing the following poses:
1. 'hi'
2. 'fist'
3. 'ok'

Note: data folder is missing. Contact me for dataset and calibration data.