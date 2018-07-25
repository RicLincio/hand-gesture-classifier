%% Project A1 - Riccardo Lincetto
% Hand pose detection: given a dataset of RGBD images reproducing a set of 
% hand poses, build a gesture recognition system. Apply the gesture 
% recognition system to a set of stereo images: compute the disparity and 
% generate the RGBD data to be processed by the classifier.

clear; close all; clc;
addpath('data')
addpath('functions')

%% prepare training set
% with 3 channels: (gray, depth, ~)

% load data
clear
load data\dataset_hand
% process data
trainSet = DIBRtrain;
trainLabel = trainLabel;
valSet = DIBRtest;
valLabel = testLabel;
% store data
save data\training_set trainSet trainLabel
save data\validation_set valSet valLabel

%% prepare test set
% from stereo images to 3 channels: (gray, depth, ~)

% load data
clear
load calib_data;
num_images = 9;
[imleft, imright] = importImages('stereo_images', 1080, 1920, num_images);
fixImage8
% process data
testSet = processStereoImages(imleft, imright, stereoParams);
testLabel = [1,1,1,1,2,2,2,3,3];
% store data
save data\test_set testSet testLabel

%% train the cnn model

% load training and test sets
clear; close all
load training_set.mat                                                      % size 28 x 28 x 4 x 8370
load validation_set.mat                                                    % size 28 x 28 x 4 x 1440
load test_set                                                              % size 28 x 28 x 4 x 9
% prepare training and test sets
trainSet = prepareSet(trainSet);
trainLabel = categorical(trainLabel);
valSet = prepareSet(valSet);
valLabel = categorical(valLabel);
testSet = prepareSet(testSet);
testLabel = categorical(testLabel);
% optimization parameters
params.miniBatchSize = size(trainLabel, 2) / 30;                           % minibatch size
params.numValidationsPerEpoch = 2;                                         % validations per epoch
params.validationFrequency = floor(size(trainLabel, 2) ...
                             / params.miniBatchSize ...
                             / params.numValidationsPerEpoch);
% network options
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs', 3, ...
    'ValidationData', {valSet, valLabel}, ...
    'ValidationFrequency', 30, ...
    'VerboseFrequency', 30, ...
    'Verbose', true,...                                                    % display messages
    'Plots', 'training-progress');                                         % display plots
% define layers
layers = [
    imageInputLayer(size(trainSet(:,:,:,1)),'Name','input')                % first input layer

    convolution2dLayer(3,64,'Padding',1,'Name','conv_1')                   % conv layer
    batchNormalizationLayer('Name','bn_1')                                 % normalize
    reluLayer('Name','relu_1')                                             % ReLU
    maxPooling2dLayer(2,'Stride',2,'Name','pool_1')                        % max pooling

    convolution2dLayer(3,128,'Padding',1,'Name','conv_2')                   % conv layer
    batchNormalizationLayer('Name','bn_2')                                 % normalize
    reluLayer('Name','relu_2')                                             % ReLU  
    maxPooling2dLayer(2,'Stride',2,'Name','pool_2')                        % max pooling

    convolution2dLayer(3,256,'Padding',1,'Name','conv_3')                  % conv layer
    batchNormalizationLayer('Name','bn_3')                                 % normalize
    reluLayer('Name','relu_3')                                             % ReLU
    maxPooling2dLayer(2,'Stride',2,'Name','pool_3')                        % max pooling
    
    fullyConnectedLayer(784,'Name','full_1')                               % fully connected large
    dropoutLayer(0.3,'Name','drop')
    fullyConnectedLayer(3,'Name','full_2')                                 % fully connected small
    softmaxLayer('Name','prob')                                            % softmax
    classificationLayer('Name','output')];                                 % classification
% train the network
rng('default')
net = trainNetwork(trainSet, trainLabel ,layers, options);

%% Classify the new test set

result.prob = predict(net, testSet, 'ExecutionEnvironment', 'gpu');        % predict softmax probs
[result.top, result.class] = max(result.prob');                            % classification index
result.class

% 1 -> five
% 2 -> fist
% 3 -> ok