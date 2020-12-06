%% Project A1 - Riccardo Lincetto
% Hand pose detection: given a dataset of RGBD images reproducing a set of 
% hand poses, build a gesture recognition system. Apply the gesture 
% recognition system to a set of stereo images: compute the disparity and 
% generate the RGBD data to be processed by the classifier.

clear; close all; clc;
addpath lib

%% training data
% NOTE: data shuffling is set in trainingOptions
load(fullfile('..','data','training_set'))                                 % size 28 x 28 x 4 x 8370
load(fullfile('..','data','validation_set'))                               % size 28 x 28 x 4 x 1440

% TODO create GUI to explore training data
% display the RGB image and the depth map, with relevant statistics

% matlab supports 1 or 3 channel images, use (gray, depth, ~)
trainSet = prepareSet(trainSet);                                           % size 28 x 28 x 3 x 8370
trainLabel = categorical(trainLabel,[1,2,3],{'five' 'fist' 'ok'});         % size 28 x 28 x 3 x 1440
valSet = prepareSet(valSet);
valLabel = categorical(valLabel,[1,2,3],{'five' 'fist' 'ok'});

% data augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-180 180], ...
                                    'RandXTranslation',[-3 3], ...
                                    'RandYTranslation',[-3 3], ...
                                    'RandScale',[.8 1.1], ...
                                    'RandXShear',[-20 20], ...
                                    'RandXReflection', true, ...
                                    'RandYReflection', true);
augimds = augmentedImageDatastore(size(trainSet,1:3),trainSet,trainLabel, ...
                                  'DataAugmentation',imageAugmenter);

%% test data
% load stereo images, compute disparity maps, transform them into an RGBD
% matrix and then transform them as the training data in (gray, depth, ~)

% load data
load(fullfile('..','data','calib_data'));
num_images = 9;
[imleft, imright] = importImages(fullfile('..','data','test_stereo'), ...
                                 1080, 1920, num_images);

% process data
testSet = processStereoImages(imleft, imright, stereoParams, false);       % size 28 x 28 x 4 x 9
testLabel = [1,1,1,1,2,2,2,3,3];

% store data
save(fullfile('..','data','test_set'),'testSet','testLabel')
load(fullfile('..','data','test_set'),'testSet','testLabel')

% reformat data
testSet = prepareSet(testSet);                                             % size 28 x 28 x 3 x 9
testLabel = categorical(testLabel,[1,2,3],{'five' 'fist' 'ok'});

%% cnn model
close all

% optimization parameters
params.miniBatchSize = size(trainLabel, 2) / 30;                           % minibatch size
params.numValidationsPerEpoch = 2;                                         % validations per epoch
params.validationFrequency = floor(size(trainLabel, 2) ...
                             / params.miniBatchSize ...
                             / params.numValidationsPerEpoch);
% network options
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs', 20, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valSet, valLabel}, ...
    'ValidationFrequency', params.validationFrequency, ...
    'VerboseFrequency', params.validationFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',4, ...
    'LearnRateDropFactor',0.5, ...
    'Verbose', true, ...                                                   % display messages
    'Plots', 'training-progress');                                         % display plots

% define layers
layers = [
    imageInputLayer(size(trainSet(:,:,:,1)),'Name','input')                % first input layer

    convolution2dLayer(3,32,'Padding',1,'Name','conv_1')                   % conv layer
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
net = trainNetwork(augimds,layers, options);

%% Classify the new test set

result.class = classify(net, testSet, 'ExecutionEnvironment', 'auto');     % predict softmax probs
subplot(2,1,1)
confusionchart(testLabel,categorical(result.class))

% how sure is the network on what it is predicting?
result.prob = predict(net, testSet, 'ExecutionEnvironment', 'auto');       % predict softmax probs
subplot(2,1,2)
bar(result.prob,'stacked')
legend('five','fist','ok')

%% Inspect visually the network layers

layer = 'conv_1';
channels = 1:64;

I = deepDreamImage(net,layer,channels, ...
    'PyramidLevels',1, ...
    'Verbose',0, ...
    'ExecutionEnvironment','auto');

figure
for i = 1:64
    subplot(8,8,i)
    imshow(I(:,:,:,i))
end