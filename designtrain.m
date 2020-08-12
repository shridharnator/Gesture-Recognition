%%  Load Test and train
load('C:\Users\Shri\Desktop\project\2D task\Gesture\Utilities\TestData.mat');
load('C:\Users\Shri\Desktop\project\2D task\Gesture\Utilities\TrainingData.mat');
%% Design YOLOv2 network layers
inputLayer = imageInputLayer([128 128 3],'Name','input','Normalization','none');
filterSize = [3 3];
middleLayers = [
    convolution2dLayer(filterSize, 16, 'Padding', 1,'Name','conv_1',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2, 'Stride',2,'Name','maxpool1')
    convolution2dLayer(filterSize, 32, 'Padding', 1,'Name', 'conv_2',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu_2')
    maxPooling2dLayer(2, 'Stride',2,'Name','maxpool2')
    convolution2dLayer(filterSize, 64, 'Padding', 1,'Name','conv_3',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu_3')
    maxPooling2dLayer(2, 'Stride',2,'Name','maxpool3')
    convolution2dLayer(filterSize, 128, 'Padding', 1,'Name','conv_4',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN4')
    reluLayer('Name','relu_4')
    ];

%% Create layer graph for yolov2 network.
lgraph = layerGraph([inputLayer; middleLayers]);
numClasses = size(TrainingData,2)-1;
%% Define Anchor boxes
% open(fullfile("C:\Users\Shri\Desktop\project\2D task\Gesture\Utilities\AnchorBoxes.m"));

Anchors = [43 59
    18 22
    23 29
    84 109];
%% Assemble YOLOv2 network
lgraph = yolov2Layers([128 128 3],numClasses,Anchors,lgraph,'relu_4');
analyzeNetwork(lgraph);
%% Train the Network
doTraining = false; 
% setting this flag to true will build and train a YOLOv2 detector
% false will load a pre-trained network
if doTraining
    rng(0);
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.001, ...
        'Verbose',true,'MiniBatchSize',16,'MaxEpochs',80,...
        'Shuffle','every-epoch','VerboseFrequency',50, ...
        'DispatchInBackground',true,...
        'ExecutionEnvironment','auto');
        [detectorYolo2, info] = trainYOLOv2ObjectDetector(TrainingData,lgraph,options);    
else
    load(fullfile("Utilities","detectorYoloV2.mat")); %pre-trained detector loaded from a MAT file
end
%% Detect ROI's with the detector and Calculate Human aspect ratio 
results = table('Size',[height(TestData) 3],...
    'VariableTypes',{'cell','cell','cell'},...
    'VariableNames',{'Boxes','Scores', 'Labels'});
depVideoPlayer = vision.DeployableVideoPlayer;
for i = 1:height(TestData)
    
    % Read the image
    I = imread(TestData.imagefilename{i});
    
    % Run the detector.
    [bboxes,scores,labels] = detect(detectorYolo2,I);
    
    %
    if ~isempty(bboxes)
        I = insertObjectAnnotation(I,'Rectangle',bboxes,cellstr(labels));
    end    
     depVideoPlayer(I);
%      pause(0.1);
    % Collect the results in the results table
    results.Boxes{i} = floor(bboxes);
    results.Scores{i} = scores;
    results.Labels{i} = labels;
    
end
release(depVideoPlayer);
