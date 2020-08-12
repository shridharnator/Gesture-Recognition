%% Video Resize (Training) 
vidobj = vision.VideoFileReader('C:\Users\Hancy\Desktop\Mini project\2D task\data\Video\5.mp4')
viewer = vision.DeployableVideoPlayer;
videoFWriter = vision.VideoFileWriter('C:\Users\Hancy\Desktop\Mini project\2D task\Deep learning yolov2\data\11.avi');
numFrames = 0;
    while ~isDone(vidobj)
              A = step(vidobj);  
              viewframe = imresize(A,[416,416]);
              step(viewer, viewframe);
              numFrames = numFrames + 1;
              step(videoFWriter, viewframe);
    end
release(vidobj);
% release(viewer);
release(videoFWriter);
%% Video Resize (Test)
vidobj = vision.VideoFileReader('C:\Users\Shri\Desktop\project\2D task\data\Video\4.mp4')
viewer = vision.DeployableVideoPlayer;
videoFWriter = vision.VideoFileWriter('C:\Users\Shri\Desktop\project\2D task\Deep learning yolov2\data\21.avi');
numFrames = 0;
    while ~isDone(vidobj)
              A = step(vidobj);  
              viewframe = imresize(A,[416,416]);
              step(viewer, viewframe);
              numFrames = numFrames + 1;
              step(videoFWriter, viewframe);
    end
release(vidobj);
release(viewer);
release(videoFWriter);
%% Create Video Labeller

%% Create Training Data from Ground Truth Data
if ~exist('Train', 'var')
    load (fullfile("Utilities","Train.mat")) 
end

if ~isfolder(fullfile("TrainingData"))
    mkdir TrainingData
end

TrainingData = objectDetectorTrainingData(Train,'SamplingFactor',1,...
    'WriteLocation','TrainingData');
save('C:\Users\shri\Desktop\project\2D task\Gesture\Utilities\TrainingData.mat','TrainingData');

% Display first few rows of the data set.
TrainingData(1:4,:)

%% Create Test Data
if ~isfolder(fullfile("TestData"))
    mkdir TestData
end
vid = VideoReader('C:\Users\Shri\Desktop\project\2D task\Gesture\data\Test.mp4')
TestData = table('Size',[vid.NumberOfFrames 1],...
    'VariableTypes',{'cell'},...
    'VariableNames',{'imagefilename'});
cd TestData
for img=1:vid.NumberOfFrames
    filename=strcat('Frame',num2str(img),'.jpg');
    image=read(vid,img);
    imwrite(image,filename);
    TestData.imagefilename{img} = which(filename);
end
cd ..  
save('C:\Users\Shri\Desktop\project\2D task\Gesture\Utilities\TestData.mat','TestData');
