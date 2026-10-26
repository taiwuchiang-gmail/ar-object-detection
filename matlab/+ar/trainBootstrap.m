function modelPath = trainBootstrap(objectName, groundTruthPath, baseNetworkPath, runName)
%TRAINBOOTSTRAP Train the small YOLOv2 model from groundTruth labels.
% baseNetworkPath must contain lgraph. Pass paths explicitly (prefer absolute).
cfg = ar.config(objectName);
assert(~isempty(regexp(runName,'^[A-Za-z0-9_-]+$','once')),'Invalid run name');
out = fullfile(cfg.root,'runs','matlab',runName);
assert(~isfolder(out),'Run already exists');
assert(isfile(groundTruthPath) || isfolder(groundTruthPath),'Training input missing');
assert(isfile(baseNetworkPath),'YOLOv2 base network missing');
if isfile(groundTruthPath)
    data = load(groundTruthPath,'gTruth');
    assert(isfield(data,'gTruth'),'Expected gTruth variable');
end
base = load(baseNetworkPath,'lgraph');
assert(isfield(base,'lgraph'),'Expected lgraph variable');
mkdir(out); frames = fullfile(out,'frames'); mkdir(frames);
if isfolder(groundTruthPath)
    trainingData = ar.trainingData(groundTruthPath,objectName);
else
    [images,boxes] = objectDetectorTrainingData(data.gTruth, ...
        SamplingFactor=1,WriteLocation=frames);
    trainingData = combine(images,boxes);
end
options = trainingOptions('sgdm',InitialLearnRate=0.001, ...
    MiniBatchSize=32,MaxEpochs=40,Shuffle='every-epoch', ...
    Verbose=true,VerboseFrequency=10);
[detector,info] = trainYOLOv2ObjectDetector(trainingData,base.lgraph,options);
modelPath = fullfile(out,'detector.mat');
save(modelPath,'detector','info','objectName','groundTruthPath','baseNetworkPath');
end
