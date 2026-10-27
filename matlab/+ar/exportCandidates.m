function outputDir = exportCandidates(objectName, videoPath, iteration, modelPath, threshold, frameStride)
%EXPORTCANDIDATES Export high-confidence YOLOv2 labels into a new iteration.
cfg = ar.config(objectName);
if nargin < 4, modelPath = fullfile(cfg.root,cfg.object.matlab_model); end
if nargin < 5, threshold = cfg.object.confidence; end
if nargin < 6, frameStride = 1; end
assert(frameStride>=1 && frameStride==floor(frameStride),'Frame stride must be a positive integer');
assert(threshold>=0 && threshold<=1,'Confidence must be between zero and one');
assert(~isempty(regexp(iteration,'^[A-Za-z0-9_-]+$','once')),'Invalid iteration');
assert(isfile(videoPath) && isfile(modelPath),'Video or model missing');
model = load(modelPath);
if isfield(model,'detector')
    detector = model.detector;
else
    detector = model.(cfg.object.matlab_variable);
end
outputDir = fullfile(cfg.root,'datasets',objectName,iteration);
assert(~isfolder(outputDir),'Iteration already exists');
mkdir(outputDir); mkdir(fullfile(outputDir,'images')); mkdir(fullfile(outputDir,'labels'));
fid = fopen(fullfile(outputDir,'provenance.jsonl'),'w');
assert(fid>=0,'Cannot write provenance'); cleanup = onCleanup(@() fclose(fid));
video = VideoReader(videoPath); frameNumber = 0; accepted = 0;
while hasFrame(video)
    image = readFrame(video); frameNumber = frameNumber+1;
    if mod(frameNumber-1,frameStride)~=0, continue; end
    [boxes,scores] = detect(detector,image,Threshold=threshold);
    if isempty(scores), continue; end
    [score,index] = max(scores);
    if score < threshold, continue; end
    normalized = ar.yoloBox(boxes(index,:),size(image));
    name = sprintf('frame_%08d',frameNumber);
    imwrite(image,fullfile(outputDir,'images',[name '.png']));
    label = fopen(fullfile(outputDir,'labels',[name '.txt']),'w');
    assert(label>=0,'Cannot write label');
    fprintf(label,'0 %.8f %.8f %.8f %.8f\n',normalized); fclose(label);
    record = struct('image',[name '.png'],'source',char(videoPath), ...
        'frame',frameNumber,'model',char(modelPath),'confidence',score, ...
        'threshold',threshold,'object',objectName,'iteration',iteration, ...
        'label_origin','yolov2_prediction','frame_stride',frameStride);
    fprintf(fid,'%s\n',jsonencode(record)); accepted=accepted+1;
end
fprintf('Candidate export: %d accepted from %d sampled frames, threshold %.2f\n', ...
    accepted,ceil(frameNumber/frameStride),threshold);
end
