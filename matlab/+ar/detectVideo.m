function outputPath = detectVideo(objectName, videoPath, runName, modelPath)
%DETECTVIDEO Save an annotated video using a copied YOLOv2 detector.
cfg = ar.config(objectName);
if nargin < 4, modelPath = fullfile(cfg.root,cfg.object.matlab_model); end
assert(~isempty(regexp(runName,'^[A-Za-z0-9_-]+$','once')),'Invalid run name');
assert(isfile(videoPath) && isfile(modelPath),'Video or model missing');
model = load(modelPath);
if isfield(model,'detector')
    detector = model.detector;
else
    detector = model.(cfg.object.matlab_variable);
end
out = fullfile(cfg.root,'runs','matlab',runName);
assert(~isfolder(out),'Run already exists');
reader = VideoReader(videoPath);
mkdir(out); outputPath = fullfile(out,'detected.mp4');
writer = VideoWriter(outputPath,'MPEG-4');
writer.FrameRate = reader.FrameRate;
open(writer); cleanup = onCleanup(@() close(writer));
while hasFrame(reader)
    image = readFrame(reader);
    [boxes,scores] = detect(detector,image);
    if ~isempty(scores)
        [score,index] = max(scores);
        if score >= cfg.object.confidence
            image = insertObjectAnnotation(image,'rectangle',boxes(index,:),score);
        end
    end
    writeVideo(writer,image);
end
end
