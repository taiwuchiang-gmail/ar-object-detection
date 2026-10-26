function duckDiagnosticPreview
%DUCKDIAGNOSTICPREVIEW Compare synthetic and real footage at a diagnostic threshold.
cfg = ar.config('duck');
out = fullfile(cfg.root,'runs','duck','step01_preflight');
assert(~isfile(fullfile(out,'diagnostic_preview.png')),'Diagnostic output exists');
model = load(fullfile(cfg.root,cfg.object.matlab_model));
files = {'duck_cropped.mp4','duck-test.mp4','duck1.mp4'};
frames = cell(1,3); results = struct([]);
for k = 1:numel(files)
    reader = VideoReader(fullfile(cfg.original_matlab_root,files{k}));
    sampleTime = reader.Duration/2; reader.CurrentTime = sampleTime;
    image = readFrame(reader);
    [boxes,scores] = detect(model.detector_duck,image,Threshold=0.01,ExecutionEnvironment='cpu');
    bestScore = NaN; bestBox = zeros(0,4);
    if ~isempty(scores)
        [bestScore,index] = max(scores); bestBox = boxes(index,:);
        image = insertShape(image,'rectangle',bestBox,LineWidth=4,Color='green');
    end
    image = insertText(image,[8 8],sprintf('%s score=%.3f',files{k},bestScore), ...
        FontSize=20,BoxColor='black',TextColor='white');
    frames{k} = imresize(image,[360 480]);
    results(k).file = files{k}; results(k).time = sampleTime;
    results(k).score = bestScore; results(k).box = bestBox;
    results(k).acceptedAtOriginalThreshold = bestScore>=cfg.object.confidence;
end
sheet = imtile(frames,GridSize=[1 3]);
imwrite(sheet,fullfile(out,'diagnostic_preview.png'));
imageshow(sheet);
save(fullfile(out,'diagnostic_samples.mat'),'results');
end
