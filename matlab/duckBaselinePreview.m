function duckBaselinePreview
%DUCKBASELINEPREVIEW Check three real frames with the historical duck detector.
cfg = ar.config('duck');
out = fullfile(cfg.root,'runs','duck','step01_preflight');
assert(isfolder(out),'Run duckStep01 first');
assert(~isfile(fullfile(out,'detector_preview.png')),'Preview already exists');
model = load(fullfile(cfg.root,cfg.object.matlab_model));
reader = VideoReader(fullfile(cfg.original_matlab_root,'duck1.mp4'));
fractions = [0.2 0.5 0.8]; frames = cell(1,3); results = struct([]);
for k = 1:numel(fractions)
    sampleTime = reader.Duration*fractions(k);
    reader.CurrentTime = sampleTime;
    image = readFrame(reader);
    [boxes,scores] = detect(model.detector_duck,image,ExecutionEnvironment='cpu');
    bestScore = NaN; bestBox = zeros(0,4);
    if ~isempty(scores)
        [bestScore,index] = max(scores); bestBox = boxes(index,:);
        image = insertShape(image,'rectangle',bestBox,LineWidth=4,Color='green');
    end
    if isempty(bestBox)
        caption = sprintf('duck1.mp4 t=%.1fs no boxes >=0.50',sampleTime);
    else
        caption = sprintf('duck1.mp4 t=%.1fs score=%.3f',sampleTime,bestScore);
    end
    image = insertText(image,[8 8],caption, ...
        FontSize=20,BoxColor='black',TextColor='white');
    frames{k} = imresize(image,[360 480]);
    results(k).time = sampleTime;
    results(k).score = bestScore;
    results(k).box = bestBox;
    fprintf('Time %.2f s: best score %.4f, box %s\n',sampleTime,bestScore,mat2str(bestBox));
end
sheet = imtile(frames,GridSize=[1 3]);
imwrite(sheet,fullfile(out,'detector_preview.png'));
imageshow(sheet);
save(fullfile(out,'baseline_samples.mat'),'results');
end
