function outputDir = prepareSynthetic(objectName, videoPath, iteration)
%PREPARESYNTHETIC Extract automatically labeled frames and separate object crops.
% Full frames retain detection context. Crops are assets, not full-frame labels.
cfg = ar.config(objectName);
assert(~isempty(regexp(iteration,'^[A-Za-z0-9_-]+$','once')),'Invalid iteration');
assert(isfile(videoPath),'Video missing');
outputDir = fullfile(cfg.root,'datasets',objectName,iteration);
assert(~isfolder(outputDir),'Iteration exists');
mkdir(outputDir);
for name = ["images","labels","crops"]
    mkdir(fullfile(outputDir,name));
end
fid = fopen(fullfile(outputDir,'provenance.jsonl'),'w');
assert(fid>=0,'Cannot write provenance'); cleanup = onCleanup(@() fclose(fid));
reader = VideoReader(videoPath); frameNumber = 0;
p = cfg.object;
while hasFrame(reader)
    image = readFrame(reader); frameNumber = frameNumber+1;
    box = ar.foregroundBox(image,p.threshold,p.dilation_radius,p.min_area);
    if isempty(box), continue; end
    name = sprintf('frame_%08d',frameNumber);
    normalized = ar.yoloBox(box,size(image));
    imwrite(image,fullfile(outputDir,'images',[name '.png']));
    imwrite(imcrop(image,box),fullfile(outputDir,'crops',[name '.png']));
    label = fopen(fullfile(outputDir,'labels',[name '.txt']),'w');
    assert(label>=0,'Cannot write label');
    fprintf(label,'0 %.8f %.8f %.8f %.8f\n',normalized); fclose(label);
    record = struct('image',[name '.png'],'source',char(videoPath), ...
        'frame',frameNumber,'object',objectName,'iteration',iteration, ...
        'label_origin','synthetic_foreground');
    fprintf(fid,'%s\n',jsonencode(record));
end
end
