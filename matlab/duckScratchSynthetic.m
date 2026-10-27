function outputDir = duckScratchSynthetic(runId)
%DUCKSCRATCHSYNTHETIC Generate fresh duck labels directly from captured renders.
% UI is removed by a fixed color rule; no existing labels or detectors are used.
if nargin < 1, runId = 'scratch_20261026_01'; end
cfg = ar.config('duck');
source = fullfile(cfg.original_matlab_root,'duck.mp4');
outputDir = fullfile(cfg.root,'datasets','duck',[runId '_synthetic']);
assert(isfile(source) && ~isfolder(outputDir),'Source missing or dataset exists');
rng(17,'twister');
mkdir(outputDir); mkdir(fullfile(outputDir,'images')); mkdir(fullfile(outputDir,'labels'));
runDir = fullfile(cfg.root,'runs','duck',runId);
if ~isfolder(runDir), mkdir(runDir); end
fid = fopen(fullfile(outputDir,'provenance.jsonl'),'w');
assert(fid>=0,'Cannot write provenance'); cleanup = onCleanup(@() fclose(fid));
reader = VideoReader(source); frameIndex=0; count=0; skipped=0;
previews = cell(1,12); previewCount=0;
while hasFrame(reader)
    original = readFrame(reader); frameIndex=frameIndex+1;
    if mod(frameIndex-1,3)~=0, continue; end
    frame = imresize(original,0.5);
    hsvImage = rgb2hsv(frame);
    % The duck is yellow/orange. White controls and cyan Export button are excluded.
    mask = hsvImage(:,:,1)<=0.20 & hsvImage(:,:,2)>=0.25 & hsvImage(:,:,3)>=0.08;
    mask = imfill(bwareaopen(mask,80),'holes');
    mask = imdilate(mask,strel('disk',3));
    regions = regionprops(mask,'BoundingBox','Area');
    if isempty(regions), skipped=skipped+1; continue; end
    [~,index] = max([regions.Area]); box = regions(index).BoundingBox;
    % Reject foreground touching the captured frame boundary.
    if box(1)<=1 || box(2)<=1 || box(1)+box(3)>=size(frame,2) || box(2)+box(4)>=size(frame,1)
        skipped=skipped+1; continue;
    end
    crop = imcrop(frame,box);
    scale = randi([180 330])/max(size(crop,1),size(crop,2));
    crop = imresize(crop,scale); h=size(crop,1); w=size(crop,2);
    x=randi([12 640-w-12]); y=randi([12 480-h-12]);
    image=zeros(480,640,3,'uint8');
    image(y:y+h-1,x:x+w-1,:)=crop;
    finalBox=[x-0.5 y-0.5 w h];
    normalized=ar.yoloBox(finalBox,size(image));
    name=sprintf('render_%08d',frameIndex);
    imwrite(image,fullfile(outputDir,'images',[name '.png']));
    label=fopen(fullfile(outputDir,'labels',[name '.txt']),'w');
    assert(label>=0,'Cannot write label');
    fprintf(label,'0 %.8f %.8f %.8f %.8f\n',normalized); fclose(label);
    record=struct('image',[name '.png'],'source',source,'frame',frameIndex, ...
        'label_origin','synthetic_color_foreground','original_box_half_resolution',box, ...
        'training_box',finalBox,'seed',17,'frame_stride',3);
    fprintf(fid,'%s\n',jsonencode(record)); count=count+1;
    if mod(count-1,75)==0 && previewCount<12
        previewCount=previewCount+1;
        previews{previewCount}=imresize(insertShape(image,'rectangle',finalBox,Color='green',LineWidth=3),[240 320]);
    end
    if mod(count,150)==0, fprintf('Generated %d synthetic frames (source frame %d)\n',count,frameIndex); end
end
previews=previews(1:previewCount);
imwrite(imtile(previews,GridSize=[3 4]),fullfile(runDir,'synthetic_label_preview.png'));
summary=struct('source',source,'frames_read',frameIndex,'labels_generated',count, ...
    'frames_rejected',skipped,'seed',17,'stride',3,'dataset',outputDir, ...
    'manual_annotations_used',false,'historical_duck_models_used',false);
save(fullfile(runDir,'synthetic_summary.mat'),'summary');
fidSummary=fopen(fullfile(runDir,'synthetic_summary.json'),'w');
assert(fidSummary>=0,'Cannot write summary');
fprintf(fidSummary,'%s\n',jsonencode(summary,PrettyPrint=true)); fclose(fidSummary);
fprintf('Synthetic generation complete: %d labels, %d rejected\n',count,skipped);
end
