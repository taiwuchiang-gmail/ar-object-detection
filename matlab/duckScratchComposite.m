function dataset = duckScratchComposite(runId)
%DUCKSCRATCHCOMPOSITE Add unlabeled background footage to synthetic renders.
% All object boxes still come from the automatically isolated rendered duck.
if nargin<1, runId='scratch_20261026_01'; end
cfg=ar.config('duck');
sourceDir=fullfile(cfg.root,'datasets','duck',[runId '_synthetic']);
dataset=fullfile(cfg.root,'datasets','duck',[runId '_synthetic_composite']);
assert(isfolder(sourceDir) && ~isfolder(dataset),'Input missing or output exists');
mkdir(dataset); mkdir(fullfile(dataset,'images')); mkdir(fullfile(dataset,'labels'));
backgroundPath=fullfile(cfg.original_matlab_root,'background.mp4');
background=VideoReader(backgroundPath);
files=dir(fullfile(sourceDir,'images','*.png')); rng(23,'twister');
fid=fopen(fullfile(dataset,'provenance.jsonl'),'w'); assert(fid>=0,'Cannot write provenance');
cleanup=onCleanup(@() fclose(fid)); panels=cell(1,12); panelCount=0;
for k=1:numel(files)
    source=fullfile(files(k).folder,files(k).name); image=imread(source);
    if mod(k,5)~=0
        backgroundTime=rand*(background.Duration-0.1);
        background.CurrentTime=backgroundTime;
        canvas=imresize(readFrame(background),[480 640]);
        if rand<0.5, canvas=fliplr(canvas); end
        canvas=uint8(double(canvas)*(0.65+rand*0.5));
        mask=im2gray(image)>5;
        mask=imfill(bwareaopen(mask,50),'holes');
        duck=uint8(double(image)*(0.60+rand*0.55));
        for channel=1:3
            plane=canvas(:,:,channel); foreground=duck(:,:,channel);
            plane(mask)=foreground(mask); canvas(:,:,channel)=plane;
        end
        image=canvas;
    else
        backgroundTime=NaN;
    end
    imwrite(image,fullfile(dataset,'images',files(k).name));
    [~,stem]=fileparts(files(k).name);
    copyfile(fullfile(sourceDir,'labels',[stem '.txt']),fullfile(dataset,'labels',[stem '.txt']));
    record=struct('image',files(k).name,'render_image',source, ...
        'label_origin','synthetic_foreground_composite','background',backgroundPath, ...
        'background_time',backgroundTime,'seed',23,'manual_annotations_used',false);
    fprintf(fid,'%s\n',jsonencode(record));
    if mod(k-1,75)==0 && panelCount<12
        panelCount=panelCount+1; boxData=readmatrix(fullfile(dataset,'labels',[stem '.txt']),FileType='text');
        box=[(boxData(2)-boxData(4)/2)*640+0.5 (boxData(3)-boxData(5)/2)*480+0.5 boxData(4)*640 boxData(5)*480];
        panels{panelCount}=imresize(insertShape(image,'rectangle',box,Color='green',LineWidth=3),[240 320]);
    end
    if mod(k,200)==0, fprintf('Composite images: %d/%d\n',k,numel(files)); end
end
imwrite(imtile(panels(1:panelCount),GridSize=[3 4]), ...
    fullfile(cfg.root,'runs','duck',runId,'synthetic_composite_preview.png'));
fprintf('Composite dataset complete: %d images\n',numel(files));
end
