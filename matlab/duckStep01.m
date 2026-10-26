function duckStep01
%DUCKSTEP01 Verify assets and preview historical recordings without training.
cfg = ar.config('duck');
out = fullfile(cfg.root,'runs','duck','step01_preflight');
assert(~isfolder(out),'Step output already exists: %s',out);
mkdir(out);
files = {'duck.mp4','duck_cropped.mp4','duck-test.mp4','duck1.mp4', ...
    'capture_movie_2024_04_28_18_25_17.mp4','capture_movie_2024_05_15_13_54_09.mp4'};
previews = cell(1,numel(files));
metadata = struct([]);
for k = 1:numel(files)
    source = fullfile(cfg.original_matlab_root,files{k});
    video = VideoReader(source);
    video.CurrentTime = video.Duration/2;
    frame = readFrame(video);
    previews{k} = insertText(imresize(frame,[300 480]),[8 8],files{k}, ...
        FontSize=16,BoxColor='black',TextColor='white');
    metadata(k).file = source;
    metadata(k).width = video.Width;
    metadata(k).height = video.Height;
    metadata(k).fps = video.FrameRate;
    metadata(k).duration = video.Duration;
end
sheet = imtile(previews,GridSize=[2 3]);
imwrite(sheet,fullfile(out,'recording_overview.png'));
imageshow(sheet);
inputPath = fullfile(cfg.root,cfg.object.ground_truth);
savedLabels = load(inputPath,'gTruth');
oldFile = savedLabels.gTruth.DataSource.Source;
newFile = fullfile(cfg.original_matlab_root,'duck_cropped.mp4');
outputPath = fullfile(out,'duck_ground_truth_relocated.mat');
ar.relocateGroundTruth(inputPath,oldFile,newFile,outputPath);
data = load(outputPath,'gTruth');
fprintf('Relocated ground truth source:\n'); disp(data.gTruth.DataSource.Source);
fprintf('Label definitions:\n'); disp(data.gTruth.LabelDefinitions);
fprintf('Label rows: %d\n',height(data.gTruth.LabelData));
fid = fopen(fullfile(out,'assets.json'),'w');
assert(fid>=0,'Cannot write asset metadata');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'%s',jsonencode(metadata,PrettyPrint=true));
save(fullfile(out,'session.mat'),'metadata','outputPath');
fprintf('Step 01 outputs: %s\n',out);
end
