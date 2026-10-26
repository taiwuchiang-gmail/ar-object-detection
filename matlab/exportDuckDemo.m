function exportDuckDemo
%EXPORTDUCKDEMO Package the historical annotated duck video for GitHub.
% Re-encodes existing predictions; does not run inference or alter detections.
cfg = ar.config('duck');
source = fullfile(cfg.original_python_root,'runs','detect','predict', ...
    'capture_movie_2024_05_15_13_54_09.avi');
assert(isfile(source),'Historical annotated duck video missing');
out = fullfile(cfg.root,'docs','media');
if ~isfolder(out), mkdir(out); end
mp4Path = fullfile(out,'yellow-duck-detection.mp4');
gifPath = fullfile(out,'yellow-duck-detection.gif');
assert(~isfile(mp4Path) && ~isfile(gifPath),'Demo already exists');
reader = VideoReader(source);
writer = VideoWriter(mp4Path,'MPEG-4');
writer.FrameRate = reader.FrameRate;
writer.Quality = 80;
open(writer); cleanup = onCleanup(@() close(writer));
count = 0;
while hasFrame(reader)
    frame = readFrame(reader);
    writeVideo(writer,imresize(frame,[540 720]));
    count = count+1;
    if mod(count,300)==0, fprintf('Encoded %d frames\n',count); end
end
clear cleanup;
reader.CurrentTime = reader.Duration/2;
imwrite(imresize(readFrame(reader),[540 720]),fullfile(out,'yellow-duck-detection.png'));
% A continuous 8-second excerpt at 8 fps; bounding boxes remain unmodified.
gifTimes = 20:1/8:(28-1/8);
for k=1:numel(gifTimes)
    reader.CurrentTime = gifTimes(k);
    frame = imresize(readFrame(reader),[300 400]);
    [indexed,map] = rgb2ind(frame,128);
    if k==1
        imwrite(indexed,map,gifPath,'gif',LoopCount=Inf,DelayTime=1/8);
    else
        imwrite(indexed,map,gifPath,'gif',WriteMode='append',DelayTime=1/8);
    end
end
info = struct('source_kind','historical_annotated_python_detection_output', ...
    'source_filename','capture_movie_2024_05_15_13_54_09.avi', ...
    'source_frame_size',[reader.Height reader.Width], ...
    'source_fps',reader.FrameRate,'source_duration_seconds',reader.Duration, ...
    'output_frame_size',[540 720],'output_frames',count, ...
    'gif_interval_seconds',[20 28],'gif_fps',8, ...
    'notes','Existing boxes and scores retained; resize/encode only. No new training or inference.');
fid=fopen(fullfile(out,'yellow-duck-detection.json'),'w');
assert(fid>=0,'Cannot write demo metadata'); fileCleanup=onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(info,PrettyPrint=true));
fprintf('Demo MP4: %s\n',mp4Path);
fprintf('Frames: %d, duration: %.3f seconds\n',count,reader.Duration);
end
