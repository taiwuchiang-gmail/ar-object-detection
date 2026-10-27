function duckScratchAssessBootstrap(runId)
%DUCKSCRATCHASSESSBOOTSTRAP Record new-model predictions on fixed real frames.
if nargin<1, runId='scratch_20261026_01'; end
cfg=ar.config('duck'); runDir=fullfile(cfg.root,'runs','duck',runId);
modelPath=strtrim(fileread(fullfile(runDir,'composite_bootstrap_model_path.txt')));
model=load(modelPath);
files={'duck1.mp4','duck-test.mp4','capture_movie_2024_04_28_18_41_48.mp4'};
fractions=[0.2 0.5 0.8]; panels=cell(1,9); results=struct([]); count=0;
for j=1:numel(files)
    reader=VideoReader(fullfile(cfg.original_matlab_root,files{j}));
    for k=1:numel(fractions)
        reader.CurrentTime=reader.Duration*fractions(k); image=readFrame(reader);
        [boxes,scores]=detect(model.detector,image,Threshold=0.01,ExecutionEnvironment='cpu');
        score=0; box=zeros(0,4);
        if ~isempty(scores)
            [score,index]=max(scores); box=boxes(index,:);
            image=insertShape(image,'rectangle',box,Color='green',LineWidth=4);
        end
        count=count+1; results(count).source=files{j}; results(count).fraction=fractions(k);
        results(count).best_score=score; results(count).box=box;
        panels{count}=imresize(insertText(image,[8 8],sprintf('%s %.3f',files{j},score), ...
            FontSize=18),[300 400]);
        fprintf('%s at %.1f: %.3f\n',files{j},fractions(k),score);
    end
end
imwrite(imtile(panels,GridSize=[3 3]),fullfile(runDir,'composite_bootstrap_real_preview.png'));
fid=fopen(fullfile(runDir,'composite_bootstrap_samples.json'),'w');
fprintf(fid,'%s\n',jsonencode(results,PrettyPrint=true)); fclose(fid);
end
