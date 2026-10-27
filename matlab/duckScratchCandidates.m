function duckScratchCandidates(runId, variant)
%DUCKSCRATCHCANDIDATES Generate recording-separated labels from the NEW model.
if nargin < 1, runId='scratch_20261026_01'; end
if nargin < 2, variant='composite'; end
prefix=''; suffix='bootstrap';
if strcmp(variant,'composite'), prefix='composite_'; suffix='composite'; end
cfg=ar.config('duck'); runDir=fullfile(cfg.root,'runs','duck',runId);
modelPath=strtrim(fileread(fullfile(runDir,[prefix 'bootstrap_model_path.txt'])));
status=jsondecode(fileread(fullfile(runDir,[prefix 'bootstrap_status.json'])));
assert(strcmp(status.status,'complete'),'Bootstrap has not completed');
diary(fullfile(runDir,'yolov2_candidates.log')); cleanup=onCleanup(@() diary('off'));
trainDir=ar.exportCandidates('duck',fullfile(cfg.original_matlab_root,'duck1.mp4'), ...
    [runId '_' suffix '_train'],modelPath,0.70,6);
valDir=ar.exportCandidates('duck',fullfile(cfg.original_matlab_root,'duck-test.mp4'), ...
    [runId '_' suffix '_val'],modelPath,0.70,6);
manifest=struct('directory',{trainDir,valDir},'split',{'train','val'});
if strcmp(variant,'composite')
    manifest(3).directory=fullfile(cfg.root,'datasets','duck',[runId '_synthetic_composite']);
    manifest(3).split='train';
end
fid=fopen(fullfile(runDir,[suffix '_split.json']),'w');
assert(fid>=0,'Cannot write split'); fprintf(fid,'%s\n',jsonencode(manifest,PrettyPrint=true)); fclose(fid);
% Release MATLAB's CUDA context before the Python training stages.
gpuDevice([]);
fprintf('CANDIDATES_COMPLETE\n');
end
