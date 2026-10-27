function duckScratchBootstrapComposite(runId)
%DUCKSCRATCHBOOTSTRAPCOMPOSITE Train a new detector using automatic composites.
if nargin<1, runId='scratch_20261026_01'; end
cfg=ar.config('duck'); runDir=fullfile(cfg.root,'runs','duck',runId);
dataset=fullfile(cfg.root,'datasets','duck',[runId '_synthetic_composite']);
assert(isfolder(dataset),'Composite dataset missing');
rng(17,'twister');
diary(fullfile(runDir,'yolov2_composite_training.log')); cleanup=onCleanup(@() diary('off'));
statusFile=fullfile(runDir,'composite_bootstrap_status.json'); started=tic;
writeStatus(statusFile,'running','',0);
try
    modelPath=ar.trainBootstrap('duck',dataset,which('yolov2VehicleDetector.mat'),[runId '_composite_yolov2']);
    fid=fopen(fullfile(runDir,'composite_bootstrap_model_path.txt'),'w');
    fprintf(fid,'%s\n',modelPath); fclose(fid);
    writeStatus(statusFile,'complete',modelPath,toc(started));
    fprintf('COMPOSITE_BOOTSTRAP_COMPLETE\n');
catch ME
    fprintf('%s\n',getReport(ME,'extended'));
    writeStatus(statusFile,'failed',getReport(ME,'extended','hyperlinks','off'),toc(started));
end
end
function writeStatus(file,status,detail,elapsed)
fid=fopen(file,'w'); fprintf(fid,'%s\n',jsonencode(struct('status',status,'detail',detail,'elapsed_seconds',elapsed))); fclose(fid);
end
