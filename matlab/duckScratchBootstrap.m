function duckScratchBootstrap(runId)
%DUCKSCRATCHBOOTSTRAP Reproduce the original 40-epoch small-detector training.
if nargin < 1, runId='scratch_20261026_01'; end
cfg=ar.config('duck');
runDir=fullfile(cfg.root,'runs','duck',runId);
dataset=fullfile(cfg.root,'datasets','duck',[runId '_synthetic']);
assert(isfolder(dataset),'Generate fresh synthetic data first');
assert(~isfile(fullfile(runDir,'bootstrap_model_path.txt')),'Bootstrap already trained');
rng(17,'twister');
diary(fullfile(runDir,'yolov2_training.log'));
cleanup=onCleanup(@() diary('off'));
started=tic;
statusFile=fullfile(runDir,'bootstrap_status.json');
writeStatus(statusFile,'running','',0);
try
modelPath=ar.trainBootstrap('duck',dataset,which('yolov2VehicleDetector.mat'),[runId '_yolov2']);
fid=fopen(fullfile(runDir,'bootstrap_model_path.txt'),'w');
assert(fid>=0,'Cannot write model path'); fprintf(fid,'%s\n',modelPath); fclose(fid);
fprintf('BOOTSTRAP_COMPLETE: %s\n',modelPath);
writeStatus(statusFile,'complete',modelPath,toc(started));
catch ME
    fprintf('%s\n',getReport(ME,'extended'));
    writeStatus(statusFile,'failed',getReport(ME,'extended','hyperlinks','off'),toc(started));
end
end
function writeStatus(file,status,detail,elapsed)
fid=fopen(file,'w');
fprintf(fid,'%s\n',jsonencode(struct('status',status,'detail',detail,'elapsed_seconds',elapsed)));
fclose(fid);
end
