function outputPath = relocateGroundTruth(inputPath, oldRoot, newRoot, outputPath)
%RELOCATEGROUNDTRUTH Rewrite embedded source paths in a NEW groundTruth MAT file.
% This operation never overwrites the input or an existing output file.
assert(isfile(inputPath) && (isfolder(newRoot) || isfile(newRoot)),'Input or new source root missing');
assert(~isfile(outputPath),'Output already exists');
data = load(inputPath,'gTruth');
assert(isfield(data,'gTruth'),'Expected gTruth variable');
missing = changeFilePaths(data.gTruth,[string(oldRoot),string(newRoot)]);
assert(isempty(missing),'Some ground truth source paths could not be resolved');
gTruth = data.gTruth;
save(outputPath,'gTruth');
end
