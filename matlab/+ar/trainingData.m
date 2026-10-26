function data = trainingData(datasetDir, objectName)
%TRAININGDATA Read images/labels candidate folders into a MATLAB datastore.
images = imageDatastore(fullfile(datasetDir,'images'));
positions = cell(numel(images.Files),1);
for k = 1:numel(images.Files)
    image = readimage(images,k); [h,w,~] = size(image);
    [~,name] = fileparts(images.Files{k});
    labelPath = fullfile(datasetDir,'labels',[name '.txt']);
    assert(isfile(labelPath),'Missing label: %s',labelPath);
    values = readmatrix(labelPath,FileType='text');
    if isempty(values)
        positions{k} = zeros(0,4);
        continue;
    end
    assert(size(values,2)==5 && all(values(:,1)==0),'Expected class 0 labels');
    normalized = values(:,2:5);
    assert(all(isfinite(normalized(:))) && all(normalized(:)>=0 & normalized(:)<=1), ...
        'Invalid normalized box');
    assert(all(normalized(:,3:4)>0,'all'),'Invalid box size');
    positions{k} = [(normalized(:,1)-normalized(:,3)/2)*w+0.5, ...
        (normalized(:,2)-normalized(:,4)/2)*h+0.5,normalized(:,3)*w,normalized(:,4)*h];
end
labels = table(positions,VariableNames={objectName});
data = combine(images,boxLabelDatastore(labels));
end
