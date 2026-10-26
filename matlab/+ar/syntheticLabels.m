function labels = syntheticLabels(image, objectName)
%SYNTHETICLABELS Generate labels from black-background renders.
cfg = ar.config(objectName); p = cfg.object;
box = ar.foregroundBox(image,p.threshold,p.dilation_radius,p.min_area);
labels = struct('Name',{},'Type',{},'Position',{});
if ~isempty(box)
    labels(1).Name = objectName;
    labels(1).Type = labelType('Rectangle');
    labels(1).Position = box;
end
end
