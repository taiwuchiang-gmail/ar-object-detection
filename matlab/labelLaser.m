function labels = labelLaser(image)
%LABELLASER Automatic labeling adapter for black-background renders.
labels = ar.syntheticLabels(image,'laser');
end
