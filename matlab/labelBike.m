function labels = labelBike(image)
%LABELBIKE Automatic labeling adapter for black-background renders.
labels = ar.syntheticLabels(image,'bike');
end
