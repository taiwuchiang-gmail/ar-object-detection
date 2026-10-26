function box = foregroundBox(image, threshold, radius, minArea)
%FOREGROUNDBOX Find the largest region in a black-background render.
mask = im2gray(image) > threshold;
if minArea > 0
    mask = imfill(bwareaopen(mask,minArea),'holes');
end
mask = imdilate(mask,strel('disk',radius));
regions = regionprops(mask,'BoundingBox','Area');
box = zeros(0,4);
if ~isempty(regions)
    [~,index] = max([regions.Area]);
    box = regions(index).BoundingBox;
end
end
