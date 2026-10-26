function normalized = yoloBox(box, imageSize)
%YOLOBOX Convert MATLAB pixel rectangles to normalized YOLO rectangles.
% imageSize is [height width ...]. Shift pixel boundary 0.5 to zero origin.
assert(size(box,2)==4 && all(isfinite(box(:))),'Invalid boxes');
assert(all(box(:,3:4)>0,'all'),'Box sizes must be positive');
h = imageSize(1); w = imageSize(2);
left = max(0,min(w,box(:,1)-0.5));
top = max(0,min(h,box(:,2)-0.5));
right = max(0,min(w,box(:,1)-0.5+box(:,3)));
bottom = max(0,min(h,box(:,2)-0.5+box(:,4)));
assert(all(right>left & bottom>top),'Box outside image');
normalized = [(left+right)/(2*w),(top+bottom)/(2*h), ...
    (right-left)/w,(bottom-top)/h];
end
