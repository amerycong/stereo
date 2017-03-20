close all
clear all

% read in 2 images
im_set = 1;
switch im_set
    case 1
        color1 = imread('cast-left.jpg'); %building
        color2 = imread('cast-right.jpg');
    case 2
        color1 = imread('Cones_im2.jpg'); %cones
        color2 = imread('Cones_im6.jpg');
    otherwise
        disp('pick different image set')
end

image1 = rgb2gray(color1);
image2 = rgb2gray(color2);

% apply Harris corner detection
corners1 = harris(image1,1,4,25000,0); 
corners2 = harris(image2,1,4,25000,0);
% corners1 = detectFASTFeatures(image1,'MinContrast',0.1);
% corners2 = detectFASTFeatures(image2,'MinContrast',0.1);
% corners1 = corners1.Location;
% corners2 = corners2.Location;

% show detected corners
figure(1)
imshow(color1)
hold on
plot(corners1(:,2),corners1(:,1),'r*')
hold off
figure(2)
imshow(color2)
hold on
plot(corners2(:,2),corners2(:,1),'r*')
hold off

% normalized cross correlation
correspondences = [];
for i = 1:size(corners1,1)
    
    y1 = corners1(i,1);
    x1 = corners1(i,2);
    %"radius" of patch, patch length = 2*patch_size
    patch_size = 10;
    
    %ignore corners with insufficient window area
    if x1 <= patch_size || y1 <= patch_size || x1 >= size(image1,2)-patch_size || y1 >= size(image1,1)-patch_size
        continue
    end
    
    %creates patch around each corner for both images
    patch1 = image1(y1-patch_size:y1+patch_size,x1-patch_size:x1+patch_size);
    NCC = zeros(1,size(corners2,1));
    for j = 1:size(corners2,1)
        y2 = corners2(j,1);
        x2 = corners2(j,2);

        if x2 <= patch_size || y2 <= patch_size || x2 >= size(image2,2)-patch_size || y2 >= size(image2,1)-patch_size
            continue
        end
        patch2 = image2(y2-patch_size:y2+patch_size,x2-patch_size:x2+patch_size);

        correlation = normxcorr2(patch1, patch2);
        
        %extract center value of NCC matrix for correlation value between
        %two patches
        NCC(j) = correlation(1+2*patch_size,1+2*patch_size);
        
        if NCC(j) == max(NCC)
            corr_index = j;
        end
    end
    
    %check threshold
    NCC_threshold = 0.90;
    if NCC(corr_index) > NCC_threshold
        %correspondences = [x1 y1 x2 y2]
        correspondences = [correspondences;x1 y1 corners2(corr_index,2) corners2(corr_index,1)];
    end
end

figure(3)
imshow([color1;color2])
hold all
plot(corners1(:,2), corners1(:,1), 'rx');
plot(corners2(:,2), corners2(:,1)+size(image1,1), 'bx');
for i = 1:size(correspondences,1)
    plot([correspondences(i,1), correspondences(i,3)], ...
        [correspondences(i,2), correspondences(i,4)+size(image1,1)]);
end
hold off

%estimate fundamental matrix w/ ransac
dist_thresh = 1e-9;
[F,inliersIndex] = estimateFundamentalMatrix(correspondences(:,1:2),correspondences(:,3:4),'Method','RANSAC','DistanceThreshold',dist_thresh,'NumTrials',20000);

%inliers
allPoints1 = correspondences(:,1:2);
allPoints2 = correspondences(:,3:4);
inliers = correspondences(inliersIndex,:);
inlierPoints1 = inliers(:,1:2);
inlierPoints2 = inliers(:,3:4);
figure(4)
imshow([color1;color2])
hold on
scatter(allPoints1(:,1), allPoints1(:,2), 'rx')
scatter(allPoints2(:,1), allPoints2(:,2)+size(image1,1), 'bx')
markersize = 36;
scatter(inlierPoints1(:,1), inlierPoints1(:,2),markersize, 'g', 'fill', 'LineWidth', 5);
scatter(inlierPoints2(:,1), inlierPoints2(:,2)+size(image1,1),markersize, 'g', 'fill','LineWidth', 5);
for i = 1:size(inlierPoints1,1)
    plot([inlierPoints1(i,1), inlierPoints2(i,1)], ...
        [inlierPoints1(i,2), inlierPoints2(i,2)+size(image1,1)],'g');
end
hold off

%dense disparity map

%horizontal disparity
disparityRange = [0 80];
dmap = disparity(image1,image2,'DisparityRange',disparityRange,'Method','SemiGlobal');
%rescale 0  - 255
dmap(dmap<0)=0;
min_val = min(min(dmap));
max_val = max(max(dmap));
dmap_scaled = (dmap-min_val)*255/(max_val-min_val);
figure(5)
imshow(dmap_scaled, [0 255])

%vertical disparity
disparityRange = [-64 64];
dmap = disparity(imrotate(image1,90),imrotate(image2,90),'DisparityRange',disparityRange,'Method','SemiGlobal');
%rescale 0  - 255
dmap(dmap<0)=0;
min_val = min(min(dmap));
max_val = max(max(dmap));
dmap_scaled = (dmap-min_val)*255/(max_val-min_val);
figure(6)
imshow(imrotate(dmap_scaled,-90), [0 255])

