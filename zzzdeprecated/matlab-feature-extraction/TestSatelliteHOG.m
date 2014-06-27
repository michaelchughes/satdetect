addpath(genpath('~/code/voc/'));
sbin = 5;
P = load('/data/burners/set1/scenes/before3_huts.pxbbox');
Im = imread('/data/burners/set1/scenes/before3_image.jpg');

B = 15;
hutIDs = [1 2 3];
hh = 0;
for hutID = hutIDs
    hh = hh +1;
    p = P(hutID,:);
    hutIm = Im( p(1)-B:p(2)+B, p(3)-B:p(4)+B, :);
    hutIm = double(hutIm)/255;
    F = features( hutIm, sbin);
    HogIm = HOGpicture( F, sbin);
    
    subplot(length(hutIDs),2, 2*hh-1);
    imshow(hutIm);
    subplot(length(hutIDs),2, 2*hh); %length(hutIDs) + hutID);
    imshow(HogIm);
end