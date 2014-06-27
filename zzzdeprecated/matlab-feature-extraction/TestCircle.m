addpath(genpath('~/code/voc/'));

N = 32;
sbin = 5;
R = 10;

[X,Y] = meshgrid(1:N, 1:N);
Xc = X(:) - N/2;
Yc = Y(:) - N/2;
cmask = sqrt(Xc.^2 + Yc.^2) < R;
smask = abs(Xc) < R & abs(Yc) < R;
Circle = zeros(N,N);
Circle(cmask) = 1;
Square = zeros(N,N);
Square(smask) = 1;
Blank = zeros(N, N);
Blank(1,1) = 1;

Cross = zeros(N, N);
Cross( :, N/2-3:N/2+3) = 1;
Cross( N/2-3:N/2+3, :) = 1;


return
%Im = ones(N, N);
%Im(1:N, 1:5) = .01;

%Im = Circle;
Im = [Circle Square; Square Circle];
Im = Im + 0.05 * randn(size(Im));
Im = Im - min(min(Im));
Im = Im / max(max(Im));
Im = color(Im);
F = features(Im, sbin);
HogIm = HOGpicture( F, sbin)

subplot(1,2,1);
imshow(Im);
subplot(1,2,2);
imshow(HogIm)



