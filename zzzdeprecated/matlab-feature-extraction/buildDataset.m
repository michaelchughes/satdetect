addpath(genpath('~/code/voc/'));
addpath(genpath('~/code/selfsim/'));
rng(0);

B = 32;
sbins = [3 5 7];
Params = struct();
Params.patch_size = 3; % must be odd
Params.desc_rad = B/2 - 2;
Params.nrad = 3;
Params.nang = 12;
Params.var_noise = 30000;
Params.saliency_thresh = 1;
Params.homogeneity_thresh = 1;
Params.snn_thresh = 1;
CenterLoc = [B/2 B/2];

rootdir = '/data/burners/set1/scenes/';
bboxfilepattern = '/data/burners/set1/scenes/before*_huts.pxbbox';
imagefilepattern = '/data/burners/set1/scenes/before*.jpg';
featureoutputdir = ['/data/burners/set1/featuresB' num2str(B)];
[~, ~] = mkdir(featureoutputdir);

ImList = dir(imagefilepattern);
BboxList = dir(bboxfilepattern);

for aa = 1:length(ImList)
   Im = imread(fullfile(rootdir,ImList(aa).name));
   Im = double(Im)/255;
   PosBbox = load(fullfile(rootdir, BboxList(aa).name));
   PosBbox = makeStandardSizeBoundingBoxes(PosBbox, B);
   
   [Pos, Neg, keepIDs] = extractPosAndNegExamples(Im, PosBbox, 100, 'knn');
   [~, Neg2] = extractPosAndNegExamples(Im, PosBbox, 100, 'random', keepIDs);
   Neg(end+1:end+length(Neg2)) = Neg2;
   
   
   %% RGB Image
   PosIm = zeros(length(Pos), numel(Pos{1}));
   for pp = 1:length(Pos)
      PosIm(pp,:) = Pos{pp}(:); 
   end   
   NegIm = zeros(length(Neg), numel(Neg{1}));
   for pp = 1:length(Neg)
      NegIm(pp,:) = Neg{pp}(:); 
   end
   rawoutdir = fullfile(featureoutputdir, 'rgbimg');
   [~, ~] = mkdir(rawoutdir);
   rawoutpath = fullfile(rawoutdir, ['group' num2str(aa) '.mat']);
   save(rawoutpath, 'PosIm', 'NegIm');
   
   PosIm = zeros(length(Pos), numel(Pos{1})/3);
   for pp = 1:length(Pos)
      gIm = rgb2gray(Pos{pp});
      PosIm(pp,:) = gIm(:); 
   end   
   NegIm = zeros(length(Neg), numel(Neg{1})/3);
   for pp = 1:length(Neg)
      gIm = rgb2gray(Neg{pp});
      NegIm(pp,:) = gIm(:); 
   end
   rawoutdir = fullfile(featureoutputdir, 'grayimg');
   [~, ~] = mkdir(rawoutdir);
   rawoutpath = fullfile(rawoutdir, ['group' num2str(aa) '.mat']);
   save(rawoutpath, 'PosIm', 'NegIm');
   
   %% HOG
   for sbin = sbins
   F = features(Neg{1}, sbin); 
   Fdim = length(F(:));
   PosFeat = zeros(length(Pos), Fdim);
   for pp = 1:length(Pos)
      F = features(Pos{pp}, sbin);
      PosFeat(pp,:) = F(:)';
   end      
   
   NegFeat = zeros(length(Neg), Fdim);
   for pp = 1:length(Neg)
      F = features(Neg{pp}, sbin);
      NegFeat(pp,:) = F(:)';
   end
   
   hogoutdir = fullfile(featureoutputdir, ['hog' num2str(sbin)]);
   [~, ~] = mkdir(hogoutdir);
   hogoutpath = fullfile(hogoutdir, ['group' num2str(aa) '.mat']);
   save(hogoutpath, 'PosFeat', 'NegFeat');
   end
   
   %% Self-similarity   
   Fdim = Params.nang * Params.nrad;
   PosFeat = zeros(length(Pos), Fdim);
   for pp = 1:length(Pos)
      [F,dC, sC, hC, nC] = mexCalcSsdescs(Pos{pp}, Params, CenterLoc);
      PosFeat(pp,:) = F(:)';
   end
      
   NegFeat = zeros(length(Neg), Fdim);
   for pp = 1:length(Neg)
      [F,dC, sC, hC, nC] = mexCalcSsdescs(Neg{pp}, Params, CenterLoc);
      if any(isnan(F(:)))
          disp('ohno');
      end
      NegFeat(pp,:) = F(:)';      
   end
   selfsimoutdir = fullfile(featureoutputdir, ['selfsim' num2str(Params.nang) 'x' num2str(Params.nrad)]);
   [~, ~] = mkdir(selfsimoutdir);
   selfsimoutpath = fullfile(selfsimoutdir, ['group' num2str(aa) '.mat']);
   save(selfsimoutpath, 'PosFeat', 'NegFeat');
   
end