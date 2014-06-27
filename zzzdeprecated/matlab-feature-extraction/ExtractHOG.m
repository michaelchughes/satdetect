function [PosFeat] = ExtractHOG(featName)

sbin = 5;
for imID = [1 2 3 4]
   Q = load(sprintf('/data/burners/set1/%s/%d.mat', featName, imID));
   
   Im = squeeze(Q.Pos(1,:,:,:));
   F = features(Im, sbin); 
   Fdim = length(F(:));   

   PosFeat = zeros(size(Q.Pos,1), Fdim);  
   for pp = 1:size(Q.Pos,1)
      Im = squeeze(Q.Pos(pp,:,:,:));
      F = features(Im, sbin);
      PosFeat(pp,:) = F(:)';      
   end
   
   NegFeat = zeros(size(Q.Neg,1), Fdim);  
   for pp = 1:size(Q.Neg,1)
      Im = squeeze(Q.Neg(pp,:,:,:));
      F = features(Im, sbin);
      NegFeat(pp,:) = F(:)';      
   end
   Pos = PosFeat;
   Neg = NegFeat;
   
   [H, W, nC] = size(Im);
   fpath = sprintf('/data/burners/set1/voc-%dx%d/', H, W);
   [~,~] = mkdir(fpath);
   fpath = sprintf('%s/%d.mat', fpath, imID)
   save(fpath, 'Pos', 'Neg'); 
   
end