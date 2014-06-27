function [PosSamples, NegSamples, keepIDs] = sampleNegativeExamples(Im, PosBbox, nSamples, method, avoidIDs)

B = PosBbox(1,4) - PosBbox(1,3) + 1;
p = PosBbox(1,:);
grayPosIm = rgb2gray(Im( p(1):p(2),  p(3):p(4), :));
posVec = reshape( grayPosIm, B*B, 1);

GrayIm = rgb2gray(Im);
nRow = B*floor(size(GrayIm,1) / B);
nCol = B*floor(size(GrayIm,2) / B);
GrayCandidates = im2col(GrayIm(1:nRow, 1:nCol), [B B], 'distinct');

PosSamples = cell(size(PosBbox,1),1);
for rowID = 1:size(PosBbox,1)
   p = PosBbox(rowID,:);
   PosSamples{rowID} = Im( p(1):p(2),  p(3):p(4), :);
end
% Now mask out the positive examples!
for rowID = 1:size(PosBbox,1)
   p = PosBbox(rowID,:);
   Im( p(1):p(2),  p(3):p(4), :) = -1;   
end

% Candidates: B^2 * nCandidates
RCandidates = im2col(Im(1:nRow,1:nCol,1), [B B], 'distinct');
GCandidates = im2col(Im(1:nRow,1:nCol,2), [B B], 'distinct');
BCandidates = im2col(Im(1:nRow,1:nCol,3), [B B], 'distinct');
% 
badIDs = any( RCandidates < 0, 1);
RCandidates = RCandidates( :, ~badIDs);
GCandidates = GCandidates( :, ~badIDs);
BCandidates = BCandidates( :, ~badIDs);
GrayCandidates = GrayCandidates(:, ~badIDs);

if strcmp(method, 'knn')
    [keepIDs, dist] = knnsearch(GrayCandidates', posVec', 'K', nSamples, 'dist', 'seuclidean');
else
    nCandidates = size(RCandidates,2);
    permIDs = randperm(nCandidates);
    for aa = 1:length(avoidIDs)
       permIDs = permIDs(permIDs ~= avoidIDs(aa));
    end
    keepIDs = permIDs(1:nSamples);

    
end

NegSamples = cell(nSamples, 1);
for ss = 1:nSamples
   sIm = zeros(B, B, 3);
   sIm(:,:,1) = reshape( RCandidates(:,keepIDs(ss)), B, B);
   sIm(:,:,2) = reshape( GCandidates(:,keepIDs(ss)), B, B);
   sIm(:,:,3) = reshape( BCandidates(:,keepIDs(ss)), B, B);

   if sum(sIm(:)==0) > 15
       assert( ~all( sIm(1,:,1) == 0));
       assert( ~all( sIm(:,1,1) == 0));
       assert( ~all( sIm(end,:,1) == 0));
   end
   NegSamples{ss} = sIm;
end
