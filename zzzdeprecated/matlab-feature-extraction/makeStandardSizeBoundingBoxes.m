function [Bboxs] = makeStandardSizeBoundingBoxes(Bboxs, B) 
% Make every row in Bboxes matrix be a box of size BxB

for rowID = 1:size(Bboxs,1)
   p = Bboxs(rowID,:);
   pH = p(2) - p(1);
   pW = p(4) - p(3);
   BH = (B - pH)/2;
   BW = (B - pW)/2;
   assert( pH < B);
   
   pWmin = p(3) - floor(BW);
   pWmax = p(4) + floor(BW);
   pHmin = p(1) - floor(BH);
   pHmax = p(2) + floor(BH);
   if floor(BH) == BH
      pHmax = pHmax - 1;      
   end
   if floor(BW) == BW
      pWmax = pWmax - 1;
   end
   assert( pHmax - pHmin + 1 == B);
   assert( pWmax - pWmin + 1 == B);
   
   Bboxs(rowID, :) = [pHmin pHmax pWmin pWmax];

end