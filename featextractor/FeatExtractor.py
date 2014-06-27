'''
'''
import numpy as np
import skimage.feature

import satdetect.ioutil as ioutil

hogParams = dict(
  orientations=9,
  pixels_per_cell=(5,5),
  cells_per_block=(1,1),
  )

lbpParamList = list()
for (nP, R) in [ (4,3),  (8, 6)]:
  lbpP = dict(
  n_points=nP,
  radius=R
  )
  lbpParamList.append(lbpP)

def extract_fvector_from_im(Im, featName):
  if featName == 'hog':
    fvec = skimage.feature.hog(Im, **hogParams)  
  return fvec

def extract_lbp_features_for_dataset(setName, window_shape=(25,25)):
  nImages = ioutil.get_num_images(setName)
  for imageID in range(1, nImages+1):
    Pos, Neg = ioutil.load_labeled_images(setName, imageID, window_shape, 'gray')
    PosFeat = _extract_lbp_for_imageset(Pos)
    NegFeat = _extract_lbp_for_imageset(Neg)
    ioutil.save_labeled_feats(setName, imageID, window_shape, 'lbp', PosFeat,
                                                                     NegFeat)
  return PosFeat

def _extract_lbp_for_imageset(Pos, B=5):  
  for pp in xrange(Pos.shape[0]):
    Im = Pos[pp]
    for ll, lbpParams in enumerate(lbpParamList):
      LBPIm = skimage.feature.local_binary_pattern(Im, lbpParams['n_points'],
                                                       lbpParams['radius'])
      LBPIm = LBPIm[B:-B, B:-B]
      n_bins = 2**lbpParams['n_points']
      LBPhist, bins = np.histogram(LBPIm.flatten(), bins=n_bins, 
                                                    range=(0, n_bins))
      LBPhist = LBPhist / np.sqrt(np.sum(np.square(LBPhist)))
      if ll == 0:
        fvec = LBPhist
      else:
        fvec = np.hstack([fvec, LBPhist])
    if pp == 0:
      Fmat = np.zeros( (Pos.shape[0], fvec.size))
    Fmat[pp] = fvec
  return Fmat

def extract_hog_features_for_dataset(setName, window_shape=(25,25)):
  nImages = ioutil.get_num_images(setName)
  for imageID in range(1, nImages+1):
    Pos, Neg = ioutil.load_labeled_images(setName, imageID, window_shape, 'gray')
    PosFeat = extract_hog_for_imageset(Pos)
    NegFeat = extract_hog_for_imageset(Neg)
    ioutil.save_labeled_feats(setName, imageID, window_shape, 'hog', PosFeat,
                                                                    NegFeat)
  return PosFeat

def extract_hog_for_imageset(Pos):
  for pp in xrange(Pos.shape[0]):
    fvec = skimage.feature.hog(Pos[pp], **hogParams)      
    if pp == 0:
      PosFeat = np.zeros((Pos.shape[0],fvec.size))
    PosFeat[pp,:] = fvec
  return PosFeat
