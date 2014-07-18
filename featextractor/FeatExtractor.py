'''
'''
import numpy as np
import skimage.feature
import glob
import os
from distutils.dir_util import mkpath

import satdetect.ioutil as ioutil


def extractRawPixelsForDataset(datapath, color='gray', window_shape=(25,25)):
  ''' Extract raw gray-scale features
  '''
  if color not in ['gray', 'rgb']:
    raise ValueError('Unrecognized value: %s' % (color))
  bboxFiles = glob.glob(os.path.join(datapath, '*.npz'))
  for curpath in bboxFiles:
    B = np.load(curpath)
    IM = ioutil.loadImage(B['imgpath'], color=color)
    PosIm = extractWindowsFromImage(IM, B['PosBBox'])
    NegIm = extractWindowsFromImage(IM, B['NegBBox'])
    outpath = os.path.join(datapath, color)
    mkpath(outpath)
    outpath = os.path.join(outpath, str(B['basename'])+'.npz')
    np.savez_compressed(outpath, PosIm=PosIm, NegIm=NegIm, **B)

def extractWindowsFromImage(IM, BBox):
  N = BBox.shape[0]
  for n in xrange(N):
    y0,y1,x0,x1 = BBox[n,:]
    if n == 0:
      if IM.ndim == 3:
        WindowIm = np.zeros( (N, y1-y0, x1-x0, IM.shape[2]), dtype=IM.dtype)
      else:
        WindowIm = np.zeros( (N, y1-y0, x1-x0), dtype=IM.dtype)
    WindowIm[n,:] = IM[y0:y1, x0:x1]
  return WindowIm


########################################################### HOG
########################################################### 
hogParams = dict(
  orientations=9,
  pixels_per_cell=(5,5),
  cells_per_block=(1,1),
  )

def extractHOGForDataset(datapath, color='gray'):
  bboxFiles = glob.glob(os.path.join(datapath, '*.npz'))
  for curpath in bboxFiles:
    B = np.load(curpath)
    IM = ioutil.loadImage(B['imgpath'], color=color)
    PosIm = extractWindowsFromImage(IM, B['PosBBox'])
    NegIm = extractWindowsFromImage(IM, B['NegBBox'])
    PosFeat = extract_hog_for_imageset(PosIm)
    NegFeat = extract_hog_for_imageset(NegIm)
    outpath = os.path.join(datapath, 'hog')
    mkpath(outpath)
    outpath = os.path.join(outpath, str(B['basename'])+'.npz')
    np.savez_compressed(outpath, Pos=PosFeat, Neg=NegFeat, **B)



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

def extract_fvector_from_im(Im, featName):
  if featName == 'hog':
    fvec = skimage.feature.hog(Im, **hogParams)  
  return fvec


"""
lbpParamList = list()
for (nP, R) in [ (4,3),  (8, 6)]:
  lbpP = dict(
  n_points=nP,
  radius=R
  )
  lbpParamList.append(lbpP)


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

"""