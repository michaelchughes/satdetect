'''
'''
import numpy as np
import skimage.color
import sklearn.neighbors
import time

import BoundBox
import satdetect.ioutil as ioutil

OFFLIMITSVAL = -255

def make_and_save_window_dataset(setName, window_shape=(25,25),
                          nPosCopy=3, xmag=3, ymag=3, nNegSamples=400):
  '''
  '''
  nImages = ioutil.get_num_images(setName)
  for imageID in range(1, nImages+1):
    PosIm = load_pos_image_windows(setName, imageID, window_shape,
                                 nCopy=0, xmag=xmag, ymag=ymag)
    PosJitterIm = load_pos_image_windows(setName, imageID, window_shape,
                                 nCopy=nPosCopy, xmag=xmag, ymag=ymag)

    NegIm, NegBbox = load_neg_image_windows(setName, imageID, window_shape,
                                      nNegSamples)
    ioutil.save_labeled_images(setName, imageID, window_shape, PosIm, NegIm, PosJitterIm, NegBbox)

    GrayPos = convert_dataset_to_gray(PosIm)
    GrayPosJitter = convert_dataset_to_gray(PosJitterIm)
    GrayNeg = convert_dataset_to_gray(NegIm)  
    ioutil.save_labeled_images(setName, imageID, window_shape, GrayPos, GrayNeg, GrayPosJitter, NegBbox)

def convert_dataset_to_gray(PosIm):
  N, H, W, C = PosIm.shape
  GrayPos = np.zeros((N, H, W))
  for pp in xrange(N):
    GrayPos[pp] = skimage.color.rgb2gray(PosIm[pp])
  return GrayPos 

def load_pos_image_windows(setName, imageID, window_shape=(25,25),
                                             nCopy=3, xmag=3, ymag=3):
  '''
  '''
  H, W = window_shape
  BBox = BoundBox.load_pos_bbox_standard_size(setName, imageID,
                                              window_shape=window_shape)
  if nCopy > 1:
    BBox = BoundBox.get_jittered_copies_of_pos_bbox(BBox, nCopy=nCopy,
                                                  xmag=xmag, ymag=ymag)
  nPos = BBox.shape[0]
  Im = ioutil.load_jpg(setName, imageID)
  PosIm = np.zeros((nPos, H, W, 3))
  for pID in xrange(nPos):    
    PosIm[pID, :, :] = Im[BBox[pID,0]:BBox[pID,1],
                            BBox[pID,2]:BBox[pID,3]]
  return PosIm

def load_neg_image_windows(setName, imageID, window_shape=(25,25), 
                           nNegSamples=500):
  BBox = BoundBox.load_pos_bbox_standard_size(setName, imageID,
                                              window_shape=window_shape)
  OBox = BoundBox.get_bound_box_with_significant_overlap(BBox, window_shape)

  nPos = BBox.shape[0]

  Im = ioutil.load_jpg(setName, imageID)
  NegImMat = np.zeros((nNegSamples, window_shape[0], window_shape[1], 3))
  flatIDs = list()
  curLoc = 0
  tstart = time.time()
  for pID in range(nPos):
    if pID == nPos - 1: # last one!
      nSamps = nNegSamples - curLoc
    else: 
      nSamps = nNegSamples / nPos
    RefIm = Im[BBox[pID,0]:BBox[pID,1], BBox[pID,2]:BBox[pID,3]]
    curN, fIDs = get_nearest_windows_from_masked_image(Im, BBox, RefIm,
                                              window_shape=window_shape,
                                              n_samples=nSamps,
                                              nZones=4, seed=pID)
    telapsed = time.time() - tstart
    if pID % 5 == 0:
      print ' %5.1f sec | %4d/%4d' % (telapsed, pID+1, nPos)
    NegImMat[curLoc:curLoc+nSamps] = curN
    flatIDs.extend(fIDs)
    curLoc += nSamps
  assert curLoc == NegImMat.shape[0]

  # Verify no duplicates
  assert np.min(flatIDs) >= 0
  uniqueIDs = np.unique(flatIDs)
  if len(flatIDs) > len(uniqueIDs):
    nExtra = 3 * (nNegSamples - uniqueIDs.size)
    ExtraMat, fIDs = sample_windows_from_masked_image(Im, OBox,
                                     window_shape=window_shape,
                                     n_samples=nExtra)
    dupIDs = list()
    for uniqueID in uniqueIDs:
      locs = np.flatnonzero(flatIDs == uniqueID)  
      if len(locs) > 1:
        dupIDs.extend(locs[1:])
    posID = 0

    y2, x2 = np.unravel_index(fIDs, (Im.shape[0], Im.shape[1]))
    for fpos, fval in enumerate(fIDs):
      ymatch = np.logical_and(OBox[:,0] <= y2[fpos], y2[fpos] < OBox[:,1])
      xmatch = np.logical_and(OBox[:,2] <= x2[fpos], x2[fpos] < OBox[:,3])
      if np.sum( xmatch * ymatch) > 0:
        continue
      if posID >= len(dupIDs):
        break
      if fval not in uniqueIDs:
        NegImMat[dupIDs[posID]] = ExtraMat[fpos]
        flatIDs[dupIDs[posID]] = fval
        posID += 1
  uniqueIDs = np.unique(flatIDs)
  assert len(uniqueIDs) == len(flatIDs)

  # Verify no overlap with positive examples
  ys, xs = np.unravel_index(flatIDs, (Im.shape[0], Im.shape[1]))
  NegBox = np.zeros((len(xs),4))
  for n in range(len(xs)):
    ymatch = np.logical_and(OBox[:,0] <= ys[n], ys[n] < OBox[:,1])
    xmatch = np.logical_and(OBox[:,2] <= xs[n], xs[n] < OBox[:,3])
    posMatches = np.flatnonzero(np.logical_and(ymatch, xmatch))
    if len(posMatches) > 0:
      print '************** WARNING: sampled neg bbox is actually positive!'
      print n, xs[n], ys[n], posMatches
      from IPython import embed; embed()
    NegBox[n,:] = [ys[n], ys[n]+window_shape[0], 
                   xs[n], xs[n]+window_shape[1]]
  return NegImMat, NegBox
  

def get_nearest_windows_from_masked_image(Im, BBox, RefIm,
                                          window_shape=(25,25),
                                          n_samples=1, step=4, seed=0,
                                          nZones=10, ZH=400, ZW=400):
  H, W = window_shape
  assert RefIm.shape[0] == H
  assert RefIm.shape[1] == W
  GrayRefImVec = skimage.color.rgb2gray(RefIm).flatten()

  BBox = BoundBox.get_bound_box_with_significant_overlap(BBox, window_shape)
  GrayIm = skimage.color.rgb2gray(Im)
  for pID in xrange(BBox.shape[0]):
    GrayIm[BBox[pID,0]:BBox[pID,1], BBox[pID,2]:BBox[pID,3]] = OFFLIMITSVAL
  ImH, ImW = GrayIm.shape

  # Loop through several "zones" centered on labeled huts,
  #  and find some nearest neighbors in each zone!
  randstate = np.random.RandomState(seed)
  zIDs = randstate.choice( BBox.shape[0], nZones, replace=False)
  xjitter = randstate.choice( range(step), 1, replace=False)
  yjitter = randstate.choice( range(step), 1, replace=False)
  flatIDs = list()
  for posID, zID in enumerate(zIDs):
    if posID == nZones - 1:
      nNeigh = n_samples - (nZones-1) * (n_samples//nZones)
    else:
      nNeigh = n_samples // nZones
    NNSearcher = sklearn.neighbors.NearestNeighbors(n_neighbors=nNeigh,
                                                    algorithm='brute')
    yc = (BBox[zID,1] +  BBox[zID,0])/2 + yjitter
    xc = (BBox[zID,3] +  BBox[zID,2])/2 + xjitter
    ymin = max(0, yc - ZH/2)
    ymax = min(ImH, yc + ZH/2)
    xmin = max(0, xc - ZW/2)
    xmax = min(ImW, xc + ZW/2)
    
    ZoneIm = GrayIm[ ymin:ymax, xmin:xmax]
    ZoneWindows = skimage.util.view_as_windows(ZoneIm, window_shape, step)
    # Flatten nR x nC x H x W into nWindow x (HxW)
    nR, nC, H, W = ZoneWindows.shape

    ZoneWindowMat = np.reshape(ZoneWindows, (nR*nC, H*W))
    NNSearcher.fit(ZoneWindowMat)
    dists, nIDs = NNSearcher.kneighbors(GrayRefImVec)
    yoff, xoff = np.unravel_index( np.squeeze(nIDs), (nR, nC))
    curflatIDs = np.ravel_multi_index((ymin+step*yoff,
                                       xmin+step*xoff), (ImH, ImW))
    flatIDs.extend(curflatIDs)
    GrayIm[ ymin:ymax, xmin:xmax] = OFFLIMITSVAL

  assert len(flatIDs) == n_samples
  ys, xs = np.unravel_index( flatIDs, (ImH, ImW))
  if Im.ndim == 3:
    NegIm = np.ones((n_samples, H, W, 3), dtype=Im.dtype)
  else:
    NegIm = np.ones((n_samples, H, W), dtype=Im.dtype)
  for nID in xrange(n_samples):
    NegIm[nID] = Im[ys[nID]:ys[nID]+H, xs[nID]:xs[nID]+W]
  return NegIm, flatIDs


def sample_windows_from_masked_image(Im, BBox, window_shape=(25,25),
                                     n_samples=0, seed=0):
  H, W = window_shape

  Mask = np.ones((Im.shape[0],Im.shape[1]), dtype=np.uint8)
  for pID in xrange(BBox.shape[0]):
    Mask[BBox[pID,0]:BBox[pID,1], BBox[pID,2]:BBox[pID,3]] = 0
  Mask = Mask[:-H+1, :-W+1]
  candidates = np.flatnonzero( Mask)

  randstate = np.random.RandomState(seed)
  n_samples = np.minimum(n_samples, len(candidates))
  flatIDs = randstate.choice(candidates, n_samples, replace=False)
  ys, xs = np.unravel_index(flatIDs, Mask.shape)

  if Im.ndim == 3:
    NegIm = np.ones((n_samples, H, W, 3), dtype=Im.dtype)
  else:
    NegIm = np.ones((n_samples, H, W), dtype=Im.dtype)

  for nID in xrange(n_samples):
    NegIm[nID] = Im[ys[nID]:ys[nID]+H, xs[nID]:xs[nID]+W]
  return NegIm, flatIDs
