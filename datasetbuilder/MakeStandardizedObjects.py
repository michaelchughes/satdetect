'''
'''
import numpy as np
import skimage.color
import sklearn.neighbors
import scipy.spatial.distance
import time
import warnings
import glob
import os
from distutils.dir_util import mkpath

import BoundBox
import satdetect.ioutil as ioutil

from matplotlib import pylab

OFFLIMITSVAL = -255

def makeAndSaveAllStdObjects(datapath, window_shape=(25,25), 
                                       nNegSamples=50,
                                       stride=5):
  PathList = glob.glob(os.path.join(datapath, 'xobjects', '*.jpg'))
  for curpath in sorted(PathList):
    assert curpath.count('.') == 1
    curpath = curpath.split('.')[0]
    basename = curpath.split(os.path.sep)[-1]

    print '================================================ %s' % (basename)

    curimgpath = curpath + '.jpg'
    Im = ioutil.loadImage(curimgpath, color='gray')
    PosBBox = BoundBox.loadStdBBox(curpath + '_huts.pxbbox', window_shape)
    NegBBox = makeStdObjects_Background(Im, PosBBox, 
                                          nNegSamples=nNegSamples,
                                          stride=stride)

    PosXY = PosBBox[:, [0,2]]
    NegXY = NegBBox[:, [0,2]]
    ## Verify that the negative bboxes are distinct
    ##  that is, they should all differ from each other by more than stride
    DistMat = scipy.spatial.distance.cdist(PosXY, NegXY)    
    assert DistMat.min() >= stride

    ## Verify that the negative bboxes are unique
    ##  that is, they should all differ from each other by more than stride
    DistMat = scipy.spatial.distance.cdist(NegXY, NegXY)    
    DistMat += 1e6 * np.eye(DistMat.shape[0]) # ignore self-distances
    assert DistMat.min() >= stride

    ## Write to file
    foldername = 'huts-%dx%d' % (window_shape[0], window_shape[1])
    outpath = os.path.join(datapath, 'xfeatures', foldername)
    mkpath(outpath)
    fpath = os.path.join(outpath, basename + '.npz')
    np.savez_compressed(fpath, PosBBox=PosBBox, NegBBox=NegBBox,
                               imgpath=curimgpath,
                               basename=basename)

    
    

def makeStdObjects_Background(Im, PosBBox, stride=5,
                                           nNegSamples=500, return_Img=0):
  ''' Calc bounding boxes for many candidate *background* objects

      Uses nearest neighbor search to find high-quality negatives.

      Returns
      --------
      NegBBox : 2D array, size nNegSamples x 4
                each row defines a bounding box for a single negative example
                         [ymin, ymax, xmin, xmax]
  '''
  ## Parse Inputs
  Im = np.asarray(Im.copy(), dtype=np.float32)
  assert Im.ndim == 2
  window_shape = (PosBBox[0,1] - PosBBox[0,0], PosBBox[0,3] - PosBBox[0,2])
  nPos = PosBBox.shape[0]

  ## OBox : 2D array, where each row is expanded bbox for same row in PosBBox
  ##   expanding prevents choosing an overlapping box as a neg example
  OBox = BoundBox.get_bound_box_with_significant_overlap(PosBBox,
                                                         window_shape)

  RefIms = np.zeros((nPos, window_shape[0], window_shape[1])) 
  for pID in xrange(nPos):
    RefIms[pID] = Im[PosBBox[pID,0]:PosBBox[pID,1], 
                     PosBBox[pID,2]:PosBBox[pID,3]]

  # Mask out all positive examples with offlimits values 
  for pID in xrange(nPos):
    Im[OBox[pID,0]:OBox[pID,1], OBox[pID,2]:OBox[pID,3]] = OFFLIMITSVAL

  print 'Mining hard negatives via knn search...'
  # Loop over all positive stdobjects, 
  #  find top K matches (nearest neighbors) among candidate neg windows
  K = int(np.ceil(nNegSamples / float(nPos)))
  stime = time.time()
  for pID in xrange(nPos):

    print "................. pos example %d/%d " % (pID+1,nPos)

    curXY = findKNearestWindowsInImage(Im, RefIms[pID], K, stride)
    assert curXY.shape[0] == K
    print '%d matches found in %.1f sec' % (curXY.shape[0], time.time()-stime)

    ## Mask out all selections in curXY, so they don't get picked again
    for nID in xrange(curXY.shape[0]):
      yc = curXY[nID,0] + window_shape[0]/2
      xc = curXY[nID,1] + window_shape[1]/2
      Im[yc-stride:yc+stride, xc-stride:xc+stride] = OFFLIMITSVAL

    ## Add chosen windows in curXY to the aggregated set of all neg windows
    if pID == 0:
      NegXY = curXY.copy()
    else:
      NegXY = np.vstack([NegXY, curXY])

  NegBBox = xy2BBox(NegXY, window_shape)

  assert NegBBox.shape[0] >= nNegSamples
  if NegBBox.shape[0] > nNegSamples:
    NegBBox = NegBBox[:nNegSamples]

  if return_Img:
    return NegBBox, Im
  return NegBBox

def findKNearestWindowsInImage(Im, RefIm, K, stride=5, Z=500):
  '''
      Returns
      --------
      bestXY : 2D array, size K x 2
               each row 
                  chosen bounding boxes, guaranteed not to overlap PosBBox
  '''
  window_shape = RefIm.shape

  # Searching entire image may be expensive, so we instead focus search
  # by examining all disjoint Z x Z tiles
  NNSearcher = sklearn.neighbors.NearestNeighbors(n_neighbors=K,
                                                    algorithm='brute')
  Him, Wim = Im.shape
  nRow = int(np.ceil( Him / float(Z) ))
  nCol = int(np.ceil( Wim / float(Z) ))

  bestXY = np.zeros((K,2))
  bestScores = np.inf * np.ones(K)
  for rr in xrange(nRow):
    for cc in xrange(nCol):
      CurTileIm = Im[ rr*Z:(rr+1)*Z, cc*Z:(cc+1)*Z]

      # Obtain 2D array of window data, size nWindow x (HxW)
      CurWindows = skimage.util.view_as_windows(CurTileIm, window_shape, stride)
      nR, nC, H, W = CurWindows.shape
      CurWindowMat = np.reshape(CurWindows, (nR*nC, H*W))

      # Identify best neighbor windows for the current reference image,
      # and translate each neighbor into a (x, y) tuple
      NNSearcher.fit(CurWindowMat)
      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dists, neighborIDs = NNSearcher.kneighbors(RefIm.flatten())
      dists = dists.flatten() # make it 1D array
      neighborIDs = neighborIDs.flatten()

      yoff, xoff = np.unravel_index(neighborIDs, (nR, nC))
      y = rr*Z + stride * yoff
      x = cc*Z + stride * xoff

      bP = 0
      cP = 0
      newScores = np.zeros_like(bestScores)
      newXY = np.zeros_like(bestXY)
      for k in xrange(K):
        if bestScores[bP] < dists[cP]:
          newScores[k] = bestScores[bP]
          newXY[k] = bestXY[bP]
          bP += 1
        else:
          newScores[k] = dists[cP]
          newXY[k, 0] = y[cP]
          newXY[k, 1] = x[cP]
          cP += 1
      bestScores = newScores
      bestXY = newXY

  return bestXY
  
def xy2BBox(XY, window_shape):
  ''' Convert matrix of x,y locations of top-left corner into bounding box
  '''
  BBox = np.zeros( (XY.shape[0], 4))
  for r in range(XY.shape[0]):
    BBox[r,0] = XY[r,0]
    BBox[r,1] = XY[r,0] + window_shape[0]
    BBox[r,2] = XY[r,1]
    BBox[r,3] = XY[r,1] + window_shape[1]
  return BBox


if __name__ == '__main__':
  makeAndSaveAllStdObjects('/data/tukuls/Sudan/', (25,25))

"""
  # Searching entire image may be expensive, so we instead focus search
  # by examining several ZH x ZW tiles centered on positive examples
  posXY = np.zeros((BBox.shape[0],2))
  for posID in xrange(BBox.shape[0]):
    posXY[posID,0] = ( BBox[centerID, 0] + BBox[centerID, 1] ) / 2
    posXY[posID,1] = ( BBox[centerID, 2] + BBox[centerID, 3] ) / 2

  randstate = np.random.RandomState(int(RefIm[0,0]))
  zoneXY = np.zeros((nZone, 2))
  for zoneID in range(nZone):
    if zoneID == 0:
      posID = randstate.choice(BBox.shape[0], 1)
    else:
      posID = randstate.choice(posXY.shape[0], 1, p=mindist)

    zoneXY[zoneID] = posXY[posID]
    curdist = scipy.spatial.distance.cdist(posXY, 
                                          zoneXY[zoneID][:,np.newaxis])
    mindist = np.minimum(mindist, curdist)
    mindist[ mindist < ZH // 2] = 0
    if np.all(mindist) < 1e-5:
      break
  # Keep good zone center locations
  zoneXY = zoneXY[:zoneID]


  for zID in range(zoneXY.shape[0]):
    xc, yc = zoneXY[zID, :]
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



def load_neg_image_windows(setName, imageID, window_shape=(25,25), 
                           nNegSamples=500):
  BBox = BoundBox.load_pos_bbox_standard_size(setName, imageID,
                                              window_shape=window_shape)
  OBox = BoundBox.get_bound_box_with_significant_overlap(BBox, window_shape)

  nPos = BBox.shape[0]

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
"""