import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
import skimage
import hashlib

from satdetect.ioutil import loadImage, getFilepathParts, mkpath

def transform(DataInfo, **kwargs):
  ''' Apply window extraction to all images in provided DataInfo dict.

      Params
      ---------
      DataInfo : dict with fields
      * imgpathList : list of valid paths to files on disk
      
      Keyword Args
      ---------
      --windowShape : tuple
      --stride : int

      See also
      ---------
      extractWindowsFromImage

      Returns
      ---------
      DataInfo dict, updated in-place.
  '''
  ## Unique identifier for this training data
  ustring = ''.join(DataInfo['imgpathList'])
  uid = int(hashlib.md5(ustring).hexdigest(), 16) % 10000
  DataInfo['trainuname'] = '%04d' % (uid)

  print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< This is WindowExtractor.transform'
  tilepathList = list()
  for imgpath in DataInfo['imgpathList']:
    curtpathList = extractWindowsFromImage(imgpath, DataInfo['outpath'],
                                           **kwargs)
    tilepathList.extend(curtpathList)
  DataInfo['tilepathList'] = tilepathList
  return DataInfo

def extractWindowsFromImage(imgpath, outpath, 
                            window_shape=(25,25),
                            stride=4, S=500,
                            negDistThr=2, posDistThr=0.8):
  ''' Divide image up into tiles and extract windows from each tile.

      Each tile is a managable size, while whole image may be too big
      to easily fit all extracted windows into memory.

      Returns
      ---------
      outfileList : list of filepaths to .npz files where each tile is stored.
  '''
  GrayIm = loadImage(imgpath, color='gray')
  try:
    ColorIm = loadImage(imgpath, color='rgb')
  except:
    ColorIm = None

  truebboxpath = imgpath.split('.')[0] + '_huts.pxbbox'
  if os.path.exists(truebboxpath):
    PosBBox = loadStdBBox(truebboxpath, window_shape=window_shape) 

  nRow = int(np.ceil(GrayIm.shape[0] / float(S)))
  nCol = int(np.ceil(GrayIm.shape[1] / float(S)))
  tileID = 0
  
  outfileList = list()

  print '================== %s' % (imgpath)

  ## March through all tiles, extracting windows and saving to file
  yh = 0
  xh = 0
  for r in range(nRow):
    for c in range(nCol):
      if yh >= GrayIm.shape[0]:
        continue
      if xh >= GrayIm.shape[1]:
        continue
      tileID +=1 
      
      ## Define the four corners of this tile
      yl = np.maximum(0, r*S - (window_shape[0] - stride))
      xl = np.maximum(0, c*S - (window_shape[1] - stride))
      yh = (r+1)*S
      xh = (c+1)*S

      ## Expand the tile by just a little if that's enough to cover whole image
      ## and thus avoid a very small tile
      if yh + S / 4 > GrayIm.shape[0]:
        yh = GrayIm.shape[0]        
      if xh + S / 4 > GrayIm.shape[1]:
        xh = GrayIm.shape[1]

      ## Create outfile name "scene1_tile02.npz"
      _, basename, ext = getFilepathParts(imgpath)
      basename = basename + "_tile%02d" % (tileID) + ".npz"
      mkpath(outpath)
      outfile = os.path.join(outpath, basename)
      outfileList.append(outfile)

      # If we've already built this tile, skip and continue
      if os.path.exists(outfile):
        print '  Using existing tile %d' % (tileID)
        continue

      print '  Building Tile %d...' % (tileID)

      ## Make the tile from the large image
      GrayTile = GrayIm[yl:yh, xl:xh].copy()
      if ColorIm is None:
        ColorTile = None
      else:
        ColorTile = ColorIm[yl:yh, xl:xh, :].copy()
      
      ## Extract the windows, with corresponding bounding boxes
      # WIm : 2D array, Nwindows x window_shape
      # WBox : 2D array, Nwindows x 4
      # WBox provides bbox in tile-specific coordinates [min=0, max=S]
      # BBox : 2D array, Nwindows x 4
      # BBox provides bbox in whole-image coordinates [min=0, max=GrayIm.shape]
      WIm, WBox = extractWindowsWithBBox(GrayTile, window_shape, stride)
      BBox = WBox.copy()
      BBox[:, [0,1]] += yl
      BBox[:, [2,3]] += xl

      if os.path.exists(truebboxpath):
        ## Identify positive windows (within a few strides of a PosBBox)
        DistMatrix = calcDistMatrixForBBox(BBox, PosBBox)
        DistToNearestPos = DistMatrix.min(axis=1)

        posMask = DistToNearestPos <= posDistThr * stride
        negMask = DistToNearestPos > negDistThr * stride
        posIDs = np.flatnonzero(posMask)
        negIDs = np.flatnonzero(negMask)
        

        PosWIm = WIm[posIDs]
        Y = -1 * np.ones(WIm.shape[0], np.int32)
        Y[negIDs] = 0
        Y[posIDs] = 1

        ## Determine the visible bounding boxes for current tile
        xs_match = np.logical_and(PosBBox[:,2] >= xl, 
                                  PosBBox[:,3] < xh) 
        ys_match = np.logical_and(PosBBox[:,0] >= yl, 
                                  PosBBox[:,1] < yh ) 
        visibleIDs = np.flatnonzero(np.logical_and(xs_match, ys_match))
        curPosBBox = PosBBox[visibleIDs].copy()
        curPosBBox[:, [0,1]] -= yl
        curPosBBox[:, [2,3]] -= xl
        assert curPosBBox.shape[0] <= len(posIDs)
        minDistFromVisPos = DistMatrix[:, visibleIDs].min(axis=0)
        assert np.all(minDistFromVisPos <= posDistThr*stride)
      else:
        PosWIm = None
        Y = None
        curPosBBox = None
      np.savez(outfile, GrayIm=GrayTile, ColorIm=ColorTile,
                        WIm=WIm,
                        TileBBox=WBox, ImBBox=BBox,
                        Y=Y, PosWIm=PosWIm,
                        curPosBBox=curPosBBox, imgpath=imgpath)      
    return outfileList

def extractWindowsWithBBox(Im, window_shape, stride):
  ''' Take manageable-sized image and extract all windows given shape and stride

     Returns
     -------
     WindowImSet : 3D array, size nWindow x (window_shape)
     BBox : 2D array, size nWindow x 4
  '''
  WindowImSet = skimage.util.view_as_windows(Im, window_shape, stride)
  nR, nC, H, W = WindowImSet.shape
  nWindow = nR * nC
  WindowImSet = np.reshape(WindowImSet, (nWindow, H, W))

  H, W = Im.shape
  ys = np.arange(0, H - window_shape[0] +1, stride)
  xs = np.arange(0, W - window_shape[1] +1, stride)
  Bx, By = np.meshgrid(xs, ys)
  BBox = np.zeros( (Bx.size, 4))
  BBox[:,0] = By.flatten()
  BBox[:,1] = By.flatten() + window_shape[0]
  BBox[:,2] = Bx.flatten()
  BBox[:,3] = Bx.flatten() + window_shape[1]
  return WindowImSet, BBox

def calcDistMatrixForBBox(ABox, BBox):
  ''' Calculate distances between bounding boxes

      Returns
      -------
      D : 2D array, size A x B
      D[a,b] = distance between ABox[a,:] and BBox[b,:]
  '''
  ay, ax = np.mean(ABox[:, 0:1], axis=1), np.mean(ABox[:, 2:3], axis=1)
  by, bx = np.mean(BBox[:, 0:1], axis=1), np.mean(BBox[:, 2:3], axis=1)

  # Make N x 2 matrices
  aPos = np.vstack([ay, ax]).T
  bPos = np.vstack([by, bx]).T
  if aPos.ndim == 1:
    aPos = aPos[np.newaxis,:]
  if bPos.ndim == 1:
    bPos = bPos[np.newaxis,:]
  return cdist(aPos, bPos, 'euclidean')

def loadStdBBox(bboxpath, **kwargs):
  ''' Load pixel bounding box from file, converted to std, uniform size

      Returns
      -------
      BBox : 2D array, size nPos x 4
           each row gives [ymin, ymax, xmin, xmax] pixel coords
           and where for every row the following conditions hold
          (ymax-ymin) = window_shape[0]
          (xmax-xmin) = window_shape[1]
  '''
  PBox = np.loadtxt(bboxpath, dtype=np.int32)
  return makeStdBBox(PBox, **kwargs)

def makeStdBBox(PBox, window_shape=(25,25), Himage=None, Wimage=None):
  ''' Convert pixel bounding boxes in given array to standard size 

      Returns
      -------
      BBox : 2D array, size nPos x 4
           each row gives [ymin, ymax, xmin, xmax] pixel coords
           and where for every row the following conditions hold
          (ymax-ymin) = window_shape[0]
          (xmax-xmin) = window_shape[1]
  '''
  if PBox.ndim == 1:
    assert PBox.size == 4
    PBox = PBox[np.newaxis,:]

  H, W = window_shape
  SBox = np.zeros_like(PBox)
  for rowID in xrange(PBox.shape[0]):
    curH = PBox[rowID,1]  - PBox[rowID,0]
    curW = PBox[rowID,3]  - PBox[rowID,2]
    gapH = (H - curH + 1) // 2
    gapW = (W - curW + 1) // 2
    assert gapH >= 0
    assert gapW >= 0

    ymin = PBox[rowID,0] - gapH
    ymax = PBox[rowID,1] + gapH
    xmin = PBox[rowID,2] - gapW
    xmax = PBox[rowID,3] + gapW

    if (ymax - ymin) > H:
      ymax = ymax - 1
    if (xmax - xmin) > W:
      xmax = xmax - 1
    assert (ymax - ymin) == H
    assert (xmax - xmin) == W

    ## Final bounds checking
    if ymin < 0:
      ymin = 0; ymax = H
    if xmin < 0:
      xmin = 0; xmax = W
    if Himage is not None and ymax > Himage:
      ymax = Himage; ymin = Himage - H
    if Wimage is not None and xmax > Wimage:
      xmax = Wimage; xmin = Wimage - H
    SBox[rowID,:] = [ymin, ymax, xmin, xmax]
  return SBox
