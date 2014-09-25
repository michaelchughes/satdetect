import os
import numpy as np
import scipy.stats
import warnings
import joblib
import time

import skimage

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from satdetect.ioutil import mkpath, getFilepathParts
from satdetect.viz import makeImageWithBBoxAnnotations

def runDetector(C, DInfo, decisionThr=0.5, **kwargs):
  ''' Apply classifier to all windows in provided dataset

      Args
      ----------
      DInfo : dict with fields

      Returns
      ---------
      None.
  '''
  print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< This is Detector.runDetector'
  stime = time.time()
  kwargs['decisionThr'] = decisionThr

  ## Loop over all source tiles
  nTiles = len(DInfo['tilepathList'])
  for tileID in range(nTiles):
    featpath = DInfo['featpathList'][tileID]
    tilepath = DInfo['tilepathList'][tileID]
    RInfo = applyDetectorToImageTileFeats(featpath, C, DInfo,
                                          **kwargs)

    ## Create image with annotations, to help debugging
    Im = makeImageWithDetectedBBoxes(tilepath, RInfo)

    ## Save annotated image to file
    outpath = DInfo['outpath']
    outpath = os.path.join(outpath, 'detection-results')
    mkpath(outpath)
    parentdir, basename, ext = getFilepathParts(tilepath)
    imfile = '%s-decisionThr%.2f.jpg' % (basename, decisionThr)
    imoutpath = os.path.join(outpath, imfile)
    skimage.io.imsave(imoutpath, Im)
    if tileID == 0:
      print 'Saving annotated image tiles...'
    print imoutpath

def makeImageWithDetectedBBoxes(tilepath, RInfo, **kwargs):
  ''' Make annotated image, with highlighted bboxes

      Returns
      --------
      Im : JPEG RGB image
  '''
  estPosIDs = np.flatnonzero(RInfo['Yhat'] > 0)

  TileInfo = np.load(tilepath)
  if 'ColorIm' in TileInfo and TileInfo['ColorIm'].ndim > 0:
    Im = TileInfo['ColorIm']
  else:
    Im = TileInfo['GrayIm']

  ## Grab subset of all rows in the BBox matrix that were assigned pos labels
  estPosBBox = TileInfo['TileBBox'][estPosIDs]
  truePosBBox = TileInfo['curPosBBox']

  Im = makeImageWithBBoxAnnotations(Im, truePosBBox, estPosBBox)
  return Im

def applyDetectorToImageTileFeats(featpath, C, DInfo,
                                  decisionThr=0.5, **kwargs):
  ''' Apply detector to single tile image

      Returns
      --------
      ResultInfo : dict with fields
      * Yhat : 1D array, size nWindows
      contains binary estimate [0,1] for whether hut exists in window or not
      * Ytrue : 1D array, size nWindows
      contains true, human-annotated binary label [0,1] for window
  '''
  TileInfo = np.load(featpath)
  Phat = C.predict_proba(TileInfo['Feat'])
  if Phat.ndim > 1:
    Phat = Phat[:,-1] # use final column, which is probability of 1
  assert Phat.min() >= 0
  assert Phat.max() <= 1.0

  Yhat = Phat > decisionThr
  ResultInfo = dict(Yhat=Yhat, 
                    Ytrue=TileInfo['Y'])
  return ResultInfo