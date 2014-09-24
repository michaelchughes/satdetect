import numpy as np
import os

def transform(DataInfo, nNegPerTile=5000):
  ''' Transform provided features into a labeled dataset for supervised learning

      Args
      --------
      DataInfo : dict with fields
      * tilepathList
      * imgpathList
      * featpathList

      Returns
      --------
      DataInfo : dict with updated fields
      * X : 2D array, size N x D
      * Y : 1D array, size N
      * tilepathPtr : 1D array, size len(tilepathList)
      * imgpathPtr : 1D array, size len(imgpathList)
  '''
  print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<< This is BasicTrainSetBuilder.transform'

  tilepathPtr = np.zeros(len(DataInfo['tilepathList'])+1)
  imgpathPtr = np.zeros(len(DataInfo['imgpathList'])+1)

  tilePos = 0
  tilepath = ''
  imgPos = 0
  imgpath = ''

  X = None
  Y = None
  N = 0
  PRNG = np.random.RandomState(0)
  for featpath in DataInfo['featpathList']:
    Q = np.load(featpath)
    if Q['tilepath'] != tilepath:
      tilepathPtr[tilePos] = N
      tilePos += 1
      tilepath = Q['tilepath']
    if Q['imgpath'] != imgpath:
      imgpathPtr[imgPos] = N
      imgPos += 1
      imgpath = Q['imgpath']

    curY = Q['Y']
    curX = Q['Feat']
    if nNegPerTile is not None and curY.size > nNegPerTile:
      negIDs = np.flatnonzero(curY == 0)
      PRNG.shuffle(negIDs)
      keepMask = curY.copy() > 0
      keepMask[negIDs[:nNegPerTile]] = 1
      curY = curY[keepMask]
      curX = curX[keepMask]

    if X is None:
      X = curX
      Y = curY
    else:
      X = np.vstack([X, curX])
      Y = np.hstack([Y, curY])
    N = Y.size

  print 'Dataset stats:'
  print ' pos examples %d' % (np.sum(Y))
  print ' neg examples %d' % (np.sum(1-Y))
  print '        total %d' % (N)

  tilepathPtr[-1] = N
  imgpathPtr[-1] = N

  DataInfo['X'] = X
  DataInfo['Y'] = Y
  DataInfo['tilepathPtr'] = tilepathPtr
  DataInfo['imgpathPtr'] = imgpathPtr
  return DataInfo