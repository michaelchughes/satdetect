import glob
import os
import numpy as np
import satdetect.ioutil as ioutil

def makeTrainAndTestDatasets(datapath, testNames):
  if type(testNames) == list:
    testNames = set(testNames)
  elif type(matchNames) == str:
    testNames = set([testNames])

  featFileList = glob.glob(os.path.join(datapath, '*.npz'))
  trainList = list()
  testList = list()
  for fpath in featFileList:
    if isMatchByFilename(fpath, testNames):
      trainList.append(fpath)
    else:
      testList.append(fpath)
  Train = makeXYDataset(trainList)
  Test = makeXYDataset(testList)
  return Train, Test

def makeXYDataset(fileList, matchNames=''):
  if type(fileList) == str:
    fileList = glob.glob(fileList)

  if type(matchNames) == list:
    matchNames = set(matchNames)
  elif len(matchNames) > 0 and type(matchNames) == str:
    matchNames = set([matchNames])

  XList = list()
  YList = list()
  BList = list()
  PathList = list()

  for fpath in fileList:
    if len(matchNames) > 0 and not isMatchByFilename(fpath, matchNames):
      continue
    print fpath
    Q = np.load(fpath)
    X = np.vstack([Q['Pos'], Q['Neg']])
    B = np.vstack([Q['PosBBox'], Q['NegBBox']])
    nPos = Q['Pos'].shape[0]
    nNeg = Q['Neg'].shape[0]
    Y = np.hstack([np.ones(nPos), np.zeros(nNeg)])
    PathList.extend( [str(Q['imgpath'])]*(nPos+nNeg) )
    XList.append(X)
    YList.append(Y)
    BList.append(B)
  X = np.vstack(XList)
  Y = np.hstack(YList)
  B = np.vstack(BList)
  PathList = np.hstack(PathList)
  return dict(X=X, Y=Y, BBox=B, imgpath=PathList)

def isMatchByFilename(fpath, matchNames):
  for mname in matchNames:
    if fpath.count(mname):
      return True
  return False

"""
def make_dataset(setName, featName, window_shape, imageIDs):
  ''' Create dataset with pos/neg examples pooled from many images
  '''
  nImages = ioutil.get_num_images(setName)
  
  PosList = list()
  NegList = list()
  nNeg = 0
  for imageID in imageIDs:
    assert imageID > 0
    assert imageID <= nImages

    Pos, Neg = ioutil.load_labeled_feats(setName, imageID, window_shape, featName)
    PosList.append(Pos)
    NegList.append(Neg)
    nNeg += Neg.shape[0]

  X = np.vstack([np.vstack(PosList), np.vstack(NegList)])
  Y = np.ones(X.shape[0], dtype=np.int32)
  Y[-nNeg:] = 0
  return X, Y

def make_images_for_dataset(setName, window_shape, imageIDs):
  PosList = list()
  NegList = list()
  for imageID in imageIDs:
    P, N = ioutil.load_labeled_images(setName, imageID,
                                      window_shape, 'rgb')
    PosList.append(P)
    NegList.append(N)

  return np.vstack([np.vstack(PosList), np.vstack(NegList)])
"""