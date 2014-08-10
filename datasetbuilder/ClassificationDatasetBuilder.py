import glob
import os
import numpy as np
import satdetect.ioutil as ioutil

def makeTrainAndTestDatasets(datapath, testNames):
  ''' Load all stored features from scene files in specified directory.

      Returns
      --------
      Train : dict definining Train dataset, with fields
              X : features, 2D array, size N x D
              Y : labels, 1D array, size N
              BBox : bounding boxes, 2D array, size N x 4
              imgpath : full file paths, list, size N 
      Test : dict defining Test dataset, or None if no test data specified
  '''
  if type(testNames) == list:
    testNames = set(testNames)
  elif type(matchNames) == str:
    testNames = set([testNames])

  featFileList = glob.glob(os.path.join(datapath, '*.npz'))
  trainList = list()
  testList = list()
  for fpath in featFileList:
    if isMatchByFilename(fpath, testNames):
      testList.append(fpath)
    else:
      trainList.append(fpath)
  if len(trainList) == 0:
    raise ValueError('No training data found')
  Train = makeXYDataset(trainList)
  printDataSummary(trainList, Train, "Train")

  if len(testList) > 0:
    Test = makeXYDataset(testList)
    printDataSummary(testList, Test, "Test")
  else:
    Test = None
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

def printDataSummary(fileList, Data, Name):
  ''' Print human-readable summary of a dataset
  '''
  print Name
  for tpath in fileList:
    print '  %s' % (tpath.split(os.path.sep)[-1])
  print ' Stats:'
  print '  %6d pos' % (np.sum(Data['Y']==1))
  print '  %6d neg' % (np.sum(Data['Y']==0))
