import os
import numpy as np
import scipy.stats
import warnings
import joblib
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from satdetect.ioutil import mkpath, getFilepathParts

def trainClassifier(Train, cname='logistic'):
  ''' Train classifier on provided dataset

      Args
      ----------
      Train : dict with fields
      * X : training data. 2D array, size Ntrain x D
            each row is a feature vector
      * Y : training labels, 1D array, size Ntrain

      Returns
      ---------
      C : trained Classifier object
  '''
  print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< This is Classifier.trainClassifier'
  print 'Training classifier of type: ', cname
  stime = time.time()

  Xtrain = Train['X']
  Ytrain = Train['Y']
  assert Xtrain.shape[0] == Ytrain.size
  assert Xtrain.ndim == 2
  assert Ytrain.ndim == 1
  assert len(np.unique(Ytrain)) == 2

  cname = cname.lower()
  if cname == 'randomforest':
    C = RandomForestClassifier(n_estimators=10, max_depth=None,
                             min_samples_leaf=10, random_state=0)
  elif cname == 'logistic':
    C = LogisticRegression(penalty='l2', C=1.0)
  elif cname == 'nearestneighbor1':
    C = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
  elif cname == 'nearestneighbor3':
    C = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
  elif cname == 'svm-rbf':
    C = SVC(C=1.0, kernel='rbf', probability=True)
  elif cname == 'svm-linear':
    C = SVC(C=1.0, kernel='linear', probability=True)
  else:
    raise NotImplementedError('Not recognized: ' + cname)
  
  ## Unique name for classifier pipeline
  ## must uniquely identify the training set + features + classifier
  uname = '%s-%s-%s' % (Train['trainuname'], Train['featuname'], cname)
  Train['pipelineuname'] = uname

  ## Train the classifier!
  C.fit(Xtrain, Ytrain)
  elapsedtime = time.time() - stime

  print 'Training complete after %.3f sec' % (elapsedtime)
  return C

def testClassifier(C, Data):
  ''' Apply pre-trained classifier to provided dataset
  '''
  print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< This is Classifier.testClassifier'

  Phat = C.predict_proba(Data['X'])
  if Phat.ndim > 1:
    Phat = Phat[:,-1] # use final column, which is probability of 1
  assert Phat.min() >= 0
  assert Phat.max() <= 1.0
  Ytest = Data['Y']

  ## Loop over thresholds that give distinct False Negative counts
  thrVals = np.unique(Phat[Ytest == 1])
  FNmax = np.maximum(10, np.sum(Ytest)/2)
  FNvals = np.arange(FNmax)
  thrVals = thrVals[:FNmax]
  for decisionThresh in thrVals:
    Yhat = np.asarray(Phat >= decisionThresh, dtype=Ytest.dtype)
    nCorrect = np.sum(Yhat == Ytest)
    nTotal = Ytest.size
    acc = nCorrect / float(nTotal)
  
    TP = np.sum(np.logical_and(Ytest == 1, Yhat == 1))
    TN = np.sum(np.logical_and(Ytest == 0, Yhat == 0))

    falseNegMask = np.logical_and(Ytest == 1, Yhat == 0)

    falsePosMask = np.logical_and(Ytest == 0, Yhat == 1)
    falsePosIDs = np.flatnonzero(falsePosMask)
    sortIDs = np.argsort(-1*Phat[falsePosIDs])
    falsePosIDs = falsePosIDs[sortIDs]

    FP = np.sum(falsePosMask)
    FN = np.sum(falseNegMask)

    msg  = 'FN:%5d/%d   '  % (FN, TP+FN)
    msg += 'FP:%5d/%d   '  % (FP, TN+FP)
    msg += 'acc:%3d/%3d %.3f   ' % (nCorrect, nTotal, acc)
    msg += 'decisionThr %.3f' % (decisionThresh)
    print msg

  return FNvals, thrVals

def saveClassifier(C, DInfo):
  ''' Save classifier to disk, using prescribed location in DInfo['outpath']

      Returns
      --------
      None. Classifier object serialized to disk using joblib.dump.
  '''
  print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< This is Classifier.saveClassifier'
  outpath = DInfo['outpath']
  outpath = os.path.join(outpath, 'trained-classifiers')
  mkpath(outpath)
    
  outpathfile = os.path.join(outpath, DInfo['pipelineuname'] + '.dump')
  SaveVars = dict(ClassifierObj=C, TrainDataInfo=DInfo)
  joblib.dump(SaveVars, outpathfile)
  print 'Classifier and TrainDataInfo saved to:'
  print outpathfile
