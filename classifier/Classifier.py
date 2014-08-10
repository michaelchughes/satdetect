"""
RunClassifierExperiment.py
"""
import os
import numpy as np
import scipy.stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings

import satdetect.ioutil as ioutil
import satdetect.viz as viz


def trainClassifier(classifierType, Train):
  ''' Train classifier on provided dataset

      Args
      ----------
      Train : dict with fields
              X : training data. 2D array, size Ntrain x D
                    each row is a feature vector
              Y : training labels, 1D array, size Ntrain

      Returns
      ---------
      C : trained classifier object
  '''
  Xtrain = Train['X']
  Ytrain = Train['Y']
  assert Xtrain.shape[0] == Ytrain.size
  assert Xtrain.ndim == 2
  assert Ytrain.ndim == 1
  assert len(np.unique(Ytrain)) == 2

  classifierType = classifierType.lower()
  if classifierType == 'randomforest':
    C = RandomForestClassifier(n_estimators=10, max_depth=None,
                             min_samples_leaf=10, random_state=0)
  elif classifierType == 'logistic':
    C = LogisticRegression(penalty='l2', C=1.0)
  elif classifierType == 'nearestneighbor1':
    C = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
  elif classifierType == 'nearestneighbor3':
    C = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
  elif classifierType == 'svm-rbf':
    C = SVC(C=1.0, kernel='rbf', probability=True)
  elif classifierType == 'svm-linear':
    C = SVC(C=1.0, kernel='linear', probability=True)
  else:
    raise NotImplementedError('Not recognized: ' + classifierType)

  ## Train the classifier!
  C.fit(Xtrain, Ytrain)
  return C

def testClassifier(C, Data):
  ''' Apply pre-trained classifier to provided dataset
  '''
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

    msg  = 'FN:%3d/%d   '  % (FN, TP+FN)
    msg += 'FP:%3d/%d   '  % (FP, TN+FP)
    msg += 'acc:%3d/%3d %.3f   ' % (nCorrect, nTotal, acc)
    msg += ' %.3f' % (decisionThresh)
    print msg

  return FNvals, thrVals


"""
def trainAndMakePredictions(classifierType, Train, Test, baseline=None, **kwargs):
  Xtrain, Ytrain = Train
  Xtest, Ytest = Test
  if classifierType == 'RandomForest':
    C = RandomForestClassifier(n_estimators=10, max_depth=None,
                             min_samples_leaf=10, random_state=0)
    C.fit(Xtrain, Ytrain)
    phat =  C.predict_proba(Xtest)
  elif classifierType == 'Logistic':
    C = LogisticRegression(penalty='l2', C=1.0)
    C.fit(Xtrain, Ytrain)
    phat =  C.predict_proba(Xtest)
  elif classifierType == 'NearestNeighbor1':
    C = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    C.fit(Xtrain, Ytrain)
    phat =  C.predict_proba(Xtest)
  elif classifierType == 'NearestNeighbor3':
    C = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
    C.fit(Xtrain, Ytrain)
    phat =  C.predict_proba(Xtest)
  else:
    raise NotImplementedError('Not recognized' + classifierType)
  return C, phat

  
def evalPredictions(Phat, Test, TestImages, args, testIDs, decisionThresh=0.5):
  ''' Evaluate predicted probabilities against true labels
  '''
  Xtest, Ytest = Test
  if Phat.ndim > 1:
    Phat = Phat[:,-1] # use final column, which is probability of 1
  assert Phat.min() >= 0
  assert Phat.max() <= 1.0

  # Loop over thresholds that give distinct False Negative counts
  thrVals = np.unique(Phat[Ytest == 1])

  FNmax = np.sum(Ytest)/2
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

    msg  = 'FN:%3d/%d   '  % (FN, TP+FN)
    msg += 'FP:%3d/%d   '  % (FP, TN+FP)
    msg += 'acc:%3d/%3d %.3f   ' % (nCorrect, nTotal, acc)
    print msg

    if args.do_viz:
      cpath = ioutil.get_path_to_classifier_data(args.setName, testIDs,
                         args.featName, args.window_shape)
      fnpath = os.path.join(cpath, 'FalseNegExamples-FN%d.png' % (FN))
      fppath = os.path.join(cpath, 'FalsePosExamples-FN%d.png' % (FN))

      # Show most confident false negatives
      viz.show_examples(TestImages[falseNegMask], figID=1)
      viz.save_fig_as_png(fnpath, figID=1)

      # Show most confident false positives
      viz.show_examples(TestImages[falsePosIDs], figID=2)
      viz.save_fig_as_png(fppath, figID=2)

    print falsePosIDs[:5] - np.sum(Ytest) # pos examples before neg ones

  return FNvals, thrVals
"""