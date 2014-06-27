"""
RunClassifierExperiment.py
"""
import os
import numpy as np
import scipy.io
import scipy.stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pylab
import warnings

ROOTDIR='/data/burners/set1/featuresB32/'
FIGDIR='/data/burners/set1/figsB32/'

trainGroupIDs=[1,2,3]
testGroupIDs=[4]

def load_dataset(featname):
  Xtrain, Ytrain = _load_dataset_single_split(featname, trainGroupIDs)
  Xtest, Ytest = _load_dataset_single_split(featname, testGroupIDs)
  return Xtrain, Ytrain, Xtest, Ytest

def _load_dataset_single_split(featname, groupIDs):
  if featname.count('img') > 0:
    poskey = 'PosIm'
    negkey = 'NegIm'
  else:
    poskey = 'PosFeat'
    negkey = 'NegFeat'
  Xlist = list()
  Ylist = list()
  for posID, g in enumerate(groupIDs):
    groupmatfile = os.path.join(ROOTDIR, featname, 'group%d.mat' % (g))
    M = scipy.io.loadmat(groupmatfile)    
    X = np.vstack([ M[poskey],
                    M[negkey]])
    Y = np.hstack([ np.ones(M[poskey].shape[0], dtype=np.int32),
                    np.zeros(M[negkey].shape[0], dtype=np.int32)])
    Xlist.append(X)
    Ylist.append(Y)
  return np.vstack(Xlist), np.hstack(Ylist)

def evalClassifier_RandomForest(featname, baseline=None, **kwargs):
  Xtrain, Ytrain, Xtest, Ytest = load_dataset(featname)
  C = RandomForestClassifier(n_estimators=10, max_depth=None,
                             min_samples_leaf=10, random_state=0)
  C.fit(Xtrain, Ytrain)
  phat =  C.predict_proba(Xtest)
  return evalPredictions(Ytest, phat, featname + '+Neighbor', 
                              baseline=baseline, **kwargs)
  
def evalClassifier_Logistic(featname, baseline=None, **kwargs):
  Xtrain, Ytrain, Xtest, Ytest = load_dataset(featname)
  C = LogisticRegression(penalty='l2', C=1.0)
  C.fit(Xtrain, Ytrain)
  phat =  C.predict_proba(Xtest)
  return evalPredictions(Ytest, phat, featname + '+Logistic', 
                              baseline=baseline, **kwargs)

def evalClassifier_NaiveBayes(featname, baseline=None, **kwargs):
  Xtrain, Ytrain, Xtest, Ytest = load_dataset(featname)
  C = GaussianNB()
  C.fit(Xtrain, Ytrain)
  phat =  C.predict_proba(Xtest)
  return evalPredictions(Ytest, phat, featname + '+NaiveBayes',
                              baseline=baseline, **kwargs)


def evalClassifier_NearestNeighbor(featname, baseline=None, **kwargs):
  Xtrain, Ytrain, Xtest, Ytest = load_dataset(featname)
  C = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
  C.fit(Xtrain, Ytrain)
  phat =  C.predict_proba(Xtest)
  return evalPredictions(Ytest, phat, featname + '+Neighbor',
                              baseline=baseline, **kwargs)

  
def evalPredictions(Ytest, Phat, mname, baseline=None, doSave=False):
  ''' Evaluate predicted probabilities against true labels
  '''
  if Phat.ndim > 1:
    Phat = Phat[:,-1] # use final column, which is probability of 1
  assert Phat.min() >= 0
  assert Phat.max() <= 1.0
  Yhat = np.asarray(Phat > 0.5, dtype=Ytest.dtype)
  nCorrect = np.sum(Yhat == Ytest)
  nTotal = Ytest.size
  acc = nCorrect / float(nTotal)
  posnCorrect = np.sum(np.logical_and(Ytest == 1, Yhat == Ytest))
  posnTotal = np.sum(Ytest==1)
  posacc = posnCorrect / float(posnTotal)
  msg =  '%4d/%4d %.3f pos   ' % (posnCorrect, posnTotal, posacc)
  msg += '%4d/%4d %.3f all   ' % (nCorrect, nTotal, acc)
  msg += '%20s  ' % (mname)

  if baseline is not None:
    errPercDiff = ((1-baseline) - (1-acc))/(1-baseline)
    msg += '% 6.2f error reduction'  % (errPercDiff)

  pylab.close('all')
  N = showMostConfidentFalseNegatives(Ytest, Phat)
  if N is not None and doSave:
    pylab.suptitle('False Negatives')
    figpath = mname + "_TestGroup%d_FalseNeg.png" % (testGroupIDs[0])
    figpath = os.path.join(FIGDIR, figpath)
    pylab.savefig(figpath, transparent=True, 
                           bbox_inches='tight', pad_inches=0)
    

  N = showMostConfidentFalsePositives(Ytest, Phat)
  if N is not None and doSave:
    pylab.suptitle('False Positives')
    figpath = mname + "_TestGroup%d_FalsePos.png" % (testGroupIDs[0])
    figpath = os.path.join(FIGDIR, figpath)
    pylab.savefig(figpath, transparent=True, 
                           bbox_inches='tight', pad_inches=0)

  print msg
  return acc
  
def evalPredictions_PredictMostCommonClass(Ytest, **kwargs):
  guess, count = scipy.stats.mode(Ytest)
  Yhat = guess * np.ones_like(Ytest)
  return evalPredictions(Ytest, Yhat, 'mostCommonClass', **kwargs)

def showMostConfidentFalseNegatives(Ytrue, Phat, Ktop=9):
  if Phat.ndim > 1:
    Phat = Phat[:,-1] # use final column, which is probability of 1
  Yhat = np.asarray(Phat > 0.5, dtype=Ytrue.dtype)
  falseNegIDs = np.flatnonzero( np.logical_and(Yhat == 0, Yhat != Ytrue))
  print 'FALSE NEG: %d/%d' % (len(falseNegIDs), np.sum(Ytrue==1))
  if len(falseNegIDs) == 0:
    return None
  # Sort false positives from smallest probability to largest
  sortIDs = np.argsort( Phat[falseNegIDs] )
  falseNegIDs = falseNegIDs[sortIDs[:Ktop]]
  #print '                ', falseNegIDs, Phat[falseNegIDs]
  PosIms, _ = loadTestImages(testGroupIDs, falseNegIDs, None)
  return plotImages(PosIms, Ktop=Ktop)


def showMostConfidentFalsePositives(Ytrue, Phat, Ktop=9):
  if Phat.ndim > 1:
    Phat = Phat[:,-1] # use final column, which is probability of 1
  Yhat = np.asarray(Phat > 0.5, dtype=Ytrue.dtype)
  falsePosIDs = np.flatnonzero( np.logical_and(Yhat == 1, Yhat != Ytrue))
  print 'FALSE POS: %d/%d' % (len(falsePosIDs), np.sum(Ytrue==0))
  if len(falsePosIDs) == 0:
    return None
  # Sort false positives from largest probability to smallest
  sortIDs = np.argsort( -1*Phat[falsePosIDs] )
  falsePosIDs = falsePosIDs[sortIDs[:Ktop]]
  #print '                ',  falsePosIDs, Phat[falsePosIDs]
  _, NegIms = loadTestImages(testGroupIDs, None, falsePosIDs)
  return plotImages(NegIms, Ktop=Ktop)


def loadTestImages(testIDs, posIDs=None, negIDs=None):
  groupmatfile = os.path.join(ROOTDIR, 'rgbimg', 'group%d.mat' % (testIDs[0]))
  M = scipy.io.loadmat(groupmatfile)
  if posIDs is None:
    PosIms = M['PosIm']
  else:
    PosIms = M['PosIm'][posIDs,:]

  NegIms = M['NegIm'][negIDs,:]
  return PosIms, NegIms

def vec2RGB(ImVec):
  B = int(np.sqrt(ImVec.size / 3))
  Im = np.reshape(ImVec, (3, B, B))
  return np.rollaxis(np.rollaxis(Im, 1),2)

def plotImages(ImMat, Ktop=9):
  nRow = int(np.floor(np.sqrt(Ktop)))
  nCol = int(np.ceil( Ktop/ float(nRow)))
  figH, axH = pylab.subplots(nRow, nCol)
  Kplot = np.minimum(ImMat.shape[0], Ktop)
  for kk in range(Kplot):
    Im = vec2RGB(ImMat[kk,:]) 
    pylab.subplot(nRow, nCol, kk+1)
    pylab.imshow(Im)
    pylab.axis('image')
    pylab.xticks([])
    pylab.yticks([])
  # Disable visibility for unused subplots
  for kk in range(Kplot, nRow*nCol):
    pylab.subplot(nRow, nCol, kk+1)
    pylab.axis('off')

  #with warnings.catch_warnings():
  #  warnings.simplefilter('ignore', category=Warning)
  #  figH.tight_layout()
  return figH

def showPosExamples(doSave=False):
  for group in [1,2,3,4]:
    PosIm, NegIm = loadTestImages([group])
    plotImages(PosIm, Ktop=16)
    if doSave:
      pngoutfile = os.path.join(FIGDIR, 'PosExamples_Group%d.png' % (group))
      pylab.savefig(pngoutfile, transparent=True, 
                                bbox_inches='tight', pad_inches=0)
    #NegIm = np.random.shuffle(NegIm)
    #plotImages(NegIm, Ktop=16)
      

doSave=False
print '----------------------------------------- Plot All Positive Examples'
showPosExamples(doSave=doSave)
if not doSave:
  pylab.show(block=True)            
sys.exit(0)

print '----------------------------------------- MostCommonClass'
Xtrain, Ytrain, Xtest, Ytest = load_dataset('grayimg')
baseline = evalPredictions_PredictMostCommonClass(Ytest, doSave=doSave)
if not doSave:
  pylab.show(block=True)            

cFuncs = [evalClassifier_NearestNeighbor, 
          #evalClassifier_NaiveBayes, 
          evalClassifier_Logistic,
          #evalClassifier_RandomForest,
         ] 
for cFunc in cFuncs:
  print '-----------------------------------------', cFunc.func_name
  for featname in ['hog5']:
    acc = cFunc(featname, baseline, doSave=doSave)
  if not doSave:
    pylab.show(block=True)
