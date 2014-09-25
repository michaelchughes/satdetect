import argparse
import glob
import joblib
import os

from satdetect.ioutil import imgpath2list
from satdetect.featextract import WindowExtractor, HOGFeatExtractor
from satdetect.detect import BasicTrainSetBuilder, Detector

def runDetector(imgpath='', cpath='', decisionThr=0.5,
                **kwargs):
  ''' Run pre-trained detector (stored in cpath) on imagery in imgpath

      Returns
      --------
      None.
  '''
  try:
    SaveVars = joblib.load(cpath)
    C = SaveVars['ClassifierObj']
    TrainInfo = SaveVars['TrainDataInfo']
  except Exception as e:
    print 'ERROR: Unable to load from file:\n' + cpath
    raise e

  imgpathList = imgpath2list(imgpath)
  DInfo = dict()
  DInfo['imgpathList'] = imgpathList
  DInfo['outpath'] = TrainInfo['outpath']

  ## Break up satellite image into 25x25 pixel windows
  WindowExtractor.transform(DInfo) 

  ## Extract HOG feature vector for each window
  featExtractor = HOGFeatExtractor()
  featExtractor.transform(DInfo)

  ## Evaluate the classifier on the training set
  Detector.runDetector(C, DInfo, decisionThr=decisionThr, **kwargs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('imgpath', type=str,
                      help='path(s) to load training images from')
  parser.add_argument('cpath', type=str,
                      help='path where pre-trained classifier dump-file lives')
  parser.add_argument('--decisionThr', type=float, default=0.5,
                      help='decision threshold for classifier')
  args = parser.parse_args()
  runDetector(**args.__dict__)