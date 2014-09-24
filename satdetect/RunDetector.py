import argparse
import glob
import joblib
import os

from satdetect.featextract import WindowExtractor, HOGFeatExtractor
from satdetect.detect import BasicTrainSetBuilder, Detector

def runDetector(imgpath='', outpath='', cpath='', 
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
    print 'ERROR: Unable to load info from file:\n' + cpath
    raise e

  if type(imgpath) == list:
    imgpathList = imgpath
  elif imgpath.count('*') > 0:
    imgpathList = glob.glob(imgpath)
  else:
    imgpathList = [imgpath]
  DInfo = dict()
  DInfo['imgpathList'] = imgpathList
  DInfo['outpath'] = outpath

  ## Break up satellite image into 25x25 pixel windows
  WindowExtractor.transform(DInfo) 

  ## Extract HOG feature vector for each window
  featExtractor = HOGFeatExtractor()
  featExtractor.transform(DInfo)

  ## Evaluate the classifier on the training set
  Detector.runDetector(C, DInfo)


if __name__ == "__main__":
  IMGPATH = '/data/tukuls/sudan/data/scene1.jpg'
  OUTPATH = '/data/tukuls/sudan/xfeatures/huts_25x25_stride4/'
  CPATH = OUTPATH + 'trained-classifiers/logistic.dump'

  parser = argparse.ArgumentParser()
  parser.add_argument('imgpath', type=str,
                      default=IMGPATH)
  parser.add_argument('outpath', type=str,
                      default=OUTPATH)
  parser.add_argument('cpath', type=str,
                      default=CPATH)
  args = parser.parse_args()

  runDetector(**args.__dict__)