from optparse import OptionParser
import glob
import os, sys


from satdetect.ioutil import imgpath2list
from satdetect.featextract import WindowExtractor, HOGFeatExtractor
from satdetect.detect import BasicTrainSetBuilder, Classifier


def trainDetector(imgpath='', outpath='',
                  feat='hog',
                  cname='logistic',
                  **kwargs):
  ''' Train a detector on imagery in imgpath, save results to outpath

      Returns
      --------
      C : trained classifier for individual windows
  '''
  imgpathList = imgpath2list(imgpath)

  DInfo = dict()
  DInfo['imgpathList'] = imgpathList
  DInfo['outpath'] = outpath

  ## Break up satellite image into 25x25 pixel windows
  WindowExtractor.transform(DInfo)

  ## TODO: add options for other features
  ## Extract HOG feature vector for each window
  featExtractor = HOGFeatExtractor()
  featExtractor.transform(DInfo)

  BasicTrainSetBuilder.transform(DInfo)

  ## Train a classifier and save to outpath
  C = Classifier.trainClassifier(DInfo, cname=cname)
  Classifier.saveClassifier(C, DInfo)

  ## Evaluate the classifier on the training set
  Classifier.testClassifier(C, DInfo)

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('--imgpath', type=str, dest="imgpath",
                      help='path(s) to load training images from')
  parser.add_option('--outpath', type=str, dest="outpath",
                      help='path where results are saved')
  parser.add_option('--cname', type=str, default='logistic',
                      help="name of classifier choices=['logistic', 'svm-linear', 'svm-rgb']")
  (options, args) = parser.parse_args()
  trainDetector(options.imgpath, options.outpath)
