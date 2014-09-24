import glob

from satdetect.featextract import WindowExtractor, HOGFeatExtractor
from satdetect.detect import BasicTrainSetBuilder, Classifier


def trainDetector(imgpath, outpath, 
                  cname='logistic',
                  **kwargs):
  ''' Train a detector on imagery in imgpath, save results to outpath

      Returns
      --------
      C : trained classifier for individual windows
  '''
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

  BasicTrainSetBuilder.transform(DInfo)

  ## Train a classifier and save to outpath
  C = Classifier.trainClassifier(DInfo, cname=cname)
  Classifier.saveClassifier(C, DInfo)

  ## Evaluate the classifier on the training set
  Classifier.testClassifier(C, DInfo)

if __name__ == "__main__":
  imgpath = '/data/tukuls/sudan/data/scene1.jpg'
  outpath = '/data/tukuls/sudan/xfeatures/huts_25x25_stride4/'
  trainDetector(imgpath, outpath)