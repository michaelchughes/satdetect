'''
'''
import argparse
from distutils.dir_util import mkpath
import os
import joblib 

from satdetect.datasetbuilder import makeTrainAndTestDatasets
import satdetect.classifier as classifier
import satdetect.ioutil as ioutil

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('datapath', type=str)
  parser.add_argument('cpath', type=str)
  parser.add_argument('--holdoutNames', type=str, default='')
  parser.add_argument('--classifier', type=str, default='NearestNeighbor1')
  parser.add_argument('--do_viz', type=int, default=1)
  args = parser.parse_args()

  args.holdoutNames = args.holdoutNames.split(',')   

  ## Train : tuple of (X, Y, BBox)
  Train, Test = makeTrainAndTestDatasets(args.datapath,
                                         args.holdoutNames)
  C = classifier.trainClassifier(args.classifier, Train)

  classifier.testClassifier(C, Train)

  print 'Saved trained classifier to %s' % (args.cpath)
  mkpath(args.cpath)
  joblib.dump(C, os.path.join(args.cpath, 'C.dump'))
  joblib.dump(Test, os.path.join(args.cpath, 'Test.dump'))


if __name__ == '__main__':
  main()
