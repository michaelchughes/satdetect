'''
'''
import argparse

import ArgParseUtil
from satdetect.datasetbuilder import make_dataset, make_images_for_dataset
import satdetect.classifier as classifier
import satdetect.ioutil as ioutil

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('setName', type=str)
  parser.add_argument('featName', type=str)
  parser.add_argument('--testIDs', type=str, default='4')
  parser.add_argument('--classifier', type=str, default='NearestNeighbor')
  parser.add_argument('--do_viz', type=int, default=1)
  parser.add_argument('--window_shape', type=str, default='25,25')
  args = parser.parse_args()
  ArgParseUtil.parse_window_shape_inplace(args)

  testIDs = [int(x) for x in args.testIDs.split(',')]
  nTotal = ioutil.get_num_images(args.setName)
  trainIDs = [x for x in xrange(1, nTotal+1) if x not in testIDs]

  Train = make_dataset(args.setName, args.featName, args.window_shape, trainIDs)
  Test = make_dataset(args.setName, args.featName, args.window_shape, testIDs)
  TestImages = make_images_for_dataset(args.setName,
                                       args.window_shape, testIDs)


  C, Phat = classifier.trainAndMakePredictions(args.classifier, Train, Test)
  FNvals, thrs = classifier.evalPredictions(Phat, Test, TestImages, args, testIDs)

  CInfo = dict( Classifier=C, FNvals=FNvals, threshvals=thrs)
  ioutil.save_classifier(args.setName, testIDs, args.featName,
                         args.window_shape, CInfo)


if __name__ == '__main__':
  main()
