'''
'''
import argparse
import joblib
import os
from distutils.dir_util import mkpath
import numpy as np
import glob
from skimage.io import imsave

from satdetect.datasetbuilder import makeXYDataset, loadStdBBox, view_as_windows_with_bbox
import satdetect.featextractor as featextractor
import satdetect.classifier as classifier
import satdetect.ioutil as ioutil
import satdetect.viz as viz

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('cpath', type=str)
  parser.add_argument('imgpath', type=str, default='')
  parser.add_argument('--tmppath', default='/tmp/detector/')
  parser.add_argument('--tileSize', type=int, default=500)
  parser.add_argument('--stride', type=int, default=4)
  parser.add_argument('--thresh', type=float, default=0.5)
  parser.add_argument('--doMakeTiles', type=int, default=0)
  args = parser.parse_args()
  args.window_shape = (25,25)

  C = joblib.load(os.path.join(args.cpath, 'C.dump'))

  basepath = args.imgpath.split('.')[0]
  tmppath = args.tmppath + basepath
  mkpath(tmppath)
  print tmppath

  doTilesExist = os.path.exists(os.path.join(tmppath, 'tile01.dump'))
  
  if args.doMakeTiles or not doTilesExist:
    Im = ioutil.loadImage(args.imgpath, color='gray')

    ## Divide image into 500x500 tiles, each one with many bounding boxes
    S = args.tileSize
    nRow = int(np.ceil(Im.shape[0] / float(S)))
    nCol = int(np.ceil(Im.shape[1] / float(S)))
    tileID = 0
    for r in range(nRow):
      for c in range(nCol):
        tileID += 1
        yl = r*S; yh = (r+1)*S
        xl = c*S; xh = (c+1)*S
        
        ## Make true bounding box for current tile
        truebboxpath = basepath + '_huts.pxbbox'
        if os.path.exists(truebboxpath):
          TrueBox = loadStdBBox(basepath + '_huts.pxbbox', args.window_shape) 

          xs_match = np.logical_and(TrueBox[:,2] >= xl, 
                                    TrueBox[:,3] < xh) 
          ys_match = np.logical_and(TrueBox[:,0] >= yl, 
                                    TrueBox[:,1] < yh ) 
          matchIDs = np.flatnonzero(np.logical_and(xs_match, ys_match))
          TBox = TrueBox[matchIDs].copy()
          TBox[:, [0,1]] -= yl
          TBox[:, [2,3]] -= xl
        else:
          TBox = None

        ## Convert tile to windows and run feature extraction
        Tile = Im[yl:yh, xl:xh].copy()
        WMat, BBox = view_as_windows_with_bbox(Tile,
                                                args.window_shape, 
                                                args.stride)
        X = featextractor.extract_hog_for_imageset(WMat)
        nTrue = TBox.shape[0]
        print 'Tile %d/%d done. %d true objects.' % (tileID, nRow*nCol, nTrue)
        outpath = os.path.join(tmppath, 'tile%02d.dump' % (tileID))
        joblib.dump(dict(imgpath=args.imgpath,
                         X=X, BBox=BBox, TBox=TBox, TileIm=Tile), outpath)

  tileList = glob.glob(os.path.join(tmppath, '*.dump'))
  for tilepath in tileList:
    Q = joblib.load(tilepath)
    
    Phat = C.predict_proba(Q['X'])[:, -1] # final column = Pr(item in class 1)
    mask = Phat > args.thresh
    BBoxOn=Q['BBox'][mask]
    AIm = viz.show_image_with_bbox(Q['TileIm'], BBoxOn, Q['TBox'], block=0)
    
    outpath = tilepath.replace('.dump', '.jpg')
    imsave(outpath, AIm)

    nTotal = Q['X'].shape[0]
    print '%6d /%6d detections' % (np.sum(mask), nTotal)
  from IPython import embed; embed()

if __name__ == '__main__':
  main()

"""
import ArgParseUtil
from satdetect.datasetbuilder import make_dataset, make_images_for_dataset
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

"""