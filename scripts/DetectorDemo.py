''' DetectorDemo
'''

import argparse
import skimage.io
import skimage.color
import numpy as np

import ArgParseUtil
from satdetect.datasetbuilder import view_as_windows_with_bbox
import satdetect.ioutil as ioutil
import satdetect.featextractor as featextractor
import satdetect.viz as viz
import satdetect.datasetbuilder as builder

def prep(setName, imageID, window_shape, featName, step=4):
  xs = [25,  750,  1250,  250]
  ys = [0,  250,   100,   250]
  BigIm = ioutil.load_jpg(setName, imageID)
  for prepID in range(len(xs)):
    SmallIm = BigIm[ys[prepID]:ys[prepID]+500, xs[prepID]:xs[prepID]+500]
    
    GrayIm = skimage.color.rgb2gray(SmallIm)
    if GrayIm.max() <= 1.0:
      GrayIm *= 255

    fpath = '/data/burners/classifier-results/prep/image%d-%d.jpg' % (imageID, prepID)
    skimage.io.imsave(fpath, SmallIm)
    fpath = '/data/burners/classifier-results/prep/image_bbox%d-%d.txt' % (imageID, prepID)
    np.savetxt(fpath, [[ys[prepID], ys[prepID]+500,
                        xs[prepID], xs[prepID]+500]])

    fpath = '/data/burners/classifier-results/prep/gray%d-%d.jpg' % (imageID, prepID)
    skimage.io.imsave(fpath, GrayIm)

    print 'Extracting sliding window images....'
    WindowImSet, BBox = view_as_windows_with_bbox(GrayIm, window_shape, step)
    fpath = '/data/burners/classifier-results/prep/bbox%d-%d.txt' % (imageID, prepID)
    np.savetxt(fpath, BBox)

    print 'Extracting features....'
    Xmat = featextractor.extract_hog_for_imageset(WindowImSet)

    fpath = '/data/burners/classifier-results/prep/feat%d-%d.npz' % (imageID, prepID)
    np.savez_compressed(fpath, Xmat=Xmat)



def detect_on_prep(imageID, prepID, CInfo, thresh=0.1):
  fpath = '/data/burners/classifier-results/prep/image%d-%d.jpg' % (imageID, prepID)
  Im = skimage.io.imread(fpath)

  fpath = '/data/burners/classifier-results/prep/bbox%d-%d.txt' % (imageID, prepID)
  BBox = np.loadtxt(fpath)

  fpath = '/data/burners/classifier-results/prep/feat%d-%d.npz' % (imageID, prepID)
  Q =  np.load(fpath)
  Xmat = Q['Xmat']
  
  print 'Classifying...'
  ps = CInfo['Classifier'].predict_proba(Xmat)[:,-1] #final col=prob class 1
  if thresh == None:
    thresh = np.min( CInfo['threshvals'])
  onIDs = np.flatnonzero( ps > thresh)
  pOn = ps[onIDs]

  Info = dict(Im=Im,
         BBoxOn=BBox[onIDs],
         pOn=pOn,
         BBoxAll=BBox, 
         pAll=ps,
         onIDs=onIDs,
         )
  return Info
  

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('setName', type=str)
  parser.add_argument('featName', type=str)
  parser.add_argument('--imageID', type=int, default=1)
  parser.add_argument('--prepID', type=int, default=0)
  parser.add_argument('--testIDs', type=str, default='4')
  parser.add_argument('--thresh', type=float, default=0.5)
  parser.add_argument('--do_prep', type=int, default=0)
  parser.add_argument('--window_shape', type=str, default='25,25')
  args = parser.parse_args()
  ArgParseUtil.parse_window_shape_inplace(args)

  testIDs = [int(x) for x in args.testIDs.split(',')]
  CInfo = ioutil.load_classifier(args.setName, testIDs, args.featName, args.window_shape)

  if args.do_prep:
    prep(args.setName, args.imageID, args.window_shape, args.featName)
  else:
    fpath = '/data/burners/classifier-results/prep/image_bbox%d-%d.txt' % (args.imageID, args.prepID)
    ImBox = np.reshape(np.loadtxt(fpath, dtype=np.int32), (1,4))

    DInfo = detect_on_prep(args.imageID, args.prepID, CInfo, thresh=args.thresh)
    TrueBox = builder.load_pos_bbox_standard_size(args.setName, args.imageID, args.window_shape)
    xs_match = np.logical_and( TrueBox[:,2] >= ImBox[0,2], 
                               TrueBox[:,3] < ImBox[0,3] ) 
    ys_match = np.logical_and( TrueBox[:,0] >= ImBox[0,0], 
                               TrueBox[:,1] < ImBox[0,1] ) 
    matchIDs = np.flatnonzero(np.logical_and(xs_match, ys_match))
    TrueBox = TrueBox[matchIDs]
    TrueBox[:, [0,1]] -= ImBox[0,0]
    TrueBox[:, [2,3]] -= ImBox[0,2]

    fpath = '/data/burners/classifier-results/prep/image%d-%d_annotated.png' % (args.imageID, args.prepID)
    viz.show_image_with_bbox(DInfo['Im'], DInfo['BBoxOn'], TrueBox, 
                             block=0)
    viz.save_fig_as_png(fpath)


if __name__ == '__main__':
  main()
