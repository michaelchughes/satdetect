'''
Show pictures of examples we suspect are really tukuls, but just weren't annotated as positive examples.
'''
import argparse
import sys
import os
import numpy as np

import ArgParseUtil
import satdetect.ioutil as ioutil
import satdetect.viz as viz

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('setName', type=str)
  parser.add_argument('--imageID', type=int)
  parser.add_argument('--exampleID', type=int, default=-1)
  parser.add_argument('--bbox', type=str, default='')
  parser.add_argument('--window_shape', type=str, default='25,25')
  args = parser.parse_args()
  ArgParseUtil.parse_window_shape_inplace(args)

  P, N, NBox = ioutil.load_labeled_images(args.setName, args.imageID,
                                    args.window_shape, 'rgb', include_bbox=1)
  Im = ioutil.load_jpg(args.setName, args.imageID)

  if len(args.bbox) > 0:
    NBox = [int(x) for x in args.bbox.split(',')]
    NBox = np.asarray(NBox)[np.newaxis,:]

  if args.exampleID < 0:
    exList = xrange(N.shape[0])
  else:
    exList = [args.exampleID-1, args.exampleID, args.exampleID+1]

  M = 25
  for n in exList:
    if NBox.shape[0] > 1:
      print '------------ %d' % (n)
      viz.imshow(N[n], figID=1)
    else:
      ImPiece = Im[NBox[n,0]:NBox[n,1], NBox[n,2]:NBox[n,3]]
      viz.imshow(ImPiece, figID=1)


    ImPiece = Im[NBox[n,0]-M:NBox[n,1]+M, NBox[n,2]-M:NBox[n,3]+M]
    viz.imshow(ImPiece, figID=2)

    try:
      keypress = raw_input(">>")
    except KeyboardInterrupt:
      sys.exit(1)

    if keypress.lower().count('y'):
      print NBox[n,:]
      sys.exit(1)

if __name__ == '__main__':
  main()