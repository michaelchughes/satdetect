'''
'''
import argparse
import numpy as np
import sys
import os
from matplotlib import pylab

import ArgParseUtil
import satdetect.ioutil as ioutil
import satdetect.viz as viz

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('setName', type=str)
  parser.add_argument('label', type=str, default='neg')
  parser.add_argument('--imageID', type=int, default=1)
  parser.add_argument('--pstart', type=int, default=0)
  parser.add_argument('--nstart', type=int, default=0)
  parser.add_argument('--Ktop', type=int, default=16)
  parser.add_argument('--window_shape', type=str, default='25,25')
  args = parser.parse_args()
  ArgParseUtil.parse_window_shape_inplace(args)

  P, N = ioutil.load_labeled_images(args.setName, args.imageID,
                                    args.window_shape, 'rgb')
    
  pstart = args.pstart
  nstart = args.nstart
  while 1:
    #viz.pylab.close('all')
    print 'p %5d/%d | n %5d' % (pstart, P.shape[0], nstart)
    if args.label.count('pos') > 0:
      viz.show_examples(P[pstart:], Ktop=args.Ktop, figID=1)
    elif args.label.count('jitter') > 0:
      viz.show_examples(PJ[pstart:], Ktop=args.Ktop, figID=1)
    else:
      viz.show_examples(N[nstart:], Ktop=args.Ktop, figID=2)
    try:
      keypress = raw_input('Press any key >>')
    except KeyboardInterrupt:
      sys.exit(1)
    if os.path.exists(keypress):
      savefile = os.path.join(keypress, 'image%02d-%s.png' % (args.imageID, args.label))
      pylab.savefig(savefile, bbox=0)
      print 'SAVED TO:', savefile
    if keypress.count('exit') > 0:
      sys.exit(1)

    try:
      nstart = int(keypress)
    except:
      pass
    if len(keypress) == 0:
      nstart += args.Ktop
      pstart += args.Ktop

    nstart = np.maximum(0, np.minimum(N.shape[0]-args.Ktop, nstart))
    pstart = np.maximum(0, np.minimum(P.shape[0]-args.Ktop, pstart))

if __name__ == '__main__':
  main()