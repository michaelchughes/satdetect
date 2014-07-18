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
  parser.add_argument('datapath', type=str)
  parser.add_argument('label', type=str, default='neg')
  parser.add_argument('--startID', type=int, default=0)
  parser.add_argument('--Ktop', type=int, default=16)
  args = parser.parse_args()

  Q = np.load(args.datapath)
  assert 'PosIm' in Q and 'NegIm' in Q

  if args.label.count('pos') > 0:
    Im = Q['PosIm']
    Smax = Im.shape[0]
  else:
    Im = Q['NegIm']
    Smax = Im.shape[0]

  s = args.startID
  while 1:
    s = np.maximum(0, np.minimum(Im.shape[0]-args.Ktop, s))
    print 'items %d-%d' % (s, s+args.Ktop-1)

    viz.show_examples(Im[s:], Ktop=args.Ktop, figID=1)
    try:
      keypress = raw_input('Press any key >>')
    except KeyboardInterrupt:
      sys.exit(1)
    if os.path.exists(keypress):
      savefile = os.path.join(keypress, 
                              'image%02d-%s.png' % (args.imageID, args.label))
      pylab.savefig(savefile, bbox=0)
      print 'SAVED TO:', savefile
    if keypress.count('exit') > 0:
      sys.exit(1)
    try:
      s = int(keypress)
    except:
      pass

    if len(keypress) == 0:
      s += args.Ktop

    
if __name__ == '__main__':
  main()