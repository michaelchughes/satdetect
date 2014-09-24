'''
'''
import argparse
import numpy as np
import sys
import os
from matplotlib import pylab

def showExamplesFromPath(datapath='', 
               label='neg',
               startID=0,
               Nsubplots=0):
  ''' Show example positive windows for provided data path
  '''
  import satdetect.viz as viz ## Avoid circular import

  Q = np.load(datapath)
  assert 'PosWIm' in Q

  if label.count('pos') > 0:
    Im = Q['PosWIm']
    Smax = Im.shape[0]

  if Im.shape[0] == 0:
    raise ValueError('No Pos Windows available.')

  s = startID
  while 1:
    s = np.maximum(0, np.minimum(Im.shape[0]-Nsubplots, s))

    viz.showExamples(Im[s:], Nsubplots=Nsubplots, figID=1)
    try:
      keypress = raw_input('Press any key >>')
    except KeyboardInterrupt:
      sys.exit(1)

    if keypress.count('exit') > 0:
      sys.exit(1)
    try:
      s = int(keypress)
    except:
      pass

    if len(keypress) == 0:
      s += Nsubplots
    """
    if os.path.exists(keypress):
      savefile = os.path.join(keypress, 
                              'image%02d-%s.png' % (args.imageID, args.label))
      pylab.savefig(savefile, bbox=0)
      print 'SAVED TO:', savefile
    """
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('datapath', type=str)
  parser.add_argument('label', type=str, default='neg')
  parser.add_argument('--startID', type=int, default=0)
  parser.add_argument('--Nsubplots', type=int, default=16)
  args = parser.parse_args()
  showExamplesFromPath(**args.__dict__)