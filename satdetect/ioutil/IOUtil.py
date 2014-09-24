''' IOUtil.py
'''
import os
import numpy as np
import glob
import joblib
import scipy.io
from skimage.data import imread
from distutils.dir_util import mkpath

def getFilepathParts(path):
  ''' Transform string defining a filesystem absolute path into component parts

      Example
      ---------
      >> getFilepathParts('/data/mhughes/myimage.jpg')
      ('/data/mhughes/', 'myimage', '.jpg')
  '''
  pathdirs = path.split(os.path.sep)
  pathdir = os.path.sep.join(pathdirs[:-1])
  basefields = pathdirs[-1].split('.')
  basename = basefields[0]
  ext = '.' + basefields[1]
  return pathdir, basename, ext

def loadImage(path, basename='', color='rgb'):
  ''' Load JPEG image from file, 

      Returns
      --------
      IM : 2D or 3D array, size H x W x nColors
           dtype will be float64, with each pixel in range (0,1)
  '''
  path = str(path)
  if len(basename) > 0:
    path = os.path.join(path, basename)
  if color == 'gray' or color == 'grey':
    IM = imread(path, as_grey=True)
    assert IM.ndim == 2
  else:
    IM = imread(path, as_grey=False)
    if not IM.ndim == 3:
      raise ValueError('Color image not available.')

  if IM.dtype == np.float:
    MaxVal = 1.0
  elif IM.dtype == np.uint8:
    MaxVal = 255
  else:
    raise ValueError("Unrecognized dtype: %s" % (IM.dtype))
  assert IM.min() >= 0.0
  assert IM.max() <= MaxVal

  IM = np.asarray(IM, dtype=np.float64)
  if MaxVal > 1:
    IM /= MaxVal
  return IM