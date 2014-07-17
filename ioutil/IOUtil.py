''' IOUtil.py
'''
import os
import numpy as np
import glob
import joblib
import scipy.io
from skimage.data import imread
from distutils.dir_util import mkpath

rootpath = '/data/burners/'

def load_classifier(setName, testIDs, featName, window_shape):
  path = get_path_to_classifier_data(setName, testIDs, featName, window_shape)
  path = os.path.join(path, 'Classifier.dump')
  CInfo = joblib.load(path)
  return CInfo

def save_classifier(setName, testIDs, featName, window_shape, CInfo):
  path = get_path_to_classifier_data(setName, testIDs, featName, window_shape)
  path = os.path.join(path, 'Classifier.dump')
  joblib.dump(CInfo, path)

def get_path_to_classifier_data(setName, testIDs, featName, window_shape):
  splitName = setName + '-' + ''.join(['%d'%(s) for s in testIDs])
  foldername = '%s-%dx%d' % (featName, window_shape[0], window_shape[1])
  path = os.path.join(rootpath, 'classifier-results', splitName, foldername)
  mkpath(path)
  return path

########################################################### labeled_feats
########################################################### 
def load_labeled_feats(setName, imageID, window_shape, featName):
  fpath = get_path_to_labeled_feats(setName, imageID, window_shape, featName)
  if not os.path.exists(fpath):
    matpath = fpath.replace('.npz', '.mat')
    try:
      Q = scipy.io.loadmat(matpath)
      save_labeled_feats(setName, imageID, window_shape, featName,
                       Q['Pos'], Q['Neg'])
    except:
      raise ValueError('BAD PATH: %s' % (fpath))
  Data = np.load(fpath)
  if Data['Pos'].ndim == 3:
    N, H, W = Data['Pos'].shape
    N2, H2, W2 = Data['Neg'].shape
    Pos = np.reshape(Data['Pos'], (N, H*W))
    Neg = np.reshape(Data['Neg'], (N2, H*W))
    return Pos, Neg
  return Data['Pos'], Data['Neg']

def save_labeled_feats(setName, imageID, window_shape, featName, Pos, Neg):
  fpath = get_path_to_labeled_feats(setName, imageID, window_shape, featName)
  np.savez_compressed(fpath, Pos=Pos, Neg=Neg)

def get_path_to_labeled_feats(setName, imageID, window_shape, featName, prefix=''):
  basename = '%s%d.npz' % (prefix,imageID)
  foldername = '%s-%dx%d' % (featName, window_shape[0], window_shape[1])
  fpath = os.path.join(rootpath, setName, foldername)
  mkpath(fpath)
  return os.path.join(fpath, basename)


def loadImage(path, basename='', color='rgb'):
  ''' Load JPG image from file
  '''
  path = str(path)
  if len(basename) > 0:
    path = os.path.join(path, basename)
  if color == 'gray' or color == 'grey':
    IM = imread(path, as_grey=True)
    assert IM.ndim == 2
  else:
    IM = imread(path)
    assert IM.ndim == 3

  IM = np.asarray(IM, dtype=np.float)
  assert IM.min() >= 0.0
  assert IM.max() <= 1.0
  return IM

def loadBBox(path):
  BBox = np.loadtxt(path, dtype=np.int32)
  return BBox

"""
########################################################### labeled_images
########################################################### 
def load_labeled_images(setName, imageID, window_shape, color, include_bbox=0):
  fpath = get_path_to_labeled_images(setName, imageID, window_shape, color)
  Data = np.load(fpath)
  if include_bbox:
    return Data['Pos'], Data['Neg'], Data['NegBBox']
  return Data['Pos'], Data['Neg']

def save_labeled_images(setName, imageID, window_shape, Pos, Neg, PosJitter=None, NegBBox=None):
  if Pos.ndim == 4:
    color = 'rgb'
  elif Pos.ndim == 3:
    color = 'gray'
  fpath = get_path_to_labeled_images(setName, imageID, window_shape, color)
  np.savez_compressed(fpath, Pos=Pos, Neg=Neg, PosJitter=PosJitter,
                                               NegBBox=NegBBox)
  matpath = fpath.replace('.npz', '.mat')
  scipy.io.savemat(matpath, dict(Pos=Pos, Neg=Neg), oned_as='row')

def get_path_to_labeled_images(setName, imageID, window_shape, color, prefix=''):
  basename = '%s%d.npz' % (prefix,imageID)
  foldername = '%s-%dx%d' % (color, window_shape[0], window_shape[1])
  fpath = os.path.join(rootpath, setName, foldername)
  mkpath(fpath)
  return os.path.join(fpath, basename)

def get_path_to_jpg(setName, imageID):
  return os.path.join(rootpath, setName, 'scenes',
                                'before%d_image.jpg' % (imageID))

def get_path_to_pos_pxbbox(setName, imageID):
  return os.path.join(rootpath, setName, 'scenes',
                                'before%d_huts.pxbbox' % (imageID))

def get_num_images(setName, prefix='before'):
  imdir = os.path.join(rootpath, setName, 'scenes')
  impattern = os.path.join(imdir, prefix + '*image.jpg')
  imList = glob.glob(impattern)
  return len(imList)

def load_jpg(setName, imageID):
  return imread(get_path_to_jpg(setName, imageID))

def load_pos_pxbbox(setName, imageID):
  P = np.loadtxt(get_path_to_pos_pxbbox(setName, imageID), dtype=np.int32)
  return P
"""