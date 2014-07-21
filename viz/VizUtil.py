import numpy as np
from matplotlib import pylab
import os
import skimage.color

DEFAULTSAVEPATH='/data/burners/figures/'

def imshow( Im, block=False, figID=1):
  figH = pylab.figure(num=figID)
  figH.clf()
  pylab.imshow(Im)
  pylab.draw()
  pylab.show(block=block)

def show_examples( PMat, Ktop=9, block=False, figID=1):
  nRow = int(np.floor(np.sqrt(Ktop)))
  nCol = int(np.ceil( Ktop/ float(nRow)))
  figH = pylab.figure(num=figID)
  figH.clf()
  figH, axH = pylab.subplots(nRow, nCol, num=figID)  
  Kplot = np.minimum(PMat.shape[0], Ktop)
  for kk in range(Kplot):
    pylab.subplot(nRow, nCol, kk+1)
    if PMat[kk].ndim == 3:
      pylab.imshow(PMat[kk], interpolation='nearest')
    else:
      pylab.imshow(PMat[kk], interpolation='nearest', cmap='gray')

    pylab.axis('image')
    pylab.xticks([])
    pylab.yticks([])
  # Disable visibility for unused subplots
  for kk in range(Kplot, nRow*nCol):
    pylab.subplot(nRow, nCol, kk+1)
    pylab.axis('off')
  pylab.draw()
  pylab.show(block=block)

def save_fig_as_png(savepath, figID=1):
  figH = pylab.figure(num=figID)
  pylab.draw()
  if not os.path.exists(savepath) and not savepath.count(os.path.sep):
    savepath = os.path.join(DEFAULTSAVEPATH, savepath)
  pylab.xticks([])
  pylab.yticks([])
  pylab.savefig(savepath, bbox_inches = 'tight', pad_inches = 0)

def show_image_with_bbox(Im, BBox, BBox2=None, 
                             boxcolor=[1,1,0],
                             boxcolor2=[0,1,0],
                             block=False):
  ''' Plot color image with bounding boxes shown
  '''
  if Im.ndim < 3:
    AIm = skimage.color.gray2rgb(Im)
  else:
    AIm = Im.copy() # annotation shouldn't happen to original array
  _add_bbox_to_im_inplace(AIm, BBox, boxcolor)
  _add_bbox_to_im_inplace(AIm, BBox-1, boxcolor)
  _add_bbox_to_im_inplace(AIm, BBox+1, boxcolor)
  if BBox2 is not None:
    _add_bbox_to_im_inplace(AIm, BBox2, boxcolor2)
  imshow(AIm, figID=1, block=block)
  return AIm
  
def _add_bbox_to_im_inplace(Im, BBox, boxcolor):
  boxcolor = np.asarray(boxcolor)
  if boxcolor.max() > 1:
    boxcolor = boxcolor / 255
  for r in xrange(BBox.shape[0]):
    Im[ BBox[r,0]:BBox[r,1], BBox[r,2]] = boxcolor[np.newaxis,:]
    Im[ BBox[r,0]:BBox[r,1], BBox[r,3]-1] = boxcolor[np.newaxis,:]
    Im[ BBox[r,0], BBox[r,2]:BBox[r,3]] = boxcolor[np.newaxis,:]
    Im[ BBox[r,1]-1, BBox[r,2]:BBox[r,3]] = boxcolor[np.newaxis,:]
    
"""
def showMostConfidentFalseNegatives(Ytrue, Phat, Ktop=9):
  if Phat.ndim > 1:
    Phat = Phat[:,-1] # use final column, which is probability of 1
  Yhat = np.asarray(Phat > 0.5, dtype=Ytrue.dtype)
  falseNegIDs = np.flatnonzero( np.logical_and(Yhat == 0, Yhat != Ytrue))
  print 'FALSE NEG: %d/%d' % (len(falseNegIDs), np.sum(Ytrue==1))
  if len(falseNegIDs) == 0:
    return None
  # Sort false positives from smallest probability to largest
  sortIDs = np.argsort( Phat[falseNegIDs] )
  falseNegIDs = falseNegIDs[sortIDs[:Ktop]]
  #print '                ', falseNegIDs, Phat[falseNegIDs]
  PosIms, _ = loadTestImages(testGroupIDs, falseNegIDs, None)
  return plotImages(PosIms, Ktop=Ktop)


def showMostConfidentFalsePositives(Ytrue, Phat, Ktop=9):
  if Phat.ndim > 1:
    Phat = Phat[:,-1] # use final column, which is probability of 1
  Yhat = np.asarray(Phat > 0.5, dtype=Ytrue.dtype)
  falsePosIDs = np.flatnonzero( np.logical_and(Yhat == 1, Yhat != Ytrue))
  print 'FALSE POS: %d/%d' % (len(falsePosIDs), np.sum(Ytrue==0))
  if len(falsePosIDs) == 0:
    return None
  # Sort false positives from largest probability to smallest
  sortIDs = np.argsort( -1*Phat[falsePosIDs] )
  falsePosIDs = falsePosIDs[sortIDs[:Ktop]]
  #print '                ',  falsePosIDs, Phat[falsePosIDs]
  _, NegIms = loadTestImages(testGroupIDs, None, falsePosIDs)
  return plotImages(NegIms, Ktop=Ktop)
"""