'''
VizUtil.py

Utilities for displaying satellite images,
with (optional) bound-box annotations
'''
import numpy as np
from matplotlib import pylab
import os
import skimage.color

def imshow(Im, block=False, figID=1):
  figH = pylab.figure(num=figID)
  figH.clf()
  pylab.imshow(Im)
  pylab.draw()
  pylab.show(block=block)

def showExamples(PMat, Nsubplots=9, block=False, figID=1, W=1, H=1):
  nRow = int(np.floor(np.sqrt(Nsubplots)))
  nCol = int(np.ceil(Nsubplots/ float(nRow)))

  figH, axH = pylab.subplots(nRow, nCol, num=figID, figsize=(W*nCol, H*nRow))  
  Kplot = np.minimum(PMat.shape[0], Nsubplots)
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

def makeImageWithBBoxAnnotations(Im, BBox, BBox2=None, 
                                 boxcolor=[0,1,0],    # green
                                 boxcolor2=[1,1,0],   # yellow
                                 **kwargs):
  ''' Create color image with bounding boxes highlighted in color
  '''
  if Im.ndim < 3:
    AIm = skimage.color.gray2rgb(Im)
  else:
    AIm = Im.copy() # annotation shouldn't happen to original array
  _add_bbox_to_im_inplace(AIm, BBox, boxcolor)
  if BBox2 is not None:
    _add_bbox_to_im_inplace(AIm, BBox2, boxcolor2)
  return AIm
  
def _add_bbox_to_im_inplace(Im, BBox, boxcolor, doThickLines=1):
  BBox = np.asarray(BBox, dtype=np.int32)
  boxcolor = np.asarray(boxcolor, dtype=np.float64)
  if boxcolor.max() > 1:
    boxcolor = boxcolor / 255
  for r in xrange(BBox.shape[0]):
    Im[BBox[r,0]:BBox[r,1], BBox[r,2]] = boxcolor[np.newaxis,:]
    Im[BBox[r,0]:BBox[r,1], BBox[r,3]-1] = boxcolor[np.newaxis,:]
    Im[BBox[r,0], BBox[r,2]:BBox[r,3]] = boxcolor[np.newaxis,:]
    Im[BBox[r,1]-1, BBox[r,2]:BBox[r,3]] = boxcolor[np.newaxis,:]

  ## Draw thick lines by repeating this cmd with slightly shifted BBox coords
  if doThickLines:
    _add_bbox_to_im_inplace(Im, BBox-1, boxcolor, doThickLines=0)
    _add_bbox_to_im_inplace(Im, BBox+1, boxcolor, doThickLines=0)

    
"""
def showMostConfidentFalseNegatives(Ytrue, Phat, Nsubplots=9):
  if Phat.ndim > 1:
    Phat = Phat[:,-1] # use final column, which is probability of 1
  Yhat = np.asarray(Phat > 0.5, dtype=Ytrue.dtype)
  falseNegIDs = np.flatnonzero( np.logical_and(Yhat == 0, Yhat != Ytrue))
  print 'FALSE NEG: %d/%d' % (len(falseNegIDs), np.sum(Ytrue==1))
  if len(falseNegIDs) == 0:
    return None
  # Sort false positives from smallest probability to largest
  sortIDs = np.argsort( Phat[falseNegIDs] )
  falseNegIDs = falseNegIDs[sortIDs[:Nsubplots]]
  #print '                ', falseNegIDs, Phat[falseNegIDs]
  PosIms, _ = loadTestImages(testGroupIDs, falseNegIDs, None)
  return plotImages(PosIms, Nsubplots=Nsubplots)


def showMostConfidentFalsePositives(Ytrue, Phat, Nsubplots=9):
  if Phat.ndim > 1:
    Phat = Phat[:,-1] # use final column, which is probability of 1
  Yhat = np.asarray(Phat > 0.5, dtype=Ytrue.dtype)
  falsePosIDs = np.flatnonzero( np.logical_and(Yhat == 1, Yhat != Ytrue))
  print 'FALSE POS: %d/%d' % (len(falsePosIDs), np.sum(Ytrue==0))
  if len(falsePosIDs) == 0:
    return None
  # Sort false positives from largest probability to smallest
  sortIDs = np.argsort( -1*Phat[falsePosIDs] )
  falsePosIDs = falsePosIDs[sortIDs[:Nsubplots]]
  #print '                ',  falsePosIDs, Phat[falsePosIDs]
  _, NegIms = loadTestImages(testGroupIDs, None, falsePosIDs)
  return plotImages(NegIms, Nsubplots=Nsubplots)
"""