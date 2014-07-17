import unittest
import numpy as np
import scipy.spatial.distance

import satdetect.ioutil
import BoundBox
import MakeStandardizedObjects as M

class MakeStdObjects_Background(unittest.TestCase):

  def testSimpleExample(self):
    print ''
    Im = np.zeros((10, 10))
    Q = np.arange(1, 5).reshape((2,2))

    PBox = np.zeros((4,4))
    PBox[0,:] = [0,2,  0,2]
    PBox[1,:] = [0,2,  8,10]
    PBox[2,:] = [8,10,  0,2]
    PBox[3,:] = [8,10,  8,10]
    for p in xrange(PBox.shape[0]):
      Im[PBox[p,0]:PBox[p,1], PBox[p,2]:PBox[p,3]] = Q

    ## Try several possible "training" sets, verify that good ones are found
    for keepIDs in [[0,1], [1,2], [0,2]]:
      RefBox = PBox[keepIDs]
      RemBox = PBox[[x for x in range(4) if x not in keepIDs]]
      EstBox = M.makeStdObjects_Background(Im, RefBox, stride=1, nNegSamples=2)
      VerifyBBoxEquivalent(RemBox, EstBox)


  def testBigExample(self, stride=5):
    print ''
    Im = np.zeros((1000, 1000))
    Q = np.ones((10,10))
    Q[4:6] = 2
    Q[:, 4:6] = 2

    # Annotate one in each of the four quadrants 
    PBox = np.zeros((4,4))
    PBox[0,:] = [100, 110,     100,110]
    PBox[1,:] = [100, 110,     700,710]
    PBox[2,:] = [651, 661,     52,62]
    PBox[3,:] = [880, 890,     666, 676]
    for p in xrange(PBox.shape[0]):
      Im[PBox[p,0]:PBox[p,1], PBox[p,2]:PBox[p,3]] = Q

    ## Try several possible "training" sets, verify that good ones are found
    for keepIDs in [ [0,1], [1,2], [0,3]]:
      RefBox = PBox[keepIDs]
      RemBox = PBox[[x for x in range(4) if x not in keepIDs]]
      EstBox = M.makeStdObjects_Background(Im, RefBox, stride=stride,
                                                         nNegSamples=2)
      VerifyBBoxEquivalent(RemBox, EstBox, stride)

  def testBigExample_ManySamples(self, stride=5):
    print ''
    Im = np.zeros((1000, 1000))
    Im[ np.random.rand(1000,1000) < 0.05] = 1
    Q = 1 * np.ones((10,10))
    Q[4:6] = 2
    Q[:, 4:6] = 2

    # Annotate one in each of the four quadrants 
    PBox = np.zeros((4,4))
    PBox[0,:] = [100, 110,     100,110]
    PBox[1,:] = [100, 110,     700,710]
    PBox[2,:] = [651, 661,     52,62]
    PBox[3,:] = [880, 890,     666, 676]
    for p in xrange(PBox.shape[0]):
      Im[PBox[p,0]:PBox[p,1], PBox[p,2]:PBox[p,3]] = Q

    ## Try several possible "training" sets, verify that good ones are found
    for keepIDs in [ [0,1], [1,2], [0,3]]:
      RefBox = PBox[keepIDs]
      RemBox = PBox[[x for x in range(4) if x not in keepIDs]]
      negBB, Im = M.makeStdObjects_Background(Im, RefBox, stride=stride,
                                                      return_Img=1,
                                                      nNegSamples=10000)
      from matplotlib import pylab
      pylab.imshow(Im, vmin=-1, vmax=3)
      pylab.show(block=1)


  def testRealImage(self, stride=5):
    print ''
    import skimage.io
    Im = skimage.io.imread('/data/burners/set1/scenes/before1_image.jpg',
                           flatten=1)
    PBox = BoundBox.load_pos_bbox_standard_size('set1', 1)

    negBBox, Im = M.makeStdObjects_Background(Im, PBox, stride=stride,
                                                      return_Img=1,
                                                      nNegSamples=25)
      

''' Test that find K-nearest neighbors works
'''
class FindKNearestWindowsInImage(unittest.TestCase):

  def testSimpleExample(self):
    print ''
    Im = np.zeros((10, 10))
    Q = np.arange(1, 5).reshape((2,2))

    PBox = np.zeros((4,4))
    PBox[0,:] = [0,2,  0,2]
    PBox[1,:] = [0,2,  8,10]
    PBox[2,:] = [8,10,  0,2]
    PBox[3,:] = [8,10,  8,10]
    for p in xrange(PBox.shape[0]):
      Im[PBox[p,0]:PBox[p,1], PBox[p,2]:PBox[p,3]] = Q
    nearXY = M.findKNearestWindowsInImage(Im, Q, PBox.shape[0], stride=1, Z=100)
    EstBox = M.xy2BBox(nearXY, Q.shape) 
    VerifyBBoxEquivalent(PBox, EstBox)


  def testRealImage(self, hutID=5, stride=5):
    ''' remember, we don't mask anything, so the top result *will* be the query
    '''
    print ''
    import skimage.io
    Im = skimage.io.imread('/data/burners/set1/scenes/before1_image.jpg',
                           flatten=1)
    PBox = BoundBox.load_pos_bbox_standard_size('set1', 1)
    RefIm = Im[PBox[hutID,0]:PBox[hutID,1], PBox[hutID,2]:PBox[hutID,3]]
    nearXY = M.findKNearestWindowsInImage(Im, RefIm, 10, 
                                              stride=3, Z=500)
    NBox = M.xy2BBox(nearXY, RefIm.shape)
    from matplotlib import pylab
    pylab.subplots(nrows=3, ncols=3)
    for ii in xrange(9):
      curImg = Im[NBox[ii,0]:NBox[ii,1], NBox[ii,2]:NBox[ii,3]]
      pylab.subplot(3, 3, ii+1)
      pylab.imshow(curImg, cmap='gray') 
    pylab.show(block=1)

def VerifyBBoxEquivalent(PBox, EstBox, atol=1e-5):
  # Verify that every PBox was found exactly
  DistMat = scipy.spatial.distance.cdist(PBox, EstBox)
  for r in xrange(DistMat.shape[0]):
    assert DistMat[r,:].min() < atol
