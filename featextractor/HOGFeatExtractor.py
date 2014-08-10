import numpy as np
import skimage.feature

import satdetect.ioutil as ioutil

class MyPipeline(object):

  def __init__(self, steps):
    self.named_steps = dict(steps)
    names, estimators = zip(*steps)
    if len(self.named_steps) != len(steps):
      raise ValueError("Names provided are not unique: %s" % (names,))

    # shallow copy of steps
    self.steps = list(zip(names, estimators))

  def fit(self, Data, y=None):
    for name, estimator in self.steps[:-1]:
      Data = estimator.transform(Data)
    assert 'X' in Data
    assert 'y' in Data
    name, C = self.steps[-1]
    C.fit(Data['X'], Data['y'])

class CDatasetBuilder(object):

  def __init__(self):
    pass

  def get_params(self):
    return dict()

  def set_params(self, **params):
    pass

  def fit(self, Data, y, **kwargs):
    return self

  def transform(self, Data, **kwargs):
    nScenes = len(Data['imgpaths'])
    XList = list()
    yList = list()
    for ii in xrange(nScenes):
      imgpath = Data['imgpaths'][ii]
      Pos = Data['PosFeat'][ii]
      Neg = Data['NegFeat'][ii]
      X = np.vstack([Pos, Neg])
      y = np.hstack([np.ones(Pos.shape[0]), np.zeros(Neg.shape[0])])
      XList.append(X)
      yList.append(y)
    return dict(X=np.vstack(XList), y=np.hstack(yList))

class HOGFeatExtractor(object):

  def __init__(self, orientations=9, pixels_per_cell=(5,5)):
    self.orientations = orientations
    self.pixels_per_cell = pixels_per_cell
    self.cells_per_block = (1,1)
    self.color = 'gray'

  def get_params(self):
    return dict(orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                )

  def set_params(self, **params):
    for key, val in params.items():
      setattr(self, key, val)    

  def fit(self, Data, y, **kwargs):
    self.transform(Data)
    return self

  def transform(self, Data, **kwargs):
    ''' Extract HOG features for specified windows in provided images
    '''
    Data['PosFeat'] = list()
    Data['NegFeat'] = list()
    nScenes = len(Data['imgpaths'])
    for ii in xrange(nScenes):
      imgpath = Data['imgpaths'][ii]
      PosBBox = Data['PosBBox'][ii]
      NegBBox = Data['NegBBox'][ii]

      Im = ioutil.loadImage(imgpath, color=self.color)
      Pos = self._extractHOGForEachBBox(Im, PosBBox)
      Neg = self._extractHOGForEachBBox(Im, NegBBox)
      Data['PosFeat'].append(Pos)
      Data['NegFeat'].append(Neg)

    return Data

  def _extractHOGForEachBBox(self, Im, BBox):
    N = BBox.shape[0]
    for n in xrange(N):
      y0,y1,x0,x1 = BBox[n,:]
      if n == 0:
        if Im.ndim == 3:
          WindowIm = np.zeros( (N, y1-y0, x1-x0, Im.shape[2]), dtype=Im.dtype)
        else:
          WindowIm = np.zeros( (N, y1-y0, x1-x0), dtype=Im.dtype)
      WindowIm[n,:] = Im[y0:y1, x0:x1]
    return self._extractHOGFromWindows(WindowIm)

  def _extractHOGFromWindows(self, ImMat):
    for ii in xrange(ImMat.shape[0]):
      fvec = skimage.feature.hog(ImMat[ii], **self.get_params())      
      if ii == 0:
        FeatMat = np.zeros((ImMat.shape[0],fvec.size))
      FeatMat[ii,:] = fvec
    return FeatMat
      
