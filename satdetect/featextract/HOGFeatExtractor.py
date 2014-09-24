import os
import numpy as np
import skimage.feature

from satdetect.ioutil import getFilepathParts, mkpath

class HOGFeatExtractor(object):

  def __init__(self, orientations=9, pixels_per_cell=(5,5)):
    ''' Create HOGFeatExtractor instance
    '''
    self.orientations = orientations
    self.pixels_per_cell = pixels_per_cell
    self.cells_per_block = (1,1)
    self.color = 'gray'

  def transform(self, DataInfo, **kwargs):
    ''' Extract HOG features for provided windows

        Args
        -------
        DataInfo : dict, with fields
        * tileFilePaths

        Returns
        -------
        DataInfo : dict with updated fields
        * featFilePaths
    '''
    print '<<<<<<<<<<<<<<<<<<<<<<<<<<<< This is HOGFeatExtractor.transform'

    featpathList = list()
    for path in DataInfo['tilepathList']:
      pathdir, basename, ext = getFilepathParts(path)
      outpath = os.path.join(pathdir, str(self))
      mkpath(outpath)
      outfile = os.path.join(outpath, basename + ext)      

      featpathList.append(outfile)
      if os.path.exists(outfile):
        print '  Using pre-computed features for %s' % (basename)
        continue

      print '  Extracting features for %s...' % (basename)
      print outfile
      WindowInfo = np.load(path)
      WIm = WindowInfo['WIm']
      nWindow = WIm.shape[0]

      ## Build Feature Matrix
      ## each row is HOG feat vector for one window image
      for wID in xrange(nWindow):
        fvec = skimage.feature.hog(WIm[wID], **self.get_params())      
        if wID == 0:
          FeatMat = np.zeros((nWindow,fvec.size))
        FeatMat[wID,:] = fvec

      ## Write to file
      np.savez(outfile, Feat=FeatMat, Y=WindowInfo['Y'],
                        imgpath=WindowInfo['imgpath'],
                        tilepath=path)

    DataInfo['featpathList'] = featpathList
    return DataInfo

  def get_params(self):
    return dict(orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                )

  def set_params(self, **params):
    for key, val in params.items():
      setattr(self, key, val)    

  def __str__(self):
    return 'hog-nBins%d-pxPerCell%d' % (self.orientations,
                                        self.pixels_per_cell[0])