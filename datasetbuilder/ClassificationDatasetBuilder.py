import numpy as np
import satdetect.ioutil as ioutil

def make_dataset(setName, featName, window_shape, imageIDs):
  ''' Create dataset with pos/neg examples pooled from many images
  '''
  nImages = ioutil.get_num_images(setName)
  
  PosList = list()
  NegList = list()
  nNeg = 0
  for imageID in imageIDs:
    assert imageID > 0
    assert imageID <= nImages

    Pos, Neg = ioutil.load_labeled_feats(setName, imageID, window_shape, featName)
    PosList.append(Pos)
    NegList.append(Neg)
    nNeg += Neg.shape[0]

  X = np.vstack([np.vstack(PosList), np.vstack(NegList)])
  Y = np.ones(X.shape[0], dtype=np.int32)
  Y[-nNeg:] = 0
  return X, Y

def make_images_for_dataset(setName, window_shape, imageIDs):
  PosList = list()
  NegList = list()
  for imageID in imageIDs:
    P, N = ioutil.load_labeled_images(setName, imageID,
                                      window_shape, 'rgb')
    PosList.append(P)
    NegList.append(N)

  return np.vstack([np.vstack(PosList), np.vstack(NegList)])