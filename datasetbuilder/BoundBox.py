''' BoundBox.py
'''
import numpy as np
import skimage
import satdetect.ioutil as ioutil

def view_as_windows_with_bbox(Im, window_shape, step):
  '''
     Returns
     -------
     WindowImSet : 3D array, size nWindow x (window_shape)
     BBox : 2D array, size nWindow x 4
  '''
  WindowImSet = skimage.util.view_as_windows(Im, window_shape, step)
  nR, nC, H, W = WindowImSet.shape
  nWindow = nR * nC
  WindowImSet = np.reshape(WindowImSet, (nWindow, H, W))

  H, W = Im.shape
  ys = np.arange(0, H - window_shape[0] +1, step)
  xs = np.arange(0, W - window_shape[1] +1, step)
  Bx, By = np.meshgrid(xs, ys)
  BBox = np.zeros( (Bx.size, 4))
  BBox[:,0] = By.flatten()
  BBox[:,1] = By.flatten() + window_shape[0]
  BBox[:,2] = Bx.flatten()
  BBox[:,3] = Bx.flatten() + window_shape[1]
  return WindowImSet, BBox

def load_pos_bbox_standard_size(setName, imageID, window_shape=(25,25)):
  PBox = ioutil.load_pos_pxbbox(setName, imageID)
  return convert_pos_bbox_to_standard_size(PBox, window_shape=window_shape)

def get_jittered_copies_of_pos_bbox(PBox, nCopy=2, xmag=3, ymag=3, seed=0):
  xopts = np.arange(-xmag, xmag+1)
  yopts = np.arange(-ymag, ymag+1)
  X, Y = np.meshgrid( xopts, yopts)
  jitteropts = np.arange( len(xopts)*len(yopts))

  # Assign zero probability to the "no jitter" option
  zeroIDs = ((X == 0) * (Y == 0)).flatten()
  ps = np.ones(len(jitteropts))
  ps[ zeroIDs] = 0
  ps /= ps.sum()

  # Now, create "jittered" copies of the pos examples
  PRNG = np.random.RandomState(seed)
  JBox = np.zeros( (nCopy*PBox.shape[0],4))
  for pID in xrange(PBox.shape[0]):
    choiceIDs = PRNG.choice(jitteropts, nCopy, replace=False, p=ps)
    xs, ys = np.unravel_index(choiceIDs, (len(xopts), len(yopts)))
    for ii in range(len(xs)):
      JBox[nCopy*pID + ii, [0,1]] = PBox[pID,[0,1]] + yopts[ys[ii]]
      JBox[nCopy*pID + ii, [2,3]] = PBox[pID,[2,3]] + xopts[xs[ii]]
  assert np.min(JBox[:,0]) >= 0
  assert np.min(JBox[:,2]) >= 0

  return JBox


def convert_pos_bbox_to_standard_size(PBox, window_shape=(25,25)):
  ''' Convert pixel bounding boxes in given array to standard size 

    Returns
    -------
    BBox : 2D array, size nPos x 4
           each row gives [ymin, ymax, xmin, xmax] pixel coords

  '''
  H, W = window_shape
  SBox = np.zeros_like(PBox)
  for rowID in xrange(PBox.shape[0]):
    curH = PBox[rowID,1]  - PBox[rowID,0]
    curW = PBox[rowID,3]  - PBox[rowID,2]
    gapH = (H - curH + 1) // 2
    gapW = (W - curW + 1) // 2
    assert gapH >= 0
    assert gapW >= 0

    ymin = PBox[rowID,0] - gapH
    ymax = PBox[rowID,1] + gapH
    xmin = PBox[rowID,2] - gapW
    xmax = PBox[rowID,3] + gapW

    if (ymax - ymin) > H:
      ymax = ymax - 1
    if (xmax - xmin) > W:
      xmax = xmax - 1
    assert (ymax - ymin) == H
    assert (xmax - xmin) == W
    SBox[rowID,:] = [ymin, ymax, xmin, xmax]
  return SBox

def get_bound_box_with_significant_overlap(BBox, window_shape=(25,25)):
  ''' Get bounding box for pixels whose window would have too much overlap 
  ''' 
  H, W = window_shape
  BBox = BBox.copy()
  BBox[:,0] -= H // 2
  BBox[:,2] -= W // 2
  BBox[:,1] -= H // 5
  BBox[:,3] -= W // 5
  np.maximum(BBox, 0, out=BBox)
  return BBox