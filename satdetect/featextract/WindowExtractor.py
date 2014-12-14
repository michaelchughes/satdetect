import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
import skimage
import hashlib

from satdetect.ioutil import loadImage, getFilepathParts, mkpath, loadLabelConfig, saveImage
import matplotlib.pyplot as plt
import pdb
import cPickle
from PIL import Image

def transform(DataInfo, **kwargs):
  ''' Apply window extraction to all images in provided DataInfo dict.

      Params
      ---------
      DataInfo : dict with fields
      * imgpathList : list of valid paths to files on disk

      Keyword Args
      ---------
      --windowShape : tuple
      --stride : int

      See also
      ---------
      extractWindowsFromImage

      Returns
      ---------
      DataInfo dict, updated in-place.
  '''
  ## Unique identifier for this training data
  ustring = ''.join(DataInfo['imgpathList'])
  uid = int(hashlib.md5(ustring).hexdigest(), 16) % 10000
  DataInfo['trainuname'] = '%04d' % (uid)
  configpath="/home/hasnain/documents/projects/tukuldata/datasets/labels.cfg"

  print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< This is WindowExtractor.transform'
  tilepathList = list()
  for imgpath in DataInfo['imgpathList']:
    #curtpathList = extractWindowsFromImage(imgpath, DataInfo['outpath'], **kwargs)
    curtpathList = extractLabelsFromImage(imgpath, configpath, DataInfo['outpath']);
    tilepathList.extend(curtpathList)
  DataInfo['tilepathList'] = tilepathList
  return DataInfo

def extractLabelsFromImage(imgpath, configpath, outpath, window_shape=(25,25), stride=4, S=500, negDistThr=2, posDistThr=0.8):

  '''
    First extract all labels from the image.
    Then divide image into tiles and extract windows from unlaballed patches
    i/p:
        imgpath: Path to input file
        outpath: Path to output files
        window_shape: Shape of extracted patch
        stride: Distance between two successive patches
        S: Size of a tile

    o/p:
        outfileList: list of filepaths to .npz files where each tile is stored1
         ____x
        |
        |
       y
'''
  labelstride = 3
  lextra = 10
  outfilelist = list()
  basename = "tmp_hasnain"
  outfile = os.path.join(outpath, basename)
  outfilelist.append(outfile)

  GrayIm = loadImage(imgpath, color='gray')
  try:
    ColorIm = loadImage(imgpath, color='rgb')
  except:
    ColorIm = None

  labels = loadLabelConfig(configpath)
  labelpaths = []
  PosBBox = {}
  for label in labels:
    truebboxpath = imgpath.split('.')[0] + '_' + label + '.pxbbox'
    if os.path.exists(truebboxpath):
      PosBBox[label] = loadStdBBox(truebboxpath, window_shape=window_shape)
  height, width = GrayIm.shape
  #Extract label patches
  #Start from -20 in each dimension of bounding box and go till +20
  outfileList = list()

  for label in labels:
    labelTiles = np.empty((1, window_shape[0], window_shape[1]))
    colorLabelTiles = np.empty((1, window_shape[0], window_shape[1], 3))
    basename = os.path.basename(imgpath).split('.')[0]
    outfile = outpath + basename + "_" + label + ".dump"
    if label in PosBBox.keys():
      bboxs = PosBBox[label]
      for i in range(0, len(bboxs)):
        bbox = bboxs[i]
        (yl, yh) = (min(max(0, bbox[0]-lextra),height-1), min(max(0, bbox[1]+lextra),height-1))
        (xl, xh) = (min(max(0, bbox[2]-lextra),width-1), min(max(0, bbox[3]+lextra),width-1))
        GrayTile = GrayIm[yl:yh, xl:xh].copy()
        if (GrayTile.shape[0] < window_shape[0] or GrayTile.shape[1] < window_shape[1]):
            continue

        WIm, WBox = extractWindowsWithBBox(GrayTile, window_shape, labelstride)
        labelTiles = np.append(labelTiles, WIm, 0)

        if ColorIm is None:
          ColorTile = None
        else:
          ColorTile = ColorIm[yl:yh, xl:xh, :].copy()

        #Extract non label patches
        #np.savez(outfile, GrayIm=GrayTile, ColorIm=ColorTile,
        #         WIm=WIm, TileBBox=WBox, imgpath=imgpath)
        #pdb.set_trace()
        WIm, WBox = extractColorWindowsWithBBox(ColorTile, (window_shape[0], window_shape[1], 3), labelstride)
        colorLabelTiles = np.append(colorLabelTiles, WIm, 0)

      labelTiles = np.array(labelTiles[1::, :, :], dtype=np.float16)
      colorLabelTiles = np.array(colorLabelTiles[1::, :, :, :], dtype=np.float16)

      #dic = {}
      #dic['label'] = labelTiles
      #dic['color'] = colorLabelTiles
      dic = colorLabelTiles
      #pdb.set_trace()
      f = open(outfile, "wb")
      cPickle.dump(dic, f, protocol=2)
      f.close()
      outfileList.append(outfile)

  return outfileList

def createLabelImage(imgpath, configpath, outpath, window_shape=(25,25)):

  if not os.path.exists(imgpath):
    raise Exception("Image path: " + imgpath + " :does not exist")
  if not os.path.exists(configpath):
    raise Exception("Config path: " + configpath + " :does not exist")
  if not os.path.isdir(outpath):
    raise Exception("Outpath path: " + outpath + " :is not a directotry")

  GrayIm = loadImage(imgpath, color='gray')

  labels = loadLabelConfig(configpath)
  labelpaths = []
  PosBBox = {}
  for label in labels:
    truebboxpath = imgpath.split('.')[0] + '_' + label + '.pxbbox'
    if os.path.exists(truebboxpath):
      PosBBox[label] = loadStdBBox(truebboxpath, window_shape=window_shape)
  height, width = GrayIm.shape

  outfileList = list()

  labelimage = np.zeros((GrayIm.shape[0], GrayIm.shape[1]))

  basename = os.path.basename(imgpath).split('.')[0]
  outfile = outpath + basename + "_" + "labels.jpg"

  for label in labels:
    if label in PosBBox:
      bboxs = PosBBox[label]
      labelnum = int(labels[label])
      for i in range(0, len(bboxs)):
        bbox = bboxs[i]
        (yl, yh, xl, xh) = (bbox[0], bbox[1], bbox[2], bbox[3])
        labelimage[yl:yh, xl:xh] = labelnum
  saveImage(labelimage, outfile)
  return

def reconstructPathces(img, indexes, preds, imgpath, configpath, outpath, stride=3):

  if not (len(indexes) == len(preds)):
    raise Exception("len(indexes): " + len(indexes) + " len(preds): " + len(preds))

  imgname = os.path.basename(imgpath)
  #Load config file
  labels = loadLabelConfig(configpath)

  for keys in range(0, len(labels.keys())+1):                                        #labels + 1 background label
    label = labels[keys]
    labelimg = np.zeros((img.shape[0], img.shape[1]))
    outfile  = outpath + "/" + imgname.split('.')[0] + "_" + label + "_labels.jpg"
    for k in range(0, len(indices)):
      index = indices[k]
      i, j = (index[0], index[1])
      labelimg[i, j] = preds[k][label]
      labelimg[i-stride/2 : i+stride/2, j-stride/2 : j+stride/2] = preds[k][label]
    saveImage(labelimg, outfile)
  return

def reconstructPredictedLabels(preds, img_dimensions, configpath, outpath, outname, window_shape=(25,25), stride=3):
  '''
    Reconstructs a prediction image for each label
    Go over all the labels. For each label, create a new image.
  '''
  if not os.path.exists(configpath):
    raise Exception("Config path: " + configpath + " :does not exist")
  if not os.path.isdir(outpath):
    raise Exception("Outpath path: " + outpath + " :is not a directotry")
  if not len(img_dimensions.shape):
    raise Exception("Invalid image dimensions")
  if not len(winshape.shape):
    raise Exception("Invalid patch dimensions")    

  height, width = (img_dimensions.shape[0], img_dimensions.shape[1])
  winshape = window_shape[0]
  labels = loadLabelConfig(configpath)


  for label in preds.shape[1]:
    #For each label, create a new image
    predimg = np.zeros(img_dimensions)
    count = 0
    for i in range(winshape, height-winshape, stride):
      for j in range(winshape, width-winshape, stride):
        predimg[i, j] = preds[count, label]
        count += 1
    assert count == preds.shape[0]    #assert that all labels are remapped to the image.
    
    for key in labels.keys():
      if labels[key] == label:
        #Save the image
        scipy.misc.imsave(outpath + "_" + outname + "_" + key + ".jpg", predimg)

  return

def extractPatches(imgpath, configpath, outpath, window_shape=(25,25), for_training=1):
  '''
    Extracts patches using a label image 
  '''
  if not os.path.exists(imgpath):
    raise Exception("Image path: " + imgpath + " :does not exist")
  if not os.path.exists(configpath):
    raise Exception("Config path: " + configpath + " :does not exist")
  if not os.path.isdir(outpath):
    raise Exception("Outpath path: " + outpath + " :is not a directotry")

  GrayIm = loadImage(imgpath, color='gray')
  try:
    ColorIm = loadImage(imgpath, color='rgb')
  except:
    raise Exception("Image is not a color image")
  Im = ColorIm
  height, width = (Im.shape[0], Im.shape[1])

  print 'Extracting labels from patches ...'

  patches, BBox, stride, window_shape = DenseGrid(Im)
  if not for_training:
    return Im, BBox, patches, stride

  winshape = window_shape[0]
  #Load config file
  labels = loadLabelConfig(configpath)

  #Load label image
  imgname = os.path.basename(imgpath)
  labelImgName = os.path.dirname(imgpath) + "/" + imgname.split('.')[0] + '_labels.jpg'
  LabelIm = loadImage(labelImgName, color='gray')

  dic = {}
  patch_labels = {}
  print 'Grouping patches into labels'

  #Go over all the labels and find which are in some bbox
  for k in range(0, len(labels.keys())+1):
    patch_labels[k] = []

  for k in range(0, BBox.shape[0]):
    patch_center = (int(BBox[k][0]+BBox[k][1])/2), int((BBox[k][2] + BBox[k][3])/2)
    imglabel = int(LabelIm[patch_center])
    if str(imglabel) not in labels.values():
      imglabel = 0
    patch_labels[imglabel].append( (int(BBox[k][0]), int(BBox[k][1]), int(BBox[k][2]), int(BBox[k][3])) )

  for k in range(0, len(patch_labels.keys())):
    dic[k] = np.zeros((len(patch_labels[k]), window_shape[0], window_shape[1], 3))

  for k in range(0, len(patch_labels.keys())):
    label_coords = patch_labels[k]
    for i in range(0, len(label_coords)):
      patch_coords = label_coords[i]
      dic[k][i, :, :, :] = Im[patch_coords[0]: patch_coords[1], patch_coords[2]: patch_coords[3]]
  
  dic['stride'] = stride
  dic['imgname'] = imgname
  dic['img_dimensions'] = (Im.shape[0], Im.shape[1])

  outfilename = outpath + '/' + imgname.split('.')[0] + '.dump'

  print 'Saving the extracted patches at path: ' + outfilename
  f = open(outfilename, 'wb')
  cPickle.dump(dic, f, protocol=2)
  f.close()

  print 'Returning from extractPatches()'
  return

def DenseGrid(im, stride=3, window_shape=(25,25,3)):
  if not window_shape[0] == window_shape[1]:
    raise Exception("Window height and width should be equal. The passed window shape is: " + window_shape)

  patches, BBox = extractColorWindowsWithBBox(im, window_shape, stride)
  return patches, BBox, stride, window_shape
  
def extractWindowsFromImage(imgpath, outpath,
                            window_shape=(25,25),
                            stride=4, S=500,
                            negDistThr=2, posDistThr=0.8):
  ''' Divide image up into tiles and extract windows from each tile.

      Each tile is a managable size, while whole image may be too big
      to easily fit all extracted windows into memory.

      Returns
      ---------
      outfileList : list of filepaths to .npz files where each tile is stored.
  '''
  GrayIm = loadImage(imgpath, color='gray')
  try:
    ColorIm = loadImage(imgpath, color='rgb')
  except:
    ColorIm = None

  truebboxpath = imgpath.split('.')[0] + '_huts.pxbbox'
  if os.path.exists(truebboxpath):
    PosBBox = loadStdBBox(truebboxpath, window_shape=window_shape)

  nRow = int(np.ceil(GrayIm.shape[0] / float(S)))
  nCol = int(np.ceil(GrayIm.shape[1] / float(S)))
  tileID = 0

  outfileList = list()

  print '================== %s' % (imgpath)
  labelTiles = np.empty((1, window_shape[0], window_shape[1]))
  colorLabelTiles = np.empty((1, window_shape[0], window_shape[1], 3))
  ## March through all tiles, extracting windows and saving to file
  yh = 0
  xh = 0
  for r in range(nRow):
    for c in range(nCol):
      if yh >= GrayIm.shape[0]:
        continue
      if xh >= GrayIm.shape[1]:
        continue
      tileID +=1

      ## Define the four corners of this tile
      yl = np.maximum(0, r*S - (window_shape[0] - stride))
      xl = np.maximum(0, c*S - (window_shape[1] - stride))
      yh = (r+1)*S
      xh = (c+1)*S

      ## Expand the tile by just a little if that's enough to cover whole image
      ## and thus avoid a very small tile
      if yh + S / 4 > GrayIm.shape[0]:
        yh = GrayIm.shape[0]
      if xh + S / 4 > GrayIm.shape[1]:
        xh = GrayIm.shape[1]

      ## Create outfile name "scene1_tile02.npz"
      _, basename, ext = getFilepathParts(imgpath)
      basename = basename + "_tile%02d" % (tileID) + ".npz"
      mkpath(outpath)
      outfile = os.path.join(outpath, basename)
      outfileList.append(outfile)

      # If we've already built this tile, skip and continue
      if os.path.exists(outfile):
        print '  Using existing tile %d' % (tileID)
        continue

      print '  Building Tile %d...' % (tileID)

      ## Make the tile from the large image
      GrayTile = GrayIm[yl:yh, xl:xh].copy()
      if ColorIm is None:
        ColorTile = None
      else:
        ColorTile = ColorIm[yl:yh, xl:xh, :].copy()
        WIm, WBox = extractColorWindowsWithBBox(ColorTile, (window_shape[0],window_shape[1],3), stride)
        colorLabelTiles = np.append(colorLabelTiles, WIm, 0)
      ## Extract the windows, with corresponding bounding boxes
      # WIm : 2D array, Nwindows x window_shape
      # WBox : 2D array, Nwindows x 4
      # WBox provides bbox in tile-specific coordinates [min=0, max=S]
      # BBox : 2D array, Nwindows x 4
      # BBox provides bbox in whole-image coordinates [min=0, max=GrayIm.shape]

      WIm, WBox = extractWindowsWithBBox(GrayTile, window_shape, stride)
      labelTiles = np.append(labelTiles, WIm, 0)

      BBox = WBox.copy()
      BBox[:, [0,1]] += yl
      BBox[:, [2,3]] += xl

      if os.path.exists(truebboxpath):
        ## Identify positive windows (within a few strides of a PosBBox)
        DistMatrix = calcDistMatrixForBBox(BBox, PosBBox)
        DistToNearestPos = DistMatrix.min(axis=1)

        posMask = DistToNearestPos <= posDistThr * stride
        negMask = DistToNearestPos > negDistThr * stride
        posIDs = np.flatnonzero(posMask)
        negIDs = np.flatnonzero(negMask)


        PosWIm = WIm[posIDs]
        Y = -1 * np.ones(WIm.shape[0], np.int32)
        Y[negIDs] = 0
        Y[posIDs] = 1

        ## Determine the visible bounding boxes for current tile
        xs_match = np.logical_and(PosBBox[:,2] >= xl,
                                  PosBBox[:,3] < xh)
        ys_match = np.logical_and(PosBBox[:,0] >= yl,
                                  PosBBox[:,1] < yh )
        visibleIDs = np.flatnonzero(np.logical_and(xs_match, ys_match))
        curPosBBox = PosBBox[visibleIDs].copy()
        curPosBBox[:, [0,1]] -= yl
        curPosBBox[:, [2,3]] -= xl
        assert curPosBBox.shape[0] <= len(posIDs)
        minDistFromVisPos = DistMatrix[:, visibleIDs].min(axis=0)
        assert np.all(minDistFromVisPos <= posDistThr*stride)
      else:
        PosWIm = None
        Y = None
        curPosBBox = None
      #np.savez(outfile, GrayIm=GrayTile, ColorIm=ColorTile,
      #                  WIm=WIm,
      #                  TileBBox=WBox, ImBBox=BBox,
      #                  Y=Y, PosWIm=PosWIm,
      #                  curPosBBox=curPosBBox, imgpath=imgpath)
      #pdb.set_trace()
    basename = os.path.basename(imgpath).split('.')[0]

    labelTiles = np.array(labelTiles[1::, :, :], dtype=np.float16)
    colorLabelTiles = np.array(colorLabelTiles[1::, :, :, :], dtype=np.float16)
    outfile = outpath + basename + "_" + "unlaballed" + ".dump"

    #dic = {}
    #dic['label'] = labelTiles
    #dic['color'] = colorLabelTiles
    #pdb.set_trace()
    dic = colorLabelTiles
    f = open(outfile, "wb")
    cPickle.dump(dic, f, protocol=2)
    f.close()

    return outfileList

def extractWindowsWithBBox(Im, window_shape, stride):
  ''' Take manageable-sized image and extract all windows given shape and stride

     Returns
     -------
     WindowImSet : 3D array, size nWindow x (window_shape)
     BBox : 2D array, size nWindow x 4
  '''
  WindowImSet = skimage.util.view_as_windows(Im, window_shape, stride)
  nR, nC, H, W = WindowImSet.shape
  nWindow = nR * nC
  WindowImSet = np.reshape(WindowImSet, (nWindow, H, W))

  H, W = Im.shape
  ys = np.arange(0, H - window_shape[0] +1, stride)
  xs = np.arange(0, W - window_shape[1] +1, stride)
  Bx, By = np.meshgrid(xs, ys)
  BBox = np.zeros( (Bx.size, 4))
  BBox[:,0] = By.flatten()
  BBox[:,1] = By.flatten() + window_shape[0]
  BBox[:,2] = Bx.flatten()
  BBox[:,3] = Bx.flatten() + window_shape[1]
  return WindowImSet, BBox

def extractColorWindowsWithBBox(Im, window_shape, stride):
  ''' Take manageable-sized image and extract all windows given shape and stride

     Returns
     -------
     WindowImSet : 3D array, size nWindow x (window_shape)
     BBox : 2D array, size nWindow x 4
  '''
  #pdb.set_trace()
  WindowImSet = skimage.util.view_as_windows(Im, window_shape, stride)
  nR, nC, nc, H, W, c = WindowImSet.shape
  nWindow = nR * nC
  WindowImSet = np.reshape(WindowImSet, (nWindow, H, W, c))

  H, W, c = Im.shape
  ys = np.arange(0, H - window_shape[0] +1, stride)
  xs = np.arange(0, W - window_shape[1] +1, stride)
  Bx, By = np.meshgrid(xs, ys)
  BBox = np.zeros( (Bx.size, 4))
  BBox[:,0] = By.flatten()
  BBox[:,1] = By.flatten() + window_shape[0]
  BBox[:,2] = Bx.flatten()
  BBox[:,3] = Bx.flatten() + window_shape[1]
  return WindowImSet, BBox



def calcDistMatrixForBBox(ABox, BBox):
  ''' Calculate distances between bounding boxes

      Returns
      -------
      D : 2D array, size A x B
      D[a,b] = distance between ABox[a,:] and BBox[b,:]
  '''
  ay, ax = np.mean(ABox[:, 0:1], axis=1), np.mean(ABox[:, 2:3], axis=1)
  by, bx = np.mean(BBox[:, 0:1], axis=1), np.mean(BBox[:, 2:3], axis=1)

  # Make N x 2 matrices
  aPos = np.vstack([ay, ax]).T
  bPos = np.vstack([by, bx]).T
  if aPos.ndim == 1:
    aPos = aPos[np.newaxis,:]
  if bPos.ndim == 1:
    bPos = bPos[np.newaxis,:]
  return cdist(aPos, bPos, 'euclidean')

def loadStdBBox(bboxpath, **kwargs):
  ''' Load pixel bounding box from file, converted to std, uniform size

      Returns
      -------
      BBox : 2D array, size nPos x 4
           each row gives [ymin, ymax, xmin, xmax] pixel coords
           and where for every row the following conditions hold
          (ymax-ymin) = window_shape[0]
          (xmax-xmin) = window_shape[1]
  '''
  PBox = np.loadtxt(bboxpath, dtype=np.int32)
  return PBox
  #return makeStdBBox(PBox, **kwargs)

def makeStdBBox(PBox, window_shape=(25,25), Himage=None, Wimage=None):
  ''' Convert pixel bounding boxes in given array to standard size

      Returns
      -------
      BBox : 2D array, size nPos x 4
           each row gives [ymin, ymax, xmin, xmax] pixel coords
           and where for every row the following conditions hold
          (ymax-ymin) = window_shape[0]
          (xmax-xmin) = window_shape[1]
  '''
  if PBox.ndim == 1:
    assert PBox.size == 4
    PBox = PBox[np.newaxis,:]

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

    ## Final bounds checking
    if ymin < 0:
      ymin = 0; ymax = H
    if xmin < 0:
      xmin = 0; xmax = W
    if Himage is not None and ymax > Himage:
      ymax = Himage; ymin = Himage - H
    if Wimage is not None and xmax > Wimage:
      xmax = Wimage; xmin = Wimage - H
    SBox[rowID,:] = [ymin, ymax, xmin, xmax]
  return SBox
