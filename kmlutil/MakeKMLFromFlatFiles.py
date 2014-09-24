
import os
import sys
import numpy as np
import glob

TDIR = '../templates/'

def loadTemplates():
  T = dict()
  T['header'] = readFileIntoStr(TDIR + 'TEMPLATEHeader.kml')
  T['stylemap'] = readFileIntoStr(TDIR + 'TEMPLATEStyleMap.kml')
  T['placemark'] = readFileIntoStr(TDIR + 'TEMPLATEPlacemark.kml')
  T['imgoverlay'] = readFileIntoStr(TDIR + 'TEMPLATEImageOverlay.kml')
  return T

def readFileIntoStr(fpath):
  with open(fpath, 'r') as f:
    lineList = f.readlines()
  return ''.join(lineList)

def pxbbox2llbbox(path):
  ''' Convert pxbbox text files into lat/long format
  '''
  from ParseKMLIntoXObjects import convert_pixel_to_latlong_bbox
  from ParseKMLIntoXObjects import prettyprint_latlong_bbox

  sceneList = glob.glob(os.path.join(path, '*.jpg'))
  for spath in sorted(sceneList):
    imLL = np.loadtxt(spath.replace('.jpg', '.llbbox'))
    imBBox = np.loadtxt(spath.replace('.jpg', '.pxbbox'))
    hutBBox = np.loadtxt(spath.replace('.jpg', '_huts.pxbbox'))
    hutLL = np.loadtxt(spath.replace('.jpg', '_huts.llbbox'))

    outpath = spath.replace('.jpg', '_huts.llbbox')
    with open(outpath , 'w') as f:
      for n in xrange(hutBBox.shape[0]):
        hutLL = convert_pixel_to_latlong_bbox(imBBox[1], imBBox[3],
                                             llvec2dict(imLL),
                                             pxvec2dict(hutBBox[n,:]))
        f.write(prettyprint_latlong_bbox(hutLL) + '\n')

def llvec2dict( llvec):
  return dict(latMin=llvec[0], latMax=llvec[1], 
              lonMin=llvec[2], lonMax=llvec[3])

def pxvec2dict(pxvec):
  return dict(yMin=pxvec[0], yMax=pxvec[1], 
              xMin=pxvec[2], xMax=pxvec[3])

def abspath2basename(fpath):
  return fpath.split(os.path.sep)[-1].split('.')[0]

def mkKML(path, outkmlfile, px2ll=False):
  ''' Write information from the flatfiles in path to provided outkmlfile
  '''
  if px2ll:
    pxbbox2llbbox(path)
  dataname = abspath2basename(outkmlfile)

  print dataname

  T = loadTemplates()
  kmlstr = T['header'].replace('DATANAME', dataname)
  kmlstr += T['stylemap']
  kmlstr += "<Folder>"
  sceneList = glob.glob(os.path.join(path, '*.jpg'))
  print sceneList

  for spath in sorted(sceneList):
    basename = abspath2basename(spath)
    print spath
    print basename

    imgoverlay = '' + T['imgoverlay']
    imgoverlay = imgoverlay.replace('SCENENAME', basename)
    imgoverlay = imgoverlay.replace('IMFILE', basename + '.jpg')

    LL = loadLatLonForImage(spath.replace('.jpg', '.llbbox'))
    print LL
    for key in LL:
      imgoverlay = imgoverlay.replace(key, '%.16f' % (LL[key]))
    kmlstr += imgoverlay

    ## Read in latlong bbox for each hut
    LLbbox = np.loadtxt(spath.replace('.jpg', '_huts.llbbox'))
    kmlstr += llbbox2PlacemarkStr(LLbbox, T)

  ## Finalize it
  kmlstr += "\n"
  kmlstr += "</Folder>\n"
  kmlstr += "</Document>\n"
  kmlstr += "</kml>\n"
  with open(outkmlfile, 'w') as f:
    f.write(kmlstr)
  return kmlstr

def llbbox2PlacemarkStr(LLbbox, T):
  Plist = list()
  for n in xrange( LLbbox.shape[0] ):
    cstr = '%.16f,%.16f,0 ' % (LLbbox[n,3], LLbbox[n,0])
    cstr += '%.16f,%.16f,0 ' % (LLbbox[n,3], LLbbox[n,1])
    cstr += '%.16f,%.16f,0 ' % (LLbbox[n,2], LLbbox[n,1])
    cstr += '%.16f,%.16f,0 ' % (LLbbox[n,2], LLbbox[n,0])
    cstr += '%.16f,%.16f,0 ' % (LLbbox[n,3], LLbbox[n,0])
    Pstr = T['placemark'].replace('COORDS', cstr)
    Pstr = Pstr.replace('NAME', 'tukul_%d' % (n+1))
    Plist.append(Pstr)
  return ''.join(Plist)

def loadLatLonForImage(path):
  LLarr = np.loadtxt(path)
  LL = dict()
  LL['SOUTH'] = LLarr[0]
  LL['NORTH'] = LLarr[1]
  LL['WEST'] = LLarr[2]
  LL['EAST'] = LLarr[3]
  rot = np.loadtxt(path.replace('.llbbox', '.rotation'))
  LL['ROTATION'] = float(rot)
  return LL
