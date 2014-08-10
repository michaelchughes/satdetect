'''
ParseKMLIntoXObjects.py

Command-line script for converting KML file spec into flat-file format

Examples
--------
$$ python ParseKMLIntoXObjects.py \
              /data/tukuls/Sudan/raw/doc.kml \ # input kml spec
              /data/tukuls/Sudan/xobjects/     # output location


'''
import os
import shutil
import argparse
from collections import defaultdict
import numpy as np
from xml.dom.minidom import parseString
from scipy.misc import imread

import fastkml
from shapely.geometry import LinearRing

def ReadScenesWithXObjectsFromKML(kmlfilepath):
  ''' Read all relevant info from KML file into a list of Scene dicts

      Returns
      --------
      Scenes : list of dicts, where each dict has fields
               * items : list of xobjects
  '''
  Scenes = ReadSceneInfoFromKML(kmlfilepath)

  kmlobj = fastkml.kml.KML()
  kmlobj.from_string('\n'.join(open(kmlfilepath,'r').readlines()))
  masterdoc = [f for f in kmlobj.features()][0]
  for ff in masterdoc.features():
    print 'FOLDER: ',ff.name
        
    for fff in ff.features():
      fff.name = fix_itemname(fff.name)
      print '  ',fff.name
      if fff._geometry is not None:
       AddXObjectBBoxToScene(Scenes, fff.name, fff.geometry)

  return Scenes

def WriteScenesWithXObjectsToFile(Scenes, outpath):
  ''' Write the contents of the Scenes list to image/flattext files in outpath

     Disk output
     -----------
     each scene in Scene results in:
     * <sceneName>.jpg
     * <sceneName>.llbbox
     * <sceneName>_<objType>.pxbbox 

     Returns
     --------
     None
  '''
  for sID, Scene in enumerate(Scenes):
    ## Write latitude/longitude bbox for entire image
    with open(os.path.join(outpath, Scene['name'] + '.llbbox'),'w') as f:
      f.write(prettyprint_latlong_bbox(Scene) + '\n')

    ## Write latitude/longitude bbox for all huts
    with open(os.path.join(outpath, Scene['name'] + '_huts.llbbox'),'w') as f:
      for item in Scene['items']:
        f.write(prettyprint_latlong_bbox(item) + '\n')

    ## Make copy of image source file, for easy access
    destpath = os.path.join(outpath, Scene['name'] + '.jpg')
    shutil.copyfile(Scene['imgfile'], destpath)

    ## Read in the image (as grayscale) to obtain correct pixel sizes
    img = imread(destpath, flatten=True) 
    H, W = img.shape

    ## Write the pixel bbox of each entire Scene image
    with open(os.path.join(outpath, Scene['name'] + '.pxbbox'),'w') as f:
      pxbbox = convert_latlong_to_pixel_bbox(H, W, Scene, Scene)
      f.write(prettyprint_pixel_bbox(pxbbox) + '\n')

    ## Write the pixel bbox of each hut
    with open( os.path.join(outpath, Scene['name'] + '_huts.pxbbox'),'w') as f:
      for item in Scene['items']:
        pxbbox = convert_latlong_to_pixel_bbox(H, W, Scene, item)
        f.write(prettyprint_pixel_bbox(pxbbox) + '\n')

def ReadSceneInfoFromKML(kmlfilepath):
  ''' Read KMLFile and return list of Scene info (imgfile, latlong bbox)

      Returns
      --------
      Scenes : list of dicts, each with fields
               imgfile : string path to jpg image
               items : list (empty) for to-be-filled-in objects
               latMin : float
               latMax : float
  '''
  xmlstring = ''.join(open(kmlfilepath,'r').readlines())
  dom = parseString(xmlstring)
  sceneList = dom.getElementsByTagName('GroundOverlay')
  Scenes = list()
  for s in sceneList:
    imgfile = str(s.getElementsByTagName('href')[0].firstChild.nodeValue)
    fullpath = os.path.sep.join(kmlfilepath.split(os.path.sep)[:-1])
    fullpath = os.path.join(fullpath, imgfile)

    latMax = float(s.getElementsByTagName('north')[0].firstChild.nodeValue)
    latMin = float(s.getElementsByTagName('south')[0].firstChild.nodeValue)
    lonMax = float(s.getElementsByTagName('east')[0].firstChild.nodeValue)
    lonMin = float(s.getElementsByTagName('west')[0].firstChild.nodeValue)
    curdict = dict(imgfile=fullpath, name=GetSceneName(imgfile),
                                    latMax=latMax, latMin=latMin,
                                    lonMax=lonMax, lonMin=lonMin,
                                    items=list())
    Scenes.append(curdict)
  return Scenes

def AddXObjectBBoxToScene(Scenes, itemname, geom):
  ''' Append a dict of lat/long info for an object into a scene

      Returns
      --------
      None. Updates to Scenes list happen in-place. 
  '''
  lonMin, latMin, lonMax, latMax = geom.bounds
  itemdict = dict(name=itemname, latMax=latMax, latMin=latMin,
                                 lonMax=lonMax, lonMin=lonMin,
                  type=GetObjectTypeFromName(itemname),               
                  )
  ## Loop over existing scenes, and
  ## add the object to the one that contains its lat/long bounding box
  for s in Scenes:
    if lonMin >= s['lonMin'] and lonMax <= s['lonMax']:
      if latMin >= s['latMin'] and latMax <= s['latMax']:
        s['items'].append(itemdict)

def GetSceneName(imgfilename):
  ''' Return filebasename (without extension), which is the unique ID for scene

      Example
      --------
      >> GetSceneName('files/scene1.jpg')
      scene1
  '''
  basename = imgfilename.split(os.path.sep)[-1]
  return basename.split('.')[0]

def GetObjectTypeFromName(itemname):
  ''' Return standard lowercase string describing type of named object

      Example
      ---------
      >> GetObjectTypeFromName('tukul50')
      'hut'
  '''
  if itemname.count('tukle') or itemname.count('tukul') \
                             or itemname.count('hut'):
    return 'hut'
  print itemname
  #raise ValueError('UNKNOWN TYPE:' + itemname)
  return 'hut'

def fix_itemname(itemname):
  ''' Fix string name for scene or object to avoid whitespaces and capitals

      Example
      ---------
      >> fix_itemname('Tukle 33')
      'tukle33'
  '''
  itemname = itemname.split('_')[0]
  itemname = itemname.replace(' ', '')
  return itemname.lower()


def convert_latlong_to_pixel_bbox(H, W, Scene, item):
  ''' Convert latitude/longitude bbox to a pixel bbox, using Scene bounds
  '''
  yScale = float(H) / (Scene['latMax'] - Scene['latMin'])
  xScale = float(W) / (Scene['lonMax'] - Scene['lonMin'])
  def lat2y(lat):
    return H - np.round(yScale * (lat - Scene['latMin']))
  def lon2x(lon):
    return np.round(xScale * (lon - Scene['lonMin']))
  return dict(xMin=lon2x(item['lonMin']), xMax=lon2x(item['lonMax']),
              yMin=lat2y(item['latMax']), yMax=lat2y(item['latMin'])
             )

def prettyprint_latlong_bbox(SDict):
  return '%.16f %.16f %.16f %.16f' % (SDict['latMin'], SDict['latMax'],
                                      SDict['lonMin'], SDict['lonMax'])

def prettyprint_pixel_bbox(SDict):
  return '%.0f %.0f %.0f %.0f' % (SDict['yMin'], SDict['yMax'],
                                  SDict['xMin'], SDict['xMax'])

def calcResolutionFromFiles(llfile, pxfile):
  LBox = np.loadtxt(llfile)
  PBox = np.loadtxt(pxfile)
  Hpx = PBox[1]
  Wpx = PBox[3]
  Hm = calcDistBetweenLatLongPair(LBox[0], LBox[2], LBox[1], LBox[2])
  Wm = calcDistBetweenLatLongPair(LBox[0], LBox[2], LBox[0], LBox[3])
  print "H x W (pixels): %6.1f x %6.1f" % (Hpx, Wpx)
  print "H x W (meters): %6.1f x %6.1f" % (Hm, Wm)
  print "H x W (m/px):   %6.2f x %6.2f" % (Hm/Hpx, Wm/Wpx)

def determineMetersPerPixelResolution(SDict, H, W):
  pass

def calcDistBetweenLatLongPair(latA, lonA, latB, lonB):
  ''' Calculate dist between two pairs of lat/long coordinates

      Returns
      -------
      distance (in meters)
  '''
  ## Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
  R_earth = 6378.1 * 1000 
  #R_earth = 6371

  phiA = degToRad(latA)
  phiB = degToRad(latB)
  deltaLat = degToRad(latB - latA)
  deltaLon = degToRad(lonB - lonA)

  a = np.sin(deltaLat/2)**2 \
      + np.cos(phiA) * np.cos(phiB) * np.sin(deltaLon/2)**2
  c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a))
  return R_earth * c

def degToRad(deg):
  return np.pi / 180 * deg

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('kmlfilepath')
  parser.add_argument('outpath')
  args = parser.parse_args()


  Scenes = ReadScenesWithXObjectsFromKML(args.kmlfilepath)
  WriteScenesWithXObjectsToFile(Scenes, args.outpath)