'''
ParseKMLIntoFlatFiles.py

Command-line script for converting KML annotations into flat-file format

Examples
--------
$$ python ParseKMLIntoFlatFiles.py \
          /path/to/doc.kml \ # input kml spec
          /path/to/output/subdir/     # output location


'''
import os
import shutil
import argparse
from collections import defaultdict
import numpy as np
from xml.dom.minidom import parseString
from scipy.misc import imread

import fastkml


########################################################### Read objects
###########################################################
def ReadObjectsFromKMLIntoScenes(kmlfilepath, Scenes):
  ''' Read all relevant info from KML file into a list of Scene dicts

      Returns
      --------
      Scenes 
        list of dicts, where each dict has fields
        * items : list of objects
  '''
  print '<<<<<<<<<<<<<<<<<<<<<<<<< This is ReadObjectsFromKML'
  uid = 1
  xmlstring = ''.join(open(kmlfilepath,'r').readlines())
  dom = parseString(xmlstring)
  markList = dom.getElementsByTagName('Placemark')
  for mark in markList:
    ## Get custom KML snippet just for this placemark
    rawXMLString = "<kml>\n" + mark.toxml() + "\n</kml>"

    ## Remove non-native KML tags that start with "gx"
    markXMLString = ''
    for line in rawXMLString.split('\n'):
      if line.count('gx:') == 0:
        markXMLString += line + '\n'

    ## Convert into custom "fastkml" object, which is geometry aware
    markobj = fastkml.kml.KML()
    try:
      markobj.from_string(markXMLString)
    except Exception as e:
      print markXMLString
      raise e
    markFeat = [f for f in markobj.features()][0]

    try:
      type = GetObjectTypeFromName(markFeat.name)
    except ValueError as e:
      # Skip placemarks that don't annotate objects of interest
      continue 

    lonMin, latMin, lonMax, latMax = markFeat.geometry.bounds
    objDict = dict(uid=uid,
                   latMax=latMax, latMin=latMin,
                   lonMax=lonMax, lonMin=lonMin,
                   type=type,               
                   )
        
    didMatch = False
    ## Loop over existing scenes, and
    ## Add this object to the one that contains its lat/long bounding box
    for s in Scenes:
      if lonMin >= s['lonMin'] and lonMax <= s['lonMax']:
        if latMin >= s['latMin'] and latMax <= s['latMax']:
          s[type + '_objects'].append(objDict)
          print '   Object:', markFeat.name, ' ->', s['name'], type
          didMatch = True
          break
    if not didMatch:
      print 'NO SCENE CONTAINS BBOX FOR OBJECT', markFeat.name
      continue
    uid += 1
    
  return Scenes

def getAllObjectTypes():
  return ['huts', 'possibles', 'razed']

def GetObjectTypeFromName(itemname):
  ''' Return standard lowercase string describing type of named object

      Example
      ---------
      >> GetObjectTypeFromName('tukul50')
      'huts'
  '''
  itemname = itemname.lower()
  if itemname.count('outer') or itemname.count('razed'):
    return 'razed'
  elif itemname.count('inner'):
    raise ValueError('UNKNOWN TYPE:' + itemname)
  if itemname.count('possible'):
    return 'possibles'
  elif itemname.count('tukle') or itemname.count('tukul') \
                             or itemname.count('hut'):
    return 'huts'
  raise ValueError('UNKNOWN TYPE:' + itemname)

########################################################### Read scenes
###########################################################
def ReadScenesFromKML(kmlfilepath):
  ''' Read KMLFile and return list of Scene info (imgfile, latlong bbox)

      Returns
      --------
      Scenes : list of dicts, each with fields
               imgfile : string path to jpg image
               items : list (empty) for to-be-filled-in objects
               latMin : float
               latMax : float
  '''
  print '<<<<<<<<<<<<<<<<<<<<<<<<< This is ReadScenesFromKML'
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

    rotElements = s.getElementsByTagName('rotation')
    if len(rotElements) > 0:
      rot = float(rotElements[0].firstChild.nodeValue)
    else:
      rot = 0.0

    curdict = dict(imgfile=fullpath, name=GetSceneName(imgfile),
                                    rotation=rot,
                                    latMax=latMax, latMin=latMin,
                                    lonMax=lonMax, lonMin=lonMin,
                                    )
    for objType in getAllObjectTypes():
      curdict[objType + '_objects'] = list()

    print ' Scene: ', curdict['name']
    print '    ', imgfile
    Scenes.append(curdict)
  return Scenes


def GetSceneName(imgfilename):
  ''' Return filebasename (without extension), which is the unique ID for scene

      Example
      --------
      >> GetSceneName('files/scene1.jpg')
      scene1
  '''
  basename = imgfilename.split(os.path.sep)[-1]
  return basename.split('.')[0]

########################################################### Write Flat Files
###########################################################
def WriteScenesAndObjectsToFlatFiles(Scenes, outpath):
  ''' Write the contents of the Scenes list to flat plain-text files in outpath

     Disk output
     -----------
     Each scene results in:
     * <sceneName>.jpg
     * <sceneName>.llbbox
     * <sceneName>.pxbbox
     * <sceneName>_<objType>.llbbox 
     * <sceneName>_<objType>.pxbbox 

     Returns
     --------
     None
  '''
  print '<<<<<<<<<<<<<<<<<<<<<<<<< This is WriteScenesAndObjectsToFlatFiles'

  for sID, Scene in enumerate(Scenes):
    ## Make copy of image source file, for easy access
    destpath = os.path.join(outpath, Scene['name'] + '.jpg')
    try:
      shutil.copyfile(Scene['imgfile'], destpath)
    except Exception as e:
      if str(e).count("same file"):
        pass
      else:
        raise e

    ## Read in image as grayscale to obtain correct pixel sizes
    img = imread(destpath, flatten=True) 
    H, W = img.shape

    rotfpath = os.path.join(outpath, Scene['name'] + '.rotation')
    with open(rotfpath, 'w') as f:
      f.write('%.5f' % (Scene['rotation']) + '\n')

    ## Write latitude/longitude bbox for entire image
    sllbboxfpath = os.path.join(outpath, Scene['name'] + '.llbbox')
    with open(sllbboxfpath,'w') as f:
      f.write(prettyprint_latlong_bbox(Scene) + '\n')

    ## Write the pixel bbox of each entire Scene image
    spxbboxfpath = os.path.join(outpath, Scene['name'] + '.pxbbox')
    with open(spxbboxfpath,'w') as f:
      pxbbox = convert_latlong_to_pixel_bbox(H, W, Scene, Scene)
      f.write(prettyprint_pixel_bbox(pxbbox) + '\n')

    print '=============', Scene['name']
    resStr = calcResolutionForScene(sllbboxfpath, spxbboxfpath)
    print resStr
    with open(os.path.join(outpath, Scene['name'] + '.resolution'),'w') as f:
      f.write(resStr +'\n')

    ## Write latitude/longitude bbox and pixel bbox for all hut objects
    for objType in getAllObjectTypes():
      objbasename = Scene['name'] + '_' + objType
      objllbboxpath = os.path.join(outpath, objbasename + '.llbbox')
      with open(objllbboxpath,'w') as f:
        for obj in Scene[objType + '_objects']:
          if obj['type'] == objType:
            f.write(prettyprint_latlong_bbox(obj) + '\n')

      objpxbboxpath = os.path.join(outpath, objbasename + '.pxbbox')
      with open(objpxbboxpath,'w') as f:
        for obj in Scene[objType + '_objects']:
          if obj['type'] == objType:
            pxbbox = convert_latlong_to_pixel_bbox(H, W, Scene, obj)
            f.write(prettyprint_pixel_bbox(pxbbox) + '\n')

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

def convert_pixel_to_latlong_bbox(H, W, Scene, item):
  ''' Convert pixel bbox to latlong bbox, using Scene bounds
  '''
  yScale = float(H) / (Scene['latMax'] - Scene['latMin'])
  xScale = float(W) / (Scene['lonMax'] - Scene['lonMin'])
  def y2lat(y):
    return Scene['latMin'] + float(H - y)/yScale
  def x2lon(x):
    return Scene['lonMin'] + float(x)/xScale
  return dict(lonMin=x2lon(item['xMin']), lonMax=x2lon(item['xMax']),
              latMin=y2lat(item['yMax']), latMax=y2lat(item['yMin'])
             )

def prettyprint_latlong_bbox(SDict):
  return '%.16f %.16f %.16f %.16f' % (SDict['latMin'], SDict['latMax'],
                                      SDict['lonMin'], SDict['lonMax'])

def prettyprint_pixel_bbox(SDict):
  return '%.0f %.0f %.0f %.0f' % (SDict['yMin'], SDict['yMax'],
                                  SDict['xMin'], SDict['xMax'])

########################################################### Calculations
###########################################################

def calcResolutionForScene(llfile, pxfile, units='all'):
  LBox = np.loadtxt(llfile)
  PBox = np.loadtxt(pxfile)
  Hpx = PBox[1]
  Wpx = PBox[3]
  Hm = calcDistBetweenLatLongPair(LBox[0], LBox[2], LBox[1], LBox[2])
  Wm = calcDistBetweenLatLongPair(LBox[0], LBox[2], LBox[0], LBox[3])
  res_px ="%7d x %7d (pixels)" % (Hpx, Wpx)
  res_m = "%7.1f x %7.1f (meters)" % (Hm, Wm)
  res_mpx = "%7.2f x %7.2f (m/px)" % (Hm/Hpx, Wm/Wpx)
  return '\n'.join([res_px, res_m, res_mpx])

def calcDistBetweenLatLongPair(latA, lonA, latB, lonB):
  ''' Calculate dist between two pairs of lat/long coordinates

      Returns
      -------
      distance (in meters)
  '''
  ## Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
  R_earth = 6378.1 * 1000 

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

########################################################### Run As Script
###########################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('kmlfilepath')
  parser.add_argument('outpath')
  args = parser.parse_args()


  Scenes = ReadScenesFromKML(args.kmlfilepath)
  Scenes = ReadObjectsFromKMLIntoScenes(args.kmlfilepath, Scenes)

  WriteScenesAndObjectsToFlatFiles(Scenes, args.outpath)
  