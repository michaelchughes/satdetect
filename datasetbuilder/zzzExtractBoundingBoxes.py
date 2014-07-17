import os
import shutil
import argparse
from collections import defaultdict
import numpy as np
from xml.dom.minidom import parseString
from scipy.misc import imread

import fastkml
from shapely.geometry import LinearRing

def extract_scenes_from_kmlfile(kmlfilepath):
  xmlstring = ''.join(open(kmlfilepath,'r').readlines())
  dom = parseString(xmlstring)
  sceneList = dom.getElementsByTagName('GroundOverlay')
  Scenes = list()
  for s in sceneList:
    imgfile = str(s.getElementsByTagName('href')[0].firstChild.nodeValue)
    latMax = float(s.getElementsByTagName('north')[0].firstChild.nodeValue)
    latMin = float(s.getElementsByTagName('south')[0].firstChild.nodeValue)
    lonMax = float(s.getElementsByTagName('east')[0].firstChild.nodeValue)
    lonMin = float(s.getElementsByTagName('west')[0].firstChild.nodeValue)
    curdict = dict(imgfile=imgfile, latMax=latMax, latMin=latMin,
                                    lonMax=lonMax, lonMin=lonMin,
                                    items=list())
    Scenes.append(curdict)
  return Scenes

def do_keep_name(name, targetnames):
  if targetnames is None or len(targetnames) == 0:
    return 1
  keepcount = np.sum([name.count(kname) for kname in targetnames])
  return keepcount > 0

def add_bbox_geom_to_scene(Scenes, itemname, geom):
   #(28.42068679214893, 9.66937681767244, 28.42075083235185, 9.669443933466559)     
  lonMin, latMin, lonMax, latMax = geom.bounds
  itemdict = dict(name=itemname, latMax=latMax, latMin=latMin,
                                    lonMax=lonMax, lonMin=lonMin,
                                    )        
  for s in Scenes:
    if lonMin >= s['lonMin'] and lonMax <= s['lonMax']:
      if latMin >= s['latMin'] and latMax <= s['latMax']:
        s['items'].append( itemdict)

def fix_itemname(itemname):
  itemname = itemname.split('_')[0]
  itemname = itemname.replace(' ', '')
  return itemname.lower()

def add_bbox_geom_to_kml_dict(KMLDict, foldername, itemname, geom, ns):

  p = fastkml.kml.Placemark(ns, None, itemname, None)
  p.geometry = geom.envelope
  KMLDict[foldername].append(p)

def create_kmlfile_from_dict(KMLDict, kmlref):
  kmlout = fastkml.kml.KML()
  doc = fastkml.kml.Document(kmlref.ns, None, None, None)
  for foldername in KMLDict:
    folder = fastkml.kml.Folder(kmlref.ns, foldername, foldername, None)
    for item in KMLDict[foldername]:
      folder.append(item)
    doc.append(folder)
  kmlout.append(doc)
  return kmlout

def parse_kml_into_bbox(kmlfilepath, targetnames):
  kmlobj = fastkml.kml.KML()
  kmlobj.from_string('\n'.join(open(kmlfilepath,'r').readlines()))

  Scenes = extract_scenes_from_kmlfile(kmlfilepath)
  KMLDict = defaultdict(lambda:list())
  masterdoc = [f for f in kmlobj.features()][0]
  for ff in masterdoc.features():
    print 'FOLDER: ',ff.name
    
    for fff in ff.features():
      if not do_keep_name(fff.name, targetnames):
        continue
      fff.name = fix_itemname(fff.name)
      print '  ',fff.name
      if fff._geometry is not None:
       add_bbox_geom_to_kml_dict(KMLDict, ff.name, 
                                          fff.name, fff.geometry, kmlobj.ns)

       add_bbox_geom_to_scene(Scenes, fff.name, fff.geometry)

  # create kmlfileobj from the dict
  kmlout = create_kmlfile_from_dict(KMLDict, kmlobj)
  return kmlout, Scenes

def convert_latlong_to_pixel_bbox(H, W, Scene, item):
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

def write_scenes_to_file(Scenes, kmlfilepath, root='/data/burners/set1/scenes/', tag='after'):
  for sID, Scene in enumerate(Scenes):
    with open( os.path.join(root, tag + str(sID+1) + '_image.llbbox'),'w') as f:
      f.write(prettyprint_latlong_bbox(Scene) + '\n')
    with open( os.path.join(root, tag + str(sID+1) + '_huts.llbbox'),'w') as f:
      for item in Scene['items']:
        f.write(prettyprint_latlong_bbox(item) + '\n')

    srcpath = os.path.sep.join(kmlfilepath.split(os.path.sep)[:-1])
    srcpath = os.path.join(srcpath, Scene['imgfile'])
    destpath = os.path.join(root, tag + str(sID+1) + '_image.jpg')
    shutil.copyfile(srcpath, destpath)
    img = imread(srcpath, flatten=True)
    H, W = img.shape
    with open( os.path.join(root, tag + str(sID+1) + '_image.pxbbox'),'w') as f:
      pxbbox = convert_latlong_to_pixel_bbox(H, W, Scene, Scene)
      f.write(prettyprint_pixel_bbox(pxbbox) + '\n')
    with open( os.path.join(root, tag + str(sID+1) + '_huts.pxbbox'),'w') as f:
      for item in Scene['items']:
        pxbbox = convert_latlong_to_pixel_bbox(H, W, Scene, item)
        f.write(prettyprint_pixel_bbox(pxbbox) + '\n')

def add_bbox_to_image(Im, BB): 
  for row in range(BB.shape[0]):
    yMin = BB[row,0]; yMax=BB[row,1]; xMin=BB[row,2]; xMax = BB[row,3]
    Im[yMin:yMax, xMin:xMax] = 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('kmlfilepath', 
          default='/data/burners/set1/kmz/after_20110311/doc.kml')
  parser.add_argument('--tag', default='after')
  parser.add_argument('--targetnames', default='')
  args = parser.parse_args()

  keepnames = args.targetnames.split(',')
  kmlout, Scenes = parse_kml_into_bbox(args.kmlfilepath, keepnames)
  write_scenes_to_file(Scenes, args.kmlfilepath, tag=args.tag)
  '''
  outfilepath = '/data/burners/set1/kmz/after_20110311/bbox.kml'
  with open(outfilepath,'w') as f:
    for line in kmlout.to_string().split('\n'):
      f.write(line + '\n')
  '''