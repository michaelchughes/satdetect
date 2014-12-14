from optparse import OptionParser
from satdetect.featextract import WindowExtractor
import os
'''
	Sample run:
	old: python createPatches.py --imgpath ~/documents/projects/tukuldata/datasets/Sudan-MakerAwat-20101219/sudanma-scene1.jpg  --outpath ~/documents/projects/extracted_patches/ --configpath ~/documents/projects/tukuldata/datasets/labels.cfg
	new: python createPatches.py --imgpath ~/documents/projects/tukuldata/datasets/cropped/  --outpath ~/documents/projects/extracted_patches/ --configpath ~/documents/projects/tukuldata/datasets/labels.cfg
'''

if __name__ == "__main__":
  '''
  parser = OptionParser()
  parser.add_option('--imgpath', type=str, dest="imgpath", help='path(s) to load training images from')
  parser.add_option('--outpath', type=str, dest="outpath", help='path where results are saved')
  parser.add_option('--configpath', type=str, dest="configpath", help="path for config files")  

  (options, args) = parser.parse_args()
  
  WindowExtractor.extractPatches(options.imgpath, options.configpath, options.outpath)
  '''

  parser = OptionParser()
  parser.add_option('--inputpath', type=str, dest="inputpath", help='path(s) to load training images from')
  parser.add_option('--outpath', type=str, dest="outpath", help='path where results are saved')
  parser.add_option('--configpath', type=str, dest="configpath", help="path for config files")  

  (options, args) = parser.parse_args()
  inputpath = options.inputpath

  for filename in os.listdir(inputpath):
  	imgpath = inputpath + "/" + filename
  	if (imgpath.endswith('.jpg') and not filename.endswith("_labels.jpg")):
		WindowExtractor.extractPatches(imgpath, options.configpath, options.outpath)
