from optparse import OptionParser
from satdetect.featextract import WindowExtractor

'''
	Sample run:
	python predictImageLabels.py --imgpath ~/documents/projects/tukuldata/datasets/all_images_and_labels --outpath ~/documents/projects/tukuldata/datasets --configpath ~/documents/projects/tukuldata/datasets/labels.cfg
'''

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('--imgpath', type=str, dest="imgpath", help='path(s) to load testing images from')
  parser.add_option('--outpath', type=str, dest="outpath", help='path where results are saved')
  parser.add_option('--configpath', type=str, dest="configpath", help="path for config files")  
  parser.add_option('--load')
  (options, args) = parser.parse_args()
  
  @WindowExtractor.extractPatches(options.imgpath, options.configpath, options.outpath)

  img, patches, indexes, stride = WindowExtractor.extractPatches(imgpath, configpath, outpath, window_shape=(25,25), for_training=0):

  #Test the patches using a classifier
  preds = np.zeros(patches.shape)

  #Reconstruct the images
  reconstructPathces(img, indexes, preds, imgpath, configpath, outpath, stride)