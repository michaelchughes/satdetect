from optparse import OptionParser
from satdetect import trainDetector

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('--imgpath', type=str, dest="imgpath", help='path(s) to load training images from')
  parser.add_option('--outpath', type=str, dest="outpath", help='path where results are saved')
  parser.add_option('--cname', type=str, default='logistic', help="name of classifier choices=['logistic', 'svm-linear', 'svm-rgb']")
  (options, args) = parser.parse_args()
  trainDetector(options.imgpath, options.outpath)
