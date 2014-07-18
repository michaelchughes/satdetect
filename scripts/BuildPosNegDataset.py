'''
'''
import argparse

import ArgParseUtil
import satdetect.datasetbuilder as datasetbuilder

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('datapath', type=str)
  parser.add_argument('--window_shape', type=str, default='25,25')
  parser.add_argument('--nNegSamples', type=int, default=10)
  parser.add_argument('--stride', type=int, default=4)
  args = parser.parse_args()
  ArgParseUtil.parse_window_shape_inplace(args)

  datasetbuilder.makeAndSaveAllStdObjects(args.datapath, 
                                          window_shape=args.window_shape,
                                          nNegSamples=args.nNegSamples,
                                          stride=args.stride)
  


if __name__ == '__main__':
  main()
