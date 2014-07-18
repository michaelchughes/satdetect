'''
'''
import argparse

import ArgParseUtil
import satdetect.featextractor as featextractor

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('datapath', type=str)
  parser.add_argument('featName', type=str)
  parser.add_argument('--window_shape', type=str, default='25,25')
  args = parser.parse_args()
  ArgParseUtil.parse_window_shape_inplace(args)

  
  featextractor.extractRawPixelsForDataset(args.datapath, color='gray',
                                           window_shape=args.window_shape)

  try:
    featextractor.extractRawPixelsForDataset(args.datapath, color='rgb',
                                           window_shape=args.window_shape)
  except ValueError as e:
    print str(e)

  if args.featName == 'hog':
    featextractor.extractHOGForDataset(args.datapath)

if __name__ == '__main__':
  main()