'''
'''
import argparse

import ArgParseUtil
import satdetect.featextractor as featextractor

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('setName', type=str)
  parser.add_argument('featName', type=str)
  parser.add_argument('--window_shape', type=str, default='25,25')
  args = parser.parse_args()
  ArgParseUtil.parse_window_shape_inplace(args)

  if args.featName == 'hog':
    featextractor.extract_hog_features_for_dataset(args.setName, 
                                                 args.window_shape)
  elif args.featName == 'lbp':
    featextractor.extract_lbp_features_for_dataset(args.setName, 
                                                 args.window_shape)

if __name__ == '__main__':
  main()