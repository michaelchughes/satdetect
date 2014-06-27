'''
'''
import argparse

import ArgParseUtil
import satdetect.datasetbuilder as datasetbuilder

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('setName', type=str)
  parser.add_argument('--window_shape', type=str, default='25,25')
  parser.add_argument('--nNegSamples', type=int, default=10)
  parser.add_argument('--nPosCopies', type=int, default=2)
  args = parser.parse_args()
  ArgParseUtil.parse_window_shape_inplace(args)

  datasetbuilder.make_and_save_window_dataset(args.setName, args.window_shape,
            nNegSamples=args.nNegSamples, nPosCopy=args.nPosCopies)
  
  

if __name__ == '__main__':
  main()
