

def parse_window_shape_inplace(args):
  shp = args.window_shape.split(',')
  if len(shp) != 2:
    raise ValueError('window_shape needs len 2')
  args.window_shape = (int(shp[0]), int(shp[1]))
