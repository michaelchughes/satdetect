from optparse import OptionParser
import os
import cPickle
import numpy as np
import pdb


def LoadDumps(inputpath):
	if (not os.path.isdir(inputpath)):
		raise Exception("Input is not a directory")

	for dfile in os.listdir(inputpath):
		if os.path.isfile(dfile) and dfile.endswith('.dump'):
			fd = open(dumpfile)
			dumpdata = cPickle.load(fd)
			fd.close()
			ltype = labelType
			colordic[dfile] = dumpdata

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option('-i', '--inputpath',  type=str, dest="inputpath", help='path to load dumped data from')
	(options, args) = parser.parse_args()
	inputpath = options.inputpath
	
	colordic, colordesc = loadDumps(inputpath)
	pdb.set_trace()


'''
def labelType(name):
	nameWithoutExtension = os.path.splitext(name)[0]
	ltype = nameWithoutExtension.split('_')[-1]
	if (not (ltype == 'tree' or ltype == 'soil' or ltype == 'huts')):
		ltype = 'unlabelled'
	return ltype

def loadDumps(inputpath, colordic={}, graydic={}, colordesc={}, graydesc={}):
	if (not os.path.isdir(inputpath)):
		raise Exception("Input is not a directory")

	for dfile in os.listdir(inputpath):
		recursive_path = inputpath + dfile + "/"
		dumpfile = inputpath + dfile

		if (os.path.isfile(dumpfile)):
			
			fd = open(dumpfile)
			dumpdata = cPickle.load(fd)
			fd.close()
			ltype = labelType(dfile)
			print dumpfile + str(dumpdata.shape)

			if (len(dumpdata.shape) == 4):
				if (ltype not in colordic.keys()):
					colordesc[dumpfile] = (dfile, ltype, 0, dumpdata.shape[0])	
					colordic[ltype] = dumpdata
				else:
					colordesc[dumpfile] = (dfile, ltype, colordic[ltype].shape[0], dumpdata.shape[0])
					colordic[ltype] = np.append(colordic[ltype], dumpdata, 0)

			elif (len(dumpdata.shape) == 3):
				if (ltype not in graydic.keys()):
					graydesc[dumpfile] = (dfile, ltype, 0, dumpdata.shape[0])	
					graydic[ltype] = dumpdata
				else:
					graydesc[dumpfile] = (dfile, ltype, graydic[ltype].shape[0], dumpdata.shape[0])
					graydic[ltype] = np.append(graydic[ltype], dumpdata, 0)

			else:
				raise Exception("Data of unknown type")

		if (os.path.isdir(recursive_path)):
			loadDumps(recursive_path, colordic, graydic, colordesc, graydesc)

	return colordic, colordesc, graydic, graydesc


def loadDumpsFromDirectories(inputpath):
	if not os.path.isdir(inputpath):
		raise Exception("inputpath: " + inputpath + " is not a dir")
	colordic, colordesc, graydic, graydesc = loadDumps(inputpath)

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option('-i', '--inputpath',  type=str, dest="inputpath", help='path to load dumped data from')
	(options, args) = parser.parse_args()
	inputpath = options.inputpath
	
	colordic, colordesc, graydic, graydesc = loadDumps(inputpath)
	pdb.set_trace()
	a = 1
'''