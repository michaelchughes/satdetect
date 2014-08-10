import numpy as np
from satdetect.featextractor.HOGFeatExtractor import HOGFeatExtractor, CDatasetBuilder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

Q = np.load('/data/tukuls/Sudan/xfeatures/huts-25x25/scene1.npz')

Data = dict(imgpaths=['/data/tukuls/Sudan/xobjects/scene1.jpg'])
Data['PosBBox'] = [Q['PosBBox']]
Data['NegBBox'] = [Q['NegBBox']]

FX = HOGFeatExtractor()

print Data.keys()
#FX.transform(Data)
#print Data.keys()

pipeline = Pipeline([
    ('FeatExtract', HOGFeatExtractor()),
    ('CBuilder', CDatasetBuilder()),
    ('Classifier', LogisticRegression()),
])

from IPython import embed; embed()