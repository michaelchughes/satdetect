satdetect : python object recognition from satellite images.

This Python package provides an open-source pipeline for training and evaluating object recognition systems for humanitarian purposes.

## Contact
mike@michaelchughes.com

## Usage

To train a detector
```
python -m satdetect.TrainDetector path/to/trainimage.jpg path/for/output/ [--OPTIONS]
```

To run a trained detector on new imagery
```
python -m satdetect.RunDetector path/to/image.jpg path/to/classifier.dump [--OPTIONS]
```

For more info, see TODO.

## Dependencies

We require these python packages

* numpy
* scipy
* sci-kit learn
* sci-kit image
* joblib (for high-quality load/dump of Python objects to disk)

## Repository Organization

* satdetect/ : module code
* tests/ : unit-tests for module code
* kmlutil/ : utilities for converting between KML object annotations and flat files
* assets/ : template files for generating reports and KML visualizations
