# **satdetect**: Python module


## To Train a Detector

```
python -m satdetect.TrainDetector <imgsrcpath> <outpath> [TRAIN OPTIONS]
```
#### EXAMPLE

```

```

#### INPUT
* `imgsrcpath` : path to annotated JPEG imagery for training detector

This can be a path to a single JPEG file, or a pattern for many JPEG files, using '*' wildcards as appropriate (which will trigger a call to `glob`).

See [JPEG with Annotations Spec](Spec-JPEGWithFlatFileAnnotations.md) for details about the format of the annotations (which must be in the same directory as the provided JPEG).


* `outpath` : directory for storing intermediate and final results

See [Output Directory Spec](Spec-OutputDir.md) for details. 

* [TRAIN OPTIONS]

  * --cname : string name of classifier (logistic,svm-linear)
  * --feat : string name of feature extractor (just 'hog' for now)
  
#### OUTPUT

Trained classifier is saved in the provided output directory.


## To Run the detector on new data

```
python -m satdetect.RunDetector <imgsrcpath> <outpath> <cpath> [DETECT OPTIONS]
```

#### EXAMPLE


#### INPUT

* `imgsrcpath` : path to annotated JPEG imagery for training detector

This can be a path to a single JPEG file, or a pattern for many JPEG files, using '*' wildcards as appropriate (which will trigger a call to `glob`).

See [JPEG with Annotations Spec](Spec-JPEGWithFlatFileAnnotations.md) for details about the format of the annotations (which must be in the same directory as the provided JPEG).


* `outpath` : directory for storing intermediate and final results

See [Output Directory Spec](Spec-OutputDir.md) for details. 


* `cpath` : full path to .dump file containing stored classifier

See `Classifier.saveClassifier` for details.

#### OUTPUT

TODO