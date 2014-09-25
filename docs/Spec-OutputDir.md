## Output directory specification

This doc defines what goes in the output directory.

### Before TrainDetector

We simply need to write down a path where we'd like results saved. This directory (or parents) need not exist. However, it does need to be a valid absolute path that could be readable and writeable.

### After TrainDetector

After calling TrainDetector, this output directory will look like

```
$ ls /path/to/outdir/
hog-nBins9-pxPerCell5/
trained-classifiers/
detector-results/
scene1_tile01.npz
scene1_tile02.npz
…
scene1_tile04.npz

```

Here, the many .npz files represent image tiles, and follow the [Tile Image Spec](Spec-TileImage.md).

The several subdirectories are described below.

### Subdirectories within `outpath`

* hog/

The hog/ directory will contain pre-extracted features for each tile in each scene.

* trained-classifieres/

The trained-classifiers/ directory will contain the trained detector, as a dump of the Python object.

* detection-results/

This won't exist until after running the detector, but afterwards will store output of a detection run, such as bbox-annotated jpeg images that show performance at a given decision threshold.

