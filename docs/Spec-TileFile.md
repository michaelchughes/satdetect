## Tile file specification

Satellite images are often too big to process all at once, especially with a sliding window approach.  

To handle large images, we divide them into *tiles* of predefined size (something like 500x500 pixels), which we can handle easily. 

Currently, the tile division code will try to avoid making very small tiles. For example, if when dividing we'd make one tile of size 500x500 and another of 500x20, we'd combine it into one tile of size 500x520. 

### Relevant Code

See `WindowExtractor.py`

###  Specification

Each source image, `scene1.jpg`, is subdivided into N tiles with names `scene1_tile01`, `scene1_tile02`, …

These tile files are saved within [outpath](Spec-OutputDir.md).

Each file `scene1_tile01.npz` will have fields

* ColorIm : 3D array, RGB image data
* GrayIm : 2D array, gray-scale image data
* WIm : 3D array, shape nWindow x window_shape, window image data
* TileBBox : 2D array, shape nWindow x 4, defines the bbox of each window relative to tile coordinates

* ImBBox : 2D array, shape nWindow x 4, defines the bbox of each window relative to whole-image coordinates
* Y : 1D array, size nWindow, labels each window as positive or not
* imgpath : string path of the source image for this scene