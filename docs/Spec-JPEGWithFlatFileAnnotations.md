## JPEG with flat annotations Specification

This doc defines a file format for object annotations.

For a given dataset, we assume that all JPEG images and annotations are in the same directory. The internal structure looks like...

```
/path/to/dataset/
* sudan-scene1.jpg
* sudan-scene1.pxbbox
* sudan-scene1_huts.pxbbox
* sudan-scene2.jpg
* sudan-scene2.pxbbox
* sudan-scene2_huts.pxbbox
```
Each JPEG file named uniquename.jpg is paired with two files: 

* uniquename.pxbbox
* uniquename_huts.pxbbox

### JPEG image

This is simply a valid 8-bit JPEG image. Nothing special.

### sceneXX.pxbbox

This file is one line of plain text, defining the min/max pixel value of the whole image.

For a 640x480 image, we have

```
0 640 0 480
```

In general, the pxbbox format (each line) is

```
ymin ymax xmin xmax
```

### sceneXX_huts.pxbbox

This file is a plain-text file where each line gives a bounding box for an annotated hut object.

For one hut centered at (15,20) and another at (300,400), we have

```
10 20 15 25
290 310 390 410
```
