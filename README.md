Perlin noise (explained)
========================

Rationale
---------
This annotated implementation of Ken Perlin's k-dimensional noise
is meant to serve as an easy-to-understand companion guide to a more in-depth
treatment of the algorithm. I wrote this code and the accompanying comments
mostly for myself in order to better understand how a gradient noise algorithm
like Perlin noise works. What better way to attempt to understand a topic
than to implement it and explain it?

Requirements
------------
Only two packages are required `numpy` and `Pillow`. They can be installed
via `pip`:
* `pip install numpy`
* `pip install Pillow`

`imagemagick` was also used to create the animations.

Examples
--------
### Raw Perlin noise ###
![Raw Perlin noise](http://i.imgur.com/4Pf7rEa.gif)

### Fractal noise ###
![Fractal noise](http://i.imgur.com/IJvpqAa.gif)

## Ridged, fractal noise ###
![Ridged, fractal noise](http://i.imgur.com/exkVnu9.gif)
