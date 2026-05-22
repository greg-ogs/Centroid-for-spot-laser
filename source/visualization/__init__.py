"""Plotting and format-conversion utilities out of the main pipeline.

Modules
-------
* Module `source.visualization.image_as_3d`  -- 3-D intensity surfaces.
* Module `source.visualization.times_graph`  -- benchmark-CSV plotting.
* Module `source.visualization.to_eps`       -- raster -> EPS converter.
"""
from source.visualization.image_as_3d import ImageAs3D
from source.visualization.times_graph import TimesGraph
from source.visualization.to_eps import convert as raster_to_eps

__all__ = ["ImageAs3D", "TimesGraph", "raster_to_eps"]
