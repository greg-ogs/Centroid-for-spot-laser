"""Plotting and format-conversion utilities.

Modules
-------
* :mod:`source.visualization.image_as_3d`  -- 3-D intensity surfaces.
* :mod:`source.visualization.times_graph`  -- benchmark-CSV plotting.
* :mod:`source.visualization.to_eps`       -- raster -> EPS converter.

For convenience the public surface is re-exported:

    from source.visualization import TimesGraph, ImageAs3D, raster_to_eps
"""
from source.visualization.image_as_3d import ImageAs3D
from source.visualization.times_graph import TimesGraph
from source.visualization.to_eps import convert as raster_to_eps

__all__ = ["ImageAs3D", "TimesGraph", "raster_to_eps"]
