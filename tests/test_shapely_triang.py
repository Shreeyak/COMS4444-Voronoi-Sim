# import shapely
import numpy as np
from typing import Tuple

import shapely.errors
from shapely.geometry import MultiPoint, MultiLineString
from shapely.ops import triangulate
from matplotlib import pyplot
from descartes.patch import PolygonPatch
from shapely_figures import SIZE, set_limits, plot_coords, plot_bounds, plot_line, BLUE

import warnings
warnings.filterwarnings("ignore", category=shapely.errors.ShapelyDeprecationWarning)


class Unit:
    def __init__(self, player: int, pos: Tuple):
        """The unit of each player. New unit spawns every N days. Can be moved in any direction by 1 km"""
        assert 0 <= player < 4

        self.player = int(player)
        self.pos = pos
        self.status = 1  # 1 = alive, 0 = dead

units = [
    # Corners
    Unit(0, (0.5, 0.5)),
    Unit(1, (0.5, 9.5)),
    Unit(2, (9.5, 9.5)),
    Unit(3, (9.5, 0.5)),
    # Middle
    Unit(0, (5.7, 5.7)),
    Unit(2, (5.3, 5.3))
]

# Construct 2 lists for passing to shapely/scipy for triangulation
pts = [x.pos for x in units]
player_ids = [x.player for x in units]


# Shapely Triangulate
pts = MultiPoint(pts)
edges = triangulate(pts, edges=True)

print(type(edges))
print(type(edges[0]))

for idx, line in enumerate(edges):
    print(f"line {idx}: ", line.xy)

# Plot the Triangulation
fig = pyplot.figure(1, figsize=SIZE, dpi=90)
fig.set_frameon(True)
ax = fig.add_subplot(111)

for line in edges:
    plot_coords(ax, line)
    plot_line(ax, line, alpha=0.7, color=BLUE)

set_limits(ax, -1, 11, -1, 11)
pyplot.show()






