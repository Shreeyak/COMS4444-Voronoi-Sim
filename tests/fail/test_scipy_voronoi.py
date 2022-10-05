"""FAIL: Has -1 for infinite regions. Need to manually find the point of intersection with the boundaries and re-construct
the polygons. Pain in the arse.

Option: https://gist.github.com/pv/8036995  (adds remote points - looks like causes error due to duplicate pts)
        Article: https://www.daniweb.com/programming/computer-science/tutorials/520314/how-to-make-quality-voronoi-diagrams

Not Option: plot_voronoi_2d doesn't work, because it doesn't actually close the polyons, just clips plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
mpl.use('TkAgg')  # For macOS. Change engine.


rng = np.random.default_rng()
points = rng.random((10,2))


vor = Voronoi(points)

fig = voronoi_plot_2d(vor, show_vertices=True, line_colors='orange',
                line_width=2, line_alpha=0.6, point_size=2)

ax = plt.gca()
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)

plt.show()
