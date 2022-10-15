import numpy as np

import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch  # plotting
from tests.shapely_figures import SIZE, set_limits, plot_coords, plot_bounds, plot_line, BLUE, GRAY

import matplotlib as mpl
mpl.use('TkAgg')  # For macOS. Change engine.


# grey is player 4 (invalid)
player_colors = [(230, 25, 75), (245, 130, 49), (60, 180, 75), (67, 99, 216), (50, 50, 50)]
player_colors = [(x/255, y/255, z/255) for (x, y, z) in player_colors]


def plot_units_and_edges(edges, edge_player_id, discrete_pts, discrete_players, pt_to_poly, day):
    # Plot
    fig = plt.figure(1, figsize=(6.0, 6.0), dpi=90)
    fig.set_frameon(True)
    ax = fig.add_subplot(111)
    # set_limits(ax, 0, map_size, 0, map_size)

    # Plot the valid edges
    for (p1, p2), pl in zip(edges, edge_player_id):
        x1, y1 = discrete_pts[p1]
        x2, y2 = discrete_pts[p2]

        col = np.array(player_colors[pl])  # edge player id includes invalid edges
        plt.plot([x1, x2], [y1, y2], color=col, alpha=0.7, )

    # Plot the units
    for pt, pl in zip(discrete_pts, discrete_players):
        x, y = pt

        col = np.array(player_colors[pl])
        plt.plot(x, y, marker="o", markersize=6, markeredgecolor=col, markerfacecolor=col)

    # Plot the polys
    for pt, pl in zip(discrete_pts, discrete_players):
        poly = pt_to_poly[pt]
        fcol = np.array(player_colors[pl])
        ecol = np.clip(np.array(player_colors[pl]) * 0.4, a_min=0, a_max=1)
        patch = PolygonPatch(poly, facecolor=fcol, edgecolor=ecol, alpha=0.2, zorder=-1)  # edgecolor=BLUE
        ax.add_patch(patch)

    plt.gca().invert_yaxis()
    # plt.show()
    plt.savefig(f"/home/shrek/work/cu_course/prob-solving/COMS4444-Voronoi-Sim/tests/plot_map/{day}.png", dpi=300)


def plot_poly_list(poly_list):
    # Plot
    fig = plt.figure(1, figsize=SIZE, dpi=90)
    fig.set_frameon(True)
    ax = fig.add_subplot(111)

    set_limits(ax, 0, 100, 0, 100)

    for poly in poly_list:
        patch = PolygonPatch(poly, facecolor=BLUE, alpha=0.3, zorder=-1)  # edgecolor=BLUE
        ax.add_patch(patch)
    plt.show()
