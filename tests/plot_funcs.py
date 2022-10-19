import numpy as np

import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch  # plotting
from tests.shapely_figures import SIZE, set_limits, plot_coords, plot_bounds, plot_line, BLUE, GRAY, RED, GREEN

import matplotlib as mpl
mpl.use('TkAgg')  # For macOS. Change engine.


# grey is player 4 (invalid)
player_colors = [(230, 25, 75), (245, 130, 49), (60, 180, 75), (67, 99, 216), (50, 50, 50)]
player_colors = [(x/255, y/255, z/255) for (x, y, z) in player_colors]


def plot_dbscan(dbscan_groups, day):
    # Plot
    fig = plt.figure(1, figsize=SIZE, dpi=90)
    fig.set_frameon(True)
    ax = fig.add_subplot(111)
    # set_limits(ax, 0, map_size, 0, map_size)
    # Plot points and outliers
    for pl, pt_dict in dbscan_groups.items():
        col = np.array(player_colors[pl])
        if len(pt_dict['groups']) > 0:
            for group in pt_dict['groups'].values():
                for pt in group:
                    x,y = pt
                    plt.plot(x, y, marker="o", markersize=6, markeredgecolor=col, markerfacecolor=col)
        for out_pt in pt_dict['outliers']:
            x,y = out_pt
            plt.plot(x, y, marker="o", markersize=6, markeredgecolor="magenta", markerfacecolor="magenta")
    plt.gca().invert_yaxis()
    # plt.show()
    plt.savefig(f"/Users/yaroslavzakharov/COMS4444-Voronoi-Sim/tests/plot_map/{day}.png", dpi=300)


def plot_units_and_edges(edges, edge_player_id, units_cls, day):
    # Plot
    fig = plt.figure(1, figsize=(6.0, 6.0), dpi=90)
    fig.set_frameon(True)
    ax = fig.add_subplot(111)
    # set_limits(ax, 0, map_size, 0, map_size)
    uuids = list(units_cls.keys())

    # Plot the valid edges
    for (p1, p2), pl in zip(edges, edge_player_id):
        x1, y1 = units_cls[uuids[p1]].pos
        x2, y2 = units_cls[uuids[p2]].pos

        col = np.array(player_colors[pl])  # edge player id includes invalid edges
        plt.plot([x1, x2], [y1, y2], color=col, alpha=0.7, )

    # Plot the units
    for unit in units_cls.values():
        x, y = unit.pos
        pl = unit.player

        col = np.array(player_colors[pl])
        plt.plot(x, y, marker="o", markersize=6, markeredgecolor=col, markerfacecolor=col)

        # Plot the polys
        poly = unit.poly
        if poly is not None:
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
    plt.gca().invert_yaxis()
    plt.show()

def plot_incursions(all_polys_list, incursion_poly_list):
    # Plot
    fig = plt.figure(1, figsize=SIZE, dpi=90)
    fig.set_frameon(True)
    ax = fig.add_subplot(111)

    set_limits(ax, 0, 100, 0, 100)

    for poly in all_polys_list:
        if poly not in incursion_poly_list:
            patch = PolygonPatch(poly, facecolor=BLUE, alpha=0.3, zorder=-1)  # edgecolor=BLUE
            ax.add_patch(patch)
    for poly in incursion_poly_list:
        patch = PolygonPatch(poly, facecolor=RED, alpha=0.3, zorder=-1)  # edgecolor=BLUE
        ax.add_patch(patch)

    plt.gca().invert_yaxis()
    plt.show()


def plot_line_list(line_list):
    # Plot
    fig = plt.figure(1, figsize=SIZE, dpi=90)
    fig.set_frameon(True)
    ax = fig.add_subplot(111)

    set_limits(ax, 0, 100, 0, 100)

    # Plot the valid edges
    for linestr in line_list:
        x, y = linestr.xy
        ax.plot(x, y, color=GREEN, alpha=0.5, linewidth=3, solid_capstyle='round', zorder=2)

    plt.gca().invert_yaxis()
    plt.show()

def plot_debug_incur(superpolygon, incursions, edge_incursion_begin_list, day):
    # Plot
    fig = plt.figure(1, figsize=SIZE, dpi=90)
    fig.set_frameon(True)
    ax = fig.add_subplot(111)

    set_limits(ax, 0, 100, 0, 100)


    patch = PolygonPatch(superpolygon, facecolor=BLUE, alpha=0.3, zorder=-1)  # edgecolor=BLUE
    ax.add_patch(patch)

    for poly in incursions:
        patch = PolygonPatch(poly, facecolor=RED, alpha=0.3, zorder=-1)  # edgecolor=BLUE
        ax.add_patch(patch)

    # Plot the valid edges
    for edge_incursion_begin in edge_incursion_begin_list:

        x, y = edge_incursion_begin.xy
        ax.plot(x, y, color=GREEN, alpha=0.5, linewidth=3, solid_capstyle='round', zorder=2)

    plt.gca().invert_yaxis()
    # plt.show()
    plt.savefig(f"/home/shrek/work/cu_course/prob-solving/COMS4444-Voronoi-Sim/tests/plot_incur/{day}.png", dpi=300)
