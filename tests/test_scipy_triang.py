"""Try to construct graph of connected player units using delaunay triangulation
"""
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from typing import Tuple
import random


class Unit:
    def __init__(self, player: int, pos: Tuple):
        """The unit of each player. New unit spawns every N days. Can be moved in any direction by 1 km"""
        assert 0 <= player < 4

        self.player = int(player)
        self.pos = pos
        self.status = 1  # 1 = alive, 0 = dead


def delaunay2edges(tri_simplices):
    """Convert the delaunay tris to unique edges
    Args:
        simplices: Triangles. The .simplices param of the triangulation object from scipy.
    Ref:
        https://stackoverflow.com/questions/69512972/how-to-generate-edge-index-after-delaunay-triangulation
    """

    def less_first(a, b):
        return [a, b] if a < b else [b, a]

    list_of_edges = []

    for triangle in tri_simplices:
        for e1, e2 in [[0, 1], [1, 2], [2, 0]]:  # for all edges of triangle
            list_of_edges.append(less_first(triangle[e1], triangle[e2]))  # always lesser index first

    array_of_edges = np.unique(list_of_edges, axis=0)  # remove duplicates
    return array_of_edges


def hex_to_rgb(col: str = "#ffffff"):
    return tuple(int(col.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))


def compute_triangle_circumcenters(xy_pts, tri_arr):
    """
    Compute the centers of the circumscribing circle of each triangle in a triangulation.
    Args:
        xy_pts : np.array, points array of shape (n, 2)
        tri_arr :  np.array, triangles array of shape (m, 3), each row is a triple of indices in the xy_pts array

    Return:
        circumcenter points array of shape (m, 2)

    Ref:
        https://stackoverflow.com/questions/5596317/getting-the-circumcentres-from-a-delaunay-triangulation-generated-using-matplotl

    Usage:
        from scipy.spatial import Delaunay
        xy_pts = np.vstack([x, y]).T
        dt = Delaunay(xy_pts)
        cc = compute_triangle_circumcenters(dt.points, dt.simplices)
    """
    tri_pts = xy_pts[tri_arr]  # (m, 3, 2) - triangles as points (not indices)

    # finding the circumcenter (x, y) of a triangle defined by three points:
    # (x-x0)**2 + (y-y0)**2 = (x-x1)**2 + (y-y1)**2
    # (x-x0)**2 + (y-y0)**2 = (x-x2)**2 + (y-y2)**2
    #
    # becomes two linear equations (squares are canceled):
    # 2(x1-x0)*x + 2(y1-y0)*y = (x1**2 + y1**2) - (x0**2 + y0**2)
    # 2(x2-x0)*x + 2(y2-y0)*y = (x2**2 + y2**2) - (x0**2 + y0**2)
    a = 2 * (tri_pts[:, 1, 0] - tri_pts[:, 0, 0])
    b = 2 * (tri_pts[:, 1, 1] - tri_pts[:, 0, 1])
    c = 2 * (tri_pts[:, 2, 0] - tri_pts[:, 0, 0])
    d = 2 * (tri_pts[:, 2, 1] - tri_pts[:, 0, 1])

    v1 = (tri_pts[:, 1, 0] ** 2 + tri_pts[:, 1, 1] ** 2) - (tri_pts[:, 0, 0] ** 2 + tri_pts[:, 0, 1] ** 2)
    v2 = (tri_pts[:, 2, 0] ** 2 + tri_pts[:, 2, 1] ** 2) - (tri_pts[:, 0, 0] ** 2 + tri_pts[:, 0, 1] ** 2)

    # solve 2x2 system (see https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_2_%C3%97_2_matrices)
    det = (a * d - b * c)
    detx = (v1 * d - v2 * b)
    dety = (a * v2 - c * v1)

    x = detx / det
    y = dety / det

    return (np.vstack((x, y))).T


def convert_pt_for_plotting(x, y, map_size):
    """Convert the coords of a point for plotting b/w sim and matplotlib"""
    return y, map_size - x


def check_pts_within_bounds(pts: np.ndarray, pt_min: Tuple, pt_max: Tuple):
    """Check that each point (circumcenter) is within a given bounding box"""
    xx = (pts[:, 0] > pt_min[0]) & (pts[:, 0] < pt_max[0])
    yy = (pts[:, 1] > pt_min[1]) & (pts[:, 1] < pt_max[1])
    valid = xx & yy  # Shape: [n,]
    return valid


if __name__ == "__main__":
    # random.seed(13)
    random.seed(18)

    # NOTE: Delaunay will ignore inconsistencies such as duplicate points
    player_colors = ['#e6194B', '#f58231', '#3cb44b', '#4363d8']
    player_colors = list(map(hex_to_rgb, player_colors))
    map_size = 100

    units = [
        # Corners
        Unit(0, (0.5, 0.5)),
        Unit(1, (0.5, 99.5)),
        Unit(2, (99.5, 99.5)),
        Unit(3, (99.5, 0.5)),
        # # Middle
        # Unit(0, (57, 57)),
        # Unit(2, (53, 53)),
        # # offset
        # Unit(0, (47, 67)),
        # Unit(2, (63, 43)),
    ]

    # Scatter some points near home base of each player
    for idx in range(3):
        units.append(Unit(0, (random.random() * 30, random.random() * 30)))
        units.append(Unit(1, (100 - random.random() * 30, random.random() * 30)))
        units.append(Unit(2, (100 - random.random() * 30, 100 - random.random() * 30)))
        units.append(Unit(3, (random.random() * 30, 100 - random.random() * 30)))

    # Scatter more throughout the map
    for idx in range(4):
        for idy in range(3):
            units.append(Unit(idx, (random.random() * 100.0, random.random() * 100.0)))

    # Corner-case 1: Enemy unit close to the boundary should cut off path b/w 2 unit, even with valid delaunay tris
    #   connecting the 2 friendly units.
    units.append(Unit(2, (96, 40)))
    units.append(Unit(0, (94, 60)))

    # Construct 2 lists for passing to shapely/scipy for triangulation
    pts = [(u.pos[0], u.pos[1]) for u in units]
    player_ids = [x.player for x in units]

    points = np.array(pts)  # Shape: [N, 2]
    tri = Delaunay(pts)
    # tri.find_simplex(p)  # Find simplex associated with point p

    edges = delaunay2edges(tri.simplices)  # Shape: [N, 2]

    # Count valid edges
    edge_player_id = []  # Player each edge belongs to
    for p1, p2 in edges:
        player1 = player_ids[p1]
        player2 = player_ids[p2]
        if player1 == player2:
            edge_player_id.append(player1)
        else:
            edge_player_id.append(-1)

    edge_player_id = np.array(edge_player_id)
    valid_edges = (edge_player_id > -1)

    # Plot the valid edges
    for (p1, p2), pl in zip(edges[valid_edges], edge_player_id[valid_edges]):
        x1, y1 = points[p1]
        x2, y2 = points[p2]

        x1, y1 = convert_pt_for_plotting(x1, y1, map_size)
        x2, y2 = convert_pt_for_plotting(x2, y2, map_size)
        col = np.array((*player_colors[pl], 255)) / 255.0
        plt.plot([x1, x2], [y1, y2], color=col, alpha=0.7)

    # Plot the invalid edges
    for p1, p2 in edges[~valid_edges]:
        x1, y1 = points[p1]
        x2, y2 = points[p2]

        x1, y1 = convert_pt_for_plotting(x1, y1, map_size)
        x2, y2 = convert_pt_for_plotting(x2, y2, map_size)
        plt.plot([x1, x2], [y1, y2], color='grey', linestyle='dashed', alpha=0.4)

    # Plot the units
    for pt, pl in zip(points, player_ids):
        x, y = pt

        x, y = convert_pt_for_plotting(x, y, map_size)
        col = np.array((*player_colors[pl], 255)) / 255.0
        plt.plot(x, y, marker="o", markersize=10, markeredgecolor=col, markerfacecolor=col)

    plt.show()

    # ----- Out of Bound Circumcenters -----
    # FAIL - We thought ignoring triangles with circumcenter outside bounds would help fix corner case 1.
    #   Causes some points close to home base to not have an edge back to base.

    # cc = compute_triangle_circumcenters(tri.points, tri.simplices)  # Shape: [N, 2]
    # cc_mask_1d = check_pts_within_bounds(cc, (0, 0), (map_size, map_size))
    # cc_valid = cc[cc_mask_1d, :]  # Broadcast mask to all channels
    #
    # simplex_valid = tri.simplices[cc_mask_1d, :]
    # edges = delaunay2edges(simplex_valid)  # Shape: [N, 2]
    #
    # # Plot the out-of-bound edges
    # simplex_out = tri.simplices[~cc_mask_1d, :]
    # edges_out = delaunay2edges(simplex_out)  # Shape: [N, 2]
    # for p1, p2 in edges_out:
    #     x1, y1 = points[p1]
    #     x2, y2 = points[p2]
    #
    #     x1, y1 = convert_pt_for_plotting(x1, y1, map_size)
    #     x2, y2 = convert_pt_for_plotting(x2, y2, map_size)
    #     plt.plot([x1, x2], [y1, y2], color='black', linestyle='dashed', alpha=0.6)
    # ------------------------
