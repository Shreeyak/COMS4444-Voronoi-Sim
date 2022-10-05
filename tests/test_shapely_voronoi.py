"""Test finding graph of connected player units using shapely voronoi"""

# import shapely
import numpy as np
import random
from typing import Tuple
from collections import defaultdict

import matplotlib as mpl
import shapely.errors
import shapely.geometry
import shapely.ops
import scipy
# from shapely.geometry import MultiPoint, MultiLineString, Polygon, box
# from shapely.ops import triangulate, voronoi_diagram
# from scipy.spatial import Delaunay, KDTree
from matplotlib import pyplot as plt
from descartes.patch import PolygonPatch  # plotting
from shapely_figures import SIZE, set_limits, plot_coords, plot_bounds, plot_line, BLUE, GRAY

mpl.use('TkAgg')  # For macOS. Change engine.

import warnings
warnings.filterwarnings("ignore", category=shapely.errors.ShapelyDeprecationWarning)


class Unit:
    def __init__(self, player: int, pos: Tuple):
        """The unit of each player. New unit spawns every N days. Can be moved in any direction by 1 km"""
        assert 0 <= player < 4

        self.player = int(player)
        self.pos = pos
        self.status = 1  # 1 = alive, 0 = dead


def create_units():
    units = [
        # Corners
        Unit(0, (1.5, 1.5)),
        Unit(1, (8.5, 1.5)),
        Unit(2, (8.5, 8.5)),
        Unit(3, (1.5, 8.5)),

        # Middle
        Unit(0, (5.7, 5.7)),
        Unit(2, (5.3, 5.3)),
        # offset
        Unit(0, (4.7, 6.7)),
        Unit(2, (6.3, 4.3)),

        # Edge-case
        Unit(1, (9.0, 6.0)),
        Unit(2, (8.0, 4.0)),
    ]

    # # Scatter some points near home base of each player
    # for idx in range(4):
    #     units.append(Unit(0, (random.random() * 3, random.random() * 3)))
    #     units.append(Unit(1, (10 - random.random() * 3, random.random() * 3)))
    #     units.append(Unit(2, (10 - random.random() * 3, 10 - random.random() * 3)))
    #     units.append(Unit(3, (random.random() * 3, 10 - random.random() * 3)))
    #
    # # Scatter more throughout the map
    # for idx in range(4):
    #     for idy in range(4):
    #         units.append(Unit(idx, (random.random() * 10.0, random.random() * 10.0)))

    # Corner-case 1: Enemy unit close to the boundary should cut off path b/w 2 unit, even with valid delaunay tris
    #   connecting the 2 friendly units.
    # units.append(Unit(2, (9.6, 4.0)))
    # units.append(Unit(0, (9.4, 6.0)))
    return units


def delaunay2edges(tri_simplices):
    """Convert the delaunay tris to unique edges
    Args:
        simplices: Triangles. The .simplices param of the triangulation object from scipy.
    Ref:
        https://stackoverflow.com/questions/69512972/how-to-generate-edge-index-after-delaunay-triangulation
    """

    def less_first(a, b):
        return (a, b) if a < b else (b, a)

    edges_dict = defaultdict(list)  # Gives all the edges and the associated triangles
    for idx, triangle in enumerate(tri_simplices):
        for e1, e2 in [[0, 1], [1, 2], [2, 0]]:  # for all edges of triangle
            edge = less_first(triangle[e1], triangle[e2])  # always lesser index first
            edges_dict[edge].append(idx)  # prevent duplicates. Track associated simplexes for each edge.

    array_of_edges = np.array(list(edges_dict.keys()), dtype=int)
    return array_of_edges


def poly_are_neighbors(poly1: shapely.geometry.polygon.Polygon,
                       poly2: shapely.geometry.polygon.Polygon) -> bool:
    # Polygons are neighbors iff they share an edge. Only 1 vertex does not count.
    # Also, both polygons might be the same
    if isinstance(poly1.intersection(poly2), shapely.geometry.linestring.LineString):
        return True
    elif poly1 == poly2:
        return True
    else:
        return False


def convert_pt_for_plotting(x, y, map_size):
    """Convert the coords of a point for plotting b/w sim and matplotlib"""
    return y, map_size - x


def hex_to_rgb(col: str = "#ffffff"):
    rgb = tuple(int(col.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    rgba = rgb + (255,)
    rgba = [x/255.0 for x in rgba]
    return tuple(rgba)


def get_shapely_centroid(poly: shapely.geometry.polygon.Polygon) -> Tuple:
    return list(poly.centroid.coords)[0]


if __name__ == "__main__":
    """Algo:
    - Create list of pts from units
        HANDLE EDGECASE 1: Quantize the unit positions to grid level (cell centers).
    - Get list of polygons from voronoi diagram for given pts.
    - Find mapping from pts idx to polys - for each poly centroid, find nearest pt
        WARN - EDGECASE 1: 2 players occupy exact same point
    - Get the graph of connected pts via triangulation (include home base when triangulating) (list of edges)
    - Clean Graph: For each edge of the graph, if the 2 points belong to diff players, it is invalid. If the the 2 polys 
        corresponding to the points are not neighbors it is invalid.
    
    - Build an actual graph data structure for each player. https://www.python.org/doc/essays/graphs/
    - From each home base, traverse the full graph. Mark all units that we come across. Remove isolated units from map.
    """
    random.seed(18)

    map_size = 10
    units = create_units()

    home_offset = 0.5
    _MAP_W = map_size
    spawn_loc = {0: (home_offset, home_offset),
                 1: (_MAP_W - home_offset, home_offset),
                 2: (_MAP_W - home_offset, _MAP_W - home_offset),
                 3: (home_offset, _MAP_W - home_offset)}

    player_colors = ['#e6194B', '#f58231', '#3cb44b', '#4363d8']
    player_colors = list(map(hex_to_rgb, player_colors))

    # Construct 2 lists for passing to shapely/scipy for triangulation
    # TODO: Handle edge case where cell is occupied by 2 players: Quantize pts to grid cells.
    #   Double-check: do we need this?
    pts = []
    player_ids = []
    for u in units:
        # pts.append((u.pos[1], map_size - u.pos[0]))  # Convert coord
        pts.append(u.pos)  # Convert coord
        player_ids.append(u.player)
    # points = np.array(pts)  # Shape: [N, 2]

    # Get polygon of Voronoi regions around each pt
    _points = shapely.geometry.MultiPoint(pts)
    envelope = shapely.geometry.box(0, 0, map_size, map_size)
    vor_regions_ = shapely.ops.voronoi_diagram(_points, envelope=envelope)
    vor_regions_ = list(vor_regions_)  # Convert to a list of Polygon
    # CLEAN the polys - they aren't being bounded correctly
    vor_regions = []
    for region in vor_regions_:
        if not isinstance(region, shapely.geometry.Polygon):
            print(f"WARNING: Region returned from voronoi not a polygon: {type(region)}")

        region_bounded = region.intersection(envelope)
        vor_regions.append(region_bounded)

    # Add the home base to list of points
    pts_with_home = pts.copy()
    player_ids_with_home = player_ids.copy()
    for player in range(4):
        # Add home bases as pts
        pts_with_home.append(spawn_loc[player])
        player_ids_with_home.append(player)

    # Find mapping from pts idx to polys (via nearest unit) and poly to player
    pt_to_poly = {}  # includes home base
    # Polygon isn't hashable, so we use polygon centroid. Because polygon is convex, centroid will be unique.
    poly_centroid_to_player = {}
    kdtree = scipy.spatial.KDTree(pts)
    for region in vor_regions:
        # Voronoi regions are all convex. Nearest pt to centroid must be point belonging to region
        centroid = get_shapely_centroid(region)
        _, ii = kdtree.query(centroid, k=1)  # index of nearest pt

        pt_to_poly[pts[ii]] = region
        poly_centroid_to_player[centroid] = player_ids[ii]

    # Find mapping of each home base to poly
    for idx in range(4):
        home_coord = spawn_loc[idx]
        _, ii = kdtree.query(home_coord, k=1)  # index of nearest pt
        pt_to_poly[home_coord] = pt_to_poly[pts[ii]]  # home base same as nearest unit

    # Get the graph of connected pts via triangulation (include home base when triangulating)
    pts_with_home = pts.copy()
    player_ids_with_home = player_ids.copy()
    for key, val in spawn_loc.items():
        # Add home bases as pts
        pts_with_home.append(val)
        player_ids_with_home.append(key)
    pts_with_home = np.array(pts_with_home)

    tri = scipy.spatial.Delaunay(pts_with_home)
    edges = delaunay2edges(tri.simplices)  # Shape: [N, 2]

    # Clean edges
    edge_player_id = []  # Player each edge belongs to
    for p1, p2 in edges:
        player1 = player_ids_with_home[p1]
        player2 = player_ids_with_home[p2]

        valid_ = False
        if player1 == player2:
            poly1 = pt_to_poly[tuple(pts_with_home[p1])]
            poly2 = pt_to_poly[tuple(pts_with_home[p2])]

            # The polygons must both belong to the same player
            # This handles edge cases where home base is conquered by another player
            play1_ = poly_centroid_to_player[get_shapely_centroid(poly1)]
            play2_ = poly_centroid_to_player[get_shapely_centroid(poly2)]

            are_neighbors = poly_are_neighbors(poly1, poly2)
            if are_neighbors and play1_ == player1 and play2_ == player1:
                # Can traverse edge only if voronoi polys are neighbors
                valid_ = True

        if valid_:
            edge_player_id.append(player1)
        else:
            edge_player_id.append(-1)

    edge_player_id = np.array(edge_player_id)
    mask_edges = (edge_player_id > -1)

    # For plotting
    edges_invalid = edges[~mask_edges].copy()

    # Remove invalid edges
    edges = edges[mask_edges]
    edge_player_id = edge_player_id[mask_edges]

    # Build a graph data structure for each player
    graphs = {0: defaultdict(list), 1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}
    for player, (p1, p2) in zip(edge_player_id, edges):
        if player > -1:
            graph_p = graphs[player]
            graph_p[p1].append(p2)
            graph_p[p2].append(p1)

    # TODO: From each home base, traverse the full graph


    # Plot
    fig = plt.figure(1, figsize=SIZE, dpi=90)
    fig.set_frameon(True)
    ax = fig.add_subplot(111)
    # set_limits(ax, 0, map_size, 0, map_size)

    for poly in vor_regions:
        player_ = poly_centroid_to_player[get_shapely_centroid(poly)]
        patch = PolygonPatch(poly, facecolor=player_colors[player_], alpha=0.3, zorder=-1)  # edgecolor=BLUE
        ax.add_patch(patch)

    # Plot the valid edges
    for (p1, p2), pl in zip(edges, edge_player_id):
        x1, y1 = pts_with_home[p1]
        x2, y2 = pts_with_home[p2]

        # x1, y1 = convert_pt_for_plotting(x1, y1, map_size)
        # x2, y2 = convert_pt_for_plotting(x2, y2, map_size)
        col = np.array(player_colors[pl])
        plt.plot([x1, x2], [y1, y2], color=col, alpha=0.7)

    # Plot the invalid edges
    for p1, p2 in edges_invalid:
        x1, y1 = pts_with_home[p1]
        x2, y2 = pts_with_home[p2]

        # x1, y1 = convert_pt_for_plotting(x1, y1, map_size)
        # x2, y2 = convert_pt_for_plotting(x2, y2, map_size)
        plt.plot([x1, x2], [y1, y2], color='grey', linestyle='dashed', alpha=0.4)

    # Plot the units
    for pt, pl in zip(pts, player_ids):
        x, y = pt

        # x, y = convert_pt_for_plotting(x, y, map_size)
        col = np.array(player_colors[pl])
        plt.plot(x, y, marker="o", markersize=10, markeredgecolor=col, markerfacecolor=col)

    # Plot the home base
    for pt, pl in zip(pts_with_home[-4:], player_ids_with_home[-4:]):
        x, y = pt
        # x, y = convert_pt_for_plotting(x, y, map_size)
        col = np.array(player_colors[pl])
        plt.plot(x, y, marker="x", markersize=14, markeredgecolor=col, markerfacecolor=col)

    plt.show()


