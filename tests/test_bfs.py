"""Run BFS from each player's home base to find killed units"""

import copy
import numpy as np
from collections import deque
import random
import time
from typing import Tuple, List
import cv2
import matplotlib as mpl


def _hex_to_rgb(col: str = "#ffffff"):
    return tuple(int(col.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))


def get_colored_occ_map(occ_map: np.ndarray,
                        units: List,
                        draw_major_lines: bool = True,
                        draw_units: bool = True):
    """Visualizes an NxN Occupancy map for the voronoi game.
    Each cell is assigned a number from 0-5: 0-3 represents a player occupying it, 4 means contested

    NOTE: Occupancy map should be updated before calling this func
    """

    assert len(occ_map.shape) == 2
    assert occ_map.max() <= 5 and occ_map.min() >= 0

    if occ_map.max() > 4:
        print(f"WARNING: Occupancy status has not been computed for all cells in the grid")
        occ_map[occ_map > 4] = 4

    # Add vars
    player_back_colors = ['#fabed4', '#ffd8b1', '#aaffc3', '#42d4f4']
    player_colors = ['#e6194B', '#f58231', '#3cb44b', '#4363d8']
    player_colors = list(map(_hex_to_rgb, player_colors))
    scale_px = 10
    unit_size_px = 10

    # Colormap
    # Colors from https://sashamaps.net/docs/resources/20-colors/
    cmap = mpl.colors.ListedColormap([*player_back_colors, '#ffffff'])
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5)  # Discrete colors
    grid_rgb = cmap(norm(occ_map))[:, :, :3]
    grid_rgb = (grid_rgb * 255).astype(np.uint8)

    # Upsample img
    grid_rgb = cv2.resize(grid_rgb, None, fx=scale_px, fy=scale_px, interpolation=cv2.INTER_NEAREST)

    if draw_major_lines:
        h, w, _ = grid_rgb.shape
        # Only show major grid lines (100x100 lines too fine) - max 10
        cols = min(10, occ_map.shape[1])
        col_line = (0, 0, 0)
        thickness = 2
        for x in np.linspace(start=int(thickness / 2), stop=w - int(thickness / 2), num=cols + 1):
            x = int(round(x))
            cv2.line(grid_rgb, (x, 0), (x, h), color=col_line, thickness=thickness)
            cv2.line(grid_rgb, (0, x), (w, x), color=col_line, thickness=thickness)

    if draw_units:
        for unit in units:
            if unit.status > 0:
                col = player_colors[unit.player]
            else:
                col = (0.0, 0.0, 0.0, 1.0)

            # Draw Circle for each unit
            y, x = unit.pos
            px, py = map(lambda z: int(round(z * scale_px)), [x, y])
            pos_px = (px, py)  # (py, px)
            cv2.circle(grid_rgb, pos_px[::-1], unit_size_px, col, -1)

    return grid_rgb


def plot_maps(occ_map, connected, units, units_new):
    from matplotlib import pyplot as plt

    grid_rgb = get_colored_occ_map(occ_map, units)
    grid_rgb2 = get_colored_occ_map(connected, units_new)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(grid_rgb)
    ax1.set_title("Occupancy")
    ax2.imshow(grid_rgb2)
    ax2.set_title("Connected")
    plt.tight_layout()
    plt.show()


class Unit:
    def __init__(self, player: int, pos: Tuple):
        """The unit of each player. New unit spawns every N days. Can be moved in any direction by 1 km"""
        assert 0 <= player < 4

        self.player = int(player)
        self.pos = pos
        self.status = 1  # 1 = alive, 0 = dead


def create_units():
    units = []
    # Scatter more throughout the map
    for idx in range(4):
        for idy in range(25):
            units.append(Unit(idx, (random.random() * 100.0, random.random() * 100.0)))
    return units


def get_connectivity_map(occ_map, spawn_loc):
    connected = np.ones_like(occ_map) * 4  # Default = disputed/empty

    for player in range(4):
        que = deque()
        reached = set()

        start = spawn_loc[player]
        start = (int(start[1]), int(start[0]))
        que.append(start)

        occ_map_ = (occ_map == player)
        while len(que) > 0:
            ele = que.popleft()

            if occ_map_[ele]:
                connected[ele] = player

                # neighbors - iter
                ymin = max(0, ele[0] - 1)
                ymax = min(connected.shape[0] - 1, ele[0] + 1)
                xmin = max(0, ele[1] - 1)
                xmax = min(connected.shape[1] - 1, ele[1] + 1)
                neigh = []
                for y in range(ymin, ymax + 1):
                    for x in range(xmin, xmax + 1):
                        neigh.append((y, x))

                for n in neigh:
                    if n not in reached:
                        que.append(n)
                        reached.add(n)

    return connected


if __name__ == "__main__":
    units = create_units()

    for u in units:
        # Quantize unit pos to cell. We assume cell origin at top-left for ease of use.
        pos_int = (int(u.pos[1]), int(u.pos[0]))

    # Occ map
    occ_map = np.zeros((100, 100), dtype=np.uint8)
    occ_map[:10, -10:] = 1
    occ_map[-20:-10, -20:-10] = 2
    occ_map[-10:, :10] = 3
    occ_map[10:40, -40:-10] = 1
    occ_map[50:60, -30:-20] = 1
    occ_map[20:30, 20:30] = 3
    occ_map[-10:, -10:] = 4

    home_offset = 0.5
    _MAP_W = 100
    spawn_loc = {0: (home_offset, home_offset),
                 1: (_MAP_W - home_offset, home_offset),
                 2: (_MAP_W - home_offset, _MAP_W - home_offset),
                 3: (home_offset, _MAP_W - home_offset)}

    # BFS
    # start = time.time()
    connectivity_map = get_connectivity_map(occ_map, spawn_loc)
    # print(f"get_connected_map: ", (time.time() - start) * 1000)

    # Kill isolated units - linear check
    # Note: Better to create a new list of tuples. Don't use class.
    units_new = copy.deepcopy(units)
    for u in units_new:
        pos_int = (int(u.pos[1]), int(u.pos[0]))
        pl = u.player
        if connectivity_map[pos_int] == pl:
            u.status = 1
        else:
            u.status = 0

    # Plot Occ map and Connected map
    for pl in range(4):
        print(f"Player {pl}: {np.count_nonzero(connectivity_map == pl)} / {np.count_nonzero(occ_map == pl)} connected")

    plot = True
    if plot:
        plot_maps(occ_map, connectivity_map, units, units_new)
