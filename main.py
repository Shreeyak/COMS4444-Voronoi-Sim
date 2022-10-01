"""This is a simulator for Project 2 of COMS 4444 (Fall 2022) - Voronoi"""

import numpy as np
import cv2
import matplotlib as mpl
import pygame
import scipy
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

mpl.use('TkAgg')


class Unit:
    def __init__(self, player: int, pos: Tuple):
        """The unit of each player. New unit spawns every N days. Can be moved in any direction by 1 km"""
        assert 0 <= player < 4

        self.player = int(player)
        self.pos = pos
        self.status = 1  # 1 = alive, 0 = dead

    def kill(self):
        self.status = 0

    def move(self):
        raise NotImplementedError


class GameMap:
    def __init__(self, map_width=100):
        """Class for methods related to the game map.
        The map is 100x100 cells, each cell is 1km wide.
        Unit coordinates on the map can be floating point.
        """
        self._MAP_W = map_width  # Width of the map in km. Each cell is 1km
        self.scale_px = 10  # How many pixels wide each cell will be
        self.unit_size_px = 2
        self.img_h = self._MAP_W * self.scale_px
        self.img_w = self._MAP_W * self.scale_px
        self.units = []

        # Colors from: https://sashamaps.net/docs/resources/20-colors/
        self.player_back_colors = ['#fabed4', '#ffd8b1', '#aaffc3', '#42d4f4']
        player_colors = ['#e6194B', '#f58231', '#3cb44b', '#4363d8']
        self.player_colors = list(map(self._hex_to_rgb, player_colors))

        self.cell_origins = self._get_cell_origins()
        # Each channel represents a player. If 1, then the player has a unit in that cell.
        self.unit_map = np.zeros((self._MAP_W, self._MAP_W, 4), dtype=np.uint8)

    def _get_cell_origins(self) -> np.ndarray:
        """Calculates the origin for each cell
        Return:
            coords: Coords for each cell, indexed by cell location. np.ndarray, Shape: [100, 100, 2].
        """
        x = np.arange(0.5, self._MAP_W, 1.0)
        y = x.copy()

        xx, yy = np.meshgrid(x, y, indexing="ij")
        coords = np.stack((xx, yy), axis=-1)
        return coords

    @staticmethod
    def _hex_to_rgb(col: str = "#ffffff"):
        return tuple(int(col.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    def add_units(self, units: List[Unit]):
        self.units.extend(units)
        for unit in units:
            cx, cy = self.metric_to_cell(unit.pos)
            self.unit_map[cx, cy, unit.player] = 1
            # print("Unit Map (0):\n", self.unit_map[:, :, 0])
            # print("/n")


    def get_unit_occupied_cells(self) -> np.ndarray:
        """Calculate which cells are counted as occupied due to unit presence (based on unit map)
        If a cell contains exactly 1 unit, then it's occupied by that unit's player.

        Returns:
            unit_occupancy_map: 2D Map that shows which cells are occupied by each player due to unit presence.
                (Does not include calculation via nearest neighbors)
        """
        # TODO: Assumes updated unit map. Make sure to update unit map after unit moves
        # Get player-wise cell occupancy. If a cell has exactly 1 unit, it's occupied. More than 1, it's disputed.
        num_units = self.unit_map.sum(axis=2)
        occupied_mask_2d = (num_units == 1).reshape((self._MAP_W, self._MAP_W, 1))
        occupied_mask_2d = np.logical_and(occupied_mask_2d, self.unit_map > 0)  # Shape: [N, N, 4]

        # 2D map that shows which cells are occupied by a player's unit. 4 means contested. 5 is uncomputed.
        occ_map = np.ones((self._MAP_W, self._MAP_W), dtype=np.uint8) * 5
        occ_map[occupied_mask_2d[:, :, 0]] = 0
        occ_map[occupied_mask_2d[:, :, 1]] = 1
        occ_map[occupied_mask_2d[:, :, 2]] = 2
        occ_map[occupied_mask_2d[:, :, 3]] = 3
        occ_map[num_units > 1] = 4

        return occ_map

    def compute_occupancy_map(self) -> np.ndarray:
        """Calculates the occupancy status of each cell in the grid"""

        # Get coords of cells that are occupied due to units inside them
        occ_map = self.get_unit_occupied_cells()
        occ_cell_pts = self.cell_origins[occ_map < 4]  # Shape: [N, 2]. Cells that are occupied by units pos.
        # Get list with player id for each occ cell
        player_ids = occ_map[occ_map < 4]  # Shape: [N,]

        # Create KD-tree with all occupied cells
        kdtree = scipy.spatial.KDTree(occ_cell_pts)

        # Query points: coords of each cell that's neither occupied nor disputed
        candidate_cell_pts = self.cell_origins[occ_map > 4]  # Shape: [N, 2]

        # For each query pt, see if it is occupied or disputed
        near_dist, near_idx = kdtree.query(candidate_cell_pts, k=2)

        # Filter Disputes: Find cells with more than 1 occupied cells at same distance
        same_d = near_dist[:, 1] - near_dist[:, 0]
        disputed = np.isclose(same_d, 0)
        disputed_cells = candidate_cell_pts[disputed]  # 2 cells with same dist. Shape: [N, 2].
        if disputed_cells.shape[0] > 0:
            disputed_cells_d = near_dist[disputed, 0]  # Radius for search. Shape: [N, ]

            # TODO - Find all neighbors within a radius, see if more than 1 player in radius
            for disp_cell, radius in zip(disputed_cells, disputed_cells_d):
                # disp_cell shape: [2,]
                d_near_dist, d_near_idx = kdtree.query(disp_cell, k=4, distance_upper_bound=radius)
                disputed_ids = player_ids[near_idx[~disputed, 0]]  # Get player ids of the contesting cells
            pass

        # For other cells, mark occupancy
        not_disputed_ids = player_ids[near_idx[~disputed, 0]]  # Get player id of the nearest cell
        not_disputed_cells = candidate_cell_pts[~disputed].astype(np.uint8)  # coords to cell index of occupied cells
        occ_map[not_disputed_cells[:, 0], not_disputed_cells[:, 1]] = not_disputed_ids

        return occ_map

    def remove_killed_units(self):
        valid_units = []
        for unit in self.units:
            if unit.status > 0:
                valid_units.append(unit)
        self.units = valid_units

    # def set_indices(self):
    #     """snippet to set indices in a 2d array to set values"""
    #     c = np.arange(0, 20).reshape((2, 10))
    #     idx = np.array([[0, 1], [0, 2], [1, 9]])  # The indices to the changed
    #     print(c)
    #     c[tuple(idx.T)] = 118
    #     print(c)

    def metric_to_px(self, pos: Tuple) -> Tuple[int, int]:
        """Convert metric unit pos to pixel location on img of grid"""
        x, y = pos
        if not 0 <= x <= self._MAP_W:
            raise ValueError(f"x out of range [0, {self._MAP_W}]: {x}")
        if not 0 <= y <= self._MAP_W:
            raise ValueError(f"y out of range [0, {self._MAP_W}]: {y}")

        px, py = map(lambda z: int(round(z * self.scale_px)), [x, y])
        return px, py

    def metric_to_cell(self, pos: Tuple) -> Tuple[int, int]:
        """Convert metric unit pos to the location of the cell containing unit"""
        x, y = pos
        if not 0 <= x <= self._MAP_W:
            raise ValueError(f"x out of range [0, {self._MAP_W}]: {x}")
        if not 0 <= y <= self._MAP_W:
            raise ValueError(f"y out of range [0, {self._MAP_W}]: {y}")

        px = int(x)
        py = int(y)
        return px, py

    def get_colored_grid(self,
                         grid: np.ndarray,
                         draw_major_lines: bool = True,
                         draw_units: bool = True):
        """Visualizes a 100x100 grid for the voronoi game.
        Each cell is assigned a number from 0-5: 0-3 represents a player occupying it, 4 means contested
        """
        assert len(grid.shape) == 2
        assert grid.max() <= 5 and grid.min() >= 0

        if grid.max() == 5:
            print(f"WARNING: Occupancy status has not been computed for all cells in the grid")
            grid[grid == 5] = 4

        # Colormap
        # Colors from https://sashamaps.net/docs/resources/20-colors/
        cmap = mpl.colors.ListedColormap([*self.player_back_colors, '#ffffff'])
        norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5)  # Discrete colors
        grid_rgb = cmap(norm(grid))[:, :, :3]
        grid_rgb = (grid_rgb * 255).astype(np.uint8)

        # Upsample img
        grid_rgb = cv2.resize(grid_rgb, None, fx=self.scale_px, fy=self.scale_px, interpolation=cv2.INTER_NEAREST)

        if draw_major_lines:
            h, w, _ = grid_rgb.shape
            rows, cols = 10, 10  # Only show major grid lines (100x100 lines too fine)
            col_line = (0, 0, 0)
            thickness = 1
            for x in np.linspace(start=int(thickness/2), stop=w - int(thickness/2), num=cols + 1):
                x = int(round(x))
                cv2.line(grid_rgb, (x, 0), (x, h), color=col_line, thickness=thickness)
                cv2.line(grid_rgb, (0, x), (w, x), color=col_line, thickness=thickness)

        if draw_units:
            for unit in self.units:
                if unit.status > 0:
                    pos_px = self.metric_to_px(unit.pos)

                    # Draw Square: Ensure that we do not exceed image bounds and wrap around
                    # xmin = max(pos_px[0] - self.unit_size_px, 0)
                    # ymin = max(pos_px[1] - self.unit_size_px, 0)
                    # xmax = min(pos_px[0] + self.unit_size_px, self.img_w - 1)
                    # ymax = min(pos_px[1] + self.unit_size_px, self.img_h - 1)
                    # grid_rgb[xmin:xmax, ymin:ymax] = self.player_colors[unit.player]

                    # Draw Circle
                    cv2.circle(grid_rgb, pos_px[::-1], self.unit_size_px, self.player_colors[unit.player], -1)

        return grid_rgb


if __name__ == '__main__':
    game_map = GameMap(map_width=10)

    # Unit Test - Grid utils
    pos = (np.random.random((2,)) * 10)
    pos = tuple(pos.tolist())
    print(f"Test - Pos: {pos}\n  Pixel: {game_map.metric_to_px(pos)}\n  Cell: {game_map.metric_to_cell(pos)}")
    print(f"  Cell Coord: {game_map.cell_origins[game_map.metric_to_cell(pos)]}")

    # Viz grid
    game_map.add_units([Unit(0, (0.5, 0.5)),
                        Unit(1, (0.5, 9.5)),
                        Unit(2, (9.5, 0.5)),
                        Unit(3, (9.5, 9.5)),
                        Unit(0, (5.7, 5.7)),
                        Unit(3, (5.3, 5.3)),
                        ])
    # Add units that will result in multiple cells at same dist
    game_map.add_units([Unit(0, (0.5, 3.5)),
                        Unit(1, (0.5, 5.5))])

    # Unit Test - Unit-based occupancy
    unit_occ_grid = game_map.get_unit_occupied_cells()
    print("Test - Unit Occupancy Grid:\n", unit_occ_grid)
    # grid_rgb = game_map.get_colored_grid(unit_occ_grid, draw_units=True)
    # plt.imshow(grid_rgb)
    # plt.show()

    # Occupancy Grid
    occ_grid = game_map.compute_occupancy_map()
    print("Occupancy Grid:\n", occ_grid)
    grid_rgb = game_map.get_colored_grid(occ_grid, draw_units=True)
    plt.imshow(grid_rgb)
    plt.show()

    cv2.imwrite('grid_10x10_occupancy.png', cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR))
