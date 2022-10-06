from typing import Tuple, List

import matplotlib as mpl
import numpy as np
import scipy


class Unit:
    def __init__(self, player: int, pos: Tuple):
        """The unit of each player. New unit spawns every N days. Can be moved in any direction by 1 km"""
        assert 0 <= player < 4

        self.player = int(player)
        self.pos = pos
        self.status = 1  # 1 = alive, 0 = dead

        # TODO: Replace usage with list of tuples. Faster.

    def kill(self):
        self.status = 0

    def move(self):
        raise NotImplementedError


class VoronoiGameMap:
    def __init__(self, map_size=100):
        """Class for methods related to the game map.
        The map is 100x100 cells, each cell is 1km wide.
        Unit coordinates on the map can be floating point.
        """
        self.map_size = map_size  # Width of the map in km. Each cell is 1km

        self.home_offset = 0.5  # Home bases are offset from corner by this amt
        self.spawn_loc = {0: (self.home_offset, self.home_offset),
                          1: (self.map_size - self.home_offset, self.home_offset),
                          2: (self.map_size - self.home_offset, self.map_size - self.home_offset),
                          3: (self.home_offset, self.map_size - self.home_offset)}

        # Optimization
        self._num_contested_pts_check = 10  # Number of closest points to check in case of disputed cells.

        # Data
        self.cell_origins = self._get_cell_origins()
        # Unit Map: Each channel represents a player. If 1, then the player has a unit in that cell, 0 otherwise.
        self.unit_map = np.zeros((self.map_size, self.map_size, 4), dtype=np.uint8)
        self.unit_id = 1  # Unique ID for each point
        self.unit_id_map = np.zeros((self.map_size, self.map_size, 4), dtype=np.uint8)  # Unit pos by ID
        self.units = []  # List of all the units on the map
        self.occupancy_map = None  # Which cells belong to which player
        self.reset_game()

        # TODO: Separate func for killing units
        #   Func that does full update

    def clear_board(self):
        """Remove all the units. Clear the associated data structures"""
        self.units = []
        self.unit_map = np.zeros((self.map_size, self.map_size, 4), dtype=np.uint8)
        self.unit_id_map = np.zeros((self.map_size, self.map_size, 4), dtype=np.uint8)

    def reset_game(self):
        """New Game"""
        self.clear_board()
        self.unit_id = 1
        # Game starts with 1 unit for each player in the corners
        self.add_units([Unit(0, self.spawn_loc[0]),
                        Unit(1, self.spawn_loc[1]),
                        Unit(2, self.spawn_loc[2]),
                        Unit(3, self.spawn_loc[3])])
        self.compute_occupancy_map()

    def _get_cell_origins(self) -> np.ndarray:
        """Calculates the origin for each cell
        Return:
            coords: Coords for each cell, indexed by cell location. np.ndarray, Shape: [100, 100, 2].
        """
        x = np.arange(0.5, self.map_size, 1.0)
        y = x.copy()

        xx, yy = np.meshgrid(x, y, indexing="ij")
        coords = np.stack((xx, yy), axis=-1)
        return coords

    def add_units(self, units: List[Unit]):
        self.units.extend(units)
        for unit in units:
            x, y = unit.pos
            if not 0 <= x < self.map_size:
                raise ValueError(f"x out of range [0, {self.map_size}]: {x}")
            if not 0 <= y < self.map_size:
                raise ValueError(f"y out of range [0, {self.map_size}]: {y}")

            cx, cy = int(x), int(y)

            # TODO: Separate func to compute unit map
            self.unit_map[cx, cy, unit.player] = 1
            self.unit_id_map[cx, cy, unit.player] = self.unit_id
            self.unit_id += 1

    def get_unit_occupied_cells(self) -> np.ndarray:
        """Calculate which cells are counted as occupied due to unit presence (based on unit map)
        If a cell contains exactly 1 unit, then it's occupied by that unit's player.

        Returns:
            unit_occupancy_map: 2D Map that shows which cells are occupied by each player due to unit presence, before
                nearest neighbor calculations. Shape: [M, M].
                0-3: Player. 4: Disputed. 5: Not computed
        """
        # TODO: Replace this is hashed unit list.
        # Get player-wise cell occupancy. If a cell has exactly 1 unit, it's occupied. More than 1, it's disputed.
        num_units = self.unit_map.sum(axis=2)
        occupied_mask_2d = (num_units == 1).reshape((self.map_size, self.map_size, 1))
        occupied_mask_2d = np.logical_and(occupied_mask_2d, self.unit_map > 0)  # Shape: [N, N, 4]

        # 2D map that shows which cells are occupied by a player's unit. 4 means contested. 5 is uncomputed.
        occ_map = np.ones((self.map_size, self.map_size), dtype=np.uint8) * 5
        occ_map[occupied_mask_2d[:, :, 0]] = 0
        occ_map[occupied_mask_2d[:, :, 1]] = 1
        occ_map[occupied_mask_2d[:, :, 2]] = 2
        occ_map[occupied_mask_2d[:, :, 3]] = 3
        occ_map[num_units > 1] = 4

        return occ_map

    def compute_occupancy_map(self):
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
        disputed_cell_pts = candidate_cell_pts[disputed]  # Shape: [N, 2].
        if disputed_cell_pts.shape[0] > 0:
            # Distance of nearest cell will be radius of our search
            radius_of_dispute = near_dist[disputed, 0]  # Shape: [N, ]
            occ_map = self._filter_disputes(occ_map, kdtree, disputed_cell_pts, radius_of_dispute, player_ids)

        # For the rest of the cells (undisputed), mark occupancy
        not_disputed_ids = player_ids[near_idx[~disputed, 0]]  # Get player id of the nearest cell
        not_disputed_cells = candidate_cell_pts[~disputed].astype(int)  # cell idx from coords of occupied cells
        occ_map[not_disputed_cells[:, 0], not_disputed_cells[:, 1]] = not_disputed_ids

        self.occupancy_map = occ_map
        return

    def _filter_disputes(self, occ_map, kdtree, disputed_cell_pts, radius_of_dispute, player_ids):
        """For each cell with multiple nearby neighbors, resolve dispute
        Split into a func for profiling
        """
        # Find all neighbors within a radius: If all neigh are same player, cell not disputed
        for disp_cell, radius in zip(disputed_cell_pts, radius_of_dispute):
            # Radius needs padding to conform to < equality.
            rad_pad = 0.1
            d_near_dist, d_near_idx = kdtree.query(disp_cell,
                                                   k=self._num_contested_pts_check,
                                                   distance_upper_bound=radius + rad_pad)
            # We will get exactly as many points as requested. Extra points will have inf dist
            # Need to filter those points that are within radius (dist < inf).
            valid_pts = np.isfinite(d_near_dist)
            d_near_idx = d_near_idx[valid_pts]

            disputed_ids = player_ids[d_near_idx]  # Get player ids of the contesting cells
            all_same_ids = np.all(disputed_ids == disputed_ids[0])
            if all_same_ids:
                # Mark cell belonging to this player
                player = disputed_ids[0]
            else:
                # Mark cell as contested
                player = 4
            disp_cell = disp_cell.astype(int)  # Shape: [2,]
            occ_map[disp_cell[0], disp_cell[1]] = player

        return occ_map

    def remove_killed_units(self):
        """Remove killed units and Recompute the occupancy map"""
        units_ = self.units.copy()
        self.clear_board()

        for unit in units_:
            if unit.status > 0:
                self.add_units([unit])
        self.compute_occupancy_map()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    mpl.use('TkAgg')  # For macOS. Change engine.
    from voronoi_renderer import VoronoiRender

    game_map = VoronoiGameMap(map_size=10)
    renderer = VoronoiRender(map_size=10, scale_px=60, unit_px=5)

    # Viz grid
    # Add 2 units to the same cell
    game_map.add_units([Unit(0, (5.7, 5.7)),
                        Unit(2, (5.3, 5.3))])
    # Add units that will result in multiple cells at same dist
    game_map.add_units([Unit(0, (0.5, 3.5)),
                        Unit(1, (0.5, 5.5)),
                        Unit(0, (0.5, 2.5))])

    # # Add 100 points per player randomly
    # import random
    # units = []
    # for idx in range(4):
    #     for fdx in range(100):
    #         units.append(
    #             Unit(idx, (random.random() * 100.0, random.random() * 100.0))
    #         )
    # game_map.add_units(units)

    # Test - Unit-based occupancy
    unit_occ_grid = game_map.get_unit_occupied_cells()
    print("\nTest - Unit Occupancy Grid (5 = Not computed yet):\n", unit_occ_grid)

    # Full Occupancy Grid
    game_map.compute_occupancy_map()
    print("\nOccupancy Grid:\n", game_map.occupancy_map)

    # Plot and save
    grid_rgb = renderer.get_colored_occ_map(game_map.occupancy_map, game_map.units)
    plt.imshow(grid_rgb)
    plt.show()
    # cv2.imwrite('images/grid_10x10_occupancy.png', cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR))
