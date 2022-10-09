import logging
from collections import deque
from typing import Dict, Tuple, List, Union

import cv2
import matplotlib as mpl
import numpy as np
import scipy


def _compute_cell_coords(map_size) -> np.ndarray:
    """Calculates the origin for each cell.
    Each cell is 1km wide and origin is in the center.

    Return:
        coords: Shape: [100, 100, 2]. Coords for each cell in grid.
    """
    x = np.arange(0.5, map_size, 1.0)
    y = x
    xx, yy = np.meshgrid(x, y, indexing="ij")
    coords = np.stack((xx, yy), axis=-1)
    return coords


class VoronoiGameMap:
    def __init__(self, map_size=100, log=True):
        """Class for methods related to the game map.
        The map is 100x100 cells, each cell is 1km wide.
        Unit coordinates on the map can be floating point.
        """
        self.logger = logging.getLogger(__name__)
        if not log:
            self.logger.disabled = True

        self.map_size = map_size  # Width of the map in km. Each cell is 1km

        self.home_offset = 0.5  # Home bases are offset from corner by this amt
        self.spawn_loc = {0: (self.home_offset, self.home_offset),
                          1: (self.home_offset, self.map_size - self.home_offset),
                          2: (self.map_size - self.home_offset, self.map_size - self.home_offset),
                          3: (self.map_size - self.home_offset, self.home_offset),}

        # Optimization
        self._num_contested_pts_check = 10  # Number of closest points to check in case of disputed cells.

        # Data
        self.cell_origins = _compute_cell_coords(self.map_size)
        self.unit_id = 1  # Unique ID for each new point
        self.units = {0: {}, 1: {}, 2: {}, 3: {}}  # {player: {id: (x, y)}}
        self.occupancy_map = None  # Which cells belong to which player

        self.reset_game()

    def add_units(self, units: List[Tuple[int, Tuple[float, float]]]):
        """Add some units to the map

        Args:
            units: List of units to be added to the map. Elements are tuple -> (player, (x, y))
        """
        for (player, pos) in units:
            x, y = pos
            if not 0 <= x < self.map_size:
                raise ValueError(f"x out of range [0, {self.map_size}]: {x}")
            if not 0 <= y < self.map_size:
                raise ValueError(f"y out of range [0, {self.map_size}]: {y}")
            if not (0 <= player <= 3):
                raise ValueError(f"Player ID must be in range [0, 3]: {player}")

            self.units[player][self.unit_id] = pos  # have a unique ID for each unit on the map
            self.unit_id += 1

    def spawn_units(self):
        """Create a unit for each player at home base"""
        units = [x for x in self.spawn_loc.items()]
        self.add_units(units)

    def reset_game(self):
        """New Game"""
        self.unit_id = 1
        self.units = {0: {}, 1: {}, 2: {}, 3: {}}

        # Game starts with 1 unit for each player in the corners
        self.spawn_units()
        self.compute_occupancy_map()

    def get_unit_occupied_cells(self) -> np.ndarray:
        """Colculate which cells contain units and are occupied/disputed

        Returns:
            unit_occupancy_map: Shape: [N, N]. Maps cells to players/dispute, before nearest neighbor calculations.
                0-3: Player. 4: Disputed. 5: Not computed
        """
        occ_map = np.ones((self.map_size, self.map_size), dtype=np.uint8) * 5  # 0-3 = player, 5 = not computed

        pts_hash = {}
        for player, pl_units in self.units.items():
            for (x, y) in pl_units.values():
                pos_grid = (int(y), int(x))  # Quantize unit pos to cell idx
                if pos_grid not in pts_hash:
                    pts_hash[pos_grid] = player
                    occ_map[pos_grid] = player
                else:
                    player_existing = pts_hash[pos_grid]
                    if player_existing != player:  # Same cell, multiple players
                        occ_map[pos_grid] = 4

        return occ_map

    def compute_occupancy_map(self, mask_grid_pos: np.ndarray = None):
        """Calculates the occupancy status of each cell in the grid

        Args:
            mask_grid_pos: Shape: [N, N]. If provided, only occupancy of these cells will be computed.
                Used when updating occupancy map.
        """

        # Which cells contain units
        occ_map = self.get_unit_occupied_cells()
        occ_cell_pts = self.cell_origins[occ_map < 4]  # list of unit
        player_ids = occ_map[occ_map < 4]  # Shape: [N,]. player id for each occ cell

        # Create KD-tree with all occupied cells
        kdtree = scipy.spatial.KDTree(occ_cell_pts)

        # Query points: coords of each cell whose occupancy is not computed yet
        if mask_grid_pos is None:
            mask = (occ_map > 4)
        else:
            mask = (occ_map > 4) & mask_grid_pos
            occ_map = self.occupancy_map  # Update existing map
        candidate_cell_pts = self.cell_origins[mask]  # Shape: [N, 2]

        # For each query pt, get associated player (nearest cell with unit)
        # Find nearest 2 points to identify if multiple cells at same dist
        near_dist, near_idx = kdtree.query(candidate_cell_pts, k=2)

        # Resolve disputes for cells with more than 1 occupied cells at same distance
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

    def get_connectivity_map(self):
        """Map of all cells that have a path to their respective home base
        Uses a brute-force BFS search for each player, from their home base out through the occupancy map
        """
        occ_map = self.occupancy_map
        connected = np.ones_like(occ_map) * 4  # Default = disputed/empty

        # Neighbouring cell coords
        adj = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        h, w = occ_map.shape

        def is_safe(arr, y, x):
            # Check if it's a valid coord and value
            # arr = bool 2D map of player's occupancy map
            return 0 <= y < h and 0 <= x < w and arr[y, x]

        for player in range(4):
            que = deque()
            reached = set()

            occ_map_ = (occ_map == player)
            start = self.spawn_loc[player]
            start = (int(start[1]), int(start[0]))  # Convert to cell index
            if occ_map_[start]:
                que.append(start)

            # while len(que) > 0:
            while que:
                ele = que.popleft()
                connected[ele] = player

                # add neighbors
                for neigh in adj:
                    y = ele[0] + neigh[0]
                    x = ele[1] + neigh[1]
                    n = (y, x)
                    if n not in reached:
                        if is_safe(occ_map_, y, x):
                            que.append(n)
                            reached.add(n)

        return connected

    def remove_killed_units(self) -> List[Tuple]:
        """Remove killed units and recompute the occupancy map
        Returns:
            killed unit list: (player, (x, y), id)
        """
        killed_units = []
        connectivity_map = self.get_connectivity_map()
        for player, unit_dict in self.units.items():
            for id, pos in unit_dict.items():
                pos_grid = (int(pos[1]), int(pos[0]))
                if not connectivity_map[pos_grid] == player:
                    killed_units.append((player, pos, id))

        # Remove isolated units from unit list
        for player, pos, id in killed_units:
            self.units[player].pop(id)

        # Update the occupancy map
        self.compute_occupancy_map(connectivity_map > 3)
        return killed_units

    def move_units(self, move_cmds: Dict[int, np.ndarray], max_dist: float = 1.0):
        """Move each unit with direction and distance and update map.

        Args:
            move_cmds: For each player, direction and distance to move for each unit.
                Shape: [N, 2] - Distance, angle
            max_dist: Max distance, in km, each unit can travel
        """
        for player in range(4):
            move = move_cmds[player]

            # Vectorize calculations
            unit_pos = np.array(list(self.units[player].values()))
            unit_pos_n = np.zeros_like(unit_pos)
            if len(unit_pos) == 0:
                continue  # No units left on the map for this player

            if move.shape != unit_pos.shape:
                raise ValueError(f"Player: {player}: "
                                 f"Number of move commands ({move.shape}) must match num of units ({unit_pos.shape})")

            # Clip to max distance and move
            dist = move[:, 0]
            angle = move[:, 1]
            dist = np.clip(dist, a_max=max_dist, a_min=None)
            unit_pos_n[:, 0] = unit_pos[:, 0] + dist * np.cos(angle)
            unit_pos_n[:, 1] = unit_pos[:, 1] + dist * np.sin(angle)

            # Clip to within map bounds
            slope = np.tan(angle)
            out_bounds = (unit_pos_n < 0) | (unit_pos_n > (self.map_size - 1e-5))  # unit pos strictly less than map size
            out_bounds = out_bounds[:, 0] | out_bounds[:, 1]
            if np.count_nonzero(out_bounds) > 0:
                # X-axis out of bounds
                # new_y = y + ((new_x - x) * slope)
                pos_out = unit_pos_n[out_bounds, :]  # x1, y1
                pos_rect = np.clip(pos_out, a_min=0, a_max=self.map_size - 1e-5)  # x2, y2
                pos_rect[out_bounds, 1] = (pos_rect - pos_out)[out_bounds, 0] * slope + pos_out[out_bounds, 1]
                unit_pos_n[out_bounds, :] = pos_rect

                # Y-axis out of bounds
                # new_x = x + ((new_y - y) / slope)
                pos_out = unit_pos_n[out_bounds, :]  # x1, y1
                pos_rect = np.clip(pos_out, a_min=0, a_max=self.map_size - 1e-5)  # x2, y2
                pos_rect[out_bounds, 0] = (pos_rect - pos_out)[out_bounds, 1] / slope + pos_out[out_bounds, 0]
                unit_pos_n[out_bounds, :] = pos_rect

            # Update positions
            for id_, pos in zip(self.units[player], unit_pos_n):
                self.units[player][id_] = tuple(pos)

        return

    def update(self):
        """Update the map (after modifying units)"""
        self.compute_occupancy_map()
        self.remove_killed_units()


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    mpl.use('TkAgg')  # For macOS. Change engine.

    from voronoi_renderer import VoronoiRender

    game_map = VoronoiGameMap(map_size=10)
    renderer = VoronoiRender(map_size=10, scale_px=60, unit_px=5)

    # Viz grid
    # Add 2 units to the same cell
    game_map.add_units([(0, (5.7, 5.7)),
                        (2, (5.3, 5.3))])
    # Add units that will result in multiple cells at same dist
    game_map.add_units([(0, (3.5, 0.5)),
                        (1, (5.5, 0.5)),
                        (0, (2.5, 0.5))])

    # # Add 100 points per player randomly
    # import random
    # units = []
    # for idx in range(4):
    #     for fdx in range(10):
    #         units.append((idx, (random.random() * 10.0, random.random() * 10.0)))
    # game_map.add_units(units)

    # Test - Unit-based occupancy
    unit_occ_grid = game_map.get_unit_occupied_cells()
    print("\nTest - Unit Occupancy Grid (5 = Not computed yet):\n", unit_occ_grid)

    # Test - Occupancy Grid before killing
    game_map.compute_occupancy_map()
    print("\nOccupancy Grid (before killing units):\n", game_map.occupancy_map, "\n")

    grid_rgb = renderer.get_colored_occ_map(game_map.occupancy_map, game_map.units)

    # Test - Remove killed units
    connectivity_map = game_map.get_connectivity_map()
    grid_rgb_c = renderer.get_colored_occ_map(connectivity_map, game_map.units)
    killed_units = game_map.remove_killed_units()
    grid_rgb_k = renderer.get_colored_occ_map(game_map.occupancy_map, game_map.units)
    # Plot killed units
    for player, pos, _ in killed_units:
        pos_px = renderer.metric_to_px(pos)
        cv2.circle(grid_rgb_k, pos_px, renderer.unit_size_px, (0, 0, 0), -1)

    # Plot and save
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(grid_rgb)
    ax2.imshow(grid_rgb_c)
    ax3.imshow(grid_rgb_k)
    ax1.set_title("Occupancy before kill")
    ax2.set_title("Connectivity")
    ax3.set_title("Occupancy after kill")
    plt.show()
    # cv2.imwrite('images/grid_10x10_occupancy.png', cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR))

    # Test - Move units
    grid_rgb = renderer.get_colored_occ_map(game_map.occupancy_map, game_map.units)
    move_cmds = {}
    for player, units in game_map.units.items():
        move = np.ones((len(units), 2), dtype=float)
        move[:, 1] = 45 * np.pi / 180
        move_cmds[player] = move
    game_map.move_units(move_cmds)
    game_map.compute_occupancy_map()
    grid_rgb_m = renderer.get_colored_occ_map(game_map.occupancy_map, game_map.units)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(grid_rgb)
    ax2.imshow(grid_rgb_m)
    ax1.set_title("Occupancy before move")
    ax2.set_title("Occupancy after move")
    plt.show()
