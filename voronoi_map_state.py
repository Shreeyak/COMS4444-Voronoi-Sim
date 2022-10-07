from collections import deque
from typing import Tuple, List

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
    def __init__(self, map_size=100):
        """Class for methods related to the game map.
        The map is 100x100 cells, each cell is 1km wide.
        Unit coordinates on the map can be floating point.
        """
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

        # TODO: Separate func for killing units
        #   Func that does full update

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

    def spawn_home_units(self):
        """Create a unit for each player at home base"""
        units = [x for x in self.spawn_loc.items()]
        self.add_units(units)

    def reset_game(self):
        """New Game"""
        self.unit_id = 1
        self.units = {0: {}, 1: {}, 2: {}, 3: {}}

        # Game starts with 1 unit for each player in the corners
        self.spawn_home_units()
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

        for player in range(4):
            que = deque()
            reached = set()

            start = self.spawn_loc[player]
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
    print("\nOccupancy Grid (before killing units):\n", game_map.occupancy_map)

    grid_rgb = renderer.get_colored_occ_map(game_map.occupancy_map, game_map.units)

    # Remove killed unit
    connectivity_map = game_map.get_connectivity_map()
    grid_rgb_c = renderer.get_colored_occ_map(connectivity_map, game_map.units)
    killed_units = game_map.remove_killed_units()
    grid_rgb_k = renderer.get_colored_occ_map(game_map.occupancy_map, game_map.units)
    # Plot killed units
    for player, pos, _ in killed_units:
        # Draw Circle for each unit
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
