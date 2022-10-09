from typing import Dict, Tuple

import numpy as np

from players.player import Player


class G4Player(Player):
    def __init__(self, player_idx, rng, spawn_pt):
        super().__init__(player_idx, rng, spawn_pt)
        self.name = "G4 Player"

    def play(self,
             units_all: Dict[int, Dict[int, Tuple]],
             occupancy_map: np.ndarray,
             current_scores: np.ndarray,
             total_scores: np.ndarray,
             unit_history: Dict[int, Dict],
             curr_day: int,
             total_days: int):
        """Send movement commands to player's unit

        Args:
            units_all: Dict of all units - {player: {id: (pos_x, posy_)}}
            occupancy_map: Which unit occupies each cell on the grid. Shape: (N, N).
                Values 0-3 = player, 4 = contested
            current_scores: Current score for each player. Shape: (4,)
            total_scores: Total cumulative score for each player. Shape: (4,)
            unit_history: History of unit position over all previous days - {day: {player: {id: (posx, posy)}}}
            curr_day: Current day in simulation
            total_days: Total number of days

        Returns:
            np.ndarray: Direction and Angle for each of this player's units. Shape: [N, 2]
                0 deg = right, 90 deg = down
        """
        # Convert params to previous format
        # game_states = occupancy_map.tolist()
        unit_id = []  # List, Shape: [N]
        unit_pos = []  # List, Shape: [N, 2], N = num of units
        for player in range(4):
            unit_id.append(np.array(list(units_all[player].keys())))
            unit_pos.append(np.array(list(units_all[player].values())))

        # Move units - 0 deg = right, 90 deg = down
        units = unit_pos[self.player_idx]  # Shape: [N, 2]
        moves = np.ones_like(units)
        angle = 45 - (90 * self.player_idx)  # towards center
        moves[:, 1] = angle * np.pi / 180

        # move towards nearest enemy unit
        # units_enemy = [unit_pos[x] for x in range(4) if x != self.player_idx]
        # units_enemy = np.concatenate(units_enemy, axis=0)  # [N, 2]
        # kdtree = scipy.spatial.KDTree(units_enemy)

        return moves
