from typing import Dict

import numpy as np


class Player:
    def __init__(self):
        self.player_idx = None  # The index of this player in the game, between 0-3
        self.name = "Default Player"
        self.rng = None

        # TODO: Pass game engine, to access game data
        # TODO: Get spawn point,

    def set_player_idx(self, idx):
        """Assigns the player's index within the game engine"""
        self.player_idx = idx

    def set_rng(self, rng):
        """Assigns a random number generator"""
        self.rng = rng

    def play(self, units_all: Dict[int, Dict[int, Dict]], occupancy_map: np.ndarray, current_scores: np.ndarray,
             total_scores: np.ndarray, unit_history: Dict[int, Dict], curr_day: int, total_days: int):
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
        """
        # Convert to previous format - Modify as required
        unit_id = []
        unit_pos = []
        for player in range(4):
            unit_id.append(list(units_all[player].keys()))
            unit_pos.append(list(units_all[player].values()))
        game_states = occupancy_map


        # Move units
        units = list(units_all[self.player_idx].values())
        moves = np.ones((len(units), 2), dtype=float)
        moves[:, 1] = 45 * np.pi / 180
        return moves
