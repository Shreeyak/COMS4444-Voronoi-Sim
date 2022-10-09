from typing import Dict

import numpy as np


class Player:
    def __init__(self, rng=None):
        self.player_idx = None
        self.name = "Default Player"
        self.rng = rng

        # TODO: Pass game engine, to access game data
        # TODO: Get spawn point,
        # TODO: Pass random generator obj
        # TODO: Pass a precomp dir

    def set_player_idx(self, idx):
        """Assigns the player's index within the game engine"""
        self.player_idx = idx

    def set_rng(self, rng):
        """Assigns a random number generator"""
        self.rng = rng

    def play(self, units: Dict[int, Dict[int, Dict]]):
        """Send movement commands to player's unit
        Args:
            units: Dict of all units - {player: {id: (pos)}}

        Returns:
            np.ndarray: Direction and Angle for each of this player's units. Shape: [N, 2]
        """
        units_pl = units[self.player_idx]
        moves = np.ones((len(units_pl), 2), dtype=float)
        moves[:, 1] = 45 * np.pi / 180
        return moves
