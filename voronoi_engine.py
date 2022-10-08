"""Game engine to handle days, unit movement"""

import logging
from collections import deque
from typing import Tuple, List

import matplotlib as mpl
import numpy as np
import scipy

from voronoi_map_state import VoronoiGameMap
from voronoi_renderer import VoronoiRender

import time

class VoronoiEngine:
    def __init__(self, map_size=100, total_days=100):
        self.game_map = VoronoiGameMap(map_size=map_size)
        self.renderer = VoronoiRender(map_size=map_size, scale_px=10, unit_px=10)
        self.logger = logging.getLogger()

        self.total_days = total_days
        self.curr_day = 0
        self.score_total = np.zeros((4,), dtype=float)
        self.score_curr = np.zeros((4,), dtype=float)

        self.units = None
        self.occupancy_map = None
        self.history_units = {}

    def reset_game(self):
        self.game_map.reset_game()
        self.curr_day = -1
        self.score_total = np.zeros((4,), dtype=float)
        self.score_curr = np.zeros((4,), dtype=float)
        self.units = None
        self.occupancy_map = None
        self.history_units = {}

    def run(self):
        for day in range(self.total_days):
            self.curr_day = day
            self.progress_day()

    def progress_day(self):
        # move units - accept inputs from each player
        self.occupancy_map = self.game_map.occupancy_map

        for player in range(4):
            self.score_curr[player] = np.count_nonzero(self.occupancy_map == player)
        self.score_total += self.score_curr

        # store history
        self.units = self.game_map.units
        self.history_units[self.curr_day] = self.units


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    voronoi_engine = VoronoiEngine(map_size=100, total_days=2)
    voronoi_engine.run()

