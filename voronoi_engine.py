"""Game engine to handle days, unit movement"""

import copy
import logging
from typing import List

import numpy as np

from players.player import Player
from voronoi_map_state import VoronoiGameMap
from voronoi_renderer import VoronoiRender


class VoronoiEngine:
    def __init__(self, player_list, map_size=100, total_days=100):
        self.game_map = VoronoiGameMap(map_size=map_size)
        self.renderer = VoronoiRender(map_size=map_size, scale_px=10, unit_px=10)
        self.logger = logging.getLogger()

        self.total_days = total_days
        self.curr_day = -1
        self.score_total = np.zeros((4,), dtype=float)
        self.score_curr = np.zeros((4,), dtype=float)

        self.units = None
        self.occupancy_map = None
        self.history_units = {}
        self.players = []

        self.history_units[self.curr_day] = copy.deepcopy(self.game_map.units)  # Store the initial map state
        self.add_players(player_list)

    def add_players(self, players_list: List[Player]):
        if len(players_list) != 4:
            raise ValueError(f"Must have 4 players in the game. Provided: {len(players_list)}")

        for idx, pl in enumerate(players_list):
            pl.player_idx = idx
            self.players.append(pl)

    def run_all(self):
        for _ in range(self.total_days):
            self.progress_day()

    def progress_day(self):
        self.curr_day += 1
        if self.curr_day >= self.total_days:
            logging.warning(f"Played all days ({self.total_days}). Cannot progress day further. Ignoring")
            return

        # spawn units
        if self.curr_day > 0:
            self.game_map.spawn_units()

        # move units - accept inputs from each player
        move_cmds = {}
        for player in self.players:
            try:
                moves = player.play(self.game_map.units)
            except Exception as e:
                logging.error(f"Exception raised by Player {player.player_idx} ({player.name}). Ignoring this player.\n"
                              f"  Error Message: {e}")
                moves = np.zeros((len(self.game_map.units[player.player_idx]), 2), dtype=float)
            move_cmds[player.player_idx] = moves

        self.game_map.move_units(move_cmds)
        self.game_map.update()

        self.occupancy_map = self.game_map.occupancy_map

        for player in range(4):
            self.score_curr[player] = np.count_nonzero(self.occupancy_map == player)
        self.score_total += self.score_curr

        # store history
        self.units = self.game_map.units
        self.history_units[self.curr_day] = copy.deepcopy(self.game_map.units)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    player_list = [Player(), Player(), Player(), Player()]
    voronoi_engine = VoronoiEngine(player_list, map_size=100, total_days=2)
    voronoi_engine.run_all()
    print("Done")
