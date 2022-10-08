"""Game engine to handle days, unit movement"""

import copy
import logging
from typing import List

import cv2
import numpy as np

from players.player import Player
from voronoi_map_state import VoronoiGameMap
from voronoi_renderer import VoronoiRender


class VoronoiEngine:
    def __init__(self, player_list, map_size=100, total_days=100, save_video=None):
        self.game_map = VoronoiGameMap(map_size=map_size)
        self.renderer = VoronoiRender(map_size=map_size, scale_px=10, unit_px=5)
        self.logger = logging.getLogger()

        self.total_days = total_days
        self.curr_day = -1
        self.score_total = np.zeros((4,), dtype=float)
        self.score_curr = np.zeros((4,), dtype=float)

        self.occupancy_map = self.game_map.occupancy_map
        self.history_units = {self.curr_day: copy.deepcopy(self.game_map.units)}  # Store the initial map state
        self.players = []

        self.add_players(player_list)

        self.create_video = False
        self.writer = None
        if save_video is not None:
            self.create_video = True
            # we are using x264 codec for mp4
            # fourcc = cv2.VideoWriter_fourcc(*'avc1')
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(save_video, apiPreference=0, fourcc=fourcc,
                                          fps=10, frameSize=(int(1000), int(1000)))
            self.write_video()  # Save 1st frame

    def add_players(self, players_list: List[Player]):
        if len(players_list) != 4:
            raise ValueError(f"Must have 4 players in the game. Provided: {len(players_list)}")

        for idx, pl in enumerate(players_list):
            pl.player_idx = idx
            self.players.append(pl)

    def run_all(self):
        for _ in range(self.total_days):
            self.progress_day()
        self.cleanup()

    def write_video(self):
        if not self.create_video:
            return
        frame = self.renderer.get_colored_occ_map(self.occupancy_map, self.game_map.units)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)

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
        self.history_units[self.curr_day] = copy.deepcopy(self.game_map.units)
        self.write_video()

    def cleanup(self):
        # video - release and destroy windows
        if self.writer is not None:
            self.writer.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # save_video = "game.mp4"
    save_video = None
    player_list = [Player(), Player(), Player(), Player()]
    voronoi_engine = VoronoiEngine(player_list, map_size=100, total_days=1000, save_video=save_video)
    voronoi_engine.run_all()
    print("Done")
