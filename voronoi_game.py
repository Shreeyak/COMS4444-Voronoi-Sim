"""Game engine to handle days, unit movement"""

import atexit
import copy
import logging
import pickle
import signal
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import shapely.geometry

from players.default_player import Player
from voronoi_map_state import VoronoiGameMap
from voronoi_renderer import VoronoiRender


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


class VoronoiEngine:
    def __init__(self, player_list, map_size=100, total_days=100, save_video=None, log=True, spawn_freq=1,
                 player_timeout=120, seed=0, ignore_error=False):
        self.game_map = VoronoiGameMap(map_size=map_size, log=log)
        self.renderer = VoronoiRender(map_size=map_size, scale_px=10, unit_px=5)
        self.logger = logging.getLogger(__name__)
        if not log:
            self.logger.disabled = True
        atexit.register(self.cleanup)  # Calls destructor

        if seed == 0:
            seed = None
        self.rng = np.random.default_rng(seed)

        self.spawn_freq = spawn_freq
        self.player_timeout = player_timeout
        self.ignore_error = ignore_error
        self.total_days = total_days
        self.curr_day = -1
        self.score_total = np.zeros((4,), dtype=int)
        self.score_curr = np.zeros((4,), dtype=int)

        self.unit_history = {self.curr_day: copy.deepcopy(self.game_map.units)}  # Store the initial map state
        self.player_list = player_list
        self.players = []
        self.player_names = []
        self.add_players(self.player_list)

        self.create_video = False
        self.writer = None
        if save_video is not None:
            self.create_video = True
            self.video_path = Path(save_video)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec. Alt: 'avc1'
            self.writer = cv2.VideoWriter(save_video, apiPreference=0, fourcc=fourcc,
                                          fps=10, frameSize=(int(1000), int(1000)))
            self.write_video()  # Save 1st frame

    def reset(self):
        self.game_map.reset_game()
        self.curr_day = -1
        self.score_total = np.zeros((4,), dtype=int)
        self.score_curr = np.zeros((4,), dtype=int)
        self.unit_history = {self.curr_day: copy.deepcopy(self.game_map.units)}

        self.players = []
        self.player_names = []
        self.add_players(self.player_list)

    @property
    def units(self):
        return self.game_map.units

    @property
    def occupancy_map(self):
        return self.game_map.occupancy_map

    def add_players(self, players_list: List[Tuple[str, Callable]]):
        if len(players_list) != 4:
            raise ValueError(f"Must have 4 players in the game. Provided: {len(players_list)}")

        for idx, (name, PlayerCls) in enumerate(players_list):
            # pl = PlayerCls(idx, self.rng, self.game_map.spawn_loc[idx])

            spawn_point = shapely.geometry.Point(self.game_map.spawn_loc[idx])
            precomp_dir = None
            pl = PlayerCls(self.rng, self.logger, self.total_days, self.spawn_freq,
                           idx, spawn_point, 0, self.game_map.map_size, precomp_dir)
            self.players.append(pl)
            self.player_names.append(name)

    def run_all(self):
        for _ in range(self.total_days):
            self.progress_day()
        self.cleanup()

    def write_video(self):
        if self.create_video:
            frame = self.renderer.get_colored_occ_map(self.game_map.occupancy_map, self.game_map.units)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(frame)

    def progress_day(self):
        self.curr_day += 1
        if self.curr_day > self.total_days - 1:
            logging.warning(f"Played all days ({self.total_days}). Cannot progress day further. Ignoring")
            return

        # spawn units
        if (self.curr_day + 1) % self.spawn_freq == 0:
            self.game_map.spawn_units()

        # move units - accept inputs from each player
        move_cmds = {}
        for pl_idx, (player, pl_name) in enumerate(zip(self.players, self.player_names)):
            try:
                # Timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.player_timeout)

                # MOVE UNIT
                # moves = player.play(self.game_map.units, self.game_map.occupancy_map, self.score_curr,
                #                     self.score_total, self.unit_history, self.curr_day, self.total_days)

                # Cast all data to the format of official player
                unit_pos = []  # u[player][pos]
                unit_id = []
                for idx in range(4):
                    unit_pos.append([shapely.geometry.Point(pos) for id, pos in self.units[idx].items()])
                    unit_id.append([id for id, pos in self.units[idx].items()])
                # Convert occupancy map to list map_states. -1 is disputed, 1-4 is players.
                map_state = (self.occupancy_map + 1)
                map_state = map_state.astype(int)  # occ map is uint8, so cannot represent neg int
                map_state[map_state == 5] = -1
                map_states = map_state.T.tolist()
                current_scores = self.score_curr.tolist()
                total_scores = self.score_total.tolist()
                moves_ = player.play(unit_id, unit_pos, map_states, current_scores, total_scores)

                # Check return values are correct
                if len(moves_) != len(unit_pos[pl_idx]):
                    self.logger.warning(f"Player ({pl_name}) "
                                        f"provided invalid moves: Length of moves ({len(moves_)}) "
                                        f"must be same as num of units ({len(unit_pos[pl_idx])}). "
                                        f"Using 0 dist for all units.")
                    moves = np.zeros((len(self.game_map.units[pl_idx]), 2), dtype=float)
                else:
                    moves = np.array(moves_).astype(float)
                    # Nan angle:
                    nan_val = np.isnan(moves)
                    if np.any(nan_val):
                        nan_val = nan_val[:, 0] | nan_val[:, 1]
                        moves[nan_val] = 0
                        self.logger.warning(f"Player ({pl_name}) on day "
                                          f"{self.curr_day} provided invalid move. Using 0 dist for that unit.")

            except TimeoutException:
                self.logger.error(f" Timeout - Player ({pl_name}) "
                                  f"on day {self.curr_day} - play took longer than {self.player_timeout}s\n"
                                  f"NULL moves for this turn.")
                moves = np.zeros((len(self.game_map.units[pl_idx]), 2), dtype=float)

            except Exception as e:
                self.logger.error(
                    f" Exception raised by Player ({pl_name}) "
                    f"on day {self.curr_day}. NULL moves for this turn.\n"
                    f"  Error Message: {e}\n"
                    f"{traceback.format_exc()}")
                if self.ignore_error:
                    moves = np.zeros((len(self.game_map.units[pl_idx]), 2), dtype=float)
                else:
                    raise e

            signal.alarm(0)  # Clear timeout alarm

            move_cmds[pl_idx] = moves

        self.game_map.move_units(move_cmds)
        self.game_map.update()

        for player in range(4):
            self.score_curr[player] = np.count_nonzero(self.game_map.occupancy_map == player)
        self.score_total += self.score_curr

        # store history
        self.unit_history[self.curr_day] = copy.deepcopy(self.game_map.units)
        self.write_video()

        if (self.curr_day + 1) % 10 == 0:
            self.logger.info(f"Day: {self.curr_day} \tScore: {self.score_total}")
        self.logger.debug(f"Day: {self.curr_day} \tScore: {self.score_total}")

        if self.curr_day == self.total_days - 1:
            self.cleanup()  # Save video

    def cleanup(self):
        # video - release and destroy windows
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logging.info(f" Saved video to: {self.video_path}")

    def save_units_history_to_file(self, path="tests/units_history.pkl"):
        """Saves the units pos to file"""
        with open(path, "wb") as fd:
            pickle.dump(self.unit_history, fd, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    save_video = "game.mp4"
    # save_video = None
    player_list = [Player(), Player(), Player(), Player()]
    voronoi_engine = VoronoiEngine(player_list, map_size=100, total_days=100, save_video=save_video)
    voronoi_engine.run_all()
    logger.info(f" Video Saved: {Path(save_video).absolute()}")
