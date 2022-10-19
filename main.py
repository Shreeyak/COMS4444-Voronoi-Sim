"""This is a simulator for Project 2 of COMS 4444 (Fall 2022) - Voronoi"""

import argparse
import atexit
import datetime
import logging

import imageio_ffmpeg
import numpy as np
import pygame

from voronoi_renderer import VoronoiRender
from voronoi_game import VoronoiEngine


class VoronoiInterface:
    def __init__(self, player_list, total_days=100, map_size=100, player_timeout=120, game_window_height=720,
                 save_video=None, fps=60, spawn_freq=1, seed=0, ignore_error=False, exit_end=False):
        """Interface for the Voronoi Game.
        Uses pygame to launch an interactive window

        Args:
            player_list: List of Player objects. Must have 4 players.
            total_days: Num of days in the simulation
            map_size: Width of the map in km. Each cell is 1x1km
            game_window_height: Width of the window the game will launch in
            player_timeout: Timeout for each player in seconds. Set to 0 to disable.
            save_video: Path to save video of game. Set to None to disable.
            fps: Frames per second for the game. Controls speed.
            spawn_freq: Frequency, in days, at which new units are spawned

        Ref:
            Pygame Design Pattern: https://www.patternsgameprog.com/discover-python-and-patterns-8-game-loop-pattern/
        """
        atexit.register(self.cleanup)  # Calls destructor

        self.exit_end = exit_end
        self.spawn_freq = spawn_freq
        self.map_size = map_size
        scale_px = game_window_height // map_size
        self.total_days = total_days
        self.game_state = VoronoiEngine(player_list, map_size=map_size, total_days=total_days,
                                        save_video=None, spawn_freq=spawn_freq, player_timeout=player_timeout,
                                        seed=seed, ignore_error=ignore_error)
        self.renderer = VoronoiRender(map_size=map_size, scale_px=scale_px, unit_px=int(scale_px / 2))

        pygame.init()
        caption = "COMS 4444: Voronoi"
        pygame.display.set_caption(caption)
        self.running = True

        # pygame creates Surface objects on which it draws graphics. Surfaces can be layered on top of each other.
        # Window contains the map and a section to the right for text
        text_w = int(game_window_height * 0.33333)  # Width of text info box
        s_width = self.renderer.img_w + text_w
        s_height = self.renderer.img_h

        # Main surface (game window). X-right, Y-down (not numpy format)
        flags = pygame.SHOWN  # | pygame.OPENGL
        self.screen = pygame.display.set_mode((s_width, s_height), flags=flags)
        # Text sub-surface
        self.text_box_surf = self.screen.subsurface(pygame.Rect((self.renderer.img_w, 0),
                                                                (text_w, self.renderer.img_h)))
        font_size = int(game_window_height / 800 * 32)  # 32 is a good size for size 800
        self.font = pygame.font.SysFont(None, font_size)  # To create text
        self.info_end = ""  # Add text info
        self.player_names = ["Default" if "d" in x else f"Group {x}" for x in self.game_state.player_names]

        # Game Map sub-surface
        self.occ_surf = self.screen.subsurface(pygame.Rect((0, 0), (self.renderer.img_w, self.renderer.img_h)))

        self.clock = pygame.time.Clock()
        self.fps = fps

        self.writer = None
        if save_video is not None:
            self.video_path = save_video
            self.frame = np.empty((s_width, s_height, 3), dtype=np.uint8)

            # disable warning (due to frame size not being a multiple of 16)
            self.writer = imageio_ffmpeg.write_frames(self.video_path, (s_width, s_height), ffmpeg_log_level="error",
                                                      fps=16, quality=9)
            self.writer.send(None)  # Seed the generator

        # Game data
        self.reset = False
        self.pause = False

    def process_input(self):
        """Handle user inputs: events such as mouse clicks and key presses"""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Close window
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    break

                elif event.key == pygame.K_r:
                    # Reset map
                    self.reset = True
                    logging.debug(f"Reset the map")

                elif event.key == pygame.K_p or event.key == pygame.K_SPACE:
                    # Reset map
                    self.pause = ~self.pause
                    logging.info(f"Game paused: {bool(self.pause)}")

    def update(self):
        """Update the state of the game"""
        if self.reset:
            self.game_state.cleanup()
            self.game_state.reset()
            self.reset = False
            return

        if self.game_state.curr_day < self.total_days - 1:
            if not self.pause:
                self.game_state.progress_day()
                self.info_end = ""
            else:
                self.info_end = "Game Paused"
        else:
            self.info_end = "Game ended.\n Press R to reset\n Esc to Quit"
            if self.writer is not None:
                self.writer.close()
                self.writer = None
                print(f"Saved video to: {self.video_path}")
            if self.exit_end:
                self.running = False

    def render(self):
        self.screen.fill((255, 255, 255))  # Blank screen

        # Draw Map
        occ_img = self.renderer.get_colored_occ_map(self.game_state.occupancy_map, self.game_state.units)
        pygame.pixelcopy.array_to_surface(self.occ_surf, np.swapaxes(occ_img, 0, 1))

        self.draw_text()

        # Update the game window to see latest changes
        pygame.display.update()

        if self.writer is not None:
            if self.game_state.curr_day < self.total_days and not self.pause:
                # Don't record past end of game or if paused
                pygame.pixelcopy.surface_to_array(self.frame, self.screen)
                frame = np.ascontiguousarray(np.swapaxes(self.frame, 0, 1))
                self.writer.send(frame)

    def draw_text(self):
        """Draw Info text on screen surface"""
        # Text box
        self.text_box_surf.fill((243, 242, 234))  # Off-White background for text
        text_color = (0, 0, 0)
        text_box_rect = self.text_box_surf.get_rect()
        pad_line = 0.1 * text_box_rect.height
        pad_top = 0.25 * text_box_rect.height
        pad_left = 0.2 * text_box_rect.width

        # Day count
        info_text = f"Day: {self.game_state.curr_day} / {self.total_days - 1}"
        text_surf = self.font.render(info_text, True, text_color)
        text_rect = text_surf.get_rect(midtop=text_box_rect.midtop)  # Position surface at center of text box
        text_rect.top += pad_top
        self.text_box_surf.blit(text_surf, text_rect.topleft)  # Draw text on text box

        # Squares with player colors for easy identification
        player_ind_size = (1.4 * text_rect.height, 1.3 * text_rect.height)  # Color box will be similar size as text
        player_ind_rect = pygame.Rect((50, 60), player_ind_size)

        # Player Count + msg
        total_score = self.game_state.score_total
        pad_line = text_rect.height * 2.5
        for idx in range(4):
            line = f"{self.player_names[idx]}: {total_score[idx]:,}"
            text_surf = self.font.render(line, True, text_color)
            # Position Text left-aligned and spaced out
            text_rect = text_surf.get_rect(left=text_box_rect.left + pad_left, top=text_box_rect.top)
            # text_rect.top += pad_top + (pad_line * (idx + 1))
            text_rect.top += pad_top + ((idx + 1) * pad_line)
            self.text_box_surf.blit(text_surf, text_rect.topleft)  # Draw text on text box

            # Position Player Indicator Box
            player_ind_rect.midleft = text_rect.midleft
            player_ind_rect.left -= player_ind_rect.width * 1.3
            color = self.renderer.player_back_colors_rgb[idx]
            pygame.draw.rect(self.text_box_surf, color, player_ind_rect)
            pygame.draw.rect(self.text_box_surf, (0, 0, 0), player_ind_rect, width=2)

        # End Game text
        pad_line = text_rect.height * 4
        init_pos = text_rect.bottom + pad_line  # Start from below player names
        if self.info_end != "":
            for idx, line in enumerate(self.info_end.split("\n")):
                text_surf = self.font.render(line, True, text_color)
                text_rect = text_surf.get_rect(midtop=text_box_rect.midtop)  # Position surface at center of text box
                line_spacing = text_rect.height * 1.5
                text_rect.top = init_pos + idx * line_spacing
                self.text_box_surf.blit(text_surf, text_rect.topleft)  # Draw text on text box

    def cleanup(self):
        # video - release and destroy windows
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            logging.info(f" Saved video to: {self.video_path}")
        pygame.quit()

    def run(self):
        print(f"\nStarting pygame.")
        print(f"Keybindings:\n"
              f"  Esc: Quit the game.\n"
              f"  P: Pause the game.\n"
              f"  R: Reset game\n")

        while self.running:
            self.process_input()
            self.update()
            self.render()
            self.clock.tick(self.fps)  # Limit updates to 60 FPS. We're much slower.

        self.cleanup()


def get_player(name: str):
    """Gets an instance of Player class given name.
    Name must match the filenames in player directory
    """
    # Avoid exceptions due to errors in other players code
    if name == "d":
        from players.default_player import Player as DefPlayer
        pl_cls = DefPlayer
    elif name == "1":
        from players.g1_player import Player as G1Player
        pl_cls = G1Player
    elif name == "2":
        from players.g2_player import Player as G2Player
        pl_cls = G2Player
    elif name == "3":
        from players.g3_player import Player as G3Player
        pl_cls = G3Player
    elif name == "4":
        from players.g4_player import Player as G4Player
        pl_cls = G4Player
    elif name == "5":
        from players.g5_player import Player as G5Player
        pl_cls = G5Player
    elif name == "6":
        from players.g6_player import Player as G6Player
        pl_cls = G6Player
    elif name == "7":
        from players.g7_player import Player as G7Player
        pl_cls = G7Player
    elif name == "8":
        from players.g8_player import Player as G8Player
        pl_cls = G8Player
    else:
        raise ValueError(f"Invalid player: {name}. Must be one of 'd' or 'g1 - g8'")
    return pl_cls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COMS 4444: Voronoi')
    parser.add_argument("--map_size", "-m", help="Size of the map in km", default=100, type=int)
    parser.add_argument("--no_gui", "-g", help="Disable GUI", action="store_true")
    # parser.add_argument("--days", "-d", help="Total number of days", default=10, type=int)
    parser.add_argument("--last", help="Total number of days", default=100, type=int)
    parser.add_argument("--spawn", default=5, type=int,
                        help="Number of days after which a new unit spawns at the homebase")
    parser.add_argument("--player1", "-p1", default="d", help="Specify player 1 out of 4. Integer equal to player id.")
    parser.add_argument("--player2", "-p2", default="d", help="Specify player 2 out of 4. Integer equal to player id.")
    parser.add_argument("--player3", "-p3", default="d", help="Specify player 3 out of 4. Integer equal to player id.")
    parser.add_argument("--player4", "-p4", default="d", help="Specify player 4 out of 4. Integer equal to player id.")
    parser.add_argument("--fps", "-f", help="Max speed of simulation", default=60, type=int)
    parser.add_argument("--timeout", "-t", default=0, type=int,
                        help="Timeout for each players execution. 0 to disable")
    parser.add_argument("--seed", "-s", type=int, default=2,
                        help="Seed used by random number generator. 0 to disable.")
    parser.add_argument("--out_video", "-o", action="store_true",
                        help="If passed, save a video of the run. Slows down the simulation 2x.")
    parser.add_argument("--ignore_error", "-i", action="store_true",
                        help="If passed, errors from players will be ignored.")
    parser.add_argument("--exit_end", "-e", action="store_true",
                        help="If passed, simulation will exit when finished. Useful when debugging.")
    args = parser.parse_args()

    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    game_window_height = 800
    map_size = args.map_size
    total_days = args.last
    fps = args.fps
    player_timeout = args.timeout
    spawn = args.spawn
    seed = args.seed
    ignore_error = args.ignore_error
    exit_end = args.exit_end

    player_name_list = [args.player1, args.player2, args.player3, args.player4]
    player_list = [(name, get_player(name)) for name in player_name_list]

    if args.out_video:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        player_str = ""
        for p in player_name_list:
            player_str += p
        save_video = f"videos/{now}-game-p{player_str}.mp4"
    else:
        save_video = None

    if args.no_gui:
        voronoi_engine = VoronoiEngine(player_list, map_size=100, total_days=total_days, save_video=save_video,
                                       spawn_freq=spawn, player_timeout=player_timeout, seed=seed,
                                       ignore_error=ignore_error)
        voronoi_engine.run_all()
    else:
        user_interface = VoronoiInterface(player_list, total_days=total_days, map_size=map_size,
                                          player_timeout=player_timeout, game_window_height=game_window_height,
                                          save_video=save_video, fps=fps,
                                          spawn_freq=spawn, seed=seed, ignore_error=ignore_error, exit_end=exit_end)
        user_interface.run()

