"""This is a simulator for Project 2 of COMS 4444 (Fall 2022) - Voronoi"""

import argparse
import atexit
import logging

import cv2
import numpy as np
import pygame

from voronoi_renderer import VoronoiRender
from voronoi_game import VoronoiEngine
from players.player import Player


class VoronoiInterface:
    def __init__(self, player_list, total_days=100, map_size=100, player_timeout=300, game_window_height=800,
                 save_video=None, fps=60):
        """Interface for the Voronoi Game.
        Uses pygame to launch an interactive window

        Args:
            map_size: Width of the map in km. Each cell is 1x1km
            game_window_height: Width of the window the game will launch in
            player_timeout: Timeout for each player

        Ref:
            Pygame Design Pattern: https://www.patternsgameprog.com/discover-python-and-patterns-8-game-loop-pattern/
        """
        atexit.register(self.cleanup)  # Calls destructor

        self.map_size = map_size
        scale_px = game_window_height // map_size
        self.total_days = total_days
        # TODO: implement player list
        self.player_list = player_list
        self.game_state = VoronoiEngine(self.player_list, map_size=map_size, total_days=total_days,
                                        save_video=None)
        self.renderer = VoronoiRender(map_size=map_size, scale_px=scale_px, unit_px=int(scale_px / 2))

        pygame.init()
        caption = "COMS 4444: Voronoi"
        pygame.display.set_caption(caption)
        self.running = True

        # pygame creates Surface objects on which it draws graphics. Surfaces can be layered on top of each other.
        # Window contains the map and a section to the right for text
        text_w = int(game_window_height * 0.77)  # Width of text info box
        s_width = self.renderer.img_w + text_w
        s_height = self.renderer.img_h

        # Main surface (game window). X-right, Y-down (not numpy format)
        flags = pygame.SHOWN  # | pygame.OPENGL
        self.screen = pygame.display.set_mode((s_width, s_height), flags=flags)
        # Text sub-surface
        self.text_box_surf = self.screen.subsurface(pygame.Rect((self.renderer.img_w, 0),
                                                                (text_w, self.renderer.img_h)))
        self.font = pygame.font.SysFont(None, 32)  # To create text
        self.info_end = ""  # Add text info

        # Game Map sub-surface
        self.occ_surf = self.screen.subsurface(pygame.Rect((0, 0), (self.renderer.img_w, self.renderer.img_h)))

        # TODO: Implement player timeout
        self.player_timeout = 3000000  # milliseconds
        self.clock = pygame.time.Clock()
        self.fps = fps

        self.create_video = False
        if save_video is not None:
            self.create_video = True
            self.video_path = save_video
            self.frame = np.array((s_width, s_height, 3), dtype=np.uint8)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec. Alt: 'avc1'
            self.writer = cv2.VideoWriter(save_video, apiPreference=0, fourcc=fourcc,
                                          fps=10, frameSize=(s_width, s_height))

        # Game data
        self.reset = False

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

    def update(self):
        """Update the state of the game"""
        if self.reset:
            self.game_state.cleanup()
            self.game_state = VoronoiEngine(self.player_list, map_size=map_size, total_days=total_days,
                                            save_video=save_video)
            self.reset = False
            return

        if self.game_state.curr_day < self.total_days - 1:
            self.game_state.progress_day()
            self.info_end = ""
        else:
            self.info_end = "Game ended. Press R to reset, Esc to Quit"

    def render(self):
        self.screen.fill((255, 255, 255))  # Blank screen

        # Draw Map
        occ_img = self.renderer.get_colored_occ_map(self.game_state.occupancy_map, self.game_state.units)
        pygame.surfarray.blit_array(self.occ_surf, np.swapaxes(occ_img, 0, 1))

        self.draw_text()

        # Update the game window to see latest changes
        pygame.display.update()

        if self.create_video and self.game_state.curr_day < self.total_days - 1:
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.swapaxes(frame, 0, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(frame)

    def draw_text(self):
        # Draw Info text on screen surface
        self.text_box_surf.fill((247, 233, 218))  # White background for text
        box_rect = self.text_box_surf.get_rect()
        color = (0, 0, 0)
        pad_v = 0.1 * box_rect.height
        pad_top = 0.25 * box_rect.height
        pad_left = 0.2 * box_rect.width

        # Day count
        info_text = f"Day: {self.game_state.curr_day} / {self.total_days - 1}"
        text_surf = self.font.render(info_text, True, color)
        text_rect = text_surf.get_rect(midtop=box_rect.midtop)  # Position surface at center of text box
        text_rect.top += pad_top
        self.text_box_surf.blit(text_surf, text_rect.topleft)  # Draw text on text box

        # Player Count + msg
        info_text = f"Player 1 ({self.player_list[0].name}): {self.game_state.score_total[0]:,}\n" \
                    f"Player 2 ({self.player_list[1].name}): {self.game_state.score_total[1]:,}\n" \
                    f"Player 3 ({self.player_list[2].name}): {self.game_state.score_total[2]:,}\n" \
                    f"Player 4 ({self.player_list[3].name}): {self.game_state.score_total[3]:,}\n"
        info_text += self.info_end
        text_lines = info_text.split("\n")
        for idx, line in enumerate(text_lines):
            text_surf = self.font.render(line, True, color)
            text_rect = text_surf.get_rect(left=box_rect.left+pad_left, top=box_rect.top)  # Position surface at center of text box
            text_rect.top += pad_top + (pad_v * (idx + 1))
            self.text_box_surf.blit(text_surf, text_rect.topleft)  # Draw text on text box

    def cleanup(self):
        # video - release and destroy windows
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logging.info(f" Saved video to: {self.video_path}")
        pygame.quit()

    def run(self):
        print(f"\nStarting pygame.")
        print(f"Keybindings:\n"
              f"  Esc: Quit the game.\n"
              f"  R: Reset game\n")

        while self.running:
            self.process_input()
            self.update()
            self.render()
            self.clock.tick(self.fps)  # Limit updates to 60 FPS. We're much slower.

        self.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COMS 4444: Voronoi')
    parser.add_argument("--map_size", "-m", help="Size of the map in km", default=100, type=int)
    parser.add_argument("--no_gui", "-g", help="Disable GUI", action="store_true")
    parser.add_argument("--days", "-d", help="Total number of days", default=10, type=int)
    parser.add_argument("--fps", "-f", help="Max speed of simulation", default=60, type=int)
    args = parser.parse_args()

    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    game_window_height = 800
    map_size = args.map_size
    total_days = args.days
    fps = args.fps
    save_video = "game.mp4"

    player_list = [Player(), Player(), Player(), Player()]
    if args.no_gui:
        voronoi_engine = VoronoiEngine(player_list, map_size=100, total_days=total_days, save_video=save_video)
        voronoi_engine.run_all()
    else:
        user_interface = VoronoiInterface(player_list, total_days=total_days, map_size=map_size, player_timeout=300,
                                          game_window_height=game_window_height, save_video=save_video, fps=fps)
        user_interface.run()

