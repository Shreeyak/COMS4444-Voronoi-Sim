"""This is a simulator for Project 2 of COMS 4444 (Fall 2022) - Voronoi"""

import argparse
import logging

import numpy as np
import pygame

from voronoi_map_state import VoronoiGameMap
from voronoi_renderer import VoronoiRender
from voronoi_game import VoronoiEngine
from players.player import Player


class VoronoiInterface:
    def __init__(self, player_list, total_days=100, map_size=100, player_timeout=300, game_window_width=800,
                 save_video=None):
        """Interface for the Voronoi Game.
        Uses pygame to launch an interactive window

        Args:
            map_size: Width of the map in km. Each cell is 1x1km
            game_window_width: Width of the window the game will launch in
            player_timeout: Timeout for each player

        Ref:
            Pygame Design Pattern: https://www.patternsgameprog.com/discover-python-and-patterns-8-game-loop-pattern/
        """
        self.map_size = map_size
        scale_px = game_window_width // map_size
        self.total_days = total_days
        # TODO: implement player list
        self.player_list = player_list
        self.game_state = VoronoiEngine(self.player_list, map_size=map_size, total_days=total_days,
                                        save_video=save_video)
        self.renderer = VoronoiRender(map_size=map_size, scale_px=scale_px, unit_px=int(scale_px / 2))

        pygame.init()
        caption = "COMS 4444: Voronoi"
        pygame.display.set_caption(caption)
        self.running = True

        # pygame creates Surface objects on which it draws graphics. Surfaces can be layered on top of each other.
        # Window contains the map and a section below it for text
        text_h = 40  # Height of text info box
        s_width = self.renderer.img_w
        s_height = self.renderer.img_h + text_h

        # Main surface (game window). X-right, Y-down (not numpy format)
        flags = pygame.SHOWN  # | pygame.OPENGL
        self.screen = pygame.display.set_mode((s_width, s_height), flags=flags)
        # Text sub-surface
        self.text_box_surf = self.screen.subsurface(pygame.Rect((0, self.renderer.img_h), (self.renderer.img_w, text_h)))
        self.font = pygame.font.SysFont(None, 32)  # To create text
        # Game Map sub-surface
        self.occ_surf = self.screen.subsurface(pygame.Rect((0, 0), (self.renderer.img_w, self.renderer.img_h)))

        # TODO: Implement player timeout
        self.player_timeout = 3000000  # milliseconds
        self.clock = pygame.time.Clock()

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

    def render(self):
        self.screen.fill((255, 255, 255))  # Blank screen

        # Draw Map
        occ_img = self.renderer.get_colored_occ_map(self.game_state.occupancy_map, self.game_state.game_map.units)
        pygame.pixelcopy.array_to_surface(self.occ_surf, np.swapaxes(occ_img, 0, 1))

        # Draw Info text on screen surface
        self.text_box_surf.fill((255, 255, 255))  # White background for text
        box_rect = self.text_box_surf.get_rect()

        info_text = f"Day: {self.game_state.curr_day} / {self.total_days - 1}"
        color = (0, 0, 0)
        text_surf = self.font.render(info_text, True, color)
        text_rect = text_surf.get_rect(center=box_rect.center)  # Position surface at center of text box
        self.text_box_surf.blit(text_surf, text_rect.topleft)  # Draw text on text box

        # Update the game window to see latest changes
        pygame.display.update()

    def run(self):
        print(f"\nStarting pygame.")
        print(f"Keybindings:\n"
              f"  Esc: Quit the game.\n"
              f"  R: Reset game\n")

        while self.running:
            self.process_input()
            self.update()
            self.render()
            self.clock.tick(60)  # Limit updates to 60 FPS. We're much slower.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COMS 4444: Voronoi')
    parser.add_argument("--map_size", "-m", help="Size of the map in km", default=100, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    game_window_width = 800
    map_size = args.map_size
    total_days = 1000
    save_video = "game.mp4"

    player_list = [Player(), Player(), Player(), Player()]
    user_interface = VoronoiInterface(player_list, total_days=total_days, map_size=map_size, player_timeout=300,
                                      game_window_width=game_window_width, save_video=save_video)
    user_interface.run()
    pygame.quit()
