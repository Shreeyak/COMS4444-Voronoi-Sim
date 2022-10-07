"""This is a simulator for Project 2 of COMS 4444 (Fall 2022) - Voronoi"""

import argparse
import logging

import numpy as np
import pygame

from voronoi_map_state import VoronoiGameMap
from voronoi_renderer import VoronoiRender


class VoronoiInterface:
    def __init__(self, map_size, game_window_width=800):
        """Interface for the Voronoi Game.
        Uses pygame to launch an interactive window

        Ref:
            Pygame Design Pattern: https://www.patternsgameprog.com/discover-python-and-patterns-8-game-loop-pattern/
        """
        self.map_size = map_size
        scale_px = game_window_width // map_size
        self.game_state = VoronoiGameMap(map_size=map_size)
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

        self.timeout = 3000000  # milliseconds
        self.clock = pygame.time.Clock()

        # Game data
        self.curr_player = 0  # The player whose units will be modified
        self.reset = False
        self.add_unit = None
        self.kill_units = False

    def process_input(self):
        """Handle user inputs: events such as mouse clicks and key presses"""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Close window
                break

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.pos[1] > self.renderer.img_h:
                    continue  # Ignore clicks on the text area

                # Add unit
                pos = self.renderer.px_to_metric(event.pos)
                self.add_unit = pos
                logging.debug(f"Added unit: Player: {self.curr_player}, Pos: {pos}")

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    break

                elif pygame.K_0 <= event.key <= pygame.K_9:
                    # Set player
                    pl_map = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}
                    self.curr_player = pl_map[event.key]
                    logging.debug(f"Player set to: {self.curr_player}")

                elif event.key == pygame.K_k:
                    self.kill_units = True

                elif event.key == pygame.K_r:
                    # Reset map
                    self.reset = True
                    logging.debug(f"Reset the map")

    def update(self):
        """Update the state of the game"""
        if pygame.time.get_ticks() > self.timeout:
            logging.info(f"Timeout. Quitting")
            self.running = False

        if self.reset:
            self.game_state.reset_game()
            self.reset = False

        if self.add_unit is not None:
            self.game_state.add_units([(self.curr_player, self.add_unit)])
            self.game_state.compute_occupancy_map()
            self.add_unit = None

        if self.kill_units:
            self.game_state.update()
            self.kill_units = False

    def render(self):
        # self.screen.fill((255, 255, 255))  # Blank screen

        # Draw Map
        occ_img = self.renderer.get_colored_occ_map(self.game_state.occupancy_map, self.game_state.units)
        pygame.pixelcopy.array_to_surface(self.occ_surf, np.swapaxes(occ_img, 0, 1))

        # Draw Info text on screen surface
        self.text_box_surf.fill((255, 255, 255))  # White background for text
        box_rect = self.text_box_surf.get_rect()

        info_text = f"Day: {0}, Player: {self.curr_player}"
        color = (0, 0, 0)
        text_surf = self.font.render(info_text, True, color)
        text_rect = text_surf.get_rect(center=box_rect.center)  # Position surface at center of text box
        self.text_box_surf.blit(text_surf, text_rect.topleft)  # Draw text on text box

        # Update the game window to see latest changes
        pygame.display.update()

    def run(self):
        print(f"\nStarting pygame. Game will automatically close after {self.timeout}ms. ")
        print(f"Keybindings:\n"
              f"  Esc: Quit the game.\n"
              f"  1-4: Select player 0-3\n"
              f"  R: Reset game\n"
              f"Default Mode: Click to add units")

        while self.running:
            self.process_input()
            self.update()
            self.render()
            self.clock.tick(60)  # Limit updates to 60 FPS. We're much slower.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COMS 4444: Voronoi')
    parser.add_argument("--map_size", "-m", help="Size of the map in km", default=100, type=int)
    args = parser.parse_args()

    game_window_width = 800
    map_size = args.map_size
    logging.basicConfig(level=logging.INFO)

    user_interface = VoronoiInterface(map_size, game_window_width)
    user_interface.run()
    pygame.quit()
