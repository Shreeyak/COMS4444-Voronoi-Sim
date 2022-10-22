"""This is a simulator for Project 2 of COMS 4444 (Fall 2022) - Voronoi"""

import argparse
import logging
import math

import numpy as np
import pygame
import scipy.spatial

from voronoi_map_state import VoronoiGameMap
from voronoi_renderer import VoronoiRender


class GameState:
    ADD = 0  # add units
    MOVE = 1  # move units
    DEL = 2  # delete units


class VoronoiInterface:
    def __init__(self, map_size, game_window_width=800):
        """Interface for the Voronoi Game.
        Uses pygame to launch an interactive window

        Ref:
            Pygame Design Pattern: https://www.patternsgameprog.com/discover-python-and-patterns-8-game-loop-pattern/
        """
        self.map_size = map_size
        self.scale_px = game_window_width // map_size
        self.game_state = VoronoiGameMap(map_size=map_size)
        self.renderer = VoronoiRender(map_size=map_size, scale_px=self.scale_px, unit_px=int(self.scale_px / 2))

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

        # Game data
        self.timeout = 3000000  # milliseconds
        self.clock = pygame.time.Clock()
        self.curr_player = 0  # The player whose units will be modified
        self.kdtree = None  # To find units on the map
        radius_detect_px = 16.0  # Radius in px from mouseclick where units will be detected
        self.radius_detect = radius_detect_px / self.scale_px
        self.all_units = {
            "player": [],
            "uid": [],
            "pos": []
        }
        self.cursor_pos = (0, 0)  # Position of the mouse cursor

        # Game state
        self.reset = False
        self.add_unit = None
        self.del_unit = None
        self.move_unit_begin = None
        self.move_unit_end = None
        self.selected_unit = None  # Whether currently moving an unit
        self.kill_units_on_isolate = True
        self.state = GameState.ADD
        logging.info(f"Mode: Add units on click")
        self.click_active = False  # Whether a mouse-click action has started

    def init_state(self):
        # Game data
        self.timeout = 3000000  # milliseconds
        self.clock = pygame.time.Clock()
        self.curr_player = 0  # The player whose units will be modified
        self.all_units = {
            "player": [],
            "uid": [],
            "pos": []
        }

        # Game state
        self.reset = False
        self.add_unit = None
        self.del_unit = None
        self.move_unit_begin = None
        self.move_unit_end = None
        self.selected_unit = None  # Whether currently moving an unit
        self.kill_units_on_isolate = True
        self.state = GameState.ADD
        logging.info(f"Mode: Add units on click")
        self.click_active = False  # Whether a mouse-click action has started

    def process_input(self):
        """Handle user inputs: events such as mouse clicks and key presses"""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Close window
                break

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.pos[1] >= self.renderer.img_h:
                    continue  # Ignore clicks on the text area

                # Add/move/del unit
                if self.state == GameState.ADD:
                    self.add_unit = True
                elif self.state == GameState.DEL:
                    self.del_unit = True
                elif self.state == GameState.MOVE:
                    self.move_unit_begin = True
                else:
                    raise RuntimeError
                self.move_unit_end = False

            elif event.type == pygame.MOUSEMOTION:
                if event.pos[1] >= self.renderer.img_h:
                    continue  # Ignore clicks on the text area
                self.cursor_pos = self.renderer.px_to_metric(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.pos[1] >= self.renderer.img_h:
                    continue  # Ignore clicks on the text area

                if self.state == GameState.MOVE or self.state == GameState.ADD:
                    self.move_unit_end = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    break

                elif pygame.K_0 <= event.key <= pygame.K_9:
                    # Set player
                    pl_map = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}
                    self.curr_player = pl_map[event.key]
                    logging.debug(f"Player set to: {self.curr_player}")

                elif event.key == pygame.K_a:
                    self.state = GameState.ADD
                    logging.info(f"Mode: Add units on click")
                elif event.key == pygame.K_d:
                    self.state = GameState.DEL
                    logging.info(f"Mode: Delete units on click")
                elif event.key == pygame.K_s:
                    self.state = GameState.MOVE
                    logging.info(f"Mode: Move units with click-drag")
                elif event.key == pygame.K_k:
                    self.kill_units_on_isolate = not self.kill_units_on_isolate
                    logging.info(f"Mode: Unit killed if isolated: {self.kill_units_on_isolate}")

                elif event.key == pygame.K_r:
                    # Reset map
                    self.reset = True
                    logging.debug(f"Reset the map")

    def build_kdtree(self):
        self.all_units = {
            "player": [],
            "uid": [],
            "pos": []
        }
        # Build KDTree - to find units on map
        for player in range(4):
            for uid, pos in self.game_state.units[player].items():
                self.all_units["player"].append(player)
                self.all_units["uid"].append(uid)
                self.all_units["pos"].append(pos)

        self.kdtree = scipy.spatial.KDTree(self.all_units["pos"])

    def kill_unit_if_isolated(self, player, uid):
        connectivity_map = self.game_state.get_connectivity_map()
        pos = self.game_state.units[player][uid]
        pos_grid = (int(pos[1]), int(pos[0]))
        if not connectivity_map[pos_grid] == player:
            self.game_state.units[player].pop(uid)
            return True
        else:
            return False

    def update(self):
        """Update the state of the game"""
        if pygame.time.get_ticks() > self.timeout:
            logging.info(f"Timeout. Quitting")
            self.running = False

        if self.reset:
            self.game_state.reset_game()
            self.init_state()
            self.reset = False

        if self.add_unit:
            uid = self.game_state.add_units([(self.curr_player, self.cursor_pos)])[0]
            self.game_state.compute_occupancy_map()
            # If the unit is isolated, kill it. Otherwise, it may kill other units on game update
            if self.kill_units_on_isolate:
                self.kill_unit_if_isolated(self.curr_player, uid)
            self.selected_unit = (self.curr_player, uid)  # Track which unit to move
            self.add_unit = False
            logging.debug(f"Added unit: Player: {self.curr_player}, Pos: {self.cursor_pos}")

        self.build_kdtree()
        if self.del_unit:
            dist, ii = self.kdtree.query(self.cursor_pos, k=1, distance_upper_bound=self.radius_detect)
            if not math.isinf(dist):
                uid = self.all_units["uid"][ii]
                player = self.all_units["player"][ii]
                self.game_state.units[player].pop(uid)
                self.game_state.update()
                logging.debug(f"Deleted unit: Player: {player}, Uid: {uid}")
            else:
                logging.debug(f"No unit to delete")
            self.del_unit = False

        if self.move_unit_begin:
            dist, ii = self.kdtree.query(self.cursor_pos, k=1, distance_upper_bound=self.radius_detect)
            if not math.isinf(dist):
                uid = self.all_units["uid"][ii]
                player = self.all_units["player"][ii]
                self.selected_unit = (player, uid)  # Track which unit to move
            else:
                self.selected_unit = None
            self.move_unit_begin = False

        if self.selected_unit is not None:
            # Move the selected unit
            player, uid = self.selected_unit
            self.game_state.units[player][uid] = self.cursor_pos
            self.game_state.compute_occupancy_map()

            if self.move_unit_end:
                # If the unit is isolated, kill it. Otherwise, it may kill other units on game update
                if self.kill_units_on_isolate:
                    self.kill_unit_if_isolated(player, uid)
                self.selected_unit = None
                self.move_unit_end = False

        if self.kill_units_on_isolate and self.selected_unit is None:
            # When moving, we want to visualize the voronoi cells. Don't kill the poor unit yet.
            self.game_state.update()

    def render(self):
        # self.screen.fill((255, 255, 255))  # Blank screen

        # Draw Map
        occ_img = self.renderer.get_colored_occ_map(self.game_state.occupancy_map, self.game_state.units)
        pygame.pixelcopy.array_to_surface(self.occ_surf, np.swapaxes(occ_img, 0, 1))

        # Draw Info text on screen surface
        self.text_box_surf.fill((255, 255, 255))  # White background for text
        box_rect = self.text_box_surf.get_rect()

        info_text = f"Player Selected: {self.curr_player}"
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
              f"  A: Mode: Click to Add units\n"
              f"  S: Mode: Click and drag to move units\n"
              f"  D: Mode: Click to Delete units\n"
              f"  R: Reset game\n"
              f"  K: Toggle - Kill isolated units\n"
              f"Interactive Mode")

        while self.running:
            self.process_input()
            self.update()
            self.render()
            self.clock.tick(60)  # Limit updates to 60 FPS. We're much slower.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COMS 4444: Voronoi')
    parser.add_argument("--map_size", "-m", help="Size of the map in km", default=100, type=int)
    args = parser.parse_args()

    game_window_width = 1400
    map_size = args.map_size
    logging.basicConfig(level=logging.INFO)

    user_interface = VoronoiInterface(map_size, game_window_width)
    user_interface.run()
    pygame.quit()
