"""This is a recreation of the visualization for context steering as seen in the game devlog
https://www.youtube.com/watch?v=6BrZryMz-ac&t=178s

Game Art Credits:
    Orbs: fsvieira: https://opengameart.org/content/vector-svg-balls-with-5-colors
    Pastel Balls: VictrisGames: https://opengameart.org/content/pastel-colored-balls-spheres
"""

import argparse
import logging
import math
import random
from pathlib import Path

import numpy as np
import pygame
import scipy.spatial


class GameState:
    ADD = 0  # add units
    MOVE = 1  # move units
    DEL = 2  # delete units


class Unit(pygame.sprite.Sprite):
    def __init__(self, player: int, pos: tuple, mute: bool = True):
        super().__init__()
        path_img_fr = "assets/bluesphere.png"
        path_img_ene = "assets/magentasphere.png"
        imgs = {
            0: path_img_fr,
            1: path_img_ene,
        }

        self.image = pygame.image.load(imgs[player])
        self.rect = self.image.get_rect()
        self.set_pos(pos)

        # Sound Effects
        path_adds = list(Path("assets/add").glob("*.wav"))
        path_dels = list(Path("assets/del").glob("*.wav"))
        self.sound_add = pygame.mixer.Sound(random.choice(path_adds))
        self.sound_die = pygame.mixer.Sound(random.choice(path_dels))

        pygame.mixer.init()
        if mute:
            self.sound_add.set_volume(0.0)
            self.sound_die.set_volume(0.0)

        self.sound_add.play()

    def set_pos(self, pos):
        self.rect.center = pos

    def die(self):
        self.sound_die.play()
        self.kill()


class ContextSteer:
    def __init__(self, map_size, scale_px):
        self.origin = np.array([map_size/2.0, map_size/2.0])  # Position for vis
        self.scale_px = scale_px
        self.scale_vec = 30  # Scale unit vector to this size for drawing

        self.valid_enemy_angle = 10  # Degrees. If avoid within this angle of target, consider it as in-line with target
        self.scale_avoid = 0.33  # In reality, the scale would depend on how far the avoid vec is, plus adjusted for str
        self.context_res = 16  # number of angles for context steering, equidistant around origin
        self.context_vecs = {}  # Holds the unit vectors repr each direction to be considered
        self.context_angles = np.array([180 - (360.0 / self.context_res) * x for x in range(self.context_res)])  # Hole
        for angle in self.context_angles:
            self.context_vecs[angle] = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])

        self.avoid_radius = 50  # km. If avoid target is further than this, ignore
        self.context_chase = []  # Aggregated chase vectors for each context angle
        self.context_avoid = []  # Aggregated avoid vectors for each context angle
        self.context_chase_diff = None  # Context after subtracting avoid
        self.context_avoid_diff = None
        self.vec_target = None  # Mean of chase and avoids
        self.vec_chase = None  # Closest chase
        self.vec_avoid = None  # Mean of avoids

    def pos2context_vec(self, pos):
        """Given the position of a unit, get the corresponding context vec"""
        # vec = np.array(pos) - self.origin  # shape: [2,]
        vec = np.array(pos)  # shape: [2,]

        # Find which context dir it is closest to
        angle = np.rad2deg(np.arctan2(vec[1], vec[0]))
        cosines = np.array([np.cos(np.deg2rad(x) - np.deg2rad(angle)) for x in self.context_angles])
        idx_angle = cosines.argmax()

        # Convert chase vec to vec in line with one of the context directions
        vec_context = self.context_vecs[self.context_angles[idx_angle]]
        return vec_context, idx_angle

    def weight_chase_context_vec(self, context_vec, vec):
        """Get the contribution of a chase vector towards given context directions"""
        dot = np.dot(vec, context_vec)
        weight = (1 + dot) / 2
        # circle enemy: weight = 1.0 - abs(dot)
        weighted_vec = context_vec * weight
        return weighted_vec

    def weight_avoid_context_vec(self, context_vec, vec):
        """Get the contribution of an avoid vector towards given context directions
        Avoid targets do not contribute beyond 45 degrees. We consider an angle > 45 from avoid to be safe.
        """
        mcon = np.linalg.norm(context_vec)
        mvec = np.linalg.norm(vec)
        angle = np.arccos(np.dot(context_vec, vec) / (mcon * mvec))

        # Make it so that context has zero value at 67.5deg instead of 180
        # Mappings (int to deg) - 2: 90, 3: 67.5, 4: 45
        angle = min(angle * 2, np.pi)
        angle = max(angle, -np.pi)
        dot = np.cos(angle) * mcon * mvec
        weight = (1 + dot) / 2

        weighted_vec = context_vec * weight
        return weighted_vec

    def get_context_vecs(self, chase_vec: np.ndarray, mode: str) -> np.ndarray:
        """Given a number of units, get the aggregated context vectors
        Args:
            chase_vec: np.ndarray of vectors towards targets
            mode: Whether it's a chase or avoid vector
        Returns:
            np.ndarray: Shape: [N, 2]. Returns the N context vectors, for each of the N context angles
        """
        assert chase_vec.shape == (2,)
        assert mode in ["chase", "avoid"]

        context_chase = np.zeros((self.context_res, 2))  # contrib of vec to each context angle
        if np.linalg.norm(chase_vec) < 1e-5:
            return context_chase  # Ignore vecs with zero magnitude

        # Normalize the vector
        chase_vec /= np.linalg.norm(chase_vec)

        # Find contrib of effective target vector to each of the context directions
        cvecs = [self.context_vecs[x] for x in self.context_angles]
        if mode == "chase":
            vec_contexts = [self.weight_chase_context_vec(cvec, chase_vec) for cvec in cvecs]
        else:
            vec_contexts = [self.weight_avoid_context_vec(cvec, chase_vec) for cvec in cvecs]
        context_chase = np.array(vec_contexts)

        # Normalize the vectors so that highest value has magnitude 1
        max_mag = np.linalg.norm(context_chase, axis=1).max()
        context_chase /= max_mag
        return context_chase

    def update(self, all_units):
        """Will chase the closest target"""
        num_units = len(all_units["pos"])
        chase_pos = [all_units["pos"][idx] for idx in range(num_units) if all_units["player"][idx] == 0]
        avoid_pos = [all_units["pos"][idx] for idx in range(num_units) if all_units["player"][idx] == 1]
        # Make all vec wrt self
        chase_pos = [x - self.origin for x in chase_pos]
        avoid_pos = [x - self.origin for x in avoid_pos]

        # NOTE:
        #   Enemies outside avoid radius will be ignored
        #   Enemies in same context angle as target and further away from target will be ignored

        #   Enemies will have str = 0.5
        # Find cumulative enemy vector. Get the avoid context angles wrt cum. enemy angle.
        # Subtract mag of avoid context angles from chase context angles. If resultant is <=0, delete that chase vec
        # Delete all chase context vectors that align with avoid context vectors with mag > 0.49 (1 largest vec)
        # Reform chase context vectors with new magnitudes and normalize so largest vector is str = 1.

        # Subtract mag of chase context angles from avoid context angles. If resultant is <=0, delete that avoid vec
        chase_vec = np.zeros((2,))  # Default: No target avail
        avoid_vecs = np.zeros((1, 2))  # Default: No target avail
        if len(chase_pos) > 0:
            chase_pos = np.array(chase_pos)
            chase_mags = np.linalg.norm(chase_pos, axis=1)
            chase_vec = chase_pos[chase_mags.argmin()]

        if len(avoid_pos) > 0:
            # Ignore all avoids outside the radius. Make all avoids equal str
            avoid_vecs = [np.array(x) for x in avoid_pos if np.linalg.norm(x) < self.avoid_radius]

            # Edge case - avoid in same dir as target. Ignore if behind the target
            mag_chase = np.linalg.norm(chase_vec)
            if mag_chase > 0:
                avoid_vecs_ = []
                for idx, avoid_vec in enumerate(avoid_vecs):
                    # If the avoid vec in same dir as chase, only consider it if it is closer than target
                    cosine = np.dot(chase_vec, avoid_vec) / (np.linalg.norm(chase_vec) * np.linalg.norm(avoid_vec))
                    if cosine < np.cos(np.deg2rad(self.valid_enemy_angle)):
                        avoid_vecs_.append(avoid_vec)
                    else:
                        if np.linalg.norm(avoid_vec) < mag_chase:
                            avoid_vecs_.append(avoid_vec)
                avoid_vecs = avoid_vecs_

            if len(avoid_vecs) > 0:
                avoid_vecs = np.array(avoid_vecs)
            else:
                avoid_vecs = np.zeros((1, 2))  # Default: No target avail

        # Normalize vectors
        # TODO: Scale avoids based on distance to self?
        chase_vec /= np.linalg.norm(chase_vec)  # Str = 1
        chase_vec[np.isnan(chase_vec)] = 0  # div by zero mag
        avoid_vecs /= np.linalg.norm(avoid_vecs, axis=1)[:, np.newaxis]  # Str = 1. Treat all avoids equally
        avoid_vecs[np.isnan(avoid_vecs)] = 0
        avoid_vecs *= self.scale_avoid  # Scale the avoids so they don't affect target vec too much

        # Find effective chase vector after influence of agg avoid vector
        # Note: If no chase vector, will run away from avoids
        avoid_vec = avoid_vecs.mean(axis=0)  # Aggregate avoid vecs
        self.vec_target = chase_vec - avoid_vec
        self.vec_chase = chase_vec
        self.vec_avoid = avoid_vec
        self.context_chase = self.get_context_vecs(self.vec_target, "chase")

        # Avoid ALL chase vec that point towards an avoid
        mask_avoid = np.zeros_like(self.context_chase).astype(bool)
        for avec in avoid_vecs:
            if np.linalg.norm(avec) > 1e-5:
                _, ang_idx = self.pos2context_vec(avec)
                mask_avoid[ang_idx, :] = True
        self.context_chase[mask_avoid] = 0

        # Aggregate the context vectors for all avoids - Account for the max effect along each context action
        # If we try to mean, then the effect of multiple enemies will not be accounted for correctly
        self.context_avoid = np.zeros((self.context_res, 2))
        for avec in avoid_vecs:
            context = self.get_context_vecs(avec, "avoid")
            m_con_agg = np.linalg.norm(self.context_avoid, axis=1)
            m_con = np.linalg.norm(context, axis=1)
            mask_replace = m_con > m_con_agg
            self.context_avoid[mask_replace] = context[mask_replace]

        # Subtract influence of agg avoid contexts from the chase contexts
        # If avoid greater than chase, zero out that chase instead of inverting it
        # Scale avoid vector context vectors
        self.context_avoid *= self.scale_avoid
        self.context_chase_diff = self.context_chase - self.context_avoid
        self.context_avoid_diff = self.context_avoid - self.context_chase
        mags_chase = np.linalg.norm(self.context_chase, axis=1)
        mags_avoid = np.linalg.norm(self.context_avoid, axis=1)
        self.context_chase_diff[(mags_avoid > mags_chase), :] = (0, 0)
        self.context_avoid_diff[(mags_avoid < mags_chase), :] = (0, 0)

        # Re-normalize context chase so that largest vect has mag 1
        max_mag = np.linalg.norm(self.context_chase, axis=1).max()
        if max_mag > 1e-5:
            self.context_chase /= max_mag

    def draw_vec(self, screen, vector, color, line_width=5):
        vector_ = vector * self.scale_vec
        pstart = self.origin * self.scale_px
        pend = (self.origin + vector_) * self.scale_px

        pygame.draw.line(screen, color, pstart, pend, line_width)

    def draw(self, screen, diff):
        line_width = 5
        pygame.draw.circle(screen, (70, 100, 50, 20), self.origin * self.scale_px, 30 * self.scale_px, 5)
        pygame.draw.circle(screen, (50, 30, 30, 20), self.origin * self.scale_px, self.avoid_radius * self.scale_px, 5)

        # Effective target and mean chase/avoid vector
        self.draw_vec(screen, self.vec_target * 1.1, (120, 50, 120), line_width)
        self.draw_vec(screen, self.vec_chase * 1.1, (50, 120, 120), line_width)
        self.draw_vec(screen, self.vec_avoid * 1.1, (120, 120, 50), line_width)

        # Individual chase/avoid context vectors
        if diff:
            # After subrtracting avoid from chase and vice-versa
            context_ch = self.context_chase_diff
            context_av = self.context_avoid_diff
        else:
            context_ch = self.context_chase
            context_av = self.context_avoid
        for vec in context_ch:
            self.draw_vec(screen, vec, (50, 130, 20, 150), line_width)
        # Highlight max chase vec
        go_vec = context_ch[np.argmax(np.linalg.norm(context_ch, axis=1))]
        self.draw_vec(screen, go_vec, (120, 250, 120, 150), line_width)
        for vec in context_av:
            self.draw_vec(screen, vec, (180, 50, 20), line_width)


class VoronoiInterface:
    def __init__(self, map_size, game_window_width=800, mute=False):
        """Interface for the Voronoi Game.
        Uses pygame to launch an interactive window

        Ref:
            Pygame Design Pattern: https://www.patternsgameprog.com/discover-python-and-patterns-8-game-loop-pattern/
        """
        self.map_size = map_size
        self.scale_px = game_window_width // map_size
        
        pygame.init()
        caption = "Context Steering"
        pygame.display.set_caption(caption)
        self.running = True
        self.mute = mute

        # pygame creates Surface objects on which it draws graphics. Surfaces can be layered on top of each other.
        # Window contains the map and a section below it for text
        self.img_h = self.map_size * self.scale_px
        self.img_w = self.map_size * self.scale_px

        # Main surface (game window). X-right, Y-down (not numpy format)
        flags = pygame.SHOWN  # | pygame.OPENGL
        self.screen = pygame.display.set_mode((self.img_w, self.img_h), flags=flags)

        # Game data
        self.timeout = 3000000  # milliseconds
        self.clock = pygame.time.Clock()
        self.cursor_pos = (0, 0)  # Position of the mouse cursor
        self.curr_player = 0  # 0 - friendly, 1 - enemy
        self.kdtree = None  # To find units on the map
        radius_detect_px = 16.0  # Radius in px from mouseclick where units will be detected
        self.radius_detect = radius_detect_px / self.scale_px
        self.all_units = {
            "player": [],
            "uid": [],
            "pos": [],
            "sprite": [],
        }
        self.uid = 0
        self.sound_err = pygame.mixer.Sound("assets/err/Sexy_Icantdothat2.wav")
        if self.mute:
            self.sound_err.set_volume(0)
        self.context_steer = ContextSteer(map_size, self.scale_px)
        self.view_diff = True  # testing. View the context chase angles after subtracting avoid context

        # Sprites
        self.unit_group = pygame.sprite.Group()

        # Game state
        self.reset = False
        self.f_add_unit = None
        self.f_del_unit = None
        self.move_unit_begin = None
        self.move_unit_end = None
        self.selected_unit = None  # Whether currently moving an unit
        self.state = GameState.ADD
        logging.info(f"Mode: Add units on click")
    
    def init_state(self):
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
            "pos": [],
            "sprite": []
        }
        self.uid = 0
        self.context_steer = ContextSteer(map_size, self.scale_px)
        self.view_diff = True  # testing

        # Game state
        self.reset = False
        self.f_add_unit = None
        self.f_del_unit = None
        self.move_unit_begin = None
        self.move_unit_end = None
        self.selected_unit = None  # Whether currently moving an unit
        self.state = GameState.ADD
        logging.info(f"Mode: Add units on click")

        for unit_sprite in self.unit_group:
            unit_sprite.kill()
    
    def metric_to_px(self, pos: tuple[float, float]) -> tuple[int, int]:
        """Convert metric unit pos (x, y) to pixel location on img of grid"""
        x, y = pos
        if not 0 <= x <= self.map_size:
            raise ValueError(f"x out of range [0, {self.map_size}]: {x}")
        if not 0 <= y <= self.map_size:
            raise ValueError(f"y out of range [0, {self.map_size}]: {y}")

        px, py = map(lambda z: int(round(z * self.scale_px)), [x, y])
        return px, py

    def px_to_metric(self, pos_px: tuple) -> tuple[float, float]:
        """Convert a pixel coord on map to metric
        Note: Pixels are in (row, col) format, transpose of XY Axes.
        """
        px, py = pos_px
        if not 0 <= px <= self.img_h:
            raise ValueError(f"x out of range [0, {self.img_h}]: {px}")
        if not 0 <= py <= self.img_w:
            raise ValueError(f"y out of range [0, {self.img_w}]: {py}")

        x, y = map(lambda z: round(z / self.scale_px, 2), [px, py])
        return x, y

    def build_kdtree(self):
        # Build KDTree - to find units on map
        if len(self.all_units["pos"]) > 0:
            self.kdtree = scipy.spatial.KDTree(self.all_units["pos"])
        else:
            self.kdtree = None

    def process_input(self):
        """Handle user inputs: events such as mouse clicks and key presses"""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Close window
                break

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.pos[1] >= self.img_h:
                    continue  # Ignore clicks on the text area

                # Add/move/del unit
                if self.state == GameState.ADD:
                    self.f_add_unit = True
                elif self.state == GameState.DEL:
                    self.f_del_unit = True
                elif self.state == GameState.MOVE:
                    self.move_unit_begin = True
                else:
                    raise RuntimeError
                self.move_unit_end = False

            elif event.type == pygame.MOUSEMOTION:
                if event.pos[1] >= self.img_h:
                    continue  # Ignore clicks on the text area
                self.cursor_pos = self.px_to_metric(event.pos)
                self.cursor_pos = (int(self.cursor_pos[0]) + 0.5, int(self.cursor_pos[1]) + 0.5)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.pos[1] >= self.img_h:
                    continue  # Ignore clicks on the text area
                self.move_unit_end = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    break

                elif pygame.K_1 <= event.key <= pygame.K_2:
                    # Set player
                    pl_map = {pygame.K_1: 0, pygame.K_2: 1}
                    self.curr_player = pl_map[event.key]
                    logging.info(f"Player set to: {self.curr_player}")

                elif event.key == pygame.K_a:
                    self.state = GameState.ADD
                    logging.info(f"Mode: Add units on click")
                elif event.key == pygame.K_d:
                    self.state = GameState.DEL
                    logging.info(f"Mode: Delete units on click")
                elif event.key == pygame.K_s:
                    self.state = GameState.MOVE
                    logging.info(f"Mode: Move units with click-drag")
                elif event.key == pygame.K_w:
                    self.view_diff = not self.view_diff
                    logging.info(f"View Context with Diff: {self.view_diff}")

                elif event.key == pygame.K_r:
                    # Reset map
                    self.reset = True
                    logging.debug(f"Reset the map")

    def add_unit(self):
        # cursor pos is ingame units (km)
        unit_sprite = Unit(self.curr_player, self.metric_to_px(self.cursor_pos), self.mute)
        self.unit_group.add(unit_sprite)

        self.uid += 1
        self.all_units["player"].append(self.curr_player)
        self.all_units["pos"].append(self.cursor_pos)
        self.all_units["uid"].append(self.uid)
        self.all_units["sprite"].append(unit_sprite)
        return self.uid

    def remove_unit(self, uid):
        # Deletes a unit found at given position.
        # Note, does not handle multiple units at same pos.
        idx = self.all_units["uid"].index(uid)
        unit_sprite = self.all_units["sprite"][idx]
        unit_sprite.die()

        del self.all_units["player"][idx]
        del self.all_units["pos"][idx]
        del self.all_units["uid"][idx]
        del self.all_units["sprite"][idx]

    def update(self):
        """Update the state of the game"""
        if pygame.time.get_ticks() > self.timeout:
            logging.info(f"Timeout. Quitting")
            self.running = False

        if self.reset:
            self.init_state()
            self.reset = False

        if self.f_add_unit:
            uid = self.add_unit()
            self.selected_unit = (self.curr_player, uid)  # Track which unit to move
            self.f_add_unit = False
            logging.debug(f"Added unit: Player: {self.curr_player}, Pos: {self.cursor_pos}")
            self.build_kdtree()  # Update the pos after any event

        if self.f_del_unit:
            if len(self.all_units["pos"]) > 0:
                dist, ii = self.kdtree.query(self.cursor_pos, k=1, distance_upper_bound=self.radius_detect)
                if not math.isinf(dist):
                    uid = self.all_units["uid"][ii]
                    self.remove_unit(uid)
                    logging.debug(f"Deleted unit: Uid: {uid}")
                else:
                    self.sound_err.play()
                    logging.debug(f"No unit to delete")
            else:
                self.sound_err.play()
                logging.debug(f"No unit to delete")
            self.f_del_unit = False
            self.build_kdtree()  # Update the pos after any event

        if self.move_unit_begin and len(self.all_units["pos"]) > 0:
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
            idx = self.all_units["uid"].index(uid)
            self.all_units["pos"][idx] = self.cursor_pos

            unit_sprite = self.all_units["sprite"][idx]
            unit_sprite.set_pos(self.metric_to_px(self.cursor_pos))

            if self.move_unit_end:
                self.selected_unit = None
                self.move_unit_end = False
                self.build_kdtree()  # Update the pos after any event

        self.context_steer.update(self.all_units)

    def render(self):
        self.screen.fill((30, 30, 30))  # Blank screen

        # Draw Context Steer Vectors
        self.context_steer.draw(self.screen, self.view_diff)

        # Draw Units
        self.unit_group.draw(self.screen)

        # Update the game window to see latest changes
        pygame.display.update()

    def run(self):
        print(f"\nStarting pygame. Game will automatically close after {self.timeout}ms. ")
        print(f"Keybindings:\n"
              f"  Esc: Quit the game.\n"
              f"  1-2: Select player 0-1\n"
              f"  A: Mode: Click to Add units\n"
              f"  S: Mode: Click and drag to move units\n"
              f"  D: Mode: Click to Delete units\n"
              f"  R: Reset game\n"
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
