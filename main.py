"""This is a simulator for Project 2 of COMS 4444 (Fall 2022) - Voronoi"""

import logging

import numpy as np
import pygame

from voronoi_game import VoronoiGameMap, Unit


def pygame_main(game_map: VoronoiGameMap):
    """Main loop to run pygame"""
    pygame.init()
    caption = "COMS 4444: Voronoi"
    pygame.display.set_caption(caption)

    # We draw the map and a section below the map for text
    text_h = 40  # Height of text info box
    s_width = game_map.img_w
    s_height = game_map.img_h + text_h
    screen = pygame.display.set_mode((s_width, s_height))  # Main screen surface. X-right, Y-down (not numpy format)

    font = pygame.font.SysFont(None, 32)
    text_box_rect = pygame.rect.Rect(0, s_height - text_h, s_width, text_h)

    timeout = 300000  # milliseconds
    clock = pygame.time.Clock()
    print(f"\nStarting pygame. Game will automatically close after {timeout}ms. ")
    print(f"Keybindings:\n"
          f"  Esc: Quit the game.\n"
          f"  1-4: Select player 0-3\n"
          f"  R: Reset game\n"
          f"  Default Mode: Click to add units")

    # Create an img of the map
    _ = game_map.compute_occupancy_map()
    occ_img = game_map.get_colored_occ_map()

    _occ_img = np.swapaxes(occ_img, 0, 1)  # Req to correct coords for pygame format
    occ_surf = pygame.pixelcopy.make_surface(_occ_img)

    # Game data
    curr_player = 0  # The player whose units will be modified
    info_mode = "Click to add a unit"

    running = True
    while running:
        dt = pygame.time.get_ticks()
        if dt > timeout:
            logging.debug(f"Timeout. Quitting")
            running = False

        _ = clock.tick(30)  # Limit to 30fps

        # TODO: Remove units with click (shift-click or change mode with key)
        # TODO: Check which units are killed on keypress
        # TODO: (Low priority) Click and drag to move units
        # TODO: (Low priority) Ignore multiple clicks: If multiple clicks on the exact spot, don't add a unit

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.pos[1] > game_map.img_h:
                    continue  # Ignore clicks on the text area
                pos = game_map.px_to_metric(event.pos)
                game_map.add_units([Unit(curr_player, pos[::-1])])
                _ = game_map.compute_occupancy_map()
                occ_img = game_map.get_colored_occ_map()
                logging.debug(f"Added unit: Player: {curr_player}, Pos: {pos}")

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif pygame.K_0 <= event.key <= pygame.K_9:
                    pl_map = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}
                    curr_player = pl_map[event.key]
                    logging.debug(f"Player set to: {curr_player}")
                elif event.key == pygame.K_r:
                    game_map.reset()
                    occ_img = game_map.get_colored_occ_map()
                    logging.debug(f"Reset the map")

        # For future flexibility, keep main screen separate from the map we'll draw
        screen.fill((255, 255, 255))

        # Update the img surface
        pygame.pixelcopy.array_to_surface(occ_surf, np.swapaxes(occ_img, 0, 1))

        # Draw the img surface on the screen surface
        screen.blit(occ_surf, (0, 0))

        # Add Info text
        info_text = f"Player: {curr_player}   " + info_mode
        text_surf = font.render(info_text, True, (0, 0, 0))
        text_rect = text_surf.get_rect()  # Size of the text
        text_rect.center = text_box_rect.center  # Center text in text box
        screen.blit(text_surf, text_rect.topleft)

        pygame.display.update()

    pygame.quit()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    disp_width = 800
    map_size = 100
    scale_px = disp_width // map_size
    game_map = VoronoiGameMap(map_width=map_size, scale_px=scale_px, unit_px=5)
    pygame_main(game_map)
