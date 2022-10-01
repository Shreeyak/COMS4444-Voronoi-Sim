"""Draw a numpy image on the screen"""

import pygame
import numpy as np

BBLUE = (180, 200, 240)
BLUE = (50, 50, 180)
RED = (180, 50, 50)
GREEN = (50, 180, 50)

size = 800, 800
padding = 20
s_width = size[0] + padding * 2
s_height = size[1] + padding * 2


def main():
    pygame.init()
    screen = pygame.display.set_mode((s_width, s_height))  # Main screen surface. X-right, Y-down (not numpy format)

    # Create an img surface from a blank numpy image
    img = np.zeros((600, 300, 3), dtype=np.uint8)
    img = np.swapaxes(img, 0, 1)  # Req to correct coords for pygame format
    img_surf = pygame.pixelcopy.make_surface(img)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        background = BBLUE
        screen.fill(background)

        # Update the numpy array data
        img = np.swapaxes(img, 0, 1)
        img[:100, :100, :] = RED
        img[:100, -100:, :] = GREEN
        img[-100:, :100, :] = BLUE
        img = np.swapaxes(img, 0, 1)

        # Update the img surface with numpy array data
        pygame.pixelcopy.array_to_surface(img_surf, img)

        # Draw the img surface on the screen surface
        screen.blit(img_surf, (padding, padding))

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()

