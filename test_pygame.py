import pygame
import pygame.locals as pglocal
import numpy as np

BLUE = (50, 50, 180)
RED = (180, 50, 50)
GREEN = (50, 180, 50)

size = 800, 800
padding = 20
s_width = size[0] + padding
s_height = size[1] + padding



def main():
    pygame.init()

    caption = "COMS 4444: Voronoi"
    pygame.display.set_caption(caption)

    screen = pygame.display.set_mode((s_width, s_height))  # X-right, Y-down

    background = BLUE
    screen.fill(background)
    pygame.display.update()

    # Draw a circle on screen
    start = (0, 0)
    size = (20, 20)
    drawing = False

    ball = pygame.image.load("circle.png")
    ball = pygame.transform.scale(ball, (40, 40))

    rect = ball.get_rect()
    rect.move(400, 100)  # set initial pos
    speed = [2, 2]

    running = True
    while running:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    background = RED
                elif event.key == pygame.K_g:
                    background = GREEN
                print(event, event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                print(event)
                start = event.pos
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                print(event)
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                start = event.pos



        rect = rect.move(speed)
        if rect.left < 0 or rect.right > s_width:
            speed[0] = -speed[0]
        if rect.top < 0 or rect.bottom > s_height:
            speed[1] = -speed[1]

        screen.fill(background)
        pygame.draw.rect(screen, RED, rect, 1)
        screen.blit(ball, rect)

        # draw circle
        pygame.draw.ellipse(screen, RED, (start, size), 2)

        pygame.display.update()


    pygame.quit()


if __name__ == "__main__":
    main()

