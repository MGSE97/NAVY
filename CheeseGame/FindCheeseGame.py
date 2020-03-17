import numpy as np
import pygame

from CheeseGame.QLearn import Agent


def clear_state(screen):
    screen.fill((255, 255, 255))
    #screen.fill((0, 0, 0))


def draw_state(sizing, screen, map, mouse, types, images, start):
    for y, row in enumerate(map):
        for x, tile in enumerate(row):
            if y == start[0] and x == start[1]:
                # draw start position
                pygame.draw.rect(screen, [200, 255, 200, 0.1], (x*sizing[0], y*sizing[1], sizing[0], sizing[1]))

            # draw map
            img = images[types.index(tile)]
            if img is not None:
                screen.blit(img, (x*sizing[0], y*sizing[1]))

    # draw mouse
    screen.blit(images[1], (mouse.Position[1]*sizing[0], mouse.Position[0]*sizing[1]))
    pygame.display.flip()


def prepare_world(size, types):
    world = np.empty(size, dtype='U1')
    # empty
    world.fill(types[0])
    # reward
    world[np.random.randint(size[1])][np.random.randint(size[0])] = types[2]
    # player
    player = [np.random.randint(size[1]), np.random.randint(size[0])]
    while world[player[0]][player[1]] == types[2]:
        player = [np.random.randint(size[1]), np.random.randint(size[0])]

    i = 0
    while i < 2:
        y = np.random.randint(0, size[1])
        x = np.random.randint(0, size[0])
        if world[y][x] != types[0]:
            continue
        world[y][x] = types[3]
        i += 1

    return [world, player]


def evaluate(position, size, world, types):
    if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= size[0] or new_pos[1] >= size[1]:
        return -1
    i = world[position[0]][position[1]]
    if i == types[2]:
        return 100
    if i == types[3]:
        return -100
    return 0


# Setup render
pygame.init()
screen_size = [400, 400]
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Cheese Game')
clock = pygame.time.Clock()
clock.tick(50)

# Setup world
size = (5, 5)
types = (" ", "*", "C", "O")
# [[dy,dx]], Up, Down, Left, Right
actions = ([-1, 0], [1, 0], [0, -1], [0, 1])
world, position = prepare_world(size, types)
visits = np.zeros((7, 7))
mouse = Agent(position, size, actions)
print("Starting location: {}".format(position))

# First draw
image_size = np.divide(screen_size, size).astype(int)
type_images = [
    None,
    pygame.transform.scale(pygame.image.load('mouse.png'), image_size),
    pygame.transform.scale(pygame.image.load('cheese.png'), image_size),
    pygame.transform.scale(pygame.image.load('hole.png'), image_size)
]
clear_state(screen)
draw_state(image_size, screen, world, mouse, types, type_images, position)

# Run game
e = 0
run = True
while run:
    mouse.Position = position
    success = False

    clear_state(screen)
    draw_state(image_size, screen, world, mouse, types, type_images, position)
    pygame.time.delay(200)

    x = 0
    while not success:
        i, action = mouse.decide()
        new_pos = np.add(action, mouse.Position)
        visits_pos = new_pos+(1, 1)
        visits[visits_pos[0]][visits_pos[1]] += 1
        val = evaluate(new_pos, size, world, types)

        clear_state(screen)

        if val == -1:
            mouse.blocked(i, val)
        elif val == -100:
            mouse.remember(new_pos, i, val)
            draw_state(image_size, screen, world, mouse, types, type_images, position)
            print("[{}] Died: Step {}, Hole at {}".format(e, x, new_pos))
            pygame.time.delay(1000)
            mouse.Position = position
        else:
            mouse.remember(new_pos, i, val)
            draw_state(image_size, screen, world, mouse, types, type_images, position)
            if val == 100:
                print("[{}] Success: Step {}, Cheese at {}".format(e, x, new_pos))
                pygame.time.delay(1000)
                success = True

        pygame.time.delay(200)
        x += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    e += 1

print(world)
print(visits)
print(mouse.__str__())
