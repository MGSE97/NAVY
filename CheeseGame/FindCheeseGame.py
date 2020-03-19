import numpy as np
import pygame

from CheeseGame.QLearn import Agent

# White Theme
#theme = [(255, 255, 255), (200, 255, 200)]
# Dark Theme
theme = [(0, 0, 0), (0, 100, 0)]
# Step render pause (ms)
speed = 150


def clear_state(screen):
    """
    Clears screen with color

    :param screen: Game screen
    """
    screen.fill(theme[0])


def draw_state(sizing, screen, map, mouse, types, images, start):
    """
    Draws whole map, mouse and starting position into screen.

    :param sizing: Picture(Tile) size
    :param screen: Game screen
    :param map: Character map with types, that represents game map (World)
    :param mouse: Mouse(Agent) object
    :param types: Map character types
    :param images: Character types images for rendering (In same order!)
    :param start: Starting position
    """
    for y, row in enumerate(map):
        for x, tile in enumerate(row):
            if y == start[0] and x == start[1]:
                # draw start position
                pygame.draw.rect(screen, theme[1], (x*sizing[0], y*sizing[1], sizing[0], sizing[1]))

            # draw map
            img = images[types.index(tile)]
            if img is not None:
                screen.blit(img, (x*sizing[0], y*sizing[1]))

    # draw mouse
    screen.blit(images[1], (mouse.Position[1]*sizing[0], mouse.Position[0]*sizing[1]))
    pygame.display.flip()


def prepare_world(size, types):
    """
    Creates random world(map) using specified types.
    It creates 1 Reward, 1 Player and 2 traps.

    :param size: Size of world (x, y)
    :param types: World types characters [empty space, player, reward, trap]
    :return: [Generated world, Player starting position]
    """
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
        if world[y][x] != types[0] or (player[0] == y and player[1] == x):
            continue
        world[y][x] = types[3]
        i += 1

    return [world, player]

def ok_bounds(position, size):
    """
    Check if position is in area

    :param position: Selected position (x, y)
    :param size: Area
    :return: If is in area
    """
    return not(position[0] < 0 or position[1] < 0 or position[0] >= size[0] or position[1] >= size[1])

def evaluate(position, size, world, types):
    """
    Evaluates selected position in the world.

    :param position: Selected position (x, y)
    :param size: Size of world(map)
    :param world: World(map)
    :param types: World types characters
    :return: Value of tile - one of [-100 trap, -1 wall, 0 empty space, 100 reward]
    """
    if not ok_bounds(position, size):
        return -1
    i = world[position[0]][position[1]]
    if i == types[2]:
        return 100
    if i == types[3]:
        return -100
    return 0

def rememberTile(agent, size, position, action, value):
    """
    Mark all outer sides of tile with value

    :param agent: Agent
    :param size: World size
    :param position: Tile position
    :param action: Action index
    :param value: Evaluated value
    """
    pos = agent.Position
    # Top
    agent.Position = position + (-1, 0)
    if ok_bounds(agent.Position, size):
        agent.remember(position, 1, value)
    # Bottom
    agent.Position = position + (1, 0)
    if ok_bounds(agent.Position, size):
        agent.remember(position, 0, value)
    # Right
    agent.Position = position + (0, -1)
    if ok_bounds(agent.Position, size):
        agent.remember(position, 3, value)
    # Left
    agent.Position = position + (0, 1)
    if ok_bounds(agent.Position, size):
        agent.remember(position, 2, value)
    # Center
    agent.Position = pos
    agent.remember(position, action, value)


'''
Main function starts from here
'''
# Setup render
pygame.init()
screen_size = [400, 400]
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Cheese Game')
clock = pygame.time.Clock()
clock.tick(max(speed/4, 10))

# Setup world and Agent
size = (5, 5)
types = (" ", "*", "C", "O")
# [[dy,dx]], Up, Down, Left, Right
actions = ([-1, 0], [1, 0], [0, -1], [0, 1])
world, position = prepare_world(size, types)
visits = np.zeros((size[0] + 2, size[1] + 2))
mouse = Agent(position, size, actions)
print("Starting location: {}".format(position))

# Prepare tile images
image_size = np.divide(screen_size, size).astype(int)
type_images = [
    None,
    pygame.transform.scale(pygame.image.load('mouse.png'), image_size),
    pygame.transform.scale(pygame.image.load('cheese.png'), image_size),
    pygame.transform.scale(pygame.image.load('hole.png'), image_size)
]

# Draw world
clear_state(screen)
draw_state(image_size, screen, world, mouse, types, type_images, position)
pygame.time.delay(speed*4)

# Run game
# Can be ended by closing window or by ALT + F4
e = 0
run = True
while run:
    success = False

    # Run search till reward is acquired
    x = 1
    while not success and run:
        # Draw current state
        clear_state(screen)
        draw_state(image_size, screen, world, mouse, types, type_images, position)
        pygame.time.delay(speed)

        # Let mouse decide on action
        i, action = mouse.decide()
        # Calculate new position
        new_pos = np.add(action, mouse.Position)
        # Log position to visit matrix
        visits_pos = new_pos+(1, 1)
        visits[visits_pos[0]][visits_pos[1]] += 1

        # Clean screen
        clear_state(screen)

        # Evaluate position
        val = evaluate(new_pos, size, world, types)
        if val == -1:
            # Mouse hit wall
            mouse.blocked(i, val)
        elif val == -100:
            # Mouse hit trap
            rememberTile(mouse, size, new_pos, i, val)
            # Draw current state
            draw_state(image_size, screen, world, mouse, types, type_images, position)
            print("[{}] Died: Step {}, Hole at {}".format(e, x, new_pos))
            pygame.time.delay(speed*4)
            # Reset mouse position
            mouse.Position = position
        else:
            # Mouse did something
            rememberTile(mouse, size, new_pos, i, val)
            #mouse.remember(new_pos, i, val)
            # Draw current state
            draw_state(image_size, screen, world, mouse, types, type_images, position)
            if val == 100:
                # If mouse hit reward, end trial and start again
                print("[{}] Success: Step {}, Cheese at {}".format(e, x, new_pos))
                pygame.time.delay(speed*4)
                success = True
                # Reset mouse position
                mouse.Position = position

        #pygame.time.delay(speed)
        x += 1

        # Allows catching events and ending game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

    e += 1

# Prints final data
print(world)
print(visits)
print(mouse.__str__())
