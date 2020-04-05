import math
import time
import turtle

import numpy as np
from LSystems.Utils import ScrolledTurtle

'''
patterns to render in format:
[
    [axiom, rule, color, angle, starting angle, starting position, scale]
    , ...
]
'''
patterns = [
    ["F+F+F+F", "F+F-F-FF+F+F-F", "green", 90, 0, "bottom|left", 0.5],
    ["F++F++F", "F+F--F+F", "blue", 60, 180, "top|right", 1.0],
    ["F", "F[+F]F[-F]F", "red", math.degrees(math.pi/7.0), 90, "bottom", 2.0],
    ["F", "FF+[+F-F-F]-[-F+F+F]", "orange", math.degrees(math.pi/8.0), 90, "bottom|right", 5.0]
]

# Maximum depth to render
max_depth = 6

# Drawing cache
cache = []


def execute_system(skk, angle, rule, max_depth, depth=0, axiom=None, scale=1.0):
    """
    Render L-System using turtle up to maximal depth

    :param skk: Turtle
    :param angle: Pattern Angle
    :param rule: Pattern Rule
    :param max_depth: Maximum render depth
    :param depth: Current depth
    :param axiom: Starting Axiom
    :param scale: Rendering Scale
    :return: None
    """
    # Lower step by depth
    step = (100*scale) ** (1/float(depth)) if depth > 1 else 100

    if step <= 1:
        # Do not go deeper than 1px
        skk.forward(step)
        return
    if depth >= max_depth:
        # Stop at max depth
        skk.forward(step)
        return

    # Recursive Renderer
    data = axiom if axiom is not None else rule
    for char in data:
        if char == 'F':
            execute_system(skk, angle, rule, max_depth, depth + 1, scale=scale)
        elif char == '+':
            skk.left(angle)
        elif char == '-':
            skk.right(angle)
        elif char == '[':
            cache.append([skk.pos(), skk.heading()])
        elif char == ']':
            skk.up()
            pos, heading = cache.pop()
            skk.setpos(pos)
            skk.setheading(heading)
            skk.down()


# Create screen
wn = turtle.Screen()
wn.title("L-Systems")

skks = []
w, h = [0, 0]
# Draw all patterns with depths
for depth in range(1, max_depth+1):
    for i, (axiom, rule, color, angle, heading, position, scale) in enumerate(patterns):
        print("Depth: {}, Color: {}, Pattern: {}, Axiom: {}".format(depth, color, rule, axiom))

        # Get or Create turtle for pattern
        if len(skks) <= i:
            #skk = turtle.Turtle()
            skk = ScrolledTurtle(wn)
            skk.color(color)
            skk.speed("fastest")
            skks.append(skk)
            skk.up()
        else:
            skk = skks[i]
            skk.screen.tracer(False)
            skk.up()
            pos = (-skk.pos()[0], -skk.pos()[1])
            skk.setpos(pos)

        # Setup starting position
        pos = (0, 0)
        if "left" in position:
            pos = np.add(pos, (-skk.screen.window_width() / 3, 0))
        if "right" in position:
            pos = np.add(pos, (skk.screen.window_width() / 3, 0))
        if "top" in position:
            pos = np.add(pos, (0, skk.screen.window_height() / 3))
        if "bottom" in position:
            pos = np.add(pos, (0, -skk.screen.window_height() / 3))

        skk.setpos(pos)
        skk.setheading(heading)
        skk.clear()
        skk.down()

        # Draw system
        execute_system(skk, angle, rule, depth, axiom=axiom, scale=scale)

        # Update screen size
        #skk.update_screen_size()
        sw, sh = skk.get_screen_size()
        if sw > w:
            w = sw
        if sh > h:
            h = sh
        wn.screensize(w, h)

        # Show and wait
        skk.screen.tracer(True)
        time.sleep(1)

print("Done")
turtle.done()
