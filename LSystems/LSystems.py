import time
import turtle

patterns = [
    ["F+F-F-FF+F+F-F", "green", 90, "left"],
    ["F+F--F+F", "blue", 90, "center"],
    ["F[+F]F[-F]F", "red", 90, "center"],
    ["FF+[+F-F-F]-[-F+F+F]", "orange", 45, "left"],
    ["F+F−F−F+F", "black", 90, "center"]
]
max_depth = 10

cache = []


def executeSystem(skk, angle, pattern, max_depth, depth=0):
    step = 100 ** (1/float(depth)) if depth > 1 else 100
    if step < 2:
        skk.forward(step)
        return
    if depth >= max_depth:
        skk.forward(step)
        return

    for char in pattern:
        if char == 'F':
            executeSystem(skk, angle, pattern, max_depth, depth + 1)
        elif char == '+':
            skk.left(angle)
        elif char == '-':
            skk.right(angle)
        elif char == '[':
            cache.append(skk.pos())
        elif char == ']':
            skk.setpos(cache.pop())


wn = turtle.Screen()
wn.title("L-Systems")

skks = []
for depth in range(1, max_depth):
    for i, (pattern, color, angle, position) in enumerate(patterns):
        print("Depth: {}, Color: {}, Pattern: {}".format(depth, color, pattern))
        wn.tracer(False)
        if len(skks) <= i:
            skk = turtle.Turtle()
            skk.color(color)
            skk.speed("fastest")
            skks.append(skk)
        else:
            skk = skks[i]
            pos = (-skk.pos()[0], -skk.pos()[1])
            skk.setpos(pos)

        pos = (0, 0)
        if position == "left":
            pos = (-wn.window_width() / 2, 0)
        elif position == "right":
            pos = (wn.window_width() / 2, 0)
        elif position == "top":
            pos = (0, wn.window_height() / 2)
        elif position == "bottom":
            pos = (0, -wn.window_height() / 2)

        skk.setpos(pos)
        skk.clear()

        executeSystem(skk, angle, pattern, depth)

        wn.tracer(True)
        time.sleep(1)

turtle.done()
