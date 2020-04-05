import turtle


# https://stackoverflow.com/questions/29276229/how-to-implement-a-drag-feature-or-zoom-in-a-python-window-application
class ScrolledTurtle(turtle.RawTurtle):
    """
    Turtle with logged boundaries, that can be used to resize screen.
    """
    def __init__(self, canvas):
        turtle.RawTurtle.__init__(self, canvas)
        self.bbox = [0, 0, 0, 0]

    def _update_bbox(self):
        # keep a record of the furthers points visited
        pos = self.position()
        if pos[0] < self.bbox[0]:
            self.bbox[0] = pos[0]
        elif pos[0] > self.bbox[2]:
            self.bbox[2] = pos[0]
        if pos[1] < self.bbox[1]:
            self.bbox[1] = pos[1]
        elif pos[1] > self.bbox[3]:
            self.bbox[3] = pos[1]

    def forward(self, *args):
        turtle.RawTurtle.forward(self, *args)
        self._update_bbox()

    def backward(self, *args):
        turtle.RawTurtle.backward(self, *args)
        self._update_bbox()

    def right(self, *args):
        turtle.RawTurtle.right(self, *args)
        self._update_bbox()

    def left(self, *args):
        turtle.RawTurtle.left(self, *args)
        self._update_bbox()

    def goto(self, *args):
        turtle.RawTurtle.goto(self, *args)
        self._update_bbox()

    def setx(self, *args):
        turtle.RawTurtle.setx(self, *args)
        self._update_bbox()

    def sety(self, *args):
        turtle.RawTurtle.sety(self, *args)
        self._update_bbox()

    def setheading(self, *args):
        turtle.RawTurtle.setheading(self, *args)
        self._update_bbox()

    def home(self, *args):
        turtle.RawTurtle.home(self, *args)
        self._update_bbox()

    def get_screen_size(self):
        min_x, min_y, max_x, max_y = self.bbox  # get the furthest points the turtle has been
        width = max((0 - min_x), (max_x)) * 2 + 100  # work out what the maximum distance from 0,0 is for each axis
        height = max((0 - min_y), (max_y)) * 2 + 100  # the 100 here gives us some padding between the edge and whats drawn
        return [width, height]

    def update_screen_size(self):
        min_x, min_y, max_x, max_y = self.bbox  # get the furthest points the turtle has been
        width = max((0 - min_x), (max_x)) * 2 + 100  # work out what the maximum distance from 0,0 is for each axis
        height = max((0 - min_y), (max_y)) * 2 + 100  # the 100 here gives us some padding between the edge and whats drawn
        self.screen.screensize(width, height)