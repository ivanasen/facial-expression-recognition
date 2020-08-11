class BoundingBox(object):
    def __init__(self, x, y, w, h, objectiveness=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.objectiveness = objectiveness
