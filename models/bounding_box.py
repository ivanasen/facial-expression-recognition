class BoundingBox(object):
    def __init__(self, x: int, y: int, width: int, height: int, score: float = 1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.score = score

    def __str__(self):
        return f"(x: {self.x}, y: {self.y}, width: {self.width}, height: {self.height}, score: {self.score})"
