import random
from level import Level
from world import World
from pathlib import Path
import copy
import numpy as np

ROOT = Path("world/world")

world = World("world/world.ldtk")

level1 = Level(world, ROOT / "0001-Template.ldtkl")  # 1x3 template
level2 = Level(world, ROOT / "0003-Template2.ldtkl")  # 2x3 template

# The target level
target = Level(world, ROOT / "0002-Target.ldtkl")

size = [x // 16 for x in target.size]
target.size
tiles = size[0] * size[1]


class Tilechecker:
    def __init__(self, target, template):
        self.target = target
        self.template = template

    @property
    def tiles(self):
        if not hasattr(self, "_tiles"):
            self._tiles = self.template.layers["Ground"]["gridTiles"]
        return self._tiles

    def t_to_coord(self, t):
        d = {13: [0, 0], 39: [16, 0], 14: [32, 0]}
        return d[t]

    def t_to_name(self, t):
        d = {13: "Grass", 39: "Flower", 14: "Bush"}
        return d[t]

    def adj_allowed(self, t):
        coord = self.t_to_coord(t)
        allowed = []
        for x in self.tiles:
            if x["px"] == coord:
                continue
            dist = 0
            for i in zip(x["px"], coord):
                dist += abs(i[0] - i[1])
            if dist <= 16:
                allowed.append(x["t"])
        return allowed


def append_if_poss(poss, arr, x, y):
    try:
        if x < 0 or y < 0:
            return False
        if arr[y][x] == 0:
            if (x, y) not in poss:
                poss.append((x, y))
    except IndexError:
        return False


def coords_around(x, y):
    l = []
    l.append((x + 1, y))
    l.append((x - 1, y))
    l.append((x, y + 1))
    l.append((x, y - 1))
    return l


def append_around(poss, arr, x, y):
    for adj_x, adj_y in coords_around(x, y):
        append_if_poss(poss, arr, adj_x, adj_y)


def allowed_tiles(arr, x, y):
    checker = Tilechecker(target, level2)
    allowed = []
    for adj_x, adj_y in coords_around(x, y):
        try:
            val = arr[adj_y][adj_x]
        except IndexError:
            continue
        if not val == 0:
            allowed.append(checker.adj_allowed(val))
    # Filter out values that are allowed by all adjacent tiles
    return list(set.intersection(*map(set, allowed)))


arr = np.zeros((size[0], size[1]), int)
x, y = tuple([x // 16 for x in target.layers["Ground"]["gridTiles"][0]["px"]])
arr[y][x] = target.layers["Ground"]["gridTiles"][0]["t"]

while 0 in arr:
    print(f"{len(arr[arr==0])} tiles left to fill")
    poss = []
    for i, val in enumerate(np.nditer(arr)):
        x, y = tuple([i % size[0], i // size[1]])
        if not val == 0:
            append_around(poss, arr, x, y)

    if poss:
        # Calculate which options have the least allowed tiles
        poss_chances = [[x, allowed_tiles(arr, x[0], x[1])] for x in poss]
        least_options = min([len(x[1]) for x in poss_chances])
        poss_chances = [x for x in poss_chances if len(x[1]) == least_options]

        # Select a random allowed tile and coordinates
        selected = random.choice(poss_chances)
        x, y = selected[0]
        t = random.choice(selected[1])

        arr[y][x] = t

        # Set location related stuff
        tile = copy.deepcopy(target.layers["Ground"]["gridTiles"][0])
        tile["px"] = [x * 16, y * 16]
        tile["d"] = [target.coordToInt([x, y], size[1])]

        # Set tile related stuff
        tile["src"] = target.tToSrc(t)
        tile["t"] = t

        target.layers["Ground"]["gridTiles"].append(tile)

target.write()
print("Wrote")
