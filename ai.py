from math import ceil
import random
from typing import Iterable
from level import Level
from world import World
from pathlib import Path
import copy
import numpy as np

ROOT = Path("world/world")

world = World("world/world.ldtk")

level1 = Level(world, ROOT / "0001-Template.ldtkl")  # 1x3 template
level2 = Level(world, ROOT / "0003-Template2.ldtkl")  # 2x3 template
level3 = Level(world, ROOT / "0000-L3_TypicalTown.ldtkl")  # The big level

# The target level
target = Level(world, ROOT / "0002-Target.ldtkl")


class Tilechecker:
    def __init__(self, target, template, check_directions=False):
        self.target = target
        self.template = template
        self.check_directions = check_directions
        self.allowed = self._build_rules()

    def _build_rules(self):
        self.all_tiles = set()
        wid, hei = self.div_16(self.template.size)
        arr = np.zeros((hei, wid), int)
        allowed = {}
        # TODO: Instead of having a numpy array of tile ids, create a dict of possible tile combinations
        # for location in arr:
        #   for layer in self.template.layers()
        #       if "gridTiles" in layer:
        #           for tile in layer["gridTiles"]
        #               if location == tile["px"]
        #                   arr[location].append(layer["t"])
        # Then create the same kind of ruleset but dict keys are now "element ids"
        #
        # Create a np array of the tiles in the template
        for layer in self.template.layers.values():
            if "gridTiles" in layer:
                for tile in layer["gridTiles"]:
                    x, y = self.div_16(tile["px"])
                    if arr[y][x] == 0:
                        arr[y][x] = t = tile["t"]
                        if t not in allowed:
                            # Build a different type of ruleset if checking is allowed
                            if self.check_directions:
                                allowed[t] = {}
                            else:
                                allowed[t] = set()

                            self.all_tiles.add(t)
        # Go through each tile and append all surrounding tiles to a dict
        for i, t in enumerate(np.nditer(arr)):
            t = int(t)
            wid = self.template.size[0]
            coords = [i // self.div_16(wid), i % self.div_16(wid)]
            for coord in self.coords_around(self.template, coords):
                y, x = coord
                dr = self.get_direction(coords, coord)
                # if dr not in allowed[0]:
                #     allowed[0][dr] = set()
                if dr not in allowed[t]:
                    allowed[t][dr] = set()
                allowed[t][dr].add(arr[y][x])
                # allowed[0][dr].add(arr[y][x])
        return allowed

    def div_16(self, val):
        # Integer div val by 16. If val is iterable, integer div all values inside it by 16
        if isinstance(val, Iterable):
            return [x // 16 for x in val]
        return val // 16

    @property
    def tiles(self):
        if not hasattr(self, "_tiles"):
            self._tiles = self.template.layers["Ground"]["gridTiles"]
        return self._tiles

    def get_direction(self, from_coords, to_coords):
        # Probably a shit function
        # Calculate the direction from coord A to coord B
        diff = [x[0] - x[1] for x in list(zip(from_coords, to_coords))]
        if diff[0] == 1:
            return "North"
        if diff[0] == -1:
            return "South"
        if diff[1] == 1:
            return "West"
        if diff[1] == -1:
            return "East"
        return False

    def t_to_coord(self, t):
        d = {13: [0, 0], 39: [16, 0], 14: [32, 0]}
        return d[t]

    def t_to_name(self, t):
        d = {13: "Grass", 39: "Flower", 14: "Bush"}
        return d[t]

    def coords_around(self, level, coords):
        # Returns x+1, x-1, y+1, y-1 of given coords
        """:param A tuple of coords
        :returns A tuple of surrounding coord tuples (y, x)"""
        y, x = coords
        around = ((y, x + 1), (y, x - 1), (y + 1, x), (y - 1, x))
        # Remove coordinates with negative values
        around = tuple((i for i in around if i[0] >= 0 and i[1] >= 0))
        # Remove coordinates that exceed boundaries
        wid, hei = self.div_16(level.size)
        around = tuple((i for i in around if i[0] < hei and i[1] < wid))
        return around

    def check_allowed(self, level, arr, coords):
        # Check surrounding coords for what they allow to be in the direction of the original coord
        # Then merge the lists to only include tiles that all surrounding tiles agreed on
        y, x = coords
        allowed = self.all_tiles
        for coord in self.coords_around(level, coords):
            dr = self.get_direction(coord, coords)
            y, x = coord
            if arr[y][x]:
                # If direction isnt known, nothing can go to that side of the tile
                if dr in self.allowed[arr[y][x]]:
                    allow = self.allowed[arr[y][x]][dr]
                else:
                    allow = set()
            else:
                allow = self.all_tiles
            allowed = set.intersection(allowed, allow)
        return allowed


wid, hei = size = [x // 16 for x in target.size]

tile_template = {"px": [128, 128], "src": [96, 16], "f": 0, "t": 14, "d": [136]}

checker = Tilechecker(target, level3, check_directions=True)
arr = np.zeros((size[1], size[0]), int)

# Set known tiles into the array
for tile in target.layers["Ground"]["gridTiles"]:
    x, y = tuple([a // 16 for a in tile["px"]])
    arr[y][x] = tile["t"]

while 0 in arr:
    print(f"{len(arr[arr==0])} tiles left to fill")
    poss = []
    for coords in list(zip(*np.nonzero(arr == 0))):
        # coords = (y, x)
        allowed = checker.check_allowed(target, arr, coords)
        poss.append((coords, allowed))
        # Not having allowed is only an issue with small tilesets
        # if not allowed:
        #     break

    # Calculate whats the least amount of options any level has and remove all tiles that have more than it
    min_opt = min([len(x[1]) for x in poss])
    min_opt = max(1, min_opt)  # Ignore tiles with 0 options
    poss = [x for x in poss if len(x[1]) == min_opt]

    # Were out of options
    if not poss:
        break

    selected = random.choice(poss)
    y, x = selected[0]
    t = random.choice(list(selected[1]))

    arr[y][x] = t

    tile = copy.deepcopy(tile_template)
    tile["px"] = [int(x) * 16, int(y) * 16]
    tile["d"] = [int(target.coordToInt((x, y), wid))]

    tile["src"] = target.tToSrc(t)
    tile["t"] = int(t)
    print(arr)

    target.layers["Ground"]["gridTiles"].append(tile)

# while 0 in arr:
#     print(f"{len(arr[arr==0])} tiles left to fill")
#     poss = []
#     for y, x in list(zip(*np.nonzero(arr))):
#         val = arr[y][x]
#
#         append_around(poss, arr, x, y)
#
#     if poss:
#         # Calculate which options have the least allowed tiles
#         poss_chances = [[x, allowed_tiles(arr, x[0], x[1])] for x in poss]
#         least_options = min([len(x[1]) for x in poss_chances])
#         poss_chances = [x for x in poss_chances if len(x[1]) == least_options]
#
#         # Select a random allowed tile and coordinates
#         selected = random.choice(poss_chances)
#         x, y = selected[0]
#         t = random.choice(selected[1])
#
#         arr[y][x] = t
#
#         # Set location related stuff
#         tile = copy.deepcopy(target.layers["Ground"]["gridTiles"][0])
#         tile["px"] = [int(x) * 16, int(y) * 16]
#         tile["d"] = [int(target.coordToInt([x, y], size[0]))]
#         target.coordToInt([x, y], size[1])
#
#         # Set tile related stuff
#         tile["src"] = target.tToSrc(t)
#         tile["t"] = t
#
#         # target.layers["Ground"]["gridTiles"].append(tile)
#
target.layers["Ground"]["gridTiles"].sort(key=lambda x: x["d"][0])
target.write()
print("Wrote")
