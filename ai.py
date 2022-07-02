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
        self.all_elements = set()
        self.elements = []
        self.valid_layers = [
            x
            for x in self.template.layers.values()
            if "gridTiles" in x and x["gridTiles"]
        ]

        wid, hei = self.div_16(self.template.size)
        depth = len(self.valid_layers)
        arr = np.zeros((depth, hei, wid), int)
        flat_arr = np.zeros((hei, wid), int)

        # Map all known tiles into a numpy array
        for d, layer in enumerate(self.valid_layers):
            for tile in layer["gridTiles"]:
                x, y = self.div_16(tile["px"])
                arr[d][y][x] = tile["t"]

        # Create a list of elements and map them to the array
        for x in range(wid):
            for y in range(hei):
                element = []
                for d in range(depth):
                    element.append(arr[d][y][x])
                element = tuple(element)

                if element not in self.elements:
                    self.elements.append(element)
                    self.all_elements.add(element)

                flat_arr[y][x] = self.elements.index(element)

        # Go through each tile and append all surrounding tiles to a dict
        allowed = {}
        for i, element in enumerate(np.nditer(flat_arr)):
            elem = int(element)
            wid = self.div_16(self.template.size[0])
            coords = [i // wid, i % wid]
            for coord in self.coords_around(self.template, coords):
                y, x = coord
                dr = self.get_direction(coords, coord)
                if elem not in allowed:
                    allowed[elem] = {}
                if dr not in allowed[elem]:
                    allowed[elem][dr] = set()
                allowed[elem][dr].add(flat_arr[y][x])
        # Set some more attributes
        self.element_array = flat_arr
        self.all_elements = {x for x in range(len(self.elements))}

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

        allowed = self.all_elements
        for coord in self.coords_around(level, coords):
            dr = self.get_direction(coord, coords)
            y, x = coord
            if arr[y][x] == -1:
                allow = self.all_elements
            else:
                # If direction isnt known, nothing can go to that side of the tile
                if dr in self.allowed[arr[y][x]]:
                    allow = self.allowed[arr[y][x]][dr]
                else:
                    allow = set()
            allowed = set.intersection(allowed, allow)
        return allowed


wid, hei = size = [x // 16 for x in target.size]

tile_template = {"px": [128, 128], "src": [96, 16], "f": 0, "t": 14, "d": [136]}

checker = Tilechecker(target, level3, check_directions=True)
arr = np.full((size[1], size[0]), -1, int)

# Set known tiles into the array
# for tile in target.layers["Ground"]["gridTiles"]:
#     x, y = tuple([a // 16 for a in tile["px"]])
#     arr[y][x] = tile["t"]

while -1 in arr:
    print(f"{len(arr[arr==-1])} tiles left to fill")
    poss = []
    for coords in list(zip(*np.nonzero(arr == -1))):
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
    element_num = random.choice(list(selected[1]))

    arr[y][x] = element_num
    element = checker.elements[element_num]

    checker.valid_layers[0]
    target.layers["Above_A"]
    for i, layer in enumerate(checker.valid_layers):
        # TODO: Valid layers is a copy of the layers
        layer = target.layers[layer["__identifier"]]
        tile = copy.deepcopy(tile_template)
        t = element[i]

        tile["px"] = [int(x) * 16, int(y) * 16]
        tile["d"] = [int(target.coordToInt((x, y), wid))]

        tile["src"] = target.tToSrc(t)
        tile["t"] = int(t)

        layer["gridTiles"].append(tile)

target.layers["Ground"]["gridTiles"].sort(key=lambda x: x["d"][0])
target.write()
print("Wrote")
