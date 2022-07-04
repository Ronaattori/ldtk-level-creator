import random
from typing import Iterable
from level import Level
from world import World
from pathlib import Path
import copy
import numpy as np
import time

ROOT = Path("world/world")

world = World("world/world.ldtk")

level1 = Level(world, ROOT / "0001-Template.ldtkl")  # 1x3 template
level2 = Level(world, ROOT / "0003-Template2.ldtkl")  # 2x3 template
level3 = Level(world, ROOT / "0000-L3_TypicalTown.ldtkl")  # The big level

# The target level
target = Level(world, ROOT / "0002-Target.ldtkl")


class Tilechecker:
    def __init__(self, target, template):
        self.target = target
        self.template = template
        self.allowed = self._build_rules()

    def _build_rules(self):
        """Builds the ruleset from the template. Keys and value inside a direction are element ids,
        which are the index of the item in self.elements
        eg. {0: {"North": {1, 2, 3}, "South": {5, 4, 9}}, 1: {"North": {}.....
        :return -> The ruleset"""
        self.elements = []
        self.valid_layers = [
            x
            for x in self.template.layers.values()
            if "gridTiles" in x and x["gridTiles"]
        ]

        wid, hei = self.div_16(self.template.size)
        depth = len(self.valid_layers)

        arr = np.zeros((depth, hei, wid), int)
        flat_arr = np.empty((hei, wid), int)

        # Map all known tiles into an ndarray
        for d, layer in enumerate(self.valid_layers):
            for tile in layer["gridTiles"]:
                x, y = self.div_16(tile["px"])
                arr[d][y][x] = tile["t"]

        # Create a list of elements and map them to a 2darray
        for x in range(wid):
            for y in range(hei):
                element = []
                for d in range(depth):
                    element.append(arr[d][y][x])
                element = tuple(element)

                if element not in self.elements:
                    self.elements.append(element)

                flat_arr[y][x] = self.elements.index(element)

        # Go through each tile and append all surrounding tiles to a dict
        allowed = {}
        for i, element in enumerate(np.nditer(flat_arr)):
            elem = int(element)
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
        self.element_ids = {
            x for x in range(len(self.elements))
        }  # Used as a base of allowed element ids

        return allowed

    def div_16(self, val):
        """Integer div val by 16. If val is iterable, integer div all values inside it by 16
        :param val -> item to divide by 16. If iterable, iterate over all items and create a list of all values // 16
        :return    -> val or val values // 16"""
        if isinstance(val, Iterable):
            return [x // 16 for x in val]
        return val // 16

    def get_direction(self, from_coords, to_coords):
        """Return the direction from from_coords to to_coords
        :param from_coords -> Coords to start from
        :param to_coords   -> Coords to end up to
        :return            -> Direction from from_coords to to_coords"""
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

    def coords_around(self, level, coords):
        """Return the coordinates around a given coordinate. Doesn't return out of bounds coordinates
        :param level -> a tuple of coords
        :returns     -> A tuple of surrounding coord tuples: (y+1,x), (y-1,x), (y,x+1), (y,x-1)"""
        y, x = coords
        around = ((y, x + 1), (y, x - 1), (y + 1, x), (y - 1, x))
        # Remove coordinates with negative values
        around = tuple((i for i in around if i[0] >= 0 and i[1] >= 0))
        # Remove coordinates that exceed boundaries
        wid, hei = self.div_16(level.size)
        around = tuple((i for i in around if i[0] < hei and i[1] < wid))
        return around

    def check_allowed(self, level, arr, poss, coords):
        """Check locations around a coordinate and return a set of element ids that are allowed in the coordinate
        :param level    -> Level object youre checking surroundings in
        :param arr      -> Numpy array of already decided elements
        :param poss     -> A dict of remaining coordinates' possible elements
        :param coords   -> A tuple of coords to check surroundings from (y, x)
        :return         -> A set of allowed element ids"""
        y, x = coords

        allowed = self.element_ids
        for coord in self.coords_around(level, coords):
            dr = self.get_direction(coord, coords)
            y, x = coord

            allow = set()
            # If coord is not solved, check its possible element options instead
            if coord in poss:
                for item in poss[coord]:
                    if dr in self.allowed[item]:
                        allow = allow | self.allowed[item][dr]

            # Coord is solved, so just fetch from its allowed list
            else:
                if dr in self.allowed[arr[y][x]]:
                    allow = allow | self.allowed[arr[y][x]][dr]

            allowed = set.intersection(allowed, allow)
        return allowed

    def propagate_element(self, level, array, poss, coords):
        """Process the influence of setting an element to the rest of the level.
        Add coords surrounding coordinates to a list, check their allowed tiles and update poss if needed.
        If a coordinates poss was changed, add that coordinates surroundings to the to-be-checked list. Repeat until list is exhausted
        :param level  -> Level object youre working to fill
        :param array  -> Numpy array of already set elements
        :param poss   -> A dict of remaining coordinates' possible elements
        :param coords -> A tuple of coords from where to start the process
        """
        propagations = [x for x in self.coords_around(level, coords)]
        while propagations:
            coords = propagations.pop(0)
            if coords not in poss:
                continue
            allowed = self.check_allowed(level, array, poss, coords)
            if poss[coords] == allowed:
                continue
            poss[coords] = allowed
            propagations.extend([x for x in self.coords_around(level, coords)])


checker = Tilechecker(target, level3)

wid, hei = size = [x // 16 for x in target.size]
arr = np.full((hei, wid), -1, int)

# TODO: Fix this part. we need to assign elements here whick might not exist in the ruleset.
# I dont think implementing this is a good idea, since its very unstable and were creating levels from thin air anyway
# Map all known tiles into a numpy array
# pre_arr = np.zeros((depth, hei, wid), int)
# depth = len(checker.valid_layers)
# for d, layer in enumerate(target.layers):
#     if "gridTiles" in layer and layer["gridTiles"]:
#         for tile in layer["gridTiles"]:
#             x, y = checker.div_16(tile["px"])
#             pre_arr[d][y][x] = tile["t"]
#
# for x in range(wid):
#     for y in range(hei):
#         element = []
#         for d in range(depth):
#             element.append(pre_arr[d][y][x])
#         element = tuple(element)
#
#         arr[y][x] = checker.elements.index(element)

timer = time.perf_counter()

poss = {}
for coords in list(zip(*np.nonzero(arr == -1))):
    # coords = (y, x)
    # allowed = checker.check_allowed(target, arr, coords)
    # poss[coords] = allowed
    poss[coords] = checker.element_ids

while -1 in arr:
    print(f"{len(arr[arr==-1])} tiles left to fill")

    # Calculate whats the least amount of options any level has and remove all tiles that have more than it
    try:
        min_opt = min(
            [len(x) for x in poss.values() if len(x) > 0]
        )  # Ignore tiles with 0 options
    except ValueError:
        # Were out of options
        break
    select_poss = {k: v for k, v in poss.items() if len(v) == min_opt}

    selected = random.choice(list(select_poss.items()))
    element_id = random.choice(list(selected[1]))
    y, x = selected[0]

    arr[y][x] = element_id
    poss.pop(selected[0])  # coord is now set. Remove it from possible options

    checker.propagate_element(target, arr, poss, (y, x))

# Go over each location and write all mapped elements into the level
tile_template = {"px": [128, 128], "src": [96, 16], "f": 0, "t": 14, "d": [136]}
for y in range(arr.shape[0]):
    for x in range(arr.shape[1]):
        element_id = arr[y][x]
        if element_id == -1:
            continue

        element = checker.elements[element_id]
        for i, layer in enumerate(checker.valid_layers):
            t = element[i]
            if t == 0:
                continue
            # TODO: Mby make sure the layers exist? Probably not needed
            # valid_layers is the layers from the template. Write to the same layers
            layer = target.layers[layer["__identifier"]]
            tile = copy.deepcopy(tile_template)

            tile["px"] = [int(x) * 16, int(y) * 16]
            tile["d"] = [int(target.coordToInt((x, y), wid))]

            tile["src"] = target.tToSrc(t)
            tile["t"] = int(t)

            layer["gridTiles"].append(tile)
print("Running time:", int(time.perf_counter() - timer), "s")

# Sort layer tiles by location for ease of reading
for layer in target.layers:
    if "gridTiles" in layer and layer["gridTiles"]:
        layer["gridTiles"].sort(key=lambda x: x["d"][0])

target.write()
print("Wrote")
