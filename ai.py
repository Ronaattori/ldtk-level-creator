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


class Tilechecker:
    """Object for various operations regarding elements and tile placing rules
    :param target    -> The Level object youre working to fill
    :param templates -> A list of Level object to use as level building templates"""

    def __init__(self, target, templates: list):
        self.target = target
        self.templates = templates
        self.allowed = self._build_rules()

    def _build_rules(self):
        """Builds the ruleset from the template. Keys and value inside a direction are element ids,
        which are the index of the item in self.elements
        eg. {0: {"North": {1, 2, 3}, "South": {5, 4, 9}}, 1: {"North": {}.....
        :return -> The ruleset"""
        # template
        self.elements = []
        # self.valid_layers = [
        #     x
        #     for x in self.template.layers.values()
        #     if "gridTiles" in x and x["gridTiles"]
        # ]

        allowed = {}
        for template in self.templates:
            wid, hei = self.div_16(template.size)
            arr = np.empty((hei, wid), int)

            self.map_elements(template, arr, add_new_elements=True)

            # Go through each tile and append all surrounding tiles to a dict
            for i, element in enumerate(np.nditer(arr)):
                elem = int(element)
                coords = [i // wid, i % wid]
                for coord in self.coords_around(template, coords):
                    y, x = coord
                    dr = self.get_direction(coords, coord)
                    if elem not in allowed:
                        allowed[elem] = {}
                    if dr not in allowed[elem]:
                        allowed[elem][dr] = set()
                    allowed[elem][dr].add(arr[y][x])

        # Set some more attributes
        self.element_array = arr
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

    def propagate_elements(self, level, array, poss, coords):
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

    def map_elements(self, level, array, skip_empty=False, add_new_elements=False):
        """Convert tiles in level to elements and map the ids to array
        :param level                  -> Target Level object to work in
        :param array                  -> Numpy array to map element ids into
        :param skip_empty=False       -> Skip mapping elements of only 0s
        :param add_new_elements=False -> Add new elements into self.elements as they are found"""
        wid, hei = [x // 16 for x in level.size]
        depth = len(level.layers)
        ndarray = np.zeros((depth, hei, wid), int)

        # Map all known tiles into an ndarray
        for d, layer in enumerate(level.layers.values()):
            for tile in layer["gridTiles"]:
                x, y = self.div_16(tile["px"])
                ndarray[d][y][x] = tile["t"]

        # Create a list of elements and map them to a 2darray
        for x in range(wid):
            for y in range(hei):
                element = []
                for d in range(depth):
                    element.append(ndarray[d][y][x])
                element = tuple(element)

                if add_new_elements:
                    if element not in self.elements:
                        self.elements.append(element)

                if skip_empty:
                    if sum(element) == 0:
                        continue

                if element not in self.elements:
                    raise IndexError(
                        f"Element {element} at coords ({y}, {x}) not found in elements!"
                    )

                array[y][x] = self.elements.index(element)


def write_elements(level, array, tilechecker):
    """Write elements mapped in array to level
    :param level -> Target Level object to write to
    :param array -> Numpy array that contains the mapped elements
    :param tilechecker -> Tilechecker object that contains the elements"""
    tile_template = {"px": [128, 128], "src": [96, 16], "f": 0, "t": 14, "d": [136]}

    # Empty all layers
    for layer in level.layers.values():
        layer["gridTiles"] = []

    # Go over the array and write each found element to the level
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            element_id = array[y][x]
            if element_id == -1:
                continue

            element = tilechecker.elements[element_id]
            for i, layer in enumerate(level.layers.values()):
                t = element[i]
                if t == 0:
                    continue
                tile = copy.deepcopy(tile_template)

                tile["px"] = [int(x) * 16, int(y) * 16]
                tile["d"] = [int(level.coordToInt((x, y), wid))]

                tile["src"] = level.tToSrc(t)
                tile["t"] = int(t)

                layer["gridTiles"].append(tile)


level1 = Level(world, ROOT / "0001-Template.ldtkl")  # 1x3 template
level2 = Level(world, ROOT / "0003-Template2.ldtkl")  # 2x3 template
level3 = Level(world, ROOT / "0000-L3_TypicalTown.ldtkl")  # The big level

roads2w = Level(world, ROOT / "0004-Roads.ldtkl")  # 2 wide roads
roads3w = Level(world, ROOT / "0005-Roads2.ldtkl")  # 3 wide roads

target = Level(world, ROOT / "0002-Target.ldtkl")

checker = Tilechecker(target, [roads2w])

wid, hei = size = [x // 16 for x in target.size]
ele_arr = np.full((hei, wid), -1, int)
arr = np.full((hei, wid), -1, int)

timer = time.perf_counter()

# Initialize poss with all coords having all options
poss = {coords: checker.element_ids for coords in list(zip(*np.nonzero(arr == -1)))}

# Map all targets pre-set elements to the array
checker.map_elements(target, ele_arr, skip_empty=True)

# Update poss with information about pre-set elements
for coords in list(zip(*np.nonzero(ele_arr != -1))):
    y, x = coords
    arr[y][x] = ele_arr[y][x]
    poss.pop(coords)
    checker.propagate_elements(target, arr, poss, coords)

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

    checker.propagate_elements(target, arr, poss, (y, x))

# Write elements mapped into arr to the target
write_elements(target, arr, checker)

print("Running time:", int(time.perf_counter() - timer), "s")

# Sort layer tiles by location for ease of reading
for layer in target.layers:
    if "gridTiles" in layer and layer["gridTiles"]:
        layer["gridTiles"].sort(key=lambda x: x["d"][0])

target.write()
print("Wrote")
