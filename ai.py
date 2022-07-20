import random
from typing import Iterable
from level import Level
from world import World
from pathfinder import Pathfinder
from ldtkcmanager import LdtkcManager
from pathlib import Path
import copy
import numpy as np
import time

ROOT = Path("world/world")

world = World("world/world.ldtk")


class Tilechecker:
    """Object for various operations regarding elements and tile placing rules
    :param template_ndarrays -> A list of ndarrays with level tiles mapped onto them"""

    def __init__(self, template_ndarrays: list):
        self.templates = template_ndarrays
        self.allowed = self._build_rules()

    def _build_rules(self):
        """Builds the ruleset from the template. Keys and value inside a direction are element ids,
        which are the index of the item in self.elements. Also counts how many times what element was next to what element.
        eg. {0: {"North": {1, 2, 3}, "South": {5, 4, 9}}, 1: {"North": {}.....
        :return -> The ruleset"""
        timer = time.perf_counter()
        self.elements = []

        allowed = {}
        weights = {}
        for template_ndarray in self.templates:
            arr = self.map_elements(
                template_ndarray, skip_empty=True, add_new_elements=True
            )
            hei, wid = arr.shape
            # Go through each tile and append all surrounding tiles to a dict
            # Also take a note of how many times an element was next to another, and on what side (for weights)
            for i, element in enumerate(np.nditer(arr)):
                elem = int(element)
                coords = [i // wid, i % wid]
                for coord in self.coords_around(arr, coords):
                    y, x = coord
                    dr = self.get_direction(coords, coord)
                    elem_id = arr[y][x]
                    if elem_id != -1:
                        if elem not in allowed:
                            allowed[elem] = {}
                            weights[elem] = {}
                        if dr not in allowed[elem]:
                            allowed[elem][dr] = set()
                            weights[elem][dr] = {}
                        if arr[y][x] not in weights[elem][dr]:
                            weights[elem][dr][arr[y][x]] = 1

                        allowed[elem][dr].add(arr[y][x])
                        weights[elem][dr][arr[y][x]] += 1

        # Set some more attributes
        self.weights = weights
        self.element_ids = {
            x for x in range(len(self.elements))
        }  # Used as a base of allowed element ids

        print("Building the ruleset took:", time.perf_counter() - timer)
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

    def coords_around(self, array, coords, steps=1):
        """Same as coords around, but return coords around 3 tiles away
        :param array  -> 2darray of the level youre working to fill
        :param coords -> a tuple of coords
        :param steps  -> How many steps away to return coordinates from
        :returns      -> A tuple of surrounding coord tuples: (y+steps,x), (y-steps,x), (y,x+steps), (y,x-steps)"""
        y, x = coords
        s = steps
        around = ((y, x + s), (y, x - s), (y + s, x), (y - s, x))
        # Remove coordinates with negative values
        around = tuple((i for i in around if i[0] >= 0 and i[1] >= 0))
        # Remove coordinates that exceed boundaries
        hei, wid = array.shape
        around = tuple((i for i in around if i[0] < hei and i[1] < wid))
        return around

    def cube_around(self, array, coords):
        """Return the coordinates around a given coordinate, with the corners also. Doesn't return out of bounds coordinates
        :param array  -> 2darray of the level youre working to fill
        :returns      -> A tuple of surrounding coord tuples: (y+1,x), (y-1,x), (y,x+1), (y,x-1)"""
        y, x = coords
        around = (
            (y, x + 1),
            (y, x - 1),
            (y + 1, x),
            (y - 1, x),
            (y - 1, x - 1),
            (y - 1, x + 1),
            (y + 1, x - 1),
            (y + 1, x + 1),
        )
        # Remove coordinates with negative values
        around = tuple((i for i in around if i[0] >= 0 and i[1] >= 0))
        # Remove coordinates that exceed boundaries
        hei, wid = array.shape
        around = tuple((i for i in around if i[0] < hei and i[1] < wid))
        return around

    def check_allowed(self, arr, poss, coords):
        """Check locations around a coordinate and return a set of element ids that are allowed in the coordinate
        :param arr      -> Numpy array of already decided elements
        :param poss     -> A dict of remaining coordinates' possible elements
        :param coords   -> A tuple of coords to check surroundings from (y, x)
        :return         -> A set of allowed element ids"""

        allowed = self.element_ids
        for coord in self.coords_around(arr, coords):
            dr = self.get_direction(coord, coords)

            allow = set()
            # If coord is not solved, check its possible element options instead
            if coord in poss:
                for item in poss[coord]:
                    if dr in self.allowed[item]:
                        allow = allow | self.allowed[item][dr]

            # Coord is solved, so just fetch from its allowed list
            else:
                y, x = coord
                if dr in self.allowed[arr[y][x]]:
                    allow = self.allowed[arr[y][x]][dr]

            allowed = set.intersection(allowed, allow)

        return allowed

    def scan_elements(self, array, poss, coords):
        """Brute force 'propagation' algorithm
        If a coordinates poss was changed, add that coordinates surroundings to the to-be-checked list. Repeat until list is exhausted
        :param array  -> Numpy array of already set elements
        :param poss   -> A dict of remaining coordinates' possible elements
        :param coords -> A tuple of coords from where to start the process
        """
        solving = True
        while solving:
            solving = False
            propagations = []
            for coords in list(zip(*np.nonzero(array == -1))):
                propagations.append(coords)
            # propagations = [x for x in self.coords_around(level, coords)]
            while propagations:
                coords = propagations.pop(0)
                allowed = self.check_allowed(array, array, poss, coords)
                if poss[coords] == allowed:
                    continue
                poss[coords] = allowed
                solving = True

    def propagate_elements(self, array, poss, coords):
        """Process the influence of setting an element to the rest of the level.
        Add coords surrounding coordinates to a list, check their allowed tiles and update poss if needed.
        If a coordinates poss was changed, add that coordinates surroundings to the to-be-checked list. Repeat until list is exhausted
        :param array  -> Numpy array of already set elements
        :param poss   -> A dict of remaining coordinates' possible elements
        :param coords -> A tuple of coords from where to start the process
        """
        propagations = [x for x in self.coords_around(array, coords)]
        while propagations:
            coords = propagations.pop(0)
            if coords not in poss:
                continue
            allowed = self.check_allowed(array, poss, coords)
            if poss[coords] == allowed:
                continue
            poss[coords] = allowed
            propagations.extend([x for x in self.coords_around(array, coords)])

    def debug_element(self, level, arr, poss, element_id):
        """Check all locations where element_id is currently allowed and write it to the level."""
        for k, v in poss.items():
            y, x = k
            if element_id in v:
                arr[y][x] = element_id
        self.write_elements(level, arr)

    def map_elements(self, ndarray, skip_empty=False, add_new_elements=False):
        """Convert tiles in level to elements and map the ids to array
        :param ndarray                -> ndarray that has all level tile info
        :param skip_empty=False       -> Skip mapping elements of only 0s
        :param add_new_elements=False -> Add new elements into self.elements as they are found"""

        # Create a list of elements and map them to a 2darray
        depth, hei, wid = ndarray.shape
        array = np.full((hei, wid), -1, int)
        for x in range(wid):
            for y in range(hei):
                element = []
                for d in range(depth):
                    element.append(ndarray[d][y][x])
                element = tuple(element)

                if skip_empty:
                    if sum(element) == 0:
                        continue

                if add_new_elements:
                    if element not in self.elements:
                        self.elements.append(element)

                if element not in self.elements:
                    raise IndexError(
                        f"Element {element} at coords ({y}, {x}) not found in elements!"
                    )

                array[y][x] = self.elements.index(element)
        return array

    def write_elements(self, level, array):
        """Write elements mapped in array to level
        :param level -> Target Level object to write to
        :param array -> Numpy array that contains the mapped elements"""
        tile_template = {"px": [128, 128], "src": [96, 16], "f": 0, "t": 14, "d": [136]}
        hei, wid = array.shape

        # Empty all layers
        for layer in level.layers.values():
            layer["gridTiles"] = []

        # Go over the array and write each found element to the level
        for y in range(array.shape[0]):
            for x in range(array.shape[1]):
                element_id = array[y][x]
                if element_id == -1:
                    continue

                element = self.elements[element_id]
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
        level.write()

    def get_weights(self, arr, selection):
        """Calculate the weights for a selection from poss
        :param arr       -> Array containing the mapped elements
        :param selection -> A value selected from poss eg.((y, x), {1, 2, 3})
        :returns         -> A list of weights to be used in np.random.choice(..., p=)"""
        from_coords, elements = selection
        weights = {x: 0 for x in elements}
        for coords in self.coords_around(arr, from_coords):
            y, x = coords
            from_elem = arr[y][x]
            dr = self.get_direction(coords, from_coords)
            for elem in weights.keys():
                if elem in self.weights[from_elem][dr]:
                    weights[elem] += self.weights[from_elem][dr][elem]
        # Weights doesnt always find the elements around it. In that case, dont handle this element right now
        # Not sure if this is correct behaviour
        total = sum([x for x in weights.values()])
        if total == 0:
            return False
        weights = [x / total for x in weights.values()]
        return weights


def create_ndarray(level, ldtkc=False):
    if ldtkc:
        # TODO: Implement ldtkc creation
        wid, hei = level["orig_dimensions"]
        depth = len(manager.tile_layers(level))
        ndarray = np.zeros((depth, hei, wid), int)

    else:
        wid, hei = [x // 16 for x in level.size]
        depth = len(level.layers)
        ndarray = np.zeros((depth, hei, wid), int)

        # Map all known tiles into an ndarray
        for d, layer in enumerate(level.layers.values()):
            for tile in layer["gridTiles"]:
                x, y = [a // 16 for a in tile["px"]]
                ndarray[d][y][x] = tile["t"]
    return ndarray


level1 = Level(world, ROOT / "0001-Template.ldtkl")  # 1x3 template
level2 = Level(world, ROOT / "0003-Template2.ldtkl")  # 2x3 template
level3 = Level(world, ROOT / "0000-L3_TypicalTown.ldtkl")  # The big level

roads = Level(world, ROOT / "0004-Roads.ldtkl")  # 2 wide roads
roads3w = Level(world, ROOT / "0005-Roads2.ldtkl")  # 3 wide roads

target = Level(world, ROOT / "0002-Target.ldtkl")

templates = [level3, roads]
# TODO: Create separate checkers for road and non-road sections
checker = Tilechecker([create_ndarray(x) for x in templates])
pathfinder = Pathfinder(checker)

timer = time.perf_counter()

# Map all targets pre-set elements to the array
ndarray = create_ndarray(target)
ele_arr = checker.map_elements(ndarray, skip_empty=True)

# Level dimensions
hei, wid = ele_arr.shape

# Initialize poss with all coords having all options
poss = {
    (i // wid, i % wid): checker.element_ids for i, _ in enumerate(np.nditer(ele_arr))
}

# Update poss with information about pre-set elements
arr = np.full((ele_arr.shape), -1, int)
for coords in list(zip(*np.nonzero(ele_arr != -1))):
    y, x = coords
    arr[y][x] = ele_arr[y][x]
    poss.pop(coords)
    checker.propagate_elements(arr, poss, coords)

print("Pre-set elements", time.perf_counter() - timer)

# Create the path
path = pathfinder.create_path(arr, (15, 1), [(15, 55), (30, 27)])
path = pathfinder.largen_path(arr, path)

tmp_arr = copy.deepcopy(arr)
tmp_poss = copy.deepcopy(poss)
# grass_element = checker.element_arrays[roads.name][0][0]
grass_element = 1  # TODO: This is the most stupid part of this code
for coord in list(zip(*np.nonzero(arr == -1))):
    if coord not in path:
        y, x = coord
        # TODO: This is stupid
        # Fetch top left element of the level that contains roads (usually/hopefully the grass tile)
        tmp_arr[y][x] = grass_element
        tmp_poss.pop(coord)
        checker.propagate_elements(tmp_arr, tmp_poss, coord)

# tmp_arr is full of grass with the road carved out
while -1 in tmp_arr:
    try:
        min_opt = min([len(v) for k, v in tmp_poss.items() if k in path and len(v) > 0])
    except ValueError:
        print("Out of optinos")
        break

    select_poss = {k: v for k, v in tmp_poss.items() if k in path and len(v) == min_opt}
    selected = random.choice(list(select_poss.items()))

    if not (weights := checker.get_weights(arr, selected)):
        continue
    element_id = np.random.choice(list(selected[1]), p=weights)

    coord = (y, x) = selected[0]

    tmp_poss.pop(coord)
    poss.pop(coord)

    tmp_arr[y][x] = element_id
    checker.propagate_elements(tmp_arr, tmp_poss, coord)

    arr[y][x] = element_id
    checker.propagate_elements(arr, poss, coord)

while -1 in arr:
    print(f"{len(arr[arr==-1])} tiles left to fill")

    # Calculate whats the least amount of options any level has and remove all tiles that have more than it
    try:
        min_opt = min(
            [len(x) for x in poss.values() if len(x) > 0]
        )  # Ignore tiles with 0 options
    except ValueError:
        # Were screwed, break out and write
        print("Out of states and options, breaking out")
        break

    select_poss = {k: v for k, v in poss.items() if len(v) == min_opt}

    selected = random.choice(list(select_poss.items()))

    if not (weights := checker.get_weights(arr, selected)):
        continue
    element_id = np.random.choice(list(selected[1]), p=weights)

    y, x = selected[0]

    arr[y][x] = element_id
    poss.pop(selected[0])  # coord is now set. Remove it from possible options

    checker.propagate_elements(arr, poss, (y, x))


print("Running time:", int(time.perf_counter() - timer), "s")

# Sort layer tiles by location for ease of reading
for layer in target.layers:
    if "gridTiles" in layer and layer["gridTiles"]:
        layer["gridTiles"].sort(key=lambda x: x["d"][0])

# Write elements mapped into arr to the target
checker.write_elements(target, arr)

print("Wrote")
