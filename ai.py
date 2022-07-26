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
MULTIPLIER = 1000
WEIGHTS = True

world = World("world/world.ldtk")


class Tilechecker:
    """Object for various operations regarding elements and tile placing rules
    :param template_ndarrays -> A list of ndarrays with level tiles mapped onto them"""

    def __init__(
        self, template_ndarrays: list, elements: list, non_place=[], log_weights=True
    ):
        self.elements = elements
        self.used_elements = set()
        self.templates = [
            self.map_elements(x, skip_empty=True, add_new_elements=True)
            for x in template_ndarrays
        ]
        self.non_place_templates = [
            self.map_elements(x, skip_empty=True, add_new_elements=True)
            for x in non_place
        ]
        self.log_weights = log_weights
        self.allowed = self._build_rules()

    def _build_rules(self):
        """Builds the ruleset from the template. Keys and value inside a direction are element ids,
        which are the index of the item in self.elements. Also counts how many times what element was next to what element.
        eg. {0: {"North": {1, 2, 3}, "South": {5, 4, 9}}, 1: {"North": {}.....
        :return -> The ruleset"""
        timer = time.perf_counter()

        # allowed = {}
        weights = {}

        n_ele = len(self.elements)
        allowed = np.zeros((4, n_ele, n_ele), dtype="?")

        # for template_ndarray in self.templates:
        for non_place, templates in (
            (False, self.templates),
            (True, self.non_place_templates),
        ):
            for arr in templates:
                hei, wid = arr.shape
                # Go through each tile and append all surrounding tiles to a dict
                # Also take a note of how many times an element was next to another, and on what side (for weights)
                for i, element in enumerate(np.nditer(arr)):
                    elem = int(element)
                    coords = [i // wid, i % wid]
                    for coord in self.coords_around(arr, coords):
                        y, x = coord
                        dr, dri = self.get_direction(coords, coord)
                        elem_id = arr[y][x]
                        if elem_id != -1:
                            if elem not in weights:
                                weights[elem] = {}
                            if dr not in weights[elem]:
                                weights[elem][dr] = {}
                            if elem_id not in weights[elem][dr]:
                                weights[elem][dr][elem_id] = 1

                            if not non_place:
                                self.used_elements.add(elem_id)
                            allowed[dri][elem][elem_id] = True
                            weights[elem][dr][elem_id] += 1

        # This will get explained to me later
        if self.log_weights:
            for from_elem in weights.keys():
                for dr in weights[from_elem].keys():
                    for k, v in weights[from_elem][dr].items():
                        weights[from_elem][dr][k] = np.log(v + 1)
        # Convert weights from number of occurences to a % chance
        for from_elem in weights.keys():
            for dr in weights[from_elem].keys():
                total = sum([v for v in weights[from_elem][dr].values()])
                for k, v in weights[from_elem][dr].items():
                    weights[from_elem][dr][k] = v / total

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
            return "North", 0
        if diff[0] == -1:
            return "South", 1
        if diff[1] == 1:
            return "West", 2
        if diff[1] == -1:
            return "East", 3
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
        :param poss     -> Numpy array of possibilities
        :param coords   -> A tuple of coords to check surroundings from (y, x)
        :return         -> A set of allowed element ids"""

        allowed = np.ones(len(self.elements), dtype="?")
        for coord in self.coords_around(arr, coords):
            _, dri = self.get_direction(coord, coords)

            allow = np.dot(poss[coord], self.allowed[dri])

            allowed = np.multiply(allowed, allow)

        return allowed

    def check_allowed_gemm(self, poss):
        """Updates the entire allowed matrix one iteration
        :param arr      -> Numpy array of already decided elements
        :param poss     -> Numpy array of possibilities
        :return         -> A set of allowed element ids"""

        r = np.dot(poss, self.allowed)

        new_poss = np.ones(poss.shape, dtype="?")

        new_poss[:-1] = np.multiply(new_poss[:-1], r[1:, :, 0])
        new_poss[1:] = np.multiply(new_poss[1:], r[:-1, :, 1])
        new_poss[:, :-1] = np.multiply(new_poss[:, :-1], r[:, 1:, 2])
        new_poss[:, 1:] = np.multiply(new_poss[:, 1:], r[:, :-1, 3])
        return new_poss

    def scan_elements(self, array, poss):
        """Brute force 'propagation' algorithm
        If a coordinates poss was changed, add that coordinates surroundings to the to-be-checked list. Repeat until list is exhausted
        :param array  -> Numpy array of already set elements
        :param poss     -> Numpy array of possibilities
        """
        while True:
            previous = np.copy(poss)
            poss = self.check_allowed_gemm(poss)

            for y in range(array.shape[0]):
                for x in range(array.shape[1]):
                    if array[y, x] > -1:
                        poss[y, x] = 0
                        poss[y, x][array[y, x]] = 1

            if np.all(poss == previous):
                break
        return poss

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
            y, x = coords
            if np.sum(poss[y, x]) == 1:
                continue
            allowed = self.check_allowed(array, poss, coords)
            if np.all(poss[coords] == allowed):
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

    def write_elements(self, level, array, ldtkc=False):
        # return
        """Write elements mapped in array to level
        :param level -> Target Level object to write to
        :param array -> Numpy array that contains the mapped elements"""
        hei, wid = array.shape

        # Go over the array and write each found element to the level
        if ldtkc:
            for y in range(hei):
                for x in range(wid):
                    element_id = array[y, x]
                    if element_id == -1:
                        continue
                    element = checker.elements[element_id]
                    # For tile layer in the level
                    for i, layer in enumerate(
                        [x for x in level["layers"] if x[0] == "TILES"]
                    ):
                        t = element[i]
                        layer = layer[2][0]

                        t_x = t // MULTIPLIER
                        t_y = t % MULTIPLIER
                        t_x += 1

                        layer[y, x][0] = t_x
                        layer[y, x][1] = t_y
            manager.write()
        else:
            tile_template = {
                "px": [128, 128],
                "src": [96, 16],
                "f": 0,
                "t": 14,
                "d": [136],
            }
            # Empty all layers
            for layer in level.layers.values():
                layer["gridTiles"] = []
            for y in range(hei):
                for x in range(wid):
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
        print("Wrote level")

    def get_weights(self, arr, from_coords, elements):
        """Calculate the weights for a selection from poss
        :param arr       -> Array containing the mapped elements
        :param selection -> A value selected from poss eg.((y, x), {1, 2, 3})
        :returns         -> A list of weights to be used in np.random.choice(..., p=)"""

        # If surrounding are still empty, return weights with all values valued the same
        if sum([arr[y, x] for y, x in checker.coords_around(arr, from_coords)]) == -4:
            return elements / np.sum(elements)

        weights = elements.astype(float)
        for coords in self.coords_around(arr, from_coords):
            y, x = coords
            from_elem = arr[y][x]
            dr, _ = self.get_direction(coords, from_coords)
            for elem in range(len(weights)):
                if (
                    from_elem in self.weights
                    and dr in self.weights[from_elem]
                    and elem in self.weights[from_elem][dr]
                ):
                    weights[elem] *= self.weights[from_elem][dr][elem]
        # Weights doesnt always find the elements around it. In that case, dont handle this element right now
        # Not sure if this is correct behaviour
        total = np.sum(weights)
        if total == 0:
            return False
        weights = weights / total
        return weights


def create_ndarray(level, ldtkc=False):
    if ldtkc:
        wid, hei = level["orig_dimensions"]
        depth = len(manager.tile_layers(level))
        ndarray = np.zeros((depth, hei, wid), int)

        for d, layer in enumerate(manager.tile_layers(level)):
            layer = layer[2]
            for y in range(hei):
                for x in range(wid):
                    t_x, t_y = layer[0, y, x]
                    if not t_x:
                        continue
                    t_x -= 1
                    tile_id = t_y + t_x * MULTIPLIER
                    ndarray[d][y][x] = tile_id
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


elements = []

# For LDTKC map creation
# manager = LdtkcManager("world.ldtkc")
# template = create_ndarray(manager.levels[4001], ldtkc=True)
# checker = Tilechecker([template])
# target = manager.levels[4000]
# ndarray = create_ndarray(target, ldtkc=True)
# ldtkc = True

# For LDTK map creation
template = Level(world, ROOT / "0000-L4_I1_Template.ldtkl")
road_template = Level(world, ROOT / "0005-L4_I2_Roads.ldtkl")
target = Level(world, ROOT / "0001-L4_Snowtown.ldtkl")

templates = [create_ndarray(x) for x in [template]]
non_place_templates = [create_ndarray(x) for x in [road_template]]

checker = Tilechecker(
    templates, elements, non_place=non_place_templates, log_weights=True
)
roads = Tilechecker([create_ndarray(road_template)], elements)

ndarray = create_ndarray(target)
ldtkc = False

timer = time.perf_counter()

# Map all targets pre-set elements to a 2darray
arr = checker.map_elements(ndarray, skip_empty=True)

# Level dimensions
hei, wid = arr.shape

# Initialize poss with all coords having all options
poss = np.zeros((hei, wid, len(checker.element_ids)), dtype="?")
for elem_id in checker.used_elements:
    poss[:, :, elem_id] = 1

# Update arr with information about pre-set elements
for coords in list(zip(*np.nonzero(arr != -1))):
    y, x = coords
    elem_id = arr[y, x]
    poss[y, x] = 0
    poss[y, x, elem_id] = 1

print("Pre-set elements", time.perf_counter() - timer)

# Create the path
pathfinder = Pathfinder(checker)
path = pathfinder.create_path(arr, (27, 12), [(15, 64)])
path = pathfinder.largen_path(arr, path)

road_arr = copy.deepcopy(arr)
road_poss = np.zeros((hei, wid, len(roads.element_ids)), dtype="?")
for elem_id in roads.used_elements:
    road_poss[:, :, elem_id] = 1

# Fill all tiles outside the path with grass
# Fetch the top left tile in the road template
grass_element = roads.map_elements(create_ndarray(road_template))[0, 0]
for i, _ in enumerate(np.nditer(road_arr)):
    coord = y, x = i // wid, i % wid
    if coord not in path:
        road_arr[y, x] = grass_element
        road_poss[y, x] = 0
        road_poss[y, x, grass_element] = 1
road_poss = roads.scan_elements(road_arr, road_poss)

# tmp_arr is full of grass with the road carved out
while -1 in road_arr:
    road_poss_sel = np.sum(road_poss, axis=2)
    print(f"{np.sum(road_poss_sel>1)} tiles left to fill")
    road_poss_sel[road_poss_sel < 2] = 999
    m = np.min(road_poss_sel)
    if m == 999:
        print("Road out of options!")
        break
    opt = np.array(np.where(road_poss_sel == m)).T

    y, x = opt[np.random.randint(len(opt))]

    if WEIGHTS:
        if not (weights := checker.get_weights(arr, (y, x), road_poss[y, x])).any():
            continue
        else:
            weights = road_poss[y, x] / np.sum(road_poss[y, x])

    element_id = np.random.choice(list(roads.element_ids), p=weights)

    coord = (y, x)

    road_poss[y, x] = 0
    road_poss[y, x, element_id] = 1
    poss[y, x] = 0
    poss[y, x, element_id] = 1

    road_arr[y, x] = element_id
    arr[y, x] = element_id

    roads.propagate_elements(road_arr, road_poss, coord)

for y in range(hei):
    for x in range(wid):
        if (y, x) in path:
            element_id = int(np.where(road_poss[y, x] == 1)[0])
            poss[y, x] = 0
            poss[y, x, element_id] = 1
            arr[y, x] = element_id

# Scan the whole level before we beging WFC
poss = checker.scan_elements(arr, poss)

while -1 in arr:
    poss_sel = np.sum(poss, axis=2)
    print(f"{np.sum(poss_sel>1)} tiles left to fill")
    poss_sel[poss_sel < 2] = 999
    m = np.min(poss_sel)
    if m == 999:
        print("Level out of options!")
        break
    opt = np.array(np.where(poss_sel == m)).T

    y, x = opt[np.random.randint(len(opt))]

    if WEIGHTS:
        if not (weights := checker.get_weights(arr, (y, x), poss[y, x])).any():
            continue
        else:
            weights = poss[y, x] / np.sum(poss[y, x])

    element_id = np.random.choice(list(checker.element_ids), p=weights)

    poss[y, x] = 0
    poss[y, x, element_id] = 1
    arr[y, x] = element_id

    checker.propagate_elements(arr, poss, (y, x))

for y in range(hei):
    for x in range(wid):
        if arr[y, x] == -1:
            element_id = int(np.where(poss[y, x] == 1)[0])
            poss[y, x] = 0
            poss[y, x, element_id] = 1
            arr[y, x] = element_id

print("Running time:", int(time.perf_counter() - timer), "s")
# # Write elements mapped into arr to the target
checker.write_elements(target, arr, ldtkc=ldtkc)
