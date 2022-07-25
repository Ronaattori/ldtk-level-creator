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
        self.log_weights = log_weights

        self.templates = [
            self.map_elements(x, skip_empty=True, add_new_elements=True)
            for x in template_ndarrays
        ]
        self.non_place_templates = [
            self.map_elements(x, skip_empty=True, add_new_elements=True)
            for x in non_place
        ]
        all_templates = self.templates + self.non_place_templates
        self.allowed = self._build_rules(all_templates)

    def _build_rules(self, templates):
        """Builds the ruleset from the template. Keys and value inside a direction are element ids,
        which are the index of the item in self.elements. Also counts how many times what element was next to what element.
        eg. {0: {"North": {1, 2, 3}, "South": {5, 4, 9}}, 1: {"North": {}.....
        :return -> The ruleset"""
        timer = time.perf_counter()

        allowed = {}
        weights = {}
        # for template_ndarray in self.templates:
        for arr in templates:
            hei, wid = arr.shape
            # Go through each tile and append all surrounding tiles to a dict
            # Also take a note of how many times an element was next to another, and on what side (for weights)
            for i, element in enumerate(np.nditer(arr)):
                elem = int(element)
                coords = [i // wid, i % wid]
                # print([arr[coord[0], coord[1]] for coord in self.coords_around(arr, coords)])
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
                            weights[elem][dr] = np.zeros((len(self.elements)))

                        allowed[elem][dr].add(arr[y][x])
                        weights[elem][dr][arr[y][x]] += 1

        # Figure out all tiles unique to the non-place templates
        place_elems = np.unique(np.concatenate(self.templates))
        non_place_elems = (
            np.unique(np.concatenate(self.non_place_templates))
            if self.non_place_templates
            else np.empty(0)
        )
        self.non_place_elems = np.array(
            [x for x in non_place_elems if x not in place_elems], dtype=int
        )

        # Set non_place tiles chance to happen to 0
        for from_elem in weights.keys():
            for dr in weights[from_elem].keys():
                weights[from_elem][dr][self.non_place_elems] = 0.0

        # Change weights to % chances
        for from_elem in weights.keys():
            for dr in weights[from_elem].keys():
                m = weights[from_elem][dr]
                if self.log_weights:
                    weights[from_elem][dr][m > 0] = np.log(m[m > 0] + 1)
                m = weights[from_elem][dr]
                if np.sum(m):
                    weights[from_elem][dr] = m / np.sum(m)

        # Set some more attributes
        self.element_ids = {x for x in range(len(self.elements))}
        self.weights = weights

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

        allowed = np.ones((len(self.element_ids)))
        for coord in self.coords_around(arr, coords):
            dr = self.get_direction(coord, coords)
            allow = np.zeros((len(self.element_ids)))

            # If coord is not solved, check its possible element options instead
            if poss[coord][0] != -1:
                for item, weight in enumerate(poss[coord]):
                    if weight > 0:
                        if dr in self.weights[item]:
                            allow = np.add(allow, self.weights[item][dr])  #  * weight)

            # Coord is solved, so just fetch from its allowed list
            else:
                allow = 1
                y, x = coord
                e = arr[y, x]
                if e in self.weights:
                    if dr in self.weights[e]:
                        allow = self.weights[e][dr]

            allowed = np.multiply(allowed, allow)
        if np.sum(allowed):
            allowed = allowed / np.sum(allowed)
        return allowed

    def scan_elements(self, array, poss):
        """Brute force 'propagation' algorithm
        If a coordinates poss was changed, add that coordinates surroundings to the to-be-checked list. Repeat until list is exhausted
        :param array  -> Numpy array of already set elements
        :param poss   -> A dict of remaining coordinates' possible elements
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
                y, x = coords
                allowed = self.check_allowed(array, poss, coords)
                if np.array_equal(poss[y, x] > 0, allowed > 0):
                    continue
                poss[y, x] = allowed
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
            if poss[coords][0] == -1:
                continue
            allowed = self.check_allowed(array, poss, coords)
            if np.array_equal(poss[coords] > 0, allowed > 0):
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
        """Write elements mapped in array to level
        :param level -> Target Level object to write to
        :param array -> Numpy array that contains the mapped elements"""
        tile_template = {"px": [128, 128], "src": [96, 16], "f": 0, "t": 14, "d": [136]}
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
            # Empty all layers
            for layer in level.layers.values():
                layer["gridTiles"] = []
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
        print("Wrote level")

    def get_weights(self, arr, selection):
        """Calculate the weights for a selection from poss
        :param arr       -> Array containing the mapped elements
        :param selection -> A value selected from poss eg.((y, x), {1, 2, 3})
        :returns         -> A list of weights to be used in np.random.choice(..., p=)"""
        from_coords, elements = selection

        # If surrounding are still empty, return weights with all values valued the same
        if not [
            1
            for coord in self.coords_around(arr, from_coords)
            if arr[coord[0], coord[1]] != -1
        ]:
            return [1 / len(elements) for x in range(len(elements))]

        weights = {x: 1.0 for x in elements}
        for coords in self.coords_around(arr, from_coords):
            y, x = coords
            from_elem = arr[y][x]
            dr = self.get_direction(coords, from_coords)
            for elem in weights.keys():
                if from_elem in self.weights:
                    if dr in self.weights[from_elem]:
                        if elem in self.weights[from_elem][dr]:
                            weights[elem] *= self.weights[from_elem][dr][elem]
        # Weights doesnt always find the elements around it. In that case, dont handle this element right now
        # Not sure if this is correct behaviour
        total = sum([x for x in weights.values()])
        # if total == 0:
        #     return False
        weights = [x / total for x in weights.values()]
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


def entropy(m):
    m = m[m > 0]
    if not len(m):
        return 9001.0
    l = np.log(m)
    return -1 * np.sum(np.multiply(m, l))


elements = []

# For LDTKC map creation
# manager = LdtkcManager("world.ldtkc")
# template = create_ndarray(manager.levels[4001], ldtkc=True)
# checker = Tilechecker([template])
# target = manager.levels[4000]
# ndarray = create_ndarray(target, ldtkc=True)
# ldtkc = True

# For LDTK map creation
template = Level(world, ROOT / "0001-L4_I1_Template.ldtkl")
road_template = Level(world, ROOT / "0005-L4_I2_Roads.ldtkl")
target = Level(world, ROOT / "0000-L4_Snowtown.ldtkl")

templates = [create_ndarray(x) for x in [template]]
non_place_templates = [create_ndarray(x) for x in [road_template]]

checker = Tilechecker(templates, elements, non_place=non_place_templates)
# checker = Tilechecker(templates, elements)
roads = Tilechecker([create_ndarray(road_template)], elements)

ndarray = create_ndarray(target)
ldtkc = False

pathfinder = Pathfinder(checker)

timer = time.perf_counter()

# Map all targets pre-set elements to the array
ele_arr = checker.map_elements(ndarray, skip_empty=True)

# Level dimensions
hei, wid = ele_arr.shape

# Initialize poss with all coords having all options
poss = np.ones((hei, wid, len(checker.element_ids)), float)
poss = np.apply_along_axis(lambda x: x / np.sum(x) if np.sum(x) > 0 else x, 2, poss)

# Update poss with information about pre-set elements
arr = np.full((ele_arr.shape), -1, int)
for coords in list(zip(*np.nonzero(ele_arr != -1))):
    y, x = coords
    arr[y, x] = ele_arr[y, x]
    poss[y, x] = -1
checker.scan_elements(arr, poss)

print("Pre-set elements", time.perf_counter() - timer)

# Create the path
# path = pathfinder.create_path(arr, (27, 12), [(15, 64)])
# path = pathfinder.largen_path(arr, path)
#
# tmp_arr = copy.deepcopy(arr)
# road_poss = {coords: roads.element_ids for coords in poss.keys()}
# # Fill all tiles outside the path with grass
# # Fetch the top left tile in the road template
# grass_element = roads.map_elements(create_ndarray(road_template))[0, 0]
# for i, element in enumerate(np.nditer(tmp_arr)):
#     coord = y, x = i // wid, i % wid
#     if coord not in path:
#         tmp_arr[y, x] = grass_element
#         if coord in road_poss:
#             road_poss.pop(coord)
# roads.scan_elements(tmp_arr, road_poss)
#
# # tmp_arr is full of grass with the road carved out
# road_poss = {k: set.intersection(v, roads.used_elements) for k, v in road_poss.items()}
# while -1 in tmp_arr:
#     # entropy = {k: np.sum([np.log(x) * x for x in v if x > 0]) for k, v in road_poss.items()}
#     try:
#         min_opt = min(
#             [len(v) for k, v in road_poss.items() if k in path and len(v) > 0]
#         )
#     except ValueError:
#         print("Out of optinos")
#         break
#
#     select_poss = {
#         k: v for k, v in road_poss.items() if k in path and len(v) == min_opt
#     }
#     selected = random.choice(list(select_poss.items()))
#
#     if WEIGHTS:
#         if not (weights := roads.get_weights(arr, selected)):
#             continue
#         element_id = np.random.choice(list(selected[1]), p=weights)
#     else:
#         element_id = np.random.choice(list(selected[1]))
#
#     coord = (y, x) = selected[0]
#
#     road_poss.pop(coord)
#     poss.pop(coord)
#
#     tmp_arr[y][x] = element_id
#     roads.propagate_elements(tmp_arr, road_poss, coord)
#
#     arr[y][x] = element_id
# checker.scan_elements(arr, poss)
#
rng = np.random.default_rng()
while -1 in arr:
    print(f"{len(arr[arr==-1])} tiles left to fill")

    # Calculate whats the least amount of options any level has and remove all tiles that have more than it
    entropies = np.apply_along_axis(entropy, 2, poss)
    selected = rng.choice(np.array(np.where(entropies == np.amin(entropies))), axis=1)
    y, x = selected

    element_id = rng.choice(np.arange(len(poss[y, x])), p=poss[y, x])

    arr[y, x] = element_id

    poss[y, x] = -1
    checker.propagate_elements(arr, poss, (y, x))


print("Running time:", int(time.perf_counter() - timer), "s")

# Sort layer tiles by location for ease of reading
# for layer in target.layers:
#     if "gridTiles" in layer and layer["gridTiles"]:
#         layer["gridTiles"].sort(key=lambda x: x["d"][0])

# Write elements mapped into arr to the target
checker.write_elements(target, arr, ldtkc=ldtkc)

print("Wrote")
