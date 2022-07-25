import time
import copy
import random
import numpy as np


class Pathfinder:
    def __init__(self, tilechecker):
        self.checker = tilechecker

    def end_close_enough(self, coords, end):
        y, x = coords  # Coords y and x
        e_y, e_x = end  # End loc y and x
        if x == e_x and abs(y - e_y) < 3:
            return True
        if y == e_y and abs(x - e_x) < 3:
            return True
        return False

    def find_path(self, arr, from_coords, to_coords: list):
        """Dijkstras pathfinding algo that moves in steps of 3.
        First goes from A to B, then from the middle of the just created path to C and so on....
        :param arr -> The array with mapped elements
        :param from_coords -> A tuple of coordinates to start from
        :param to_coords   -> A list of tuples of coordinates to coordinate to"""
        path_steps = []
        y, x = from_coords
        for end_coord in to_coords:
            visited = set()
            # First iteration draw a straight path. The ones after that start from the middle of the current known path
            if path_steps:
                current = path_steps[len(path_steps) // 2]
            else:
                current = from_coords
            path = {current: [current]}
            while current:
                # TODO: mby move coords_around outside of the tilechecker
                for coord in self.checker.coords_around(arr, current, steps=3):
                    y, x = coord
                    dist = len(path[current]) + 1
                    if coord not in path or dist < len(path[coord]):
                        path[coord] = path[current] + [coord]

                visited.add(current)
                if current == end_coord:
                    path_steps.extend(path[end_coord])
                    break
                # If under 3 tiles away in on direction from the end, add current path + end coord to the final path
                # fill_path will fill the skipped coordinates
                if self.end_close_enough(current, end_coord):
                    path_steps.extend(path[current] + [end_coord])
                    break
                next = None
                for c, p in path.items():
                    y, x = c
                    if arr[y][x] != -1:
                        continue
                    if c in visited:
                        continue
                    if next is None or len(p) < len(path[next]):
                        next = c
                if next is None:
                    return False
                current = next
        # Fill in the blanks caused by taking 3 steps at a time
        final_path = self.fill_path(path_steps)
        return final_path

    def create_path(self, arr, from_coords, to_coords: list):
        """Uses dijkstras pathfinding algo, and creates wiggliness in the shortest path found
        First goes from A to B, then from the middle of the just created path to C and so on....
        :param arr -> The array with mapped elements
        :param from_coords -> A tuple of coordinates to start from
        :param to_coords   -> A list of tuples of coordinates to coordinate to"""
        timer = time.perf_counter()
        arr = copy.deepcopy(arr)
        open_cells = list(zip(*np.nonzero(arr == -1)))
        random.shuffle(open_cells)
        if not (witness := self.find_path(arr, from_coords, to_coords)):
            raise Exception(f"Could not find a path from {from_coords} to {to_coords}")
        wiggliness_reduction = 0.0  # Percentage
        while True:
            if not open_cells:
                print("Finding out the path took", time.perf_counter() - timer)
                return witness
            c = open_cells.pop(0)
            y, x = c
            if c in witness:
                if random.random() < wiggliness_reduction:
                    continue  # Reduce the wiggliness
                # find_path considers everything other than -1 as an obstacle
                arr[y][x] = 0
                if new_path := self.find_path(arr, from_coords, to_coords):
                    witness = new_path

    def largen_path(self, array, path):
        """Walk through path and append all coordinates around it to the path. Widens the path by 2"""
        large_path = set()
        for coord in path:
            large_path = large_path | {
                tuple(x) for x in self.checker.cube_around(array, coord)
            }
        # large_path == 3 wide
        return large_path

    def fill_path(self, path):
        # The pathfinding moves in steps of 3, so fill in the blank coordinates
        filled_path = []
        prev_coord = False
        for coord in path:
            if prev_coord:
                y, x = coord
                prev_y, prev_x = prev_coord
                if abs(y - prev_y) > 3 or abs(x - prev_x) > 3:
                    prev_coord = coord
                    continue
                if y != prev_y:
                    if y > prev_y:
                        for y in range(prev_y + 1, y):
                            filled_path.append((y, x))
                    else:
                        for y in range(prev_y - 1, y, -1):
                            filled_path.append((y, x))
                if x != prev_x:
                    if x > prev_x:
                        for x in range(prev_x + 1, x):
                            filled_path.append((y, x))
                    else:
                        for x in range(prev_x - 1, x, -1):
                            filled_path.append((y, x))
            prev_coord = coord
            filled_path.append(coord)
        return filled_path
