from io import BytesIO
from pathlib import Path
import numpy as np


class LdtkcManager:
    def __init__(self, world_path):
        with open(Path(world_path), "rb") as file:
            data = np.load(BytesIO(file.read()), allow_pickle=True)
        # Set attributes
        self.data = data
        self.level_data, self.world_data = [data[key][()] for key in data.files]
        self.levels = {int(key): self.level_data[key] for key in self.level_data.keys()}

    def tile_layers(self, level):
        return [x for x in level["layers"] if x[0] == "TILES"]
