import json
from pathlib import Path
import copy

ESSENTIALS_FOLDER = Path(r"C:\Pythonpaskaa\Pokemon Essentials v20")


class World:
    """
    :param World file filepath

    The world object
    """

    def __init__(self, world_filepath):
        with open(world_filepath, "r", encoding="utf-8") as infile:
            self.json = json.load(infile)

        # Create some useful attributes
        self.filepath = world_filepath

    @property
    def layer_uids(self):
        return {x["identifier"]: x["uid"] for x in self.json["defs"]["layers"]}

    @property
    def layers(self):
        return {x["identifier"]: x for x in self.json["defs"]["layers"]}

    def copy_layer(self, copy_layer, layer_name):
        lc = copy.deepcopy(self.layers[copy_layer])
        lc["uid"] = max(self.layer_uids.values()) + 1
        lc["identifier"] = layer_name
        self.json["defs"]["layers"].append(lc)

    def write(self):
        with open(self.filepath, "w", encoding="utf-8") as outfile:
            json.dump(self.json, outfile, indent=4)
