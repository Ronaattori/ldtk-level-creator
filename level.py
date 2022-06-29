import json
from pathlib import Path
import copy
import string
import logging
import numpy as np

LEVEL_TEMPLATE = Path("./0000-Template.ldtkl")
ROOT = Path("world/world")
ESSENTIALS_FOLDER = Path(r"C:\Pythonpaskaa\Pokemon Essentials v20")


class Level:
    """
    :param World object
    :param level_filename=None (path to an existing level)

    The object for a level
    """

    def __init__(self, world, level_filename):
        try:
            with open(level_filename, "r", encoding="utf-8") as infile:
                self.json = json.load(infile)
        except FileNotFoundError:
            with open(LEVEL_TEMPLATE, "r", encoding="utf-8") as infile:
                self.json = json.load(infile)

        # Create some useful attributes
        self.filename = str(level_filename)
        self.name = self.json["identifier"]
        self.size = (self.json["pxWid"], self.json["pxHei"])
        self.world = world

    @property
    def layers(self):
        return {x["__identifier"]: x for x in self.json["layerInstances"]}

    # Take in a single int tile texture pointer and return the coordinate version of it
    def tToSrc(self, value):
        x = value % 8 * 16
        y = value // 8 * 16
        return [x, y]

    # Convert coords into the single int pointer format
    def coordToInt(self, coords, width):
        return coords[0] + coords[1] * width

    def set_size(self, width_px, height_px):
        """
        Set the level size
        :param Level width in pixels
        :param Level height in pixels
        """
        self.json["pxWid"] = width_px
        self.json["pxHei"] = height_px

    def set_uid(self, uid):
        """
        Set the level uid
        :param Uid to set
        """
        self.json["uid"] = uid

    def set_levelids(self):
        """Set the correct levelId to all layers. Uses the level uid as the levelId"""
        for x in self.layers.values():
            x["levelId"] = self.json["uid"]

    def set_identifier(self, identifier):
        """
        Set the identifier for the level
        :param Identifier
        """
        self.json["identifier"] = identifier
        self.filename.split("-")[1].split(".")[0]

    def set_location(self, coords):
        """
        Set the world location
        :param A tuple of (x, y, Depth)
        """
        x, y, depth = coords
        self.json["worldX"] = x * 16
        self.json["worldY"] = y * 16
        self.json["worldDepth"] = depth

    def copy_layer(self, copy_layer, layer_name):
        """
        Create a new layer from a copy
        :param Layer to copy
        :param New name for the layer
        """
        if layer_name not in self.world.layers:
            self.world.copy_layer(copy_layer, layer_name)
        lc = copy.deepcopy(self.layers[copy_layer])
        lc["__identifier"] = layer_name
        lc["layerDefUid"] = self.world.layer_uids[layer_name]
        self.json["layerInstances"].append(lc)

    def __create_filename(self, name):
        # name = self.rubydata.attributes["@name"].decode("utf-8")
        id = max(int(x.name.split("-")[0]) for x in ROOT.glob("*.ldtkl")) + 1
        mapName = "".join(l for l in name if l.isalnum())
        levelName = f"L{self.rubydata.attributes['@id']}_{mapName}"
        return f"{id:0>4}-{levelName}.ldtkl"

    def set_tiles(self, rmxp_array, autotiles):
        """
        Sets the correct tiles into the level
        :param 3 layer array the size of the level
        :param A list of autotiles used by the level
        """
        self.layers["Ground"]["gridTiles"] = []

        # Get sizes in tiles and pixels
        width_px = self.json["pxWid"]
        height_px = self.json["pxHei"]
        width_tiles = width_px // 16
        height_tiles = height_px // 16

        autotile_arrays = {}
        for layer in range(3):
            for index_x in range(width_tiles):
                for index_y in range(height_tiles):
                    t = int(rmxp_array[layer][index_y][index_x])
                    if t == 0:
                        continue

                    # Adjust t to account for 384 RMXP autotiles and 8 empty ldtk tiles
                    # Add +8 if importing legacy tilesets (empty line on top)
                    t = t - 384

                    # Negative means an autotile
                    if t < 0:
                        t += 384
                        autotile_index = t // 48 - 1
                        autotileset = autotiles[autotile_index]

                        # Create a zeros array the size of the level for the current autotileset
                        if autotileset not in autotile_arrays:
                            autotile_arrays[autotileset] = np.zeros(
                                (height_tiles, width_tiles), int
                            )

                        # Add arrays to a dict, to be written into the level later
                        autotile_arrays[autotileset][index_y][index_x] = 1

                    else:
                        src = self.tToSrc(t)
                        self.layers["Ground"]["gridTiles"].append(
                            {
                                "px": [width_px, height_px],
                                "src": src,
                                "f": 0,
                                "t": t,
                                "d": [self.coordToInt((index_x, index_y), width_tiles)],
                            }
                        )
        for at_name, at_array in autotile_arrays.items():
            if not (layer_type := self.get_autotile_layer(at_name)):
                logging.warning(f"Layer type not found for: {at_name}")
                continue
            layer_name = self.next_layer_name(layer_type)

            # Create the layer if it doesn't exist yet
            if not self.layer_exists(layer_name):
                self.copy_layer(layer_type + "A", layer_name)
                # TODO: Possibly make it add empty levels as soon as they are known
            self.layers[layer_name]["intGridCsv"] = at_array.flatten().tolist()

    def get_autotile_layer(self, autotile_name):
        """
        param: The name of the autotileset
        returns: The name of the autotile layer it should go on"""
        lower = autotile_name.lower()

        for i in ["water", "fountain", "sea", "flowers", "flowers1"]:
            if i in lower:
                return "Auto_Water_"

        for i in ["cliff"]:
            if i in lower:
                return "Auto_Cliff_"

        for i in ["brick", "path"]:
            if i in lower:
                return "Auto_Road_"

    def layer_exists(self, layer_name):
        "Check if a layer exists in the level"
        for k in self.layers:
            if k == layer_name:
                return True
        return False

    def next_layer_name(self, layer_type):
        """Find out the next needed alphabet for any layer type
        :param Layer type (eg. Auto_Water_)"""
        for l in string.ascii_uppercase:
            if not self.layer_exists(name := layer_type + l):
                return name
        return None

    def fill_autotile_layer(self, type, needed):
        """Fill needed autotile layers with empty ones
        :param Layer type (eg. Auto_Water_)
        :param How many layers are needed"""
        for i in range(needed):
            lc = copy.deepcopy(self.layers[f"{type}A"])
            lc["intGridCsv"][lc["intGridCsv"] == 1] = 0
            lc["__identifier"] = id = self.next_layer_name(type)
            lc["layerDefUid"] = self.world.layer_uids[id]

            # Add it to the class
            self.json["layerInstances"].append(lc)

    def clear_ground(self):
        "Clear ground layer tiles"
        self.layers["Ground"]["gridTiles"] = []

    def write(self):
        with open(self.filename, "w", encoding="utf-8") as outfile:
            json.dump(self.json, outfile, indent=4)

        # # Remove data that doesn't belong into the world file
        # self.json["layerInstances"] = None
        # self.json["externalRelPath"] = self.filename
        # self.json.pop("__header__")
        # self.world.json["nextUid"] += 1

        # # Save level to world file
        # self.world.json["levels"].append(self.json)
