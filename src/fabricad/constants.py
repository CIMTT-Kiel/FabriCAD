"""
This module defines several constants used throughout the project.

The module provides access to NamedTupel objects which group all constants and make them available via attribute access.

Examples
--------
>>> from project import constants
>>> # get Path object for the project root directory
>>> constants.PATHS.ROOT
"""
from pathlib import Path

from collections import namedtuple


# Paths
_ROOT = Path(__file__).parents[2]
_path_dict = {
    "ROOT":                 _ROOT,
    "PREVIEW_DATA":             _ROOT / "data/fabricad_preview_data",

}

Paths = namedtuple("Paths", list(_path_dict.keys()))
PATHS = Paths(**_path_dict)

# clean up for paths constants
del _path_dict
del Paths
del _ROOT


# general clean up
del namedtuple
del Path