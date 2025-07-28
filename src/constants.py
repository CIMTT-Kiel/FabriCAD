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
_ROOT = Path(__file__).parents[1]
_path_dict = {
    "ROOT":                 _ROOT,
    "REPORT_FIGURES":       _ROOT / "reports/figures",
    "CONFIG":               _ROOT / "config",

    "DATA_RAW":             _ROOT / "data/1_raw",
    "DATA_INTERMEDIATE":    _ROOT / "data/2_intermediate",
    "DATA_PRIMARY":         _ROOT / "data/3_primary",
    "DATA_FEATURE":         _ROOT / "data/4_feature",
    "DATA_MODEL_INPUT":     _ROOT / "data/5_model_input",
    "DATA_MODELS":          _ROOT / "data/6_models",
    "DATA_MODEL_OUTPUT":    _ROOT / "data/7_model_output",
    "DATA_REPORTING":       _ROOT / "data/8_reporting",
    "PREVIEW_DATA":         _ROOT / "data/fabricad_preview_data",

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