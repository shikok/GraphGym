import pandas as pd
import numpy as np
import torch
import re
from scipy.spatial import distance_matrix
from torch import Tensor
from pathlib import Path

pd.options.mode.chained_assignment = None
PROJECT_DIR = Path(__file__).parent.parent


class ProcessSlideFile(object):
    """ class for processing a slide file basic features """

    # ### Constants ###
    FOLDER_DIR = PROJECT_DIR.joinpath("data/HIV/segmented/")
    # Note that T32 is negative, not positive, based on dropbox data, not on annotation key, and T45 is the same.
    _KEY_TABLE = pd.read_csv(PROJECT_DIR.joinpath('data/HIV/HIV_key.csv').as_posix())
    _LABEL_NAME_REGEX = 'T[0-9]*'
    _MARKERS_COL_START_INDEX = 2
    _MARKERS_COL_STOP_INDEX = 34
    _LABEL_COL_NAME = 'HIV'
    _X_POS_COL_NAME = 'X_position'
    _Y_POS_COL_NAME = 'Y_position'
    CELL_ID_COL_INDEX = 1
    NEIGHBOR_COL_START_INDEX = 128

    def __init__(self, file: str):
        self.file_path = self.FOLDER_DIR.joinpath(file)
        self.file_name = self.file_path.name
        self.file_pointer = pd.read_csv(self.file_path)
        self._dist_mat = None
        self._vertex_ten = None
        self._pos_arr = None
        self._label = None

    @property
    def vertex_ten(self) -> Tensor:
        if self._vertex_ten is None:
            self._set_vertex_ten()
        return self._vertex_ten

    @property
    def dist_mat(self) -> np.ndarray:
        if self._dist_mat is None:
            self._set_dist_mat()
        return self._dist_mat

    @property
    def pos_arr(self) -> np.ndarray:
        if self._pos_arr is None:
            self._set_pos_arr()
        return self._pos_arr

    @property
    def label(self) -> Tensor:
        if self._label is None:
            self._set_label()
        return self._label

    def _set_vertex_ten(self, dtype=torch.double):
        rel_cols = self.file_pointer.columns[self._MARKERS_COL_START_INDEX:self._MARKERS_COL_STOP_INDEX + 1]
        self._vertex_ten = torch.tensor(self.file_pointer.loc[:, rel_cols].values, dtype=dtype)

    def _set_label(self):
        file_key = re.search(self._LABEL_NAME_REGEX, self.file_path.name)[0]
        status = self._KEY_TABLE.loc[self._KEY_TABLE.Label == file_key, self._LABEL_COL_NAME]
        self._label = torch.tensor(status.values)

    def _set_dist_mat(self):
        self._dist_mat = np.exp(-(distance_matrix(self.pos_arr, self.pos_arr) ** 2))

    def _set_pos_arr(self):
        self._pos_arr = self.file_pointer.loc[:, [self._X_POS_COL_NAME, self._Y_POS_COL_NAME]].values


if __name__ == '__main__':
    file_name = "ROI002_T1.csv"
    ex = ProcessSlideFile(file_name)
    d = ex.dist_mat
    p = ex.pos_arr
    v = ex.vertex_ten
    l = ex.label
    print(d)
