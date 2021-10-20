from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from torch import Tensor
from GraphGym.graphgym.contrib.loader.HIVCsvHandler import ProcessSlideFile


class NeighborhoodType(Enum):
    PHYSICAL_NEIGHBORHOOD = 0
    RADIUS_NEIGHBORHOOD = 1
    KNN_NEIGHBORHOOD = 2


@dataclass
class GraphEdgeValues:
    edge_index: Optional[Tensor] = None
    edge_attr: Optional[Tensor] = None


class HivProcGraphData:
    """ class for getting edges of data graph according to user wishes """

    def __init__(self, file: str):
        self.file = file
        self.file_processed = ProcessSlideFile(file)

    def get_edge_values(self, neighborhood_type: NeighborhoodType, **kwargs):
        if neighborhood_type is NeighborhoodType.PHYSICAL_NEIGHBORHOOD:
            return self._get_phy_edge_index()
        elif neighborhood_type is NeighborhoodType.RADIUS_NEIGHBORHOOD:
            return self._get_rad_edge_index(**kwargs)
        elif neighborhood_type is NeighborhoodType.KNN_NEIGHBORHOOD:
            return self._get_knn_edge_index()
        else:
            raise IOError("Invalid neighborhood type")

    def _get_phy_edge_index(self) -> GraphEdgeValues:
        # extract neighborhood information
        neigh_df = self.file_processed.file_pointer.iloc[:,
                   ProcessSlideFile.NEIGHBOR_COL_START_INDEX:self.file_processed.file_pointer.shape[1]]
        cell_id_df = self.file_processed.file_pointer.iloc[:, ProcessSlideFile.CELL_ID_COL_INDEX]  # Cell ID
        unite_df = pd.concat([cell_id_df, neigh_df], axis=1)  # Join cell IDs to neighbor data
        # Arrange into two-columns - cell ID and neighbor
        cell_nei_df = unite_df.melt(id_vars="CellId", value_vars=unite_df.columns[1:],
                                    var_name="NeighbourNumber", value_name="NeighbourId")
        cell_nei_red_df = cell_nei_df[cell_nei_df.NeighbourId != 0].drop('NeighbourNumber',
                                                                         axis=1)  # Remove all non-neighbor lines
        edge_index = torch.tensor(cell_nei_red_df.transpose().values - 1)
        return GraphEdgeValues(edge_index=edge_index)

    def _get_knn_edge_index(self) -> GraphEdgeValues:
        pass

    def _get_rad_edge_index(self, radius: float = 25.0) -> GraphEdgeValues:
        proc_dist_mat = np.copy(self.file_processed.dist_mat)
        # remove self loop edges by setting diagonal distances to be higher then radius
        np.fill_diagonal(proc_dist_mat, 2 * radius)
        edge_tensor = torch.tensor(proc_dist_mat[proc_dist_mat <= radius])
        edge_index = torch.tensor(np.argwhere(proc_dist_mat <= radius).transpose())
        return GraphEdgeValues(edge_index=edge_index, edge_attr=edge_tensor)
