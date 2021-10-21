import torch
from torch_geometric.data import InMemoryDataset
from GraphGym.graphgym.contrib.loader.HIVCsvHandler import ProcessSlideFile
from GraphGym.graphgym.contrib.loader.HIVEdgeValues import HivProcGraphData, NeighborhoodType
from torch_geometric.data import Data


# class HivSlideBaseClass(InMemoryDataset):
#     """ this data set is constructed using the naive definition of neighbors determined by physical touch between
#     cells """
#
#     def __init__(self, root=ProcessSlideFile.FOLDER_DIR.parent.as_posix(), transform=None, pre_transform=None,
#                  pre_filter=None):
#         super(HivSlideBaseClass, self).__init__(root, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def raw_file_names(self):
#         return [x.is_file() for x in ProcessSlideFile.FOLDER_DIR.iterdir()]
#
#     @property
#     def processed_file_names(self):
#         return [type(self).__name__ + ".data"]
#
#     def download(self):
#         return []
#
#     def process(self):
#         dataset = list()
#         for file in self.raw_file_names:
#             cur_graph = HivProcGraphData(file)
#             edge_values = cur_graph.get_edge_values(self.get_neighborhood_and_args())
#             dataset.append(
#                 Data(x=cur_graph.file_processed.vertex_ten,
#                      edge_index=edge_values.edge_index,
#                      edge_attr=edge_values.edge_attr,
#                      y=cur_graph.file_processed.label,
#                      pos=cur_graph.file_processed.pos_ten,
#                      dist_mat=cur_graph.file_processed.dist_mat,
#                      name=cur_graph.file_processed.file_name))
#
#         data, slices = self.collate(dataset)
#         torch.save((data, slices), self.processed_paths[0])
#
#     @abstractmethod
#     def get_neighborhood_and_args(self):
#         pass

class HivSlidePhysicalDataSet(InMemoryDataset):
    """Data set of whole slide using physical contact as edges"""

    def __init__(self, root=ProcessSlideFile.FOLDER_DIR.parent.as_posix(), transform=None, pre_transform=None,
                 pre_filter=None):
        super(HivSlidePhysicalDataSet, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [x for x in ProcessSlideFile.FOLDER_DIR.iterdir() if x.is_file()]

    @property
    def processed_file_names(self):
        return [type(self).__name__ + ".data"]

    def download(self):
        return []

    def process(self):
        dataset = list()
        for file in self.raw_file_names:
            cur_graph = HivProcGraphData(file)
            edge_values = cur_graph.get_edge_values(NeighborhoodType.PHYSICAL_NEIGHBORHOOD)
            dataset.append(
                Data(x=cur_graph.file_processed.vertex_ten,
                     edge_index=edge_values.edge_index,
                     edge_attr=edge_values.edge_attr,
                     y=cur_graph.file_processed.label,
                     pos=cur_graph.file_processed.pos_ten,
                     dist_mat=cur_graph.file_processed.dist_mat,
                     name=cur_graph.file_processed.file_name))

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])


class HivSlideRadiusDataSet(InMemoryDataset):
    """Data set of whole slide using distance as edged with a cutoff of radius"""

    def __init__(self, root=ProcessSlideFile.FOLDER_DIR.parent.as_posix(), transform=None, pre_transform=None,
                 pre_filter=None, radius=25.0):
        self.radius = radius
        super(HivSlideRadiusDataSet, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [x for x in ProcessSlideFile.FOLDER_DIR.iterdir() if x.is_file()]

    @property
    def processed_file_names(self):
        return [type(self).__name__ + ".data"]

    def download(self):
        return []

    def process(self):
        dataset = list()
        for file in self.raw_file_names:
            cur_graph = HivProcGraphData(file)
            edge_values = cur_graph.get_edge_values(NeighborhoodType.RADIUS_NEIGHBORHOOD, radius=self.radius)
            dataset.append(
                Data(x=cur_graph.file_processed.vertex_ten,
                     edge_index=edge_values.edge_index,
                     edge_attr=edge_values.edge_attr,
                     y=cur_graph.file_processed.label,
                     pos=cur_graph.file_processed.pos_ten,
                     dist_mat=cur_graph.file_processed.dist_mat,
                     name=cur_graph.file_processed.file_name))

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    a = HivSlidePhysicalDataSet()
