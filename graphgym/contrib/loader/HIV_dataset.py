from pathlib import Path
from GraphGym.graphgym.contrib.loader import datasets as ds
from graphgym.register import register_loader
from deepsnap.dataset import GraphDataset

TOP_PROJECT_DIR = Path(__file__).parent.parent.parent.parent.parent


# def load_hiv_dataset(neighbordef='initial', subgraph='none',
#                      naive_radius=25,
#                      knn=5, knn_max=50,
#                      high_exp_marker='CK', exp_radius=5, exp_depth=10,
#                      width=250, window_num=100, min_cells=0):
#     return ds.HIV_Dataset(neighbordef=neighbordef, subgraph=subgraph,
#                           naive_radius=naive_radius,
#                           knn=knn, knn_max=knn_max,
#                           high_exp_marker=high_exp_marker, exp_radius=exp_radius, exp_depth=exp_depth,
#                           width=width, window_num=window_num, min_cells=min_cells)


def load_hiv_dataset(format, name, dataset_dir):
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if format == 'PyG':
        if name == 'hiv':
            dataset_raw = ds.HIV_Dataset(neighbordef='initial', subgraph='windows',
                                         naive_radius=25,
                                         knn=5, knn_max=50,
                                         high_exp_marker='CK', exp_radius=5, exp_depth=10,
                                         width=250, window_num=100, min_cells=0)
            graphs = GraphDataset.pyg_to_graphs(dataset_raw)
            return graphs


register_loader('hiv', load_hiv_dataset)
