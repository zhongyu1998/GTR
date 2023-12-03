import torch
from torch_geometric.data import Data

from utils.process import tree_decomp_mol, tree_decomposition, position_features, find_trunk


class OGBTransform(object):
    # In OGB, atom and bond types are numbered starting from 0.
    # We need to convert them to match the numbering convention of RDKit.
    def __call__(self, data):
        data.x[:, 0] += 1
        data.edge_attr[:, 0] += 1
        return data


class TUDTransform(object):
    def __call__(self, data):
        if data.num_node_features == 4:  # PROTEINS
            data.x_tu = torch.argmax(data.x[:, 1:], dim=1, keepdim=True)
        elif data.num_node_features == 7:  # MUTAG
            data.x_tu = torch.argmax(data.x, dim=1, keepdim=True)
        elif data.num_node_features == 21:  # ENZYMES
            data.x_tu = torch.argmax(data.x[:, 18:], dim=1, keepdim=True)
        elif data.num_node_features == 0:  # IMDB-BINARY, IMDB-MULTI, COLLAB
            data.x = torch.zeros(data.num_nodes, 0)
        return data


class SkeletonTreeData(Data):
    def __inc__(self, key, item, *args):
        if key == 'tree_edge_index':
            return self.x_bag.size(0)
        elif key == 'node2bag_index':
            return torch.tensor([[self.x.size(0)], [self.x_bag.size(0)]])
        else:
            return super(SkeletonTreeData, self).__inc__(key, item, *args)


class SkeletonTree(object):
    def __init__(self, mol, max_level):
        self.mol = mol
        self.max_level = max_level

    def __call__(self, data):
        if self.mol:
            tree_edge_index, node2bag_index, num_bags, x_bag, bag_list = tree_decomp_mol(data)
        else:
            tree_edge_index, node2bag_index, num_bags, x_bag, bag_list = tree_decomposition(data)
        trunk_list = find_trunk(num_bags, tree_edge_index, self.max_level)

        data = SkeletonTreeData(**{k: v for k, v in data})

        data.tree_edge_index = tree_edge_index  # edge index in the skeleton tree
        data.node2bag_index = node2bag_index  # map each node in the original graph to the bag(s) in the skeleton tree
        data.num_bags = num_bags  # number of bags in the skeleton tree
        data.x_bag = x_bag  # identifier for each bag (0: clique, 1: cycle, 2: edge, 3: isolated node or singleton)
        # data.bag_list = bag_list  # a list of bags in the skeleton tree
        data.bag_size = torch.tensor([len(b) for b in bag_list], dtype=torch.long)  # size of each bag
        data.x_pos = position_features(data.edge_index, data.num_nodes, 16)  # position feature of each node
        data.trunk_list = trunk_list  # a list of trunks at various levels in the skeleton tree

        return data
