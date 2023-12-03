import networkx as nx
import torch

from itertools import chain
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, to_undirected, to_networkx

types = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def mol_from_data(data):
    mol = Chem.RWMol()

    atom_type = data.x if data.x.dim() == 1 else data.x[:, 0]
    for a in atom_type.tolist():
        mol.AddAtom(Chem.Atom(a))

    row, col = data.edge_index
    mask = row < col
    row, col = row[mask].tolist(), col[mask].tolist()

    bond_type = data.edge_attr if data.edge_attr.dim() == 1 else data.edge_attr[:, 0]
    bond_type = bond_type[mask].tolist()

    for i, j, bond in zip(row, col, bond_type):
        assert 1 <= bond <= 4
        mol.AddBond(i, j, types[bond - 1])

    return mol.GetMol()


def tree_decomp_mol(data):
    mol = mol_from_data(data)  # an `rdkit` molecule

    # process rings and bonds and generate their corresponding bags
    bags = [list(r) for r in Chem.GetSymmSSSR(mol)]
    x_bag = [1] * len(bags)
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            bags.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            x_bag.append(2)

    # generate the 'atom2bag' maps
    atom2bag = [[] for i in range(mol.GetNumAtoms())]
    for b in range(len(bags)):
        for atom in bags[b]:
            atom2bag[atom].append(b)

    # add singleton bags (the intersection of at least 3 bags) and construct the edges of the skeleton graph
    w_edge = {}
    for atom in range(mol.GetNumAtoms()):
        b_atom = atom2bag[atom]
        if len(b_atom) <= 1:  # 'atom' cannot be a singleton bag according to its definition
            continue

        bonds = [b for b in b_atom if len(bags[b]) == 2]  # bond bag(s) to which the 'atom' belongs
        rings = [b for b in b_atom if len(bags[b]) >= 4]  # ring bag(s) to which the 'atom' belongs

        if len(bonds) >= 3 or (len(b_atom) >= 3 and len(bonds) == 2):
            # intersection of at least 3 bonds, or intersection of at least 3 bags containing 2 bonds
            bags.append([atom])
            x_bag.append(3)
            b2 = len(bags) - 1
            for b1 in b_atom:
                w_edge[(b1, b2)] = 1
        elif len(rings) >= 3:  # intersection of at least 3 rings
            bags.append([atom])
            x_bag.append(3)
            b2 = len(bags) - 1
            for b1 in b_atom:
                w_edge[(b1, b2)] = 99
        else:  # construct an edge between bags b1 and b2 to which the 'atom' belongs
            for i in range(len(b_atom)):
                for j in range(i+1, len(b_atom)):
                    b1, b2 = b_atom[i], b_atom[j]
                    count = len(set(bags[b1]) & set(bags[b2]))
                    w_edge[(b1, b2)] = max(count, w_edge.get((b1, b2), -1))

    # update the 'atom2bag' maps
    atom2bag = [[] for i in range(mol.GetNumAtoms())]
    for b in range(len(bags)):
        for atom in bags[b]:
            atom2bag[atom].append(b)

    # construct the skeleton tree from the skeleton graph
    if len(w_edge) > 0:
        edge_index_T, weight = zip(*w_edge.items())
        edge_index = torch.tensor(edge_index_T).t()
        inv_weight = 100 - torch.tensor(weight)
        graph = to_scipy_sparse_matrix(edge_index, inv_weight, len(bags))
        skeleton_tree = minimum_spanning_tree(graph)
        edge_index, _ = from_scipy_sparse_matrix(skeleton_tree)
        edge_index = to_undirected(edge_index, num_nodes=len(bags))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    rows = [[i] * len(atom2bag[i]) for i in range(mol.GetNumAtoms())]
    row = torch.tensor(list(chain.from_iterable(rows)))
    col = torch.tensor(list(chain.from_iterable(atom2bag)))
    atom2bag = torch.stack([row, col], dim=0).to(torch.long)

    return edge_index, atom2bag, len(bags), torch.tensor(x_bag, dtype=torch.long), bags


def tree_decomposition(data):
    graph_data = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
    G = to_networkx(graph_data, to_undirected=True, remove_self_loops=True)

    # process isolated nodes and generate their corresponding bags
    bags = [[i] for i in list(nx.isolates(G))]
    x_bag = [3] * len(bags)
    # remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    # process cliques and generate their corresponding bags
    cliques = list()
    for c in nx.find_cliques(G):
        if len(c) >= 3:  # filter out edges, and they will be considered later in the 'edges'
            cliques.append(c)
            bags.append(c)
            x_bag.append(0)

    # process cycles and generate their corresponding bags
    cycles = list()
    for c in nx.cycle_basis(G):
        if len(c) >= 4:  # filter out triangles, since they have already been considered in the 'cliques'
            cycles.append(c)
            bags.append(c)
            x_bag.append(1)

    # remove cliques
    for c in cliques:
        S = G.subgraph(c)
        clique_edges = list(S.edges)
        G.remove_edges_from(clique_edges)
        G.remove_nodes_from(list(nx.isolates(G)))
    # remove cycles
    for c in cycles:
        cycle_edges = list(zip(c[:-1], c[1:]))
        cycle_edges.append((c[-1], c[0]))
        G.remove_edges_from(cycle_edges)
        G.remove_nodes_from(list(nx.isolates(G)))

    # process edges and generate their corresponding bags
    edges = list()
    for e in G.edges:
        edges.append(list(e))
        bags.append(list(e))
        x_bag.append(2)

    # generate the 'node2bag' maps
    node2bag = [[] for i in range(data.num_nodes)]
    for b in range(len(bags)):
        for node in bags[b]:
            node2bag[node].append(b)

    # add singleton bags (the intersection of at least 3 bags) and construct the edges of the skeleton graph
    w_edge = {}
    for node in range(data.num_nodes):
        b_node = node2bag[node]
        if len(b_node) <= 1:  # 'node' cannot be a singleton bag according to its definition
            continue

        edges, strcs, weights = list(), list(), list()
        for b in b_node:
            n = len(bags[b])
            if n == 2:  # 'b' represents an edge
                edges.append(b)
            elif n >= 3:  # 'b' represents a clique or a cycle
                strcs.append(b)
                weights.append(n*(n-1)//2 if x_bag[b] == 0 else n)

        if len(b_node) >= 3:  # intersection of at least 3 bags
            bags.append([node])
            x_bag.append(3)
            b = len(bags) - 1
            for e in edges:
                w_edge[(b, e)] = 1
            for s, w in zip(strcs, weights):
                w_edge[(b, s)] = w
        elif len(b_node) == 2:  # construct an edge between bags b1 and b2 to which the 'node' belongs
            b1, b2 = b_node[0], b_node[1]
            count = len(set(bags[b1]) & set(bags[b2]))
            w_edge[(b1, b2)] = max(count, w_edge.get((b1, b2), -1))

    # update the 'node2bag' maps
    node2bag = [[] for i in range(data.num_nodes)]
    for b in range(len(bags)):
        for node in bags[b]:
            node2bag[node].append(b)

    # construct the skeleton tree from the skeleton graph
    if len(w_edge) > 0:
        edge_index_T, weight = zip(*w_edge.items())
        edge_index = torch.tensor(edge_index_T).t()
        inv_weight = 50000 - torch.tensor(weight)
        graph = to_scipy_sparse_matrix(edge_index, inv_weight, len(bags))
        skeleton_tree = minimum_spanning_tree(graph)
        edge_index, _ = from_scipy_sparse_matrix(skeleton_tree)
        edge_index = to_undirected(edge_index, num_nodes=len(bags))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    rows = [[i] * len(node2bag[i]) for i in range(data.num_nodes)]
    row = torch.tensor(list(chain.from_iterable(rows)))
    col = torch.tensor(list(chain.from_iterable(node2bag)))
    node2bag = torch.stack([row, col], dim=0).to(torch.long)

    return edge_index, node2bag, len(bags), torch.tensor(x_bag, dtype=torch.long), bags


def position_features(edge_index, num_nodes, pos_dim):
    if edge_index.size(1) == 0:
        features = torch.zeros(num_nodes, pos_dim)
    else:  # random walk process
        A = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1)), torch.Size([num_nodes, num_nodes]))

        idx = torch.LongTensor([range(num_nodes), range(num_nodes)])
        elem = torch.sparse.sum(A, dim=-1).to_dense().clamp(min=1).pow(-1)
        D_inv = torch.sparse.FloatTensor(idx, elem, torch.Size([num_nodes, num_nodes])).to_dense()  # D^-1

        # iteration
        M_power = torch.sparse.mm(A, D_inv)
        M = M_power.to_sparse()
        features = list()
        for i in range(2 * pos_dim - 1):
            M_power = torch.sparse.mm(M, M_power)
            if i % 2 == 0:
                features.append(torch.diagonal(M_power))

        features = torch.stack(features, dim=-1)

    return features


def find_diameter_path(T):
    random_paths = nx.shortest_path(T, source=list(T.nodes)[0])
    source = max(random_paths, key=lambda i: len(random_paths[i]))
    source_paths = nx.shortest_path(T, source=source)
    target = max(source_paths, key=lambda i: len(source_paths[i]))
    diameter_path = source_paths[target]

    return diameter_path


def find_trunk(num_bags, tree_edge_index, max_level):
    trunk_list = list()

    T = nx.Graph()
    T.add_nodes_from(range(num_bags))
    T.add_edges_from(tree_edge_index.T.tolist())

    level = 0
    while T.nodes:
        level += 1
        if level <= max_level:
            level_list = list()  # a list of trunks at the current level

        isolated_nodes = list(nx.isolates(T))
        if level == 1 and len(isolated_nodes):
            level_list.append(isolated_nodes)
            T.remove_nodes_from(isolated_nodes)

        for c in list(nx.connected_components(T)):
            S = T.subgraph(c).copy()
            assert len(S.edges) != 0

            diameter_path = find_diameter_path(S)
            level_list.append(diameter_path)
            d_path_edges = list(zip(diameter_path[:-1], diameter_path[1:]))
            T.remove_edges_from(d_path_edges)
            T.remove_nodes_from(list(nx.isolates(T)))

        if level < max_level:
            trunk_list.append(level_list)
        elif not T.nodes:
            trunk_list.append(level_list)

    assert len(trunk_list) <= max_level

    return trunk_list


def get_node_emb_size(dataset):
    temp_node_attrs = dataset[0].x
    max_node_attrs = dataset[0].x

    for data in dataset[1:]:
        temp_node_attrs = torch.cat([max_node_attrs, data.x], 0)
        max_node_attrs = torch.max(temp_node_attrs, 0, keepdim=True)[0]

    return (max_node_attrs + 1).squeeze(0).tolist()


def get_bag_emb_size(dataset):
    max_size = 0

    for data in dataset:
        size = data.bag_size.max().item()
        if size > max_size:
            max_size = size

    return [max_size + 1]


def split_data(dataset):
    indices = [i for i in range(len(dataset))]
    labels = [data.y.item() for data in dataset]

    train_idx, val_test_idx, y_train, y_val_test = train_test_split(indices, labels, test_size=0.2,
                                                                    shuffle=True, stratify=labels)
    val_idx, test_idx, y_val, y_test = train_test_split(val_test_idx, y_val_test, test_size=0.5,
                                                        shuffle=True, stratify=y_val_test)

    return train_idx, val_idx, test_idx
