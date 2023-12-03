import torch
import torch.nn.functional as F

from torch.nn import ModuleList, Embedding, Linear, LSTM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch_scatter import scatter


class NodeEncoder(torch.nn.Module):
    def __init__(self, node_emb_size, hidden_dim):
        super(NodeEncoder, self).__init__()

        self.embeddings = ModuleList()

        for i in range(len(node_emb_size)):
            self.embeddings.append(Embedding(node_emb_size[i], hidden_dim))

    def reset_parameters(self):
        for emb in self.embeddings:
            emb.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])

        return out


class Net(torch.nn.Module):
    def __init__(self, node_emb_size, hidden_dim, output_dim, num_layers, dropout, bidirectional, max_level, device):
        super(Net, self).__init__()

        self.bidirectional = bidirectional
        self.dropout = dropout
        self.max_level = max_level
        self.device = device

        self.node_encoder = NodeEncoder(node_emb_size, hidden_dim)
        self.bag_encoder = Embedding(4, hidden_dim)
        self.pos_encoder = Linear(16, hidden_dim)

        self.lstms_attr = ModuleList()
        self.lstms_pos = ModuleList()
        self.lins_attr = ModuleList()
        self.lins_pos = ModuleList()

        for level in range(max_level):
            self.lstms_attr.append(
                LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                     dropout=dropout, bidirectional=bidirectional))
            self.lstms_pos.append(
                LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                     dropout=dropout, bidirectional=bidirectional))
            self.lins_attr.append(Linear(hidden_dim, output_dim))
            self.lins_pos.append(Linear(hidden_dim, output_dim))

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.bag_encoder.reset_parameters()
        self.pos_encoder.reset_parameters()

        for lstm_a, lstm_p, lin_a, lin_p in zip(self.lstms_attr, self.lstms_pos, self.lins_attr, self.lins_pos):
            lstm_a.reset_parameters()
            lstm_p.reset_parameters()
            lin_a.reset_parameters()
            lin_p.reset_parameters()

    def forward(self, batch_data):
        if hasattr(batch_data, 'x_tu'):
            batch_data.x = batch_data.x_tu

        row, col = batch_data.node2bag_index
        x_bag = self.bag_encoder(batch_data.x_bag)
        # if x_bag.dim() == 1:
        #     x_bag = x_bag.unsqueeze(0)

        if batch_data.x.size(1):
            x_node = self.node_encoder(batch_data.x)
            x_attr = x_bag + scatter(x_node[row], col, dim=0, dim_size=x_bag.size(0), reduce='sum')
        else:
            x_attr = x_bag + self.node_encoder(batch_data.bag_size)
        x_pos = self.pos_encoder(batch_data.x_pos)
        x_pos = scatter(x_pos[row], col, dim=0, dim_size=x_bag.size(0), reduce='sum')

        start_idx = [0]
        h_attr_list = list()
        h_pos_list = list()
        batch_size = batch_data.num_graphs

        for level in range(1, self.max_level+1):
            x_attr_list = list()
            x_pos_list = list()
            trunk_lengths = list()
            num_trunks = list()
            indices = list()

            for i in range(batch_size):
                if level == 1:
                    start_idx.append(start_idx[i] + batch_data.num_bags[i].item())

                if level > len(batch_data.trunk_list[i]):  # trunk at this level in graph 'i' is empty
                    num_trunks.append(0)
                    continue

                trunks = batch_data.trunk_list[i][level-1]
                num_trunks.append(len(trunks))
                indices.extend([i] * num_trunks[i])

                for t in trunks:
                    x_attr_list.append(x_attr[start_idx[i]:][torch.tensor(t)])
                    x_pos_list.append(x_pos[start_idx[i]:][torch.tensor(t)])
                    trunk_lengths.append(len(t))

            if sum(num_trunks) == 0:  # trunks at this level of all graphs in the batch are all empty
                break

            x_pad = pad_sequence(x_attr_list, batch_first=True, padding_value=0.0)
            x_pack = pack_padded_sequence(x_pad, trunk_lengths, batch_first=True, enforce_sorted=False)
            output, (h_n, c_n) = self.lstms_attr[level-1](x_pack)

            h_last = h_n[-1, :, :] + h_n[-2, :, :] if self.bidirectional else h_n[-1, :, :]
            h_trunk = scatter(h_last, torch.tensor(indices).to(self.device),
                              dim=0, dim_size=batch_size, reduce='sum')
            h_attr = F.dropout(h_trunk, self.dropout, training=self.training)
            h_attr_list.append(h_attr)

            x_pad = pad_sequence(x_pos_list, batch_first=True, padding_value=0.0)
            x_pack = pack_padded_sequence(x_pad, trunk_lengths, batch_first=True, enforce_sorted=False)
            output, (h_n, c_n) = self.lstms_pos[level-1](x_pack)

            h_last = h_n[-1, :, :] + h_n[-2, :, :] if self.bidirectional else h_n[-1, :, :]
            h_trunk = scatter(h_last, torch.tensor(indices).to(self.device),
                              dim=0, dim_size=batch_size, reduce='sum')
            h_pos = F.dropout(h_trunk, self.dropout, training=self.training)
            h_pos_list.append(h_pos)

        score_over_layer = 0
        # combine all trunk representations at various levels to create a representation of the skeleton tree
        for level, h_attr in enumerate(h_attr_list):
            score_over_layer += F.dropout(self.lins_attr[level](h_attr), self.dropout, training=self.training)
        for level, h_pos in enumerate(h_pos_list):
            score_over_layer += F.dropout(self.lins_pos[level](h_pos), self.dropout, training=self.training)

        return score_over_layer
