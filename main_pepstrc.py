import argparse

import torch
import torch.optim as optim

from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from transform import OGBTransform, SkeletonTree
from utils.peptides_structural import PeptidesStructuralDataset
from utils.process import get_node_emb_size
from model import Net


def train(model, device, train_loader, optimizer):
    model.train()

    total_loss = 0
    total_len = 0
    for data in train_loader:
        data = data.to(device)
        mask = ~torch.isnan(data.y)
        output = model(data)[mask]
        y = data.y[mask]
        criterion = torch.nn.L1Loss(reduction='sum')
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_len += len(output)

    return total_loss / total_len


@torch.no_grad()
def evaluate(model, device, loader):
    model.eval()

    total_perf = 0
    total_len = 0
    for data in loader:
        data = data.to(device)
        mask = ~torch.isnan(data.y)
        output = model(data)[mask]
        y = data.y[mask]
        l1_loss = torch.nn.L1Loss(reduction='sum')
        total_perf += l1_loss(output, y).cpu().item()
        total_len += len(output)

    return total_perf / total_len


def main():
    # Parameters settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of graph trunk network (GTR)')
    parser.add_argument('--name', type=str, default="Peptides-struct",
                        help='name of dataset (default: Peptides-struct)')
    parser.add_argument('--device', type=int, default=0,
                        help='which GPU to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='number of hidden units (default: 256)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of LSTM layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--bidirectional', action="store_true",
                        help='whether to use bidirectional LSTMs (default: False)')
    parser.add_argument('--lr', type=float, default=0.0004,
                        help='initial learning rate (default: 0.0004)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='reduction factor of learning rate (default: 0.5)')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='number of epochs for learning rate reduction (default: 10)')
    parser.add_argument('--lr_limit', type=float, default=2e-5,
                        help='minimum learning rate, stop training once it is reached (default: 2e-5)')
    parser.add_argument('--max_level', type=int, default=12,  # avg: 4.19, max: 37
                        help='maximum number of trunk levels (default: 12)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='maximum number of training epochs (default: 300)')
    args = parser.parse_args()
    print(args)

    transform = Compose([OGBTransform(), SkeletonTree(mol=True, max_level=args.max_level)])

    dataset = PeptidesStructuralDataset('data', pre_transform=transform)
    node_emb_size = get_node_emb_size(dataset)

    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx['train']]
    val_dataset = dataset[split_idx['val']]
    test_dataset = dataset[split_idx['test']]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    model = Net(node_emb_size=node_emb_size,
                hidden_dim=args.hidden_dim,
                output_dim=dataset[0].y.shape[1],
                num_layers=args.num_layers,
                dropout=args.dropout,
                bidirectional=args.bidirectional,
                max_level=args.max_level,
                device=device).to(device)
    model.reset_parameters()

    print()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=args.lr_factor,
                                                     patience=args.lr_patience,
                                                     verbose=True)

    train_curve = []
    val_curve = []
    test_curve = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer)
        train_perf = evaluate(model, device, train_loader)
        val_perf = evaluate(model, device, val_loader)
        test_perf = evaluate(model, device, test_loader)

        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.6f}, '
              f'Train: {train_perf:.6f}, Val: {val_perf:.6f}, Test: {test_perf:.6f}')
        # with open(filename, 'a') as f:
        #     f.write(f'{train_loss:.6f} {train_perf:.6f} {val_perf:.6f} {test_perf:.6f}')
        #     f.write("\n")

        train_curve.append(train_perf)
        val_curve.append(val_perf)
        test_curve.append(test_perf)

        scheduler.step(val_perf)

        if optimizer.param_groups[0]['lr'] < args.lr_limit:
            break

    print()
    print(f'===== Final result: {test_curve[val_curve.index(min(val_curve))]}')


if __name__ == '__main__':
    main()
