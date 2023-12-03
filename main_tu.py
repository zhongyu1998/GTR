import argparse

import torch
import torch.optim as optim

from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from transform import TUDTransform, SkeletonTree
from utils.process import get_bag_emb_size, split_data
from model import Net


def train(model, device, train_loader, optimizer):
    model.train()

    total_loss = 0
    total_len = 0
    for data in train_loader:
        data = data.to(device)
        output = model(data)
        y = data.y.to(torch.long)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
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

    correct = 0
    total_len = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        correct += pred.eq(data.y).sum().cpu().item()
        total_len += len(pred)

    return correct / total_len


def main():
    # Parameters settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of graph trunk network (GTR)')
    parser.add_argument('--name', type=str, default="MUTAG",
                        help='name of dataset: MUTAG, ENZYMES, PROTEINS, IMDB-BINARY, IMDB-MULTI, COLLAB (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which GPU to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of LSTM layers (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--bidirectional', action="store_true",
                        help='whether to use bidirectional LSTMs (default: False)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='reduction factor of learning rate (default: 0.5)')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='number of epochs for learning rate reduction (default: 10)')
    parser.add_argument('--lr_limit', type=float, default=1e-5,
                        help='minimum learning rate, stop training once it is reached (default: 1e-5)')
    parser.add_argument('--max_level', type=int, default=5,
                        help='maximum number of trunk levels (default: 5)')
    # MUTAG avg: 2.27, max: 6; ENZYMES avg: 3.74, max: 18; PROTEINS avg: 3.87, max: 47
    # IMDB-B avg: 1.99, max: 10; IMDB-M avg: 1.36, max: 8; COLLAB avg: 10.69, max: 299
    parser.add_argument('--epochs', type=int, default=200,
                        help='maximum number of training epochs (default: 200)')
    args = parser.parse_args()
    print(args)

    transform = Compose([TUDTransform(), SkeletonTree(mol=False, max_level=args.max_level)])

    dataset = TUDataset(root='data', name=args.name, pre_transform=transform)
    if dataset.num_node_features:
        node_emb_size = [dataset.num_node_features]
    else:
        node_emb_size = get_bag_emb_size(dataset)

    train_idx, val_idx, test_idx = split_data(dataset)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    model = Net(node_emb_size=node_emb_size,
                hidden_dim=args.hidden_dim,
                output_dim=dataset.num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
                bidirectional=args.bidirectional,
                max_level=args.max_level,
                device=device).to(device)
    model.reset_parameters()

    print()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=args.lr_factor,
                                                     patience=args.lr_patience,
                                                     verbose=True)

    train_curve = []
    val_curve = []
    test_curve = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer)
        train_acc = evaluate(model, device, train_loader)
        val_acc = evaluate(model, device, val_loader)
        test_acc = evaluate(model, device, test_loader)

        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.6f}, '
              f'Train: {train_acc:.6f}, Val: {val_acc:.6f}, Test: {test_acc:.6f}')
        # with open(filename, 'a') as f:
        #     f.write(f'{train_loss:.6f} {train_acc:.6f} {val_acc:.6f} {test_acc:.6f}')
        #     f.write("\n")

        train_curve.append(train_acc)
        val_curve.append(val_acc)
        test_curve.append(test_acc)

        scheduler.step(val_acc)

        if optimizer.param_groups[0]['lr'] < args.lr_limit:
            break

    print()
    print(f'===== Final result: {test_curve[-1]}')


if __name__ == '__main__':
    main()
