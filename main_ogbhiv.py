import argparse

import torch
import torch.optim as optim

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from transform import OGBTransform, SkeletonTree
from utils.process import get_node_emb_size
from model import Net


def train(model, device, train_loader, optimizer):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        output = model(data)
        y = data.y.to(output.dtype)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def evaluate(model, device, loader, evaluator):
    model.eval()

    y_preds, y_trues = [], []
    for data in loader:
        data = data.to(device)
        y_preds.append(model(data))
        y_trues.append(data.y)

    perf = evaluator.eval({
        'y_pred': torch.cat(y_preds, dim=0),
        'y_true': torch.cat(y_trues, dim=0),
    })[evaluator.eval_metric]

    return perf


def main():
    # Parameters settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of graph trunk network (GTR)')
    parser.add_argument('--name', type=str, default="ogbg-molhiv",
                        help='name of dataset (default: ogbg-molhiv)')
    parser.add_argument('--device', type=int, default=0,
                        help='which GPU to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='number of hidden units (default: 128)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of LSTM layers (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--bidirectional', action="store_true",
                        help='whether to use bidirectional LSTMs (default: False)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='initial learning rate (default: 0.0003)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='reduction factor of learning rate (default: 0.5)')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='number of epochs for learning rate reduction (default: 10)')
    parser.add_argument('--lr_limit', type=float, default=5e-5,
                        help='minimum learning rate, stop training once it is reached (default: 5e-5)')
    parser.add_argument('--max_level', type=int, default=4,  # avg: 2.26, max: 26
                        help='maximum number of trunk levels (default: 4)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='maximum number of training epochs (default: 50)')
    args = parser.parse_args()
    print(args)

    transform = Compose([OGBTransform(), SkeletonTree(mol=True, max_level=args.max_level)])

    evaluator = Evaluator(args.name)
    dataset = PygGraphPropPredDataset(args.name, 'data', pre_transform=transform)
    node_emb_size = get_node_emb_size(dataset)

    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx['train']]
    val_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    model = Net(node_emb_size=node_emb_size,
                hidden_dim=args.hidden_dim,
                output_dim=dataset.num_tasks,
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
        train_perf = evaluate(model, device, train_loader, evaluator)
        val_perf = evaluate(model, device, val_loader, evaluator)
        test_perf = evaluate(model, device, test_loader, evaluator)

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
    print(f'===== Final result: {test_curve[val_curve.index(max(val_curve))]}')


if __name__ == '__main__':
    main()
