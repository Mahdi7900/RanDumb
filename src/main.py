import os
import argparse
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.kernel_approximation import RBFSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100', 'vitbi1k_cars',  'vitbi21k_cars',  'vitbi1k_cifar100', 'vitbi1k_imagenet-r', 'vitbi1k_imagenet-a', 'vitbi1k_cub',
                        'vitbi1k_omnibenchmark', 'vitbi1k_vtab', 'vitbi1k_cars', 'vitbi21k_cifar100', 'vitbi21k_imagenet-r', 'vitbi21k_imagenet-a', 'vitbi21k_cub', 'vitbi21k_omnibenchmark', 'vitbi21k_vtab'], help='Dataset')
    parser.add_argument('--model', type=str, default='SLDA',
                    choices=['SLDA', 'NCM', 'DLP'], help='Model')
    parser.add_argument('--augment', action='store_true',
                        help='Use RandomFlip Augmentation')
    parser.add_argument('--embed', action='store_true',
                        help='Use embedding projection')
    parser.add_argument('--embed_mode', type=str, default='RanDumb',
                        choices=['RanDumb', 'RanPAC'], help='Choice of embedding')
    parser.add_argument('--embed_dim', type=int,
                        default=10000, help='Embedding dimension')
    # Default args
    parser.add_argument('--feature_path', type=str,
                        default='../', help='Path to features')
    parser.add_argument('--log_dir', type=str,
                        default='../logs/', help='Path to logs')
    args = parser.parse_args()
    return args


def get_logger(folder, name):
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s:%(name)s:%(message)s")

    # file logger
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fh = logging.FileHandler(os.path.join(
        folder, name+'_checkpoint.log'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

class DiagonalLinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_classes, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        return (self.weights * x.unsqueeze(1)).sum(dim=2) + self.bias


if __name__ == '__main__':
    args = parse_args()
    exp_name = f"{args.dataset}_{args.model}_{args.augment}_{args.embed}"
    if args.embed:
        exp_name += f"_{args.embed_mode}_{args.embed_dim}"

    console_logger = get_logger(folder=args.log_dir, name=exp_name)
    console_logger.debug(args)

    if args.augment:
        train_X = np.load(os.path.join(args.feature_path,
                          f"{args.dataset}_train_features_combined.npy"))
        train_y = np.load(os.path.join(args.feature_path,
                          f"{args.dataset}_train_labels_combined.npy"))
    else:
        train_X = np.load(os.path.join(args.feature_path,
                          f"{args.dataset}_train_features.npy"))
        train_y = np.load(os.path.join(args.feature_path,
                          f"{args.dataset}_train_labels.npy"))

    args.num_classes = len(np.unique(train_y))
    test_X = np.load(os.path.join(args.feature_path,
                     f"{args.dataset}_test_features.npy"))
    test_y = np.load(os.path.join(args.feature_path,
                     f"{args.dataset}_test_labels.npy"))

    train_y = train_y.astype(np.int32)
    test_y = test_y.astype(np.int32)
    # Insert your fancy class ordering here
    class_ordering = np.arange(args.num_classes)
    idx = []
    for i in range(args.num_classes):
        idx.append(np.where(train_y == class_ordering[i])[0])
    idx = np.concatenate(idx, axis=0)
    train_X = train_X[idx]
    train_y = train_y[idx]

    if args.embed_mode == 'RanPAC':
        W = np.random.randn(train_X.shape[1], args.embed_dim)
        train_X = np.maximum(0, np.matmul(train_X, W))
        test_X = np.maximum(0, np.matmul(test_X, W))
    elif args.embed_mode == 'RanDumb':
        embedder = RBFSampler(gamma='scale', n_components=args.embed_dim)
        # The scikit function ignores data passed to it, using on the input dimensions. We are not fitting anything here with data.
        embedder.fit(train_X)
        train_X = embedder.transform(train_X)
        test_X = embedder.transform(test_X)

    if args.model == 'SLDA':
        # Very sample-efficient shrinkage estimator
        oa = OAS(assume_centered=False)
        # Main difference between original paper code and here. Faster, easier to play but roughly equivalent to the online version: https://github.com/tyler-hayes/Deep_SLDA/blob/master/SLDA_Model.py with better-set shrinkage. Tested against original online code with hparam search for shrinkage, returns similar results (\pm 0.8)
        model = LinearDiscriminantAnalysis(
            solver='lsqr', covariance_estimator=oa)
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        # Some datasets are imbalanced, so we calculate per class accuracy to get the average incremental accuracy
        matrix = confusion_matrix(test_y, preds)
        acc_per_class = matrix.diagonal()/matrix.sum(axis=1)
        acc = np.mean(acc_per_class)

    elif args.model == 'NCM':
        model = NearestCentroid(metric='cosine')
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        matrix = confusion_matrix(test_y, preds)
        acc_per_class = matrix.diagonal()/matrix.sum(axis=1)
        acc = np.mean(acc_per_class)
    elif args.model == 'DLP':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DiagonalLinearProbe(train_X.shape[1], args.num_classes).to(device)

        # Convert data
        train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
        train_y_tensor = torch.tensor(train_y, dtype=torch.long)
        test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
        test_y_tensor = torch.tensor(test_y, dtype=torch.long).to(device)

        loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=32, shuffle=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Training loop
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = F.cross_entropy(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(test_X_tensor).argmax(dim=1).cpu().numpy()

        matrix = confusion_matrix(test_y, preds)
        acc_per_class = matrix.diagonal()/matrix.sum(axis=1)
        acc = np.mean(acc_per_class)
        
    logger_out = f"Test accuracy\t{acc}\tDataset\t{args.dataset}\tModel\t{args.model}\tAugment\t{args.augment}\tEmbed\t{args.embed}"
    if args.embed:
        logger_out += f"\tEmbed_mode\t{args.embed_mode}\tEmbed_dim\t{args.embed_dim}"
    console_logger.info(logger_out)
