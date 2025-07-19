import os
import argparse
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.kernel_approximation import RBFSampler
from NsCE import NsCEClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100'], help='Dataset')
    parser.add_argument('--model', type=str, default='SLDA',
                    choices=['SLDA', 'NCM', 'NsCE'], help='Model')
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
        model = NearestCentroid(metric='manhattan')
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        matrix = confusion_matrix(test_y, preds)
        acc_per_class = matrix.diagonal()/matrix.sum(axis=1)
        acc = np.mean(acc_per_class)
    elif args.model == 'NsCE':
        threshold = 100
        model = NsCEClassifier(threshold=threshold)
        model.partial_fit(train_X, train_y)

        # Predict on test set
        preds = model.predict(test_X)

        # Build a proper mask of only the known predictions
        mask = np.array([p is not None for p in preds])
        valid_true = test_y[mask]
        # Extract only the integer labels (cast from object to int)
        valid_pred = np.array([p for p in preds[mask]], dtype=int)
        # Now it's safe to compute confusion_matrix
        matrix = confusion_matrix(valid_true, valid_pred)
        acc_per_class = matrix.diagonal() / matrix.sum(axis=1)
        acc = np.mean(acc_per_class)
        
    logger_out = f"Test accuracy\t{acc}\tDataset\t{args.dataset}\tModel\t{args.model}\tAugment\t{args.augment}\tEmbed\t{args.embed}"
    if args.embed:
        logger_out += f"\tEmbed_mode\t{args.embed_mode}\tEmbed_dim\t{args.embed_dim}"
    console_logger.info(logger_out)
