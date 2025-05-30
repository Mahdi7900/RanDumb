import numpy as np
import torchvision
import glob
from PIL import Image
from datasets import load_dataset


def get_combined_X(a, b):
    c = np.empty((a.shape[0] + b.shape[0], a.shape[1],
                 a.shape[2], a.shape[3]), dtype=a.dtype)
    c[0::2, :, :, :] = a
    c[1::2, :, :, :] = b
    return c


def get_combined_y(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c


def process_data(X, y, mean, var, color=True, flip=True):
    if color:
        X = np.transpose(X, (0, 3, 1, 2))

    idx = []
    for i in range(len(np.unique(y))):
        idx.append(np.where(y == i)[0])
    idx = np.concatenate(idx, axis=0)

    X = X[idx]
    y = y[idx]

    X = X.astype(np.float32)/255
    X = (X - mean)/var

    if flip:
        X_flip = X[:, :, :, ::-1]
        X_full = get_combined_X(X, X_flip)
        y_full = get_combined_y(y, y)
        X = X.reshape(X.shape[0], -1)
        X_full = X_full.reshape(X_full.shape[0], -1)
        return X, y, X_full, y_full

    X = X.reshape(X.shape[0], -1)
    return X, y


mean_list = {
    'MNIST': (0.1307,),
    'CIFAR10': (0.4914, 0.4822, 0.4465),
    'CIFAR100': (0.5071, 0.4867, 0.4408),
    'TinyImagenet': (0.4802, 0.4481, 0.3975),
    'miniImagenet': (0.485, 0.456, 0.406),
}

var_list = {
    'MNIST': (0.3081,),
    'CIFAR10': (0.2470, 0.2435, 0.2616),
    'CIFAR100': (0.2675, 0.2565, 0.2761),
    'TinyImagenet': (0.2302, 0.2265, 0.2262),
    'miniImagenet': (0.229, 0.224, 0.225),
}


if __name__ == '__main__':
    d = torchvision.datasets.CIFAR100(
        root='../data', train=True, download=True)
    train_X = d.data
    train_y = np.array(d.targets)
    mean, var = np.array(mean_list['CIFAR100'])[np.newaxis, :, np.newaxis, np.newaxis], np.array(
        var_list['CIFAR100'])[np.newaxis, :, np.newaxis, np.newaxis]
    train_X, train_y, train_X_combined, train_y_combined = process_data(
        train_X, train_y, mean, var, color=True, flip=True)
    np.save(f"cifar100_train_features.npy", train_X)
    np.save(f"cifar100_train_labels.npy", train_y)
    np.save(f"cifar100_train_features_combined.npy", train_X_combined)
    np.save(f"cifar100_train_labels_combined.npy", train_y_combined)

    d = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True)
    test_X = d.data
    test_y = np.array(d.targets)
    test_X, test_y = process_data(
        test_X, test_y, mean, var, color=True, flip=False)
    np.save(f"cifar100_test_features.npy", test_X)
    np.save(f"cifar100_test_labels.npy", test_y)

    d = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
    train_X = d.data
    train_y = np.array(d.targets)
    mean, var = np.array(mean_list['CIFAR10'])[np.newaxis, :, np.newaxis, np.newaxis], np.array(
        var_list['CIFAR10'])[np.newaxis, :, np.newaxis, np.newaxis]
    train_X, train_y, train_X_combined, train_y_combined = process_data(
        train_X, train_y, mean, var, color=True, flip=True)
    np.save(f"cifar10_train_features.npy", train_X)
    np.save(f"cifar10_train_labels.npy", train_y)
    np.save(f"cifar10_train_features_combined.npy", train_X_combined)
    np.save(f"cifar10_train_labels_combined.npy", train_y_combined)

    d = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True)
    test_X = d.data
    test_y = np.array(d.targets)
    test_X, test_y = process_data(
        test_X, test_y, mean, var, color=True, flip=False)
    np.save(f"cifar10_test_features.npy", test_X)
    np.save(f"cifar10_test_labels.npy", test_y)

    d = torchvision.datasets.MNIST(root='../data', train=True, download=True)
    train_X = d.data.numpy()
    train_y = np.array(d.targets)
    mean, var = np.array(mean_list['MNIST'])[np.newaxis, :, np.newaxis], np.array(
        var_list['MNIST'])[np.newaxis, :, np.newaxis]
    train_X, train_y = process_data(
        train_X, train_y, mean, var, color=False, flip=False)
    np.save(f"mnist_train_features.npy", train_X)
    np.save(f"mnist_train_labels.npy", train_y)

    d = torchvision.datasets.MNIST(root='../data', train=False, download=True)
    test_X = d.data.numpy()
    test_y = np.array(d.targets)
    test_X, test_y = process_data(
        test_X, test_y, mean, var, color=False, flip=False)
    np.save(f"mnist_test_features.npy", test_X)
    np.save(f"mnist_test_labels.npy", test_y)
