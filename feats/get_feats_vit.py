import torch
import timm
import torchvision
import numpy as np
import os  # ← Add this
from torchvision import transforms, datasets
from transformers import ViTFeatureExtractor, ViTModel


def get_dataset(dataset, batchsize, transforms_list=None):
    assert (dataset in ['CIFAR100', 'imagenet-r',
            'imagenet-a', 'cub', 'omnibenchmark', 'vtab', 'cars'])
    print('==> Loading datasets..')
    if dataset == 'CIFAR100':
        transforms_list = transforms.Compose([transforms.Resize(
            224, interpolation=3, antialias=True), transforms.ToTensor()])
    else:
        transforms_list = transforms.Compose([transforms.Resize(
            256, interpolation=3, antialias=True), transforms.CenterCrop(224), transforms.ToTensor()])

    if dataset in ['CIFAR10', 'CIFAR100']:
        dset = getattr(torchvision.datasets, dataset)
        kwargs_train = {'train': True, 'download': True}
        kwargs_test = {'train': False, 'download': True}
        train_data = dset(
            '../data/', transform=transforms_list, **kwargs_train)
        test_data = dset('../data/', transform=transforms_list, **kwargs_test)
    elif dataset in ['imagenet-r', 'imagenet-a', 'cub', 'omnibenchmark', 'vtab', 'cars']:
        train_path = f'../data/{dataset}/train/'
        test_path = f'../data/{dataset}/test/'
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(
                f"[Warning] Skipping dataset {dataset} because paths don't exist.")
            return None, None, 0, 0
        train_data = datasets.ImageFolder(
            '../data/'+dataset+'/train/', transform=transforms_list)
        test_data = datasets.ImageFolder(
            '../data/'+dataset+'/test/', transform=transforms_list)

    trainlen, testlen = len(train_data), len(test_data)
    print(trainlen, testlen)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return trainloader, testloader, trainlen, testlen


def run_extraction(dataset_name, model, batchsize, featsize, tag):
    trainloader, testloader, trainlen, testlen = get_dataset(
        dataset_name, batchsize)
    if trainloader:
        extract_feats(model, trainloader, batchsize, featsize,
                      trainlen, f'{tag}_{dataset_name}_train')
        extract_feats(model, testloader, batchsize, featsize,
                      testlen, f'{tag}_{dataset_name}_test')
    else:
        print(f"[Skipped] {dataset_name} not found.")


def extract_feats(model, loader, batchsize, featsize, num_samples, expname):
    print('==> Extracting features..')

    model.cuda()
    model.eval()

    labelarr, featarr, flipped_featarr = np.zeros(num_samples, dtype='u2'), np.zeros(
        (num_samples, featsize), dtype=np.float32), np.zeros((num_samples, featsize), dtype=np.float32)

    with torch.inference_mode():
        for count, (image, label) in enumerate(loader):
            idx = (np.ones(batchsize)*count*batchsize +
                   np.arange(batchsize)).astype(int)
            idx = idx[:label.shape[0]]
            image = image.cuda(non_blocking=True)
            # Flip image tensor horizontally

            feat = model(image)
            labelarr[idx] = label.numpy()
            featarr[idx] = feat.cpu().numpy()

            image_flipped = image.flip(dims=[3])
            feat_flipped = model(image_flipped)
            flipped_featarr[idx] = feat_flipped.cpu().numpy()
        np.save('./'+expname+'_features_flipped.npy', flipped_featarr)
        np.save('./'+expname+'_features.npy', featarr)
        np.save('./'+expname+'_labels.npy', labelarr)
    return


if __name__ == '__main__':
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    model = model.cuda()

    model.eval()
    featsize = 768
    batchsize = 512

    model = timm.create_model('vit_base_patch16_224',
                              pretrained=True, num_classes=0)

    for ds in ['CIFAR100', 'imagenet-r', 'imagenet-a', 'cub', 'omnibenchmark', 'vtab', 'cars']:
        run_extraction(ds, model, batchsize, featsize, 'vitbi1k')

    # ADD YOUR OWN DATASETS HERE

    model = timm.create_model(
        "vit_base_patch16_224_in21k", pretrained=True, num_classes=0)

    for ds in ['CIFAR100', 'imagenet-r', 'imagenet-a', 'cub', 'omnibenchmark', 'vtab', 'cars']:
        run_extraction(ds, model, batchsize, featsize, 'vitbi21k')

    # ADD YOUR OWN DATASETS HERE
