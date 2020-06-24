import torch
import torchvision
from torchvision import datasets, transforms

def get_svhn(batch_size):
    trsnform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971))])
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('/home/ubuntu/data', 'train', download=True,
                      transform=trsnform), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('/home/ubuntu/data', 'test', download=True,
                      transform=trsnform), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_cifar(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = torchvision.datasets.CIFAR10(root='/home/ubuntu/data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/home/ubuntu/data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def get_mnist(batch_size):
    trsnform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/ubuntu/data', train=True, download=True,
                       transform=trsnform), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/ubuntu/data', train=False, download=True,
                       transform=trsnform), batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

