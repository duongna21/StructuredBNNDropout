import sys
sys.path.append('./')
from model import *
from BNNDropoutCV.data import *
import argparse
import numpy as np
import torch.optim as optim

parser = argparse.ArgumentParser()

parser.add_argument('--dset', '-d', default='cifar')
parser.add_argument('--type', '-t', default='cnn')
parser.add_argument('--p', type=float, default=0.5)
parser.add_argument('--num_flows', '-n', type=int, default=2)
parser.add_argument('--bs', '-b', type=int, default=32)
parser.add_argument('--klw', '-k', type=float, default=0.1)
parser.add_argument('--MC', '-k', type=int, default=10)
parser.add_argument('--load', '-l', type=bool, default=False)

args = parser.parse_args()

dataset = args.dset
network_type = args.type
drop_prob = args.p
num_flows = args.num_flows
bs = args.bs
klw = args.klw
MC = args.MC
load = args.load

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

if dataset == 'cifar':
    model = NetCNN(drop_prob=drop_prob, num_flows=num_flows).to(dev)
    if load:
        model = torch.load('/home/ubuntu/cifar.pth')

if dataset == 'svhn':
    model = NetCNN(drop_prob=drop_prob, num_flows=num_flows).to(dev)
    if load:
        model = torch.load('/home/ubuntu/svhn.pth')

if dataset == 'mnist':
    model = NetFC(drop_prob=drop_prob, num_flows=num_flows).to(dev)
    if load:
        model = torch.load('/home/ubuntu/mnist.pth')

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50, 60, 70, 80], gamma=0.3)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.05, cycle_momentum=False)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=10)

if dataset == 'mnist':
    train_loader, test_loader = get_mnist(batch_size=bs)

if dataset == 'svhn':
    train_loader, test_loader = get_svhn(batch_size=bs)

if dataset == 'cifar':
    train_loader, test_loader = get_cifar(batch_size=bs)


learner = Learner(model, len(train_loader), len(train_loader.dataset))
kl_weight = klw
epochs = 100

for epoch in range(1, epochs + 1):
    train_loss, train_elbo, train_acc = 0, 0, 0
    kl_weight = min(kl_weight, 1)
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        torch.cuda.empty_cache()

        if (batch_idx % 1000 == 0): print(batch_idx)
        data, target = data.to(dev), target.to(dev)
        if dataset == 'mnist':
            data = data.view(-1, 28*28)

        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1)[1]
        loss, elbo = learner(output, target, kl_weight)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_elbo += elbo.item()
        train_acc += np.sum(pred.cpu().numpy() == target.cpu().data.numpy())

        del loss, elbo

    scheduler.step()

    print(epoch, 'train_acc', train_acc / len(train_loader.dataset) * 100)

    if dataset == 'cifar':
        torch.save('/home/ubuntu/cifar.pth')
    if dataset == 'svhn':
        torch.save('/home/ubuntu/svhn.pth')
    if dataset == 'mnist':
        torch.save('/home/ubuntu/mnist.pth')

    model.eval()
    test_loss, test_elbo, test_acc = 0, 0, 0
    num_test_samples = MC

    for batch_idx, (data, target) in enumerate(test_loader):
        outputs = torch.zeros(num_test_samples, len(data), 10).to(dev)
        torch.cuda.empty_cache()
        data, target = data.to(dev), target.to(dev)
        if dataset == 'mnist':
            data = data.view(-1, 28*28)
        with torch.no_grad():
            for i in range(num_test_samples):
                outputs[i] = model(data)
            output = outputs.mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1].squeeze()
            loss, elbo = learner(output, target, kl_weight)

        test_loss += loss.item()
        test_elbo += elbo.item()
        # pred = output.data.max(1)[1]
        test_acc += np.sum(pred.cpu().numpy() == target.cpu().data.numpy())
        del loss, elbo

    print(epoch, 'test_acc', test_acc / len(test_loader.dataset) * 100, '\n')



