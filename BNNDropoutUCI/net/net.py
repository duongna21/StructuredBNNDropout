# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings

warnings.filterwarnings("ignore")

from scipy.special import logsumexp
import torch
torch.manual_seed(0)
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('./')
from model import *
import time

dev = 'cpu' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self, input_shape, num_flows=2, p=0.1):
        super(Net, self).__init__()
        self.num_flows = num_flows
        hidden_size = 50
        self.fc1 = Linear(input_shape, hidden_size, alpha=p/(1 - p),  num_flows=num_flows)
        self.fc2 = Linear(hidden_size, 1, alpha=p/(1 - p), num_flows=num_flows)

    def forward(self, x):
        x = F.relu(self.fc1(x.float().to(dev)))
        x = self.fc2(x)
        return x

    def fit(self, X_train, y_train_normalized, optimizer, learner, batch_size=128, n_epochs=400):
        # print(X_train.shape, y_train_normalized)
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train_normalized))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        kl_weight = 1
        for t in range(n_epochs):
            train_loss = 0.
            for i, (data, target) in enumerate(train_loader):
                # if i == 1: break
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                # print('train: ', data.shape, data[0], data.mean(), data.std())
                prediction = self(data.to(dev))
                loss, _ = learner(prediction, target, kl_weight)
                loss.backward()
                optimizer.step()

                train_loss += loss

            if t % 100 == 0:
                print(t, 'train_loss', train_loss / X_train.shape[0] * batch_size)
            del train_loss

    def predict(self, X_test):
        y_preds = self(X_test.cpu())
        return y_preds.cpu().detach().numpy()



class Learner(nn.Module):
    def __init__(self, net, num_samples):
        super(Learner, self).__init__()
        self.num_samples = num_samples
        self.net = net

    def forward(self, input, target, kl_weight=0.1):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        kl = kl / self.num_samples
        # print('kl: ', kl)
        mse = F.mse_loss(input, target.to(dev))
        # print(mse, kl_weight)
        elbo = - mse - kl
        return mse + kl_weight * kl, elbo


class net:

    def __init__(self, X_train, y_train, input_shape, n_epochs=400,
                 normalize=False, num_flows=2, tau=1.0, dropout=0.1):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[self.std_X_train == 0] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[1])
            self.mean_X_train = np.zeros(X_train.shape[1])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
                  np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin=2).T

        # We construct the network
        batch_size = 128

        model = Net(input_shape=input_shape, p=dropout, num_flows=num_flows).to(dev)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        learner = Learner(model, len(X_train))  # len(train_loader.dataset)


        # We iterate the learning process
        start_time = time.time()
        model.fit(X_train, y_train_normalized, optimizer=optimizer, learner=learner, batch_size=batch_size, n_epochs=n_epochs)

        self.model = model
        self.tau = tau
        self.running_time = time.time() - start_time

        # We are done!

    def predict(self, X_test, y_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data


            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        X_test = np.array(X_test, ndmin=2)
        y_test = torch.Tensor(np.array(y_test, ndmin=2).T)
        # print('\n\ntest: ', X_test.shape, X_test[0], X_test.mean(), X_test.std(), '\n\n')


        # We normalize the test set

        X_test = torch.Tensor((X_test - np.full(X_test.shape, self.mean_X_train)) / \
                 np.full(X_test.shape, self.std_X_train))
        # print('\n', self.mean_X_train, self.std_X_train, '\ntest: ', X_test.shape, X_test[0], X_test.mean(), X_test.std(), '\n\n')

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model
        standard_pred = model.predict(X_test)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        rmse_standard_pred = torch.mean( (y_test.squeeze() - torch.Tensor(standard_pred).squeeze() ) ** 2.) ** 0.5

        T = 10000
        Yt_hat = np.array([model.predict(X_test) for _ in range(T)])
        # print(Yt_hat)

        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        MC_pred = np.mean(Yt_hat, 0)
        rmse = torch.mean((y_test.squeeze() - torch.Tensor(MC_pred).squeeze()) ** 2.) ** 0.5

        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - torch.Tensor(Yt_hat)) ** 2., 0) - np.log(T)
              - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(self.tau))
        test_ll = np.mean(ll)

        # We are done!
        return rmse_standard_pred, rmse, test_ll

