from matplotlib import pyplot as plt
import numpy as np
from random import randint
from scipy import io
import torch
from torchvision import transforms

# dataset
class HandDataset(torch.utils.data.Dataset):

    def __init__(self, file_name):
        data = io.loadmat(file_name=str(file_name))
        # TODO add support for validation set (different key values)
        if 'trainSet' in data:
            self.features = np.float32(data['trainSet'])
            self.labels = np.int64(data['trainLabel'].squeeze()) - 1
        elif 'valSet' in data:
            self.features = np.float32(data['valSet'])
            self.labels = np.int64(data['valLabel'].squeeze()) - 1
        elif 'testSet' in data:
            self.features = np.float32(data['testSet'])
            self.labels = np.int64(data['testLabel'].squeeze()) - 1
        self.classes = ('hi', 'fist', 'ok')
        self.transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transforms(self.features[:,:,:,index]), torch.from_numpy(self.labels)[index]

    def __repr__(self):
        return f"{len(self.labels)} samples of shape {self.sample_shape}"

    def __str__(self):
        return f"{len(self.labels)} samples of shape {self.sample_shape}"

    def show_sample(self, index=None):
        index = randint(0, len(self.labels)) if index is None else index
        print(f"\nSample: {index}\nLabel: {self.classes[self.labels[index]]}")
        titles = ['R','G','B','D']
        plt.figure(figsize=(10,40))
        for i in range(4):
            plt.subplot(1,4,i+1)
            plt.imshow(self.features[:,:,i,index])
            plt.title(titles[i])
            plt.axis('off')
        plt.show()

    @property
    def sample_shape(self):
        return self.features.shape[:-1]


# models
class SoftmaxRegression(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs):
        super(SoftmaxRegression, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_inputs, n_outputs)
        )

    def forward(self, x):
        """Return logits."""
        return self.layer(x)


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=6, kernel_size=5, padding=2), torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 5 * 5, 120), torch.nn.Sigmoid(),
            torch.nn.Linear(120, 84), torch.nn.Sigmoid(),
            torch.nn.Linear(84, 3)
        )

    def __repr__(self):
        X = torch.randn(size=(1, 4, 28, 28), dtype=torch.float32)
        for layer in self.layers:
            X = layer(X)
            print(layer.__class__.__name__,'output shape: \t', X.shape )
        return '\n'

    def forward(self, x):
        # Expected input is of shape (4, 28, 28)
        return self.layers(x)


# training
class Accumulator:
    """Class to accumulate sums over multiple variables."""

    def __init__(self, vars=2):
        super().__init__()
        self.data = [0.] * vars

    def add(self, *args):
        self.data = [a + float(b) for (a, b) in zip(self.data, args)]

    def reset(self):
        self.data = [0.] * len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, valid_iter, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    valid_metric = Accumulator(2)  # num. samples; correct predictions.
    # validation
    net.eval()

    for X, y in valid_iter:  # cycle on mini-batches

        # move data to GPU if available
        X, y = X.to(device), y.to(device)

        # forward
        y_hat = net(X)
        
        # accumulate batch statistics
        valid_metric.add(
            y.size().numel(),
            accuracy(y_hat, y)
        )

    return valid_metric[1] / valid_metric[0]


def train(net, n_epochs, train_iter, loss_function, optimizer, valid_iter=None):
    
    # Select best available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(f"Training on device: {device}")

    train_metric = Accumulator(3)  # num. samples; running loss; correct predictions.

    for epoch in range(n_epochs):  # cycle on epochs

        # training
        net.train()
        train_metric.reset()

        for batch, (X, y) in enumerate(train_iter):  # cycle on mini-batches

            # move data to GPU if available
            X, y = X.to(device), y.to(device)

            # reset the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_hat = net(X)
            loss = loss_function(y_hat, y)
            loss.backward()
            optimizer.step()

            # accumulate batch statistics
            train_metric.add(
                y.size().numel(),
                loss.item(),
                accuracy(y_hat, y)
            )

        # print epoch statistics (training)
        print(f"epoch:{epoch:5}\tloss:{train_metric[1]/train_metric[0]:.3}\taccuracy:{train_metric[2]/train_metric[0]:.3}", end='')

        if valid_iter is not None:

            valid_accuracy = evaluate_accuracy(net, valid_iter, device)

            # print epoch statistics (validation)
            print(f"\tval-accuracy:{valid_accuracy:.3}")

        else:
            print("")




if __name__ == '__main__':

    # Test accumulator class
    metrics = Accumulator(2)
    metrics.add(3,6)
    assert metrics[0] / metrics[1] == 0.5
    metrics.reset()
    assert metrics[0] == metrics[1] == 0
