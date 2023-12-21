from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

batch_size = 100
latent_size = 20
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn_model = CNN()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.01)

# train
cnn_model.train()
for epoch in range(5):
    for i, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = cnn_model(x)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()

# test
cnn_model.eval()
n_correct = 0
with torch.no_grad():
    for x, y in test_loader:
        outputs = cnn_model(x)
        _, pred = torch.max(outputs.data, 1)
        n_correct += (pred == y).sum().item()

acc = n_correct/len(test_loader.dataset)
print("test accuracy: {:.4f}".format(acc))

# save model
torch.save(cnn_model.state_dict(), '../a4_files/cnn_model.pth')