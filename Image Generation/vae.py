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
import numpy as np

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.layer1 = nn.Linear(784, 400)
        self.hidden_layer1 = nn.Linear(400, latent_size)
        self.hidden_layer2 = nn.Linear(400, latent_size)
        self.layer2 = nn.Linear(latent_size, 400)
        self.layer3 = nn.Linear(400, 784)

    def encode(self, x):
        #The encoder will take an input of size 784, and will produce two vectors of size latent_size (corresponding to the coordinatewise means and log_variances)
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and have two linear output layers (no activations)
        v = torch.relu(self.layer1(x))
        means = self.hidden_layer1(v)
        log_vars = self.hidden_layer2(v)
        return means, log_vars

    def reparameterize(self, means, log_variances):
        #The reparameterization module lies between the encoder and the decoder
        #It takes in the coordinatewise means and log-variances from the encoder (each of dimension latent_size), and returns a sample from a Gaussian with the corresponding parameters
        stds = torch.exp(log_variances*0.5)
        c = torch.randn_like(stds)
        z = means + c*stds
        return z

    def decode(self, z):
        #The decoder will take an input of size latent_size, and will produce an output of size 784
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
        v = torch.relu(self.layer2(z))
        output = torch.sigmoid(self.layer3(v))
        return output

    def forward(self, x):
        #Apply the VAE encoder, reparameterization, and decoder to an input of size 784
        #Returns an output image of size 784, as well as the means and log_variances, each of size latent_size (they will be needed when computing the loss)
        means, log_vars = self.encode(x)
        z = self.reparameterize(means, log_vars)
        output = self.decode(z)
        return output, means, log_vars

def vae_loss_function(reconstructed_x, x, means, log_variances):
    #Compute the VAE loss
    #The loss is a sum of two terms: reconstruction error and KL divergence
    #Use cross entropy loss between x and reconstructed_x for the reconstruction error (as opposed to L2 loss as discussed in lecture -- this is sometimes done for data in [0,1] for easier optimization)
    #The KL divergence is -1/2 * sum(1 + log_variances - means^2 - exp(log_variances)) as described in lecture
    #Returns loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed_x, x, reduction="sum")
    kl_div = torch.sum(1+log_variances-means.pow(2)-torch.exp(log_variances))*(-0.5)
    loss = reconstruction_loss + kl_div
    return loss, reconstruction_loss

def train(model, optimizer):
    #Trains the VAE for one epoch on the training dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    train_loss = 0
    train_reconstruction_loss = 0
    for x, y in train_loader:
        x = x.view(batch_size, 784)
        optimizer.zero_grad()
        output, means, log_vars = model(x)
        loss, reconstruction_loss = vae_loss_function(output, x, means, log_vars)
        train_loss += loss.item()
        train_reconstruction_loss += reconstruction_loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss/len(train_loader.dataset)
    avg_train_reconstruction_loss = train_reconstruction_loss/len(train_loader.dataset)
    return avg_train_loss, avg_train_reconstruction_loss

def test(model):
    #Runs the VAE on the test dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    test_loss = 0
    test_reconstruction_loss = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.view(batch_size, 784)
            output, means, log_vars = model(x)
            loss, reconstruction_loss = vae_loss_function(output, x, means, log_vars)
            test_loss += loss.item()
            test_reconstruction_loss += reconstruction_loss.item()
    avg_test_loss = test_loss/len(test_loader.dataset)
    avg_test_reconstruction_loss = test_reconstruction_loss/len(test_loader.dataset)
    return avg_test_loss, avg_test_reconstruction_loss

epochs = 50
avg_train_losses = []
avg_train_reconstruction_losses = []
avg_test_losses = []
avg_test_reconstruction_losses = []

vae_model = VAE().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    vae_model.train()
    avg_train_loss, avg_train_reconstruction_loss = train(vae_model, vae_optimizer)
    vae_model.eval()
    avg_test_loss, avg_test_reconstruction_loss = test(vae_model)
    avg_train_losses.append(avg_train_loss)
    avg_train_reconstruction_losses.append(avg_train_reconstruction_loss)
    avg_test_losses.append(avg_test_loss)
    avg_test_reconstruction_losses.append(avg_test_reconstruction_loss)
    print("average train loss:", avg_train_loss)
    print("average test loss:", avg_test_loss)
    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(avg_train_reconstruction_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()

plt.plot(avg_test_reconstruction_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()
