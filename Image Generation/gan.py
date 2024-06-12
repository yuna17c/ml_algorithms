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


class Generator(nn.Module):
    #The generator takes an input of size latent_size, and will produce an output of size 784.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(latent_size, 400)
        self.layer2 = nn.Linear(400, 784)

    def forward(self, z):
        hidden_layer = torch.relu(self.layer1(z))
        output = torch.sigmoid(self.layer2(hidden_layer))
        return output

class Discriminator(nn.Module):
    #The discriminator takes an input of size 784, and will produce an output of size 1.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its output
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(784, 400)
        self.layer2 = nn.Linear(400,1)

    def forward(self, x):
        hidden_layer = torch.relu(self.layer1(x))
        output = torch.sigmoid(self.layer2(hidden_layer))
        return output

def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    #Trains both the generator and discriminator for one epoch on the training dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    generator.train()
    discriminator.train()
    gen_loss, dscr_loss = 0, 0
    loss_function = nn.BCELoss()
    for data, _ in train_loader:
        data = data.view(batch_size, 784)
        true_labels = torch.ones(batch_size, 1)
        false_labels = torch.zeros(batch_size, 1)
        
        # discriminator
        discriminator_optimizer.zero_grad()
        # real data
        dscr_output = discriminator(data)
        loss1 = loss_function(dscr_output, true_labels)
        # fake data
        rand_x = torch.randn(batch_size, latent_size)
        fake_data = generator(rand_x)
        fake_output = discriminator(fake_data)
        loss2 = loss_function(fake_output, false_labels)
        # total discriminator loss
        loss = (loss1 + loss2)
        dscr_loss += (loss1.item()+loss2.item())
        loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        # generator
        generator_optimizer.zero_grad()
        gen_output = discriminator(fake_data)
        loss3 = loss_function(gen_output, true_labels)
        gen_loss += loss3.item()
        loss3.backward()
        generator_optimizer.step()

    avg_generator_loss = gen_loss/len(train_loader)
    avg_discriminator_loss = dscr_loss/len(train_loader)
    return avg_generator_loss, avg_discriminator_loss

def test(generator, discriminator):
    #Runs both the generator and discriminator over the test dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    generator.eval()
    discriminator.eval()
    gen_loss, dscr_loss = 0, 0
    loss_function = nn.BCELoss()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(batch_size, 784)
            true_labels = torch.ones(batch_size, 1)
            false_labels = torch.zeros(batch_size, 1)
            
            # test discriminator
            dscr_output = discriminator(data)
            loss1 = loss_function(dscr_output, true_labels)
            rand_x = torch.randn(batch_size, latent_size)
            fake_data = generator(rand_x)
            fake_output = discriminator(fake_data)
            loss2 = loss_function(fake_output, false_labels)
            # total discriminator loss
            dscr_loss += (loss1.item()+loss2.item())

            # test generator
            gen_output = discriminator(fake_data)
            loss3 = loss_function(gen_output, true_labels)
            gen_loss += loss3.item()
            
    avg_generator_loss = gen_loss/len(test_loader)
    avg_discriminator_loss = dscr_loss/len(test_loader)
    return avg_generator_loss, avg_discriminator_loss

epochs = 50

discriminator_avg_train_losses = []
discriminator_avg_test_losses = []
generator_avg_train_losses = []
generator_avg_test_losses = []

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator, discriminator_optimizer)
    generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)
    discriminator_avg_train_losses.append(discriminator_avg_train_loss)
    generator_avg_train_losses.append(generator_avg_train_loss)
    discriminator_avg_test_losses.append(discriminator_avg_test_loss)
    generator_avg_test_losses.append(generator_avg_test_loss)

    print(discriminator_avg_train_loss, generator_avg_train_loss, discriminator_avg_test_loss, generator_avg_test_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = generator(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(discriminator_avg_train_losses)
plt.plot(generator_avg_train_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()

plt.plot(discriminator_avg_test_losses)
plt.plot(generator_avg_test_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()
