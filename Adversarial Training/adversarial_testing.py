from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from torchvision.utils import save_image

if not os.path.exists('3b_results'):
    os.mkdir('3b_results')

batch_size = 100
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

# load the model from part a
model = CNN()
model.load_state_dict(torch.load('cnn_model.pth'))
loss_function = nn.CrossEntropyLoss()

def FGSM(x,y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = loss_function(model(x_adv), y)
    loss.backward()
    adv_img = x_adv + eps*x_adv.grad.data.sign()
    adv_img = torch.clamp(adv_img, 0, 1)
    return adv_img

def PGD(x, y, eps):
    delta = torch.zeros_like(x, requires_grad=True)
    num_iter = 5
    alpha = 0.2
    for t in range(num_iter):
        loss = loss_function(model(x + delta), y)
        loss.backward()
        gradient = delta + alpha*delta.grad.detach().sign()
        delta.data = gradient.clamp(-eps,eps)
        delta.grad.zero_()
    adv_img = x + delta
    adv_img = torch.clamp(adv_img,0,1)
    return adv_img

def test_adversarial(epsilon, method):
    model.eval()
    adv_imgs = []
    correct = 0
    labels = []
    for x,y in test_loader:
        x.requires_grad = True
        # generate adversarial image
        if method=="FGSM":
            adv_img = FGSM(x, y, epsilon)
        elif method=="PGD":
            adv_img = PGD(x, y, epsilon)
        adv_imgs.append(adv_img)
        # calculate accuracy 
        pred = model(adv_img)
        final_pred = pred.max(1, keepdim=True)[1].reshape(1,100)
        labels.append(final_pred)
        correct += (final_pred==y).sum().item()
    accuracy = correct/len(test_loader.dataset)
    return adv_imgs, accuracy, labels

def display_img(imgs, num, labels):
    for i in range(5):
        img = imgs[rand_i[i]]
        label = labels[rand_i[i]][0][rand_j[i]].item()
        save_image(img[rand_j[i]],
                   '3b_results/' + str(num) + str(i) + '_' + str(label)  + '.png')

rand_i = random.sample(range(100), 5)
rand_j = random.sample(range(100), 5)

# standard training and FGSM attack
print("------\nFGSM")
adv_imgs_1, accuracy, labels = test_adversarial(0.2, "FGSM")
print(accuracy)
display_img(adv_imgs_1, 1, labels)
adv_imgs_2, accuracy, labels = test_adversarial(0.1, "FGSM")
print(accuracy)
display_img(adv_imgs_2, 2, labels)
adv_imgs_3, accuracy, labels = test_adversarial(0.5, "FGSM")
print(accuracy)
display_img(adv_imgs_3, 3, labels)

# standard training and PGD attack (part d)
print("------\nPGD")
adv_imgs_1, accuracy, labels = test_adversarial(0.2, "PGD")
display_img(adv_imgs_1, 4, labels)
print(accuracy)
adv_imgs_2, accuracy, labels = test_adversarial(0.1, "PGD")
display_img(adv_imgs_2, 5, labels)
print(accuracy)
adv_imgs_3, accuracy, labels = test_adversarial(0.5, "PGD")
display_img(adv_imgs_3, 6, labels)
print(accuracy)