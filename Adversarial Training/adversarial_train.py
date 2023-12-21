from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
import os
import random
from torchvision.utils import save_image

if not os.path.exists('3c_results'):
    os.mkdir('3c_results')

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

# load the model
model = CNN()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def PGD(x, y, eps):
    delta = torch.zeros_like(x, requires_grad=True)
    num_iter = 2
    step_size = 0.5
    for t in range(num_iter):
        loss = loss_function(model(x + delta), y)
        loss.backward()
        delta.data = (delta + step_size*delta.grad.detach().sign()).clamp(-eps,eps)
        delta.grad.zero_()
    adv_img = x + delta
    adv_img = torch.clamp(adv_img,0,1)
    return adv_img

def FGSM(x,y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = loss_function(model(x_adv), y)
    loss.backward()
    adv_img = x_adv + eps*x_adv.grad.data.sign()
    adv_img = torch.clamp(adv_img, 0, 1)
    return adv_img

# train with FGSM
model.train()
train_eps = 0.2   
for epoch in range(5):
    for i, (x,y) in enumerate(train_loader):
        x.requires_grad = True
        optimizer.zero_grad()
        # train with adversarial images 
        adv_img = FGSM(x, y, train_eps)
        adv_output = model(adv_img)
        adv_loss = loss_function(adv_output, y)
        adv_loss.backward()
        optimizer.step()
    print("epoch: "+str(epoch+1)+"/5")

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
                   '3c_results/' + str(num) + str(i) + '_' + str(label)  + '.png')

def test_no_attack():
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            _, pred = torch.max(outputs.data, 1)
            n_correct += (pred == y).sum().item()

    acc = n_correct/len(test_loader.dataset)
    return acc

rand_i = random.sample(range(100), 5)
rand_j = random.sample(range(100), 5)

# FGSM training and FGSM attack
print("------\nFGSM")
adv_imgs_1, accuracy, labels = test_adversarial(0.2, "FGSM")
display_img(adv_imgs_1, 1, labels)
print(accuracy)
adv_imgs_2, accuracy, labels = test_adversarial(0.1, "FGSM")
display_img(adv_imgs_2, 2, labels)
print(accuracy)
adv_imgs_3, accuracy, labels = test_adversarial(0.5, "FGSM")
display_img(adv_imgs_3, 3, labels)
print(accuracy)

# FGSM training and PGD attack (part d)
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

# standard test (part d)
print("------\nNo Attack")
accuracy = test_no_attack()
print(accuracy)
