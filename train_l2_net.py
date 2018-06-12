import pandas as pd
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from l2net_dataset import PhotoTour
from torchvision.datasets import PhotoTour as DefaultDataTour
from evalMetrics import ErrorRateAt95Recall
import torchvision.transforms as tf

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x

# def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1):

#     dim = input.dim()
#     if dim < 3:
#         raise ValueError('Expected 3D or higher dimensionality \
#                          input (got {} dimensions)'.format(dim))
#     div = input.mul(input).unsqueeze(1)
#     if dim == 3:
#         div = F.pad(div, (0, 0, size // 2, (size - 1) // 2))
#         div = F.avg_pool2d(div, (size, 1), stride=1).squeeze(1)
#     else:
#         sizes = input.size()
#         div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
#         div = F.pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
#         div = F.avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
#         div = div.view(sizes)
#     div = div.mul(alpha).add(k).pow(beta)
#     return input / div

# class LRN(nn.Module):

#     def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
#         super(LRN, self).__init__()
#         self.size = size
#         self.alpha = alpha
#         self.beta = beta
#         self.k = k

#     def forward(self, input):
#         return local_response_norm(input, self.size, self.alpha, self.beta,
#                                      self.k)

class L2net(nn.Module):
    def __init__(self):
        super(L2net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, affine=False)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32, affine=False)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64, affine=False)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64, affine=False)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128, affine=False)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128, affine=False)
        self.conv7 = nn.Conv2d(128, 128, 8)
        self.bn7 = nn.BatchNorm2d(128, affine=False)
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return ((x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) /
                sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x))

    def forward(self, x):
        x = self.input_norm(x)
        #print('x0:', x)
        int1 = self.bn1(self.conv1(x))
        x = F.relu(int1)
        #print('x1:', x)
        x = F.relu(self.bn2(self.conv2(x)))
        #print('x2:', x)
        x = F.relu(self.bn3(self.conv3(x)))
        #print('x3:', x)
        x = F.relu(self.bn4(self.conv4(x)))
        #print('x4:', x)
        x = F.relu(self.bn5(self.conv5(x)))
        #print('x5:', x)
        x = F.relu(self.bn6(self.conv6(x)))
        #print('x6:', x)
        x = self.bn7(self.conv7(x))
        #print('x7:', x)
        int2 = x.view(x.size(0), -1)
        return L2Norm()(int2), int1, int2

    def dump(self, file_name):
        torch.save(self.state_dict(), file_name)

def adjust_learning_rate(optimizer, epoch, start_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] =  start_lr / (10 ** (epoch // 20))

def distance_matrix(Y1, Y2):
    eye = Variable(torch.eye(Y1.size()[0])).cuda()
    one = Variable(torch.ones(Y1.size()[0])).cuda()
    mul = Y1.matmul(Y2.t())
    return torch.sqrt(2*(one - mul))

def train_l2net(model, dataloader, optimizer, eval_dataloader=None, scorer=None, epochs=50):
    
    for epoch in range(1, epochs+1):

        adjust_learning_rate(optimizer, epoch, start_lr=0.01)
        for d1, d2 in tqdm(dataloader, total=len(dataloader)):

            optimizer.zero_grad()
            d1 = Variable(d1[0]).cuda()
            d2 = Variable(d2[0]).cuda()

            y1, y1_first, y1_last = model(d1)
            y2, y2_first, y2_last = model(d2)
            y1_first, y2_first = y1_first.view(64, -1), y2_first.view(64, -1)

            D = distance_matrix(y1, y2)
            R1 = y1_last.t().matmul(y1_last) / 128
            R2 = y2_last.t().matmul(y2_last) / 128
            G_first = y1_first.matmul(y2_first.t())
            G_last = y1_last.matmul(y2_last.t())

            E1 = -0.5 * (F.log_softmax(-D, dim=0).diag().sum() +
                         F.log_softmax(-D, dim=1).diag().sum())
            #print("E1:", E1[0])
            E2 = 0.5 * ((R1 ** 2).sum() - (R1 ** 2).diag().sum() +
                        (R2 ** 2).sum() - (R2 ** 2).diag().sum())
            #print("E2:", E2[0])
            E3 = -0.5 * (F.log_softmax(G_last, dim=0).diag().sum() +
                         F.log_softmax(G_last, dim=1).diag().sum() +
                         F.log_softmax(G_first, dim=0).diag().sum() +
                         F.log_softmax(G_first, dim=1).diag().sum())
            #print("E3:", E3[0])
            loss = E1 + E2 + E3
            loss.backward()
            optimizer.step()

        score = None
        if eval_dataloader is not None and scorer is not None:
            score = evaluate(model, eval_dataloader, scorer)
        if score is not None:
            print('Score after epoch {} is {}'.format(epoch, score))

def evaluate(model, dataloader, scorer):

    distances = []
    labels = []
    for d1, d2, m in tqdm(dataloader, total=len(dataloader)):

        d1 = Variable(d1, volatile=True).cuda()
        d2 = Variable(d2, volatile=True).cuda()
        y1, _, _ = model(d1)
        y2, _, _ = model(d2)
        dists = torch.sqrt(torch.sum((y1 - y2) ** 2, 1))
        distances.append(dists.data.cpu().view(-1, 1))
        labels.append(m.view(-1, 1))

    distances = torch.cat(distances, 0).numpy().reshape(-1)
    labels = torch.cat(labels, 0).numpy().reshape(-1)
    return scorer(labels, distances)

def rotate(image, degrees):
    return image.rotate(degrees[random.randint(0, len(degrees)-1)])

if __name__ == "__main__":

    train_dataset_names = ['liberty']#, 'liberty_harris',
                           #'yosemite', 'yosemite_harris']
    test_dataset_name = 'notredame'
    degrees = [0, 90, 180, 270]
    train_transform = tf.Compose([
        tf.Lambda(lambda x: x.unsqueeze(-1).numpy()),
        tf.ToPILImage(),
        tf.Resize(32),
        tf.Lambda(lambda x: rotate(x, degrees)),
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip(),
        tf.ToTensor()
    ])
    test_transform = tf.Compose([
        tf.Lambda(lambda x: x.unsqueeze(-1).numpy()),
        tf.ToPILImage(),
        tf.Resize(32),
        tf.ToTensor()
    ])

    train_datasets = [PhotoTour(root='./brown6',
                                name=name,
                                transform=train_transform,
                                train=False,
                                download=True) for name in train_dataset_names]
    test_dataset = DefaultDataTour(root='./brown6',
                                   name=test_dataset_name,
                                   transform=test_transform,
                                   train=False,
                                   download=True)
    dataloader = DataLoader(ConcatDataset(train_datasets))
    eval_dataloader = DataLoader(dataset=test_dataset, batch_size=128)
    
    model = L2net().cuda()
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    train_l2net(model, dataloader, optimizer, eval_dataloader, ErrorRateAt95Recall, epochs=50)
    model.dump("l2net_LY_N.pt")