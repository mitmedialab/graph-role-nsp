import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import *
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torchvision
import numpy as np
import torchvision.models as models
import pdb



def knn(x_train, y_train, x_test, k, device, log_interval=100, log=True):

    test_size = x_test.shape[0]
    num_train = y_train.shape[0]

    y_test = torch.zeros((test_size), device=device, dtype=torch.float)

    for test_index in range(0, test_size):
        test_input = x_test[test_index]
        distances = torch.norm(x_train - test_input, dim=1)

        indexes = torch.topk(distances, k, largest=False)[1]
        classes = torch.gather(y_train, 0, indexes)
        mode = int(torch.mode(classes)[0])

        y_test[test_index] = mode

    return y_test

class Classifier(nn.Module):
  def __init__(self, input_feats=198, out_feats = 4, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.fc1 = nn.Linear(input_feats, 128)
    self.bn1 = nn.BatchNorm1d(num_features=128)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(num_features=64)
    self.fc3 = nn.Linear(64, 32)
    self.bn3 = nn.BatchNorm1d(num_features=32)
    self.fc4 = nn.Linear(32, out_feats)
    self.bn4 = nn.BatchNorm1d(num_features=out_feats)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()


  def forward(self, context):
    out = self.bn1(self.relu(self.fc1(context)))
    out = self.bn2(self.relu(self.fc2(out)))
    out = self.bn3(self.relu(self.fc3(out)))
    out = torch.softmax(self.fc4(out), dim =1)

    return out


class ClassifierSmall(nn.Module):
  def __init__(self, num_classes, input_feats=198, out_feats = 4, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.fc1 = nn.Linear(input_feats, num_classes + 1 )
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()
    #dropout add - generalization 


  def forward(self, context, social_rel):
    out = self.relu(self.fc1(context))
    out = torch.softmax(out, dim = 1)

    return out