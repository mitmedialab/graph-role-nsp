import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import pdb
import model_role_graph


class Decoder_NoRole(nn.Module):
    def __init__(self,num_class,time_step=3,state_dim=66):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = state_dim
        self.time_step = time_step
        self.grrn=model_role_graph.Graph_NoRole(state_dim, num_class, time_step)

        
        self.fc1 = nn.Linear(num_class**2, 6)
        self.bn1 = nn.BatchNorm1d(num_features=6)
        self.fc2 = nn.Linear(6, num_class+1)
        self.relu = nn.ReLU()
    

    def forward(self, feats, social_rel):

        #[batcn, node_num, 2048]
        
        all_scores = self.grrn(feats, social_rel)
        #pdb.set_trace()
        out = all_scores[-1].flatten(start_dim = 1)
        out = self.bn1(self.relu(self.fc1(out)))
        out = torch.softmax(self.fc2(out), dim = 1)

        return out

class Decoder_Role(Decoder_NoRole):
    def __init__(self,num_class,time_step=3,state_dim=66):
        super().__init__(num_class,time_step=3,state_dim=66)
        self.num_class = num_class
        self.hidden_dim = state_dim
        self.time_step = time_step
        self.grrn=model_role_graph.Graph_Role(state_dim, num_class, time_step)

        
        self.fc1 = nn.Linear(num_class**2, 6)
        self.bn1 = nn.BatchNorm1d(num_features=6)
        self.fc2 = nn.Linear(6, num_class+1)
        self.relu = nn.ReLU()

    def forward(self, feats, social_rel):

        #[batcn, node_num, 2048]
        
        all_scores = self.grrn(feats, social_rel)
        out = all_scores[-1].flatten(start_dim = 1)
        out = self.bn1(self.relu(self.fc1(out)))
        out = torch.softmax(self.fc2(out), dim = 1)

        return out
