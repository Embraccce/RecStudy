import torch
import torch.nn as nn


class LR(nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.b = nn.init.xavier_normal_(nn.Parameter(torch.empty(1,1)))
        self.w = nn.init.xavier_normal_(nn.Parameter(torch.empty(n_features,1)))
    def forward(self, x):        
        logits = torch.sigmoid(torch.matmal(x,self.w)+self.b)
        return logits