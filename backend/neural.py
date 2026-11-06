from torch import nn
class DefectClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(12,24),nn.ReLU(),nn.Linear(24,12),nn.ReLU(),nn.Linear(12,2))
    def forward(self,x):return self.net(x)
