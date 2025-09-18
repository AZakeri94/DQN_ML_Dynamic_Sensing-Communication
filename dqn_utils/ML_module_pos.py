import torch
import torch.nn as nn

# You can keep "node" as a global variable, or pass it into the class if you prefer
node = 1024  

class NN_beam_pred(nn.Module):
    def __init__(self, num_features, num_output=33):
        super(NN_beam_pred, self).__init__()
        self.layer_1 = nn.Linear(num_features, node)
        self.layer_2 = nn.Linear(node, node)
        self.layer_3 = nn.Linear(node, node)
        self.layer_out = nn.Linear(node, num_output)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return x
