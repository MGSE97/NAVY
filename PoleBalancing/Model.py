import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_size, device):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.device = device

        self.fc_1 = nn.Linear(num_inputs, num_size)
        self.fc_2 = nn.Linear(num_size, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input):
        x = torch.relu(self.fc_1(input))
        x = torch.softmax(self.fc_2(x), dim=-1)
        return x

    def train_model(self, memory, optimizer):
        observations, actions = memory

        observations = torch.stack(observations).to(self.device)
        actions = torch.stack(actions).to(self.device)

        results = self(observations)
        results = results.view(-1, self.num_outputs)

        loss = F.mse_loss(actions, results, reduction="sum")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        result = self.forward(input)
        result = result[0].data.cpu().numpy()

        action = np.random.choice(self.num_outputs, 1, p=result)[0]
        return action
