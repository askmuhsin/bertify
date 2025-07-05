import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model_class, path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model