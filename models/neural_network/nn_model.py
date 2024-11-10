import torch.nn as nn

class SerieANeuralNetworkModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        # Increased network capacity and added dropout
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 4, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)
