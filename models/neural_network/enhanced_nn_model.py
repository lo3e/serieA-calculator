import torch.nn as nn

class SerieAEnhancedNeuralNetworkModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        # Base feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Separate heads for different predictions
        self.match_result_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, 3),
            nn.Softmax(dim=1)
        )
        
        self.double_chance_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, 3),  # 1X, X2, 12
            nn.Sigmoid()  # Use sigmoid since these are not mutually exclusive
        )
        
        self.goals_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, 2),  # GOAL/NO GOAL
            nn.Softmax(dim=1)
        )
        
        self.over_under_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, 2),  # UNDER/OVER 2.5
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return {
            'match_result': self.match_result_head(features),
            'double_chance': self.double_chance_head(features),
            'goals': self.goals_head(features),
            'over_under': self.over_under_head(features)
        }