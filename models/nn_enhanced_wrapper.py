from typing import List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from models.neural_network.enhanced_nn_model import SerieAEnhancedNeuralNetworkModel
from models.neural_network.enhanced_nn_dataset import SerieAEnhancedDataset

class SerieAEnhancedNeuralNetworkWrapper:
    def __init__(self, teams: List[str], input_size: int = 10, hidden_size: int = 256,
                 epochs: int = 300, batch_size: int = 32, learning_rate: float = 0.0005):
        self.model = SerieAEnhancedNeuralNetworkModel(input_size, hidden_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.teams = teams
        self.team_stats = None

    def get_model_name(self) -> str:
        return "Neural Network"

    def initialize_model(self, historical_data: List[Dict]):
        dataset = SerieAEnhancedDataset(historical_data, self.teams)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6)
        
        # Different loss functions for different tasks
        criterion = {
            'match_result': nn.BCELoss(),
            'double_chance': nn.BCELoss(),
            'goals': nn.BCELoss(),
            'over_under': nn.BCELoss()
        }

        best_val_loss = float('inf')
        patience = 25
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            for features, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(features)
                
                # Calculate loss for each prediction type
                loss = sum(criterion[key](outputs[key], targets[key]) for key in outputs.keys())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for features, targets in val_loader:
                    outputs = self.model(features)
                    val_loss += sum(criterion[key](outputs[key], targets[key]).item() 
                                  for key in outputs.keys())

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}")
                print(f"Train Loss: {avg_train_loss:.4f}")
                print(f"Val Loss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered dopo {epoch + 1} epoche")
                    checkpoint = torch.load('best_model.pth')
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    break

        self.team_stats = dataset._calculate_team_stats(historical_data, self.teams)

    def predict_match(self, home_team: str, away_team: str) -> Dict[str, Any]:
        self.model.eval()
        
        if not self.team_stats:
            raise ValueError("Model not initialized. Call initialize_model first.")

        features = [
            np.mean(self.team_stats[home_team]['goals_scored']),
            np.mean(self.team_stats[away_team]['goals_scored']),
            np.mean(self.team_stats[home_team]['goals_conceded']),
            np.mean(self.team_stats[away_team]['goals_conceded']),
            np.mean(self.team_stats[home_team]['home_goals']),
            np.mean(self.team_stats[away_team]['away_goals']),
            self.team_stats[home_team]['wins'] / max(1, sum([self.team_stats[home_team]['wins'], 
                                                           self.team_stats[home_team]['draws'], 
                                                           self.team_stats[home_team]['losses']])),
            self.team_stats[away_team]['wins'] / max(1, sum([self.team_stats[away_team]['wins'], 
                                                           self.team_stats[away_team]['draws'], 
                                                           self.team_stats[away_team]['losses']])),
            np.mean(self.team_stats[home_team]['form']),
            np.mean(self.team_stats[away_team]['form'])
        ]
        
        features_tensor = torch.tensor([features], dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
        
        return {
            'match_result': {
                'home_win': outputs['match_result'][0][0].item(),
                'draw': outputs['match_result'][0][1].item(),
                'away_win': outputs['match_result'][0][2].item()
            },
            'double_chance': {
                '1X': outputs['double_chance'][0][0].item(),
                'X2': outputs['double_chance'][0][1].item(),
                '12': outputs['double_chance'][0][2].item()
            },
            'goals': {
                'goal': outputs['goals'][0][0].item(),
                'no_goal': outputs['goals'][0][1].item()
            },
            'over_under': {
                'under': outputs['over_under'][0][0].item(),
                'over': outputs['over_under'][0][1].item()
            },
            'team_strengths': {
                home_team: {
                    'attack': np.mean(self.team_stats[home_team]['goals_scored']),
                    'defense': 1/max(0.1, np.mean(self.team_stats[home_team]['goals_conceded'])),
                    'form': np.mean(self.team_stats[home_team]['form'])
                },
                away_team: {
                    'attack': np.mean(self.team_stats[away_team]['goals_scored']),
                    'defense': 1/max(0.1, np.mean(self.team_stats[away_team]['goals_conceded'])),
                    'form': np.mean(self.team_stats[away_team]['form'])
                }
            }
        }