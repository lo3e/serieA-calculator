from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from models.neural_network.nn_model import SerieANeuralNetworkModel
from models.neural_network.nn_dataset import SerieANeuralNetworkDataset

class SerieANeuralNetworkWrapper:
    def __init__(self, teams: List[str], input_size: int = 10, hidden_size: int = 256, 
                 epochs: int = 300, batch_size: int = 32, learning_rate: float = 0.0005):
        self.model = SerieANeuralNetworkModel(input_size, hidden_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.teams = teams
        self.team_stats = None

    def get_model_name(self) -> str:
        return "Neural Network"

    def initialize_model(self, historical_data: List[Dict]):
        # Prepare dataset
        dataset = SerieANeuralNetworkDataset(historical_data, self.teams)
        
        # Split data into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training setup con ottimizzazioni
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Scheduler modificato per essere più paziente
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=10,  # Aumentato da 5 a 10
            factor=0.5,   # Riduzione più graduale del learning rate
            min_lr=1e-6   # Learning rate minimo
        )
        
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        patience = 25  # Aumentato da 10 a 25
        patience_counter = 0
        best_accuracy = 0.0

        # Training loop con early stopping migliorato
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for features, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                # Gradient clipping per stabilità
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for features, targets in val_loader:
                    outputs = self.model(features)
                    val_loss += criterion(outputs, targets).item()
                    
                    predicted = torch.argmax(outputs, dim=1)
                    actual = torch.argmax(targets, dim=1)
                    correct_predictions += (predicted == actual).sum().item()
                    total_predictions += targets.size(0)

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct_predictions / total_predictions

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}")
                print(f"Train Loss: {avg_train_loss:.4f}")
                print(f"Val Loss: {avg_val_loss:.4f}")
                print(f"Val Accuracy: {accuracy:.4f}")
                print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Early stopping migliorato
            if avg_val_loss < best_val_loss:
                if accuracy > best_accuracy:  # Salva solo se migliora sia loss che accuracy
                    best_accuracy = accuracy
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'accuracy': accuracy
                    }, 'best_model.pth')
                    print(f"Nuovo miglior modello salvato! Accuracy: {accuracy:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered dopo {epoch + 1} epoche")
                    # Carica il miglior modello
                    checkpoint = torch.load('best_model.pth')
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    break

        # Save final team stats
        self.team_stats = dataset._calculate_team_stats(historical_data, self.teams)

    def predict_match(self, home_team: str, away_team: str) -> Dict[str, float]:
        self.model.eval()
        
        if not self.team_stats:
            raise ValueError("Model not initialized. Call initialize_model first.")

        # Prepare features
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
            output = self.model(features_tensor)
        
        probs = output.squeeze().tolist()
        
        return {
            'home_win': probs[0],
            'draw': probs[1],
            'away_win': probs[2],
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