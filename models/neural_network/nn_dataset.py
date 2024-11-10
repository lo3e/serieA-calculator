import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Tuple

class SerieANeuralNetworkDataset(Dataset):
    def __init__(self, historical_data: List[Dict], teams: List[str]):
        self.processed_data = self._process_data(historical_data, teams)
    
    def _calculate_team_stats(self, historical_data: List[Dict], teams: List[str]) -> Dict:
        """Calculate comprehensive team statistics"""
        team_stats = {team: {
            'goals_scored': [],
            'goals_conceded': [],
            'home_goals': [],
            'away_goals': [],
            'home_matches': 0,
            'away_matches': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'form': []  # ultimi 5 risultati
        } for team in teams}

        # Calcola le statistiche per ogni squadra
        for match in historical_data:
            home_team = match['home_team']
            away_team = match['away_team']
            
            if home_team in teams and away_team in teams:
                # Update goals
                team_stats[home_team]['goals_scored'].append(match['home_goals'])
                team_stats[home_team]['goals_conceded'].append(match['away_goals'])
                team_stats[home_team]['home_goals'].append(match['home_goals'])
                team_stats[home_team]['home_matches'] += 1

                team_stats[away_team]['goals_scored'].append(match['away_goals'])
                team_stats[away_team]['goals_conceded'].append(match['home_goals'])
                team_stats[away_team]['away_goals'].append(match['away_goals'])
                team_stats[away_team]['away_matches'] += 1

                # Update form (1: win, 0: draw, -1: loss)
                if match['home_goals'] > match['away_goals']:
                    team_stats[home_team]['wins'] += 1
                    team_stats[away_team]['losses'] += 1
                    team_stats[home_team]['form'].append(1)
                    team_stats[away_team]['form'].append(-1)
                elif match['home_goals'] < match['away_goals']:
                    team_stats[home_team]['losses'] += 1
                    team_stats[away_team]['wins'] += 1
                    team_stats[home_team]['form'].append(-1)
                    team_stats[away_team]['form'].append(1)
                else:
                    team_stats[home_team]['draws'] += 1
                    team_stats[away_team]['draws'] += 1
                    team_stats[home_team]['form'].append(0)
                    team_stats[away_team]['form'].append(0)

                # Keep only last 5 matches for form
                team_stats[home_team]['form'] = team_stats[home_team]['form'][-5:]
                team_stats[away_team]['form'] = team_stats[away_team]['form'][-5:]

        return team_stats

    def _process_data(self, historical_data: List[Dict], teams: List[str]) -> List[Tuple]:
        team_stats = self._calculate_team_stats(historical_data, teams)
        processed = []

        for match in historical_data:
            home_team = match['home_team']
            away_team = match['away_team']

            if home_team not in teams or away_team not in teams:
                continue

            # Create feature vector
            features = [
                # Goal scoring abilities
                np.mean(team_stats[home_team]['goals_scored']) if team_stats[home_team]['goals_scored'] else 0,
                np.mean(team_stats[away_team]['goals_scored']) if team_stats[away_team]['goals_scored'] else 0,
                
                # Defensive abilities
                np.mean(team_stats[home_team]['goals_conceded']) if team_stats[home_team]['goals_conceded'] else 0,
                np.mean(team_stats[away_team]['goals_conceded']) if team_stats[away_team]['goals_conceded'] else 0,
                
                # Home/Away performance
                np.mean(team_stats[home_team]['home_goals']) if team_stats[home_team]['home_goals'] else 0,
                np.mean(team_stats[away_team]['away_goals']) if team_stats[away_team]['away_goals'] else 0,
                
                # Win rates
                team_stats[home_team]['wins'] / max(1, sum([team_stats[home_team]['wins'], 
                                                          team_stats[home_team]['draws'], 
                                                          team_stats[home_team]['losses']])),
                team_stats[away_team]['wins'] / max(1, sum([team_stats[away_team]['wins'], 
                                                          team_stats[away_team]['draws'], 
                                                          team_stats[away_team]['losses']])),
                
                # Recent form (average of last 5 matches)
                np.mean(team_stats[home_team]['form']) if team_stats[home_team]['form'] else 0,
                np.mean(team_stats[away_team]['form']) if team_stats[away_team]['form'] else 0,
            ]

            # Create target (one-hot encoded)
            if match['home_goals'] > match['away_goals']:
                target = [1, 0, 0]  # home win
            elif match['home_goals'] < match['away_goals']:
                target = [0, 0, 1]  # away win
            else:
                target = [0, 1, 0]  # draw

            processed.append((features, target))

        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        features, target = self.processed_data[idx]
        return (torch.tensor(features, dtype=torch.float32), 
                torch.tensor(target, dtype=torch.float32))