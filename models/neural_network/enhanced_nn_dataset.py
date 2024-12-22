import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Tuple

class SerieAEnhancedDataset(Dataset):
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

            # Features rimangono gli stessi del tuo codice originale
            features = [
                np.mean(team_stats[home_team]['goals_scored']) if team_stats[home_team]['goals_scored'] else 0,
                np.mean(team_stats[away_team]['goals_scored']) if team_stats[away_team]['goals_scored'] else 0,
                np.mean(team_stats[home_team]['goals_conceded']) if team_stats[home_team]['goals_conceded'] else 0,
                np.mean(team_stats[away_team]['goals_conceded']) if team_stats[away_team]['goals_conceded'] else 0,
                np.mean(team_stats[home_team]['home_goals']) if team_stats[home_team]['home_goals'] else 0,
                np.mean(team_stats[away_team]['away_goals']) if team_stats[away_team]['away_goals'] else 0,
                team_stats[home_team]['wins'] / max(1, sum([team_stats[home_team]['wins'], 
                                                          team_stats[home_team]['draws'], 
                                                          team_stats[home_team]['losses']])),
                team_stats[away_team]['wins'] / max(1, sum([team_stats[away_team]['wins'], 
                                                          team_stats[away_team]['draws'], 
                                                          team_stats[away_team]['losses']])),
                np.mean(team_stats[home_team]['form']) if team_stats[home_team]['form'] else 0,
                np.mean(team_stats[away_team]['form']) if team_stats[away_team]['form'] else 0,
            ]

            total_goals = match['home_goals'] + match['away_goals']
            
            targets = {
                # 1-X-2
                'match_result': [1, 0, 0] if match['home_goals'] > match['away_goals'] else 
                               [0, 1, 0] if match['home_goals'] == match['away_goals'] else 
                               [0, 0, 1],
                
                # Doppia chance (1X, X2, 12)
                'double_chance': [
                    1 if match['home_goals'] >= match['away_goals'] else 0,  # 1X
                    1 if match['home_goals'] <= match['away_goals'] else 0,  # X2
                    1 if match['home_goals'] != match['away_goals'] else 0   # 12
                ],
                
                # Goal/No Goal
                'goals': [1, 0] if match['home_goals'] > 0 and match['away_goals'] > 0 else [0, 1],
                
                # Over/Under 2.5
                'over_under': [0, 1] if total_goals > 2.5 else [1, 0]
            }

            processed.append((features, targets))

        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        features, targets = self.processed_data[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()}
        )