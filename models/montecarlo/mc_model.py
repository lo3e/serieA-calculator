from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import poisson

class MonteCarloSerieA:
    def __init__(self, teams: List[str], n_simulations: int = 10000):
        self.teams = teams
        self.n_simulations = n_simulations
        self.team_attack = {team: 1.0 for team in teams}  # Attack strength
        self.team_defense = {team: 1.0 for team in teams}  # Defense strength
        self.home_advantage = 1.1  # Home team scoring boost
        
    def initialize_model(self, historical_data: List[Dict], decay_factor: float = 0.5):
        """
        Inizializza il modello usando i dati storici per stimare i parametri
        di attacco e difesa di ogni squadra
        """
        # Reset parameters
        goals_scored = {team: [] for team in self.teams}
        goals_conceded = {team: [] for team in self.teams}
        
        # Collect historical performance
        for match in historical_data:
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            
            if home_team in self.teams:
                goals_scored[home_team].append(home_goals)
                goals_conceded[home_team].append(away_goals)
            if away_team in self.teams:
                goals_scored[away_team].append(away_goals)
                goals_conceded[away_team].append(home_goals)
        
        # Calculate attack and defense strengths
        league_avg_goals = np.mean([np.mean(goals) for goals in goals_scored.values() if goals])
        
        for team in self.teams:
            if goals_scored[team]:
                self.team_attack[team] = np.mean(goals_scored[team]) / league_avg_goals
                self.team_defense[team] = np.mean(goals_conceded[team]) / league_avg_goals
    
    def simulate_match(self, home_team: str, away_team: str) -> Tuple[int, int]:
        """
        Simula una singola partita e ritorna i gol segnati
        """
        # Expected goals calculation
        home_expected = (self.team_attack[home_team] * self.team_defense[away_team] * 
                        self.home_advantage * 1.5)  # 1.5 è la media di gol per squadra in Serie A
        away_expected = (self.team_attack[away_team] * self.team_defense[home_team] * 
                        1.5)
        
        # Simulate actual goals using Poisson distribution
        home_goals = np.random.poisson(home_expected)
        away_goals = np.random.poisson(away_expected)
        
        return home_goals, away_goals
    
    def predict_match(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Predice il risultato di una partita usando simulazioni Monte Carlo
        """
        home_wins = 0
        away_wins = 0
        draws = 0
        
        # Run multiple simulations
        for _ in range(self.n_simulations):
            home_goals, away_goals = self.simulate_match(home_team, away_team)
            
            if home_goals > away_goals:
                home_wins += 1
            elif away_goals > home_goals:
                away_wins += 1
            else:
                draws += 1
        
        # Calculate probabilities
        return {
            'home_win': home_wins / self.n_simulations,
            'draw': draws / self.n_simulations,
            'away_win': away_wins / self.n_simulations
        }
    
    def simulate_season(self, fixture_list: List[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
        """
        Simula un'intera stagione e ritorna le statistiche per ogni squadra
        """
        stats = {team: {'points': 0, 'goals_for': 0, 'goals_against': 0} 
                for team in self.teams}
        
        for home_team, away_team in fixture_list:
            home_goals, away_goals = self.simulate_match(home_team, away_team)
            
            # Update statistics
            stats[home_team]['goals_for'] += home_goals
            stats[home_team]['goals_against'] += away_goals
            stats[away_team]['goals_for'] += away_goals
            stats[away_team]['goals_against'] += home_goals
            
            # Update points
            if home_goals > away_goals:
                stats[home_team]['points'] += 3
            elif away_goals > home_goals:
                stats[away_team]['points'] += 3
            else:
                stats[home_team]['points'] += 1
                stats[away_team]['points'] += 1
                
        return stats