from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import nbinom, poisson
from dataclasses import dataclass

@dataclass
class MatchPrediction:
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float
    expected_away_goals: float

class AdvancedEnsembleModel:
    def __init__(self, teams: List[str], n_simulations: int = 10000):
        self.teams = teams
        self.n_simulations = n_simulations
        # Parametri per ogni modello nell'ensemble
        self.team_stats = {team: {
            'attack': 1.0,
            'defense': 1.0,
            'variance': 1.2,  # Inizializzato a un valore più alto per evitare problemi
            'home_boost': 1.1,
            'form_factor': 1.0
        } for team in teams}
        
    def _negative_binomial_parameters(self, mean: float, variance: float) -> Tuple[float, float]:
        """
        Converte media e varianza nei parametri della distribuzione negative binomial
        con controlli di validità
        """
        # Assicurati che media e varianza siano positivi
        mean = max(0.1, mean)
        # La varianza deve essere maggiore della media per la binomiale negativa
        variance = max(mean + 0.1, variance)
        
        try:
            # Calcola i parametri
            r = mean ** 2 / (variance - mean)
            p = mean / variance
            
            # Assicurati che i parametri siano validi
            r = max(0.1, r)  # r deve essere positivo
            p = max(0.01, min(0.99, p))  # p deve essere tra 0 e 1
            
            return r, p
        except (ValueError, ZeroDivisionError):
            # In caso di errore, ritorna valori predefiniti sicuri
            return 1.0, 0.5
    
    def _simulate_goals_poisson(self, expected_goals: float) -> int:
        """
        Simula i goal usando la distribuzione di Poisson con controllo dei parametri
        """
        expected_goals = max(0, expected_goals)  # Assicura che sia non negativo
        return np.random.poisson(expected_goals)
    
    def _simulate_goals_negative_binomial(self, expected_goals: float, variance: float) -> int:
        """
        Simula i goal usando la distribuzione Negative Binomial con gestione errori
        """
        try:
            r, p = self._negative_binomial_parameters(expected_goals, variance)
            return nbinom.rvs(n=r, p=p)
        except ValueError:
            # Se c'è un errore, usa Poisson come fallback
            return self._simulate_goals_poisson(expected_goals)
    
    def _simulate_correlated_goals(self, home_expected: float, away_expected: float, 
                                 correlation: float = 0.2) -> Tuple[int, int]:
        """
        Simula goal correlati tra le squadre con controlli di validità
        """
        # Assicura che i valori attesi siano non negativi
        home_expected = max(0, home_expected)
        away_expected = max(0, away_expected)
        # Limita la correlazione tra -1 e 1
        correlation = max(-0.99, min(0.99, correlation))
        
        try:
            # Genera variabili normali correlate
            mean = [home_expected, away_expected]
            cov = [[1, correlation], [correlation, 1]]
            goals_norm = np.random.multivariate_normal(mean, cov)
            
            # Converti in Poisson con controllo non negatività
            home_goals = self._simulate_goals_poisson(max(0, goals_norm[0]))
            away_goals = self._simulate_goals_poisson(max(0, goals_norm[1]))
            
            return home_goals, away_goals
        except (ValueError, np.linalg.LinAlgError):
            # In caso di errore, usa simulazioni indipendenti
            return (self._simulate_goals_poisson(home_expected),
                   self._simulate_goals_poisson(away_expected))
    
    def _update_team_stats(self, home_team: str, away_team: str, 
                          home_goals: int, away_goals: int, weight: float):
        """
        Aggiorna le statistiche delle squadre con controlli di validità
        """
        # Limita il peso tra 0 e 1
        weight = max(0, min(1, weight))
        
        # Media dei goal per partita (parametro di base)
        base_goals = 1.5
        
        # Aggiorna attacco con controlli
        for team, goals in [(home_team, home_goals), (away_team, away_goals)]:
            # Attacco
            new_attack = (goals / base_goals)
            self.team_stats[team]['attack'] = (
                self.team_stats[team]['attack'] * (1 - weight) +
                new_attack * weight
            )
            self.team_stats[team]['attack'] = max(0.1, self.team_stats[team]['attack'])
            
            # Varianza (con limite minimo per evitare problemi)
            new_variance = max(1.2, (goals - base_goals) ** 2)
            self.team_stats[team]['variance'] = (
                self.team_stats[team]['variance'] * (1 - weight) +
                new_variance * weight
            )
            self.team_stats[team]['variance'] = max(1.2, self.team_stats[team]['variance'])
        
        # Aggiorna difesa
        for team, goals_against in [(home_team, away_goals), (away_team, home_goals)]:
            new_defense = (goals_against / base_goals)
            self.team_stats[team]['defense'] = (
                self.team_stats[team]['defense'] * (1 - weight) +
                new_defense * weight
            )
            self.team_stats[team]['defense'] = max(0.1, self.team_stats[team]['defense'])
    
    def initialize_model(self, historical_data: List[Dict], decay_factor: float = 0.5):
        """
        Inizializza il modello usando diversi approcci per stimare i parametri
        """
        if not historical_data:
            return
            
        recent_weight = 1.0
        for match in reversed(historical_data):
            home_team = match['home_team']
            away_team = match['away_team']
            
            if home_team in self.teams and away_team in self.teams:
                try:
                    home_goals = int(match['home_goals'])
                    away_goals = int(match['away_goals'])
                    
                    self._update_team_stats(
                        home_team, away_team, 
                        home_goals, away_goals, 
                        weight=recent_weight
                    )
                    recent_weight *= max(0.1, min(1, decay_factor))
                except (ValueError, KeyError):
                    continue
    
    def predict_match(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Predice il risultato usando un ensemble di approcci
        """
        predictions = []
        
        for _ in range(self.n_simulations):
            try:
                # Calcola i valori attesi con controlli di validità
                home_expected = max(0, (
                    self.team_stats[home_team]['attack'] * 
                    self.team_stats[away_team]['defense'] * 
                    self.team_stats[home_team]['home_boost'] * 1.5
                ))
                away_expected = max(0, (
                    self.team_stats[away_team]['attack'] * 
                    self.team_stats[home_team]['defense'] * 1.5
                ))
                
                # Scegli il modello di simulazione
                rand_choice = np.random.random()
                
                if rand_choice < 0.4:
                    # Poisson standard
                    home_goals = self._simulate_goals_poisson(home_expected)
                    away_goals = self._simulate_goals_poisson(away_expected)
                elif rand_choice < 0.8:
                    # Negative Binomial
                    home_goals = self._simulate_goals_negative_binomial(
                        home_expected,
                        self.team_stats[home_team]['variance']
                    )
                    away_goals = self._simulate_goals_negative_binomial(
                        away_expected,
                        self.team_stats[away_team]['variance']
                    )
                else:
                    # Goals correlati
                    home_goals, away_goals = self._simulate_correlated_goals(
                        home_expected, away_expected
                    )
                
                # Determina il risultato
                if home_goals > away_goals:
                    result = 'home_win'
                elif away_goals > home_goals:
                    result = 'away_win'
                else:
                    result = 'draw'
                
                predictions.append(result)
                
            except Exception:
                # In caso di errore durante la simulazione, usa Poisson standard
                home_goals = self._simulate_goals_poisson(1.5)
                away_goals = self._simulate_goals_poisson(1.5)
                predictions.append('draw' if home_goals == away_goals else
                                'home_win' if home_goals > away_goals else
                                'away_win')
        
        # Calcola probabilità finali
        total_sims = len(predictions)
        results = {
            'home_win': predictions.count('home_win') / total_sims,
            'draw': predictions.count('draw') / total_sims,
            'away_win': predictions.count('away_win') / total_sims
        }
        
        return results