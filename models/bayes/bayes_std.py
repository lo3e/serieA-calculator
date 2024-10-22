import numpy as np
from scipy.stats import poisson
from datetime import datetime
import json

class SerieABayesModel:
    def __init__(self, current_teams):
        self.current_teams = set(current_teams)
        self.attack_strength = {}
        self.defense_strength = {}
        self.home_advantage = 1.2
        self.average_goals = 2.6

    def initialize_model(self, historical_data, decay_factor=0.5):
        """
        Inizializza le forze delle squadre usando dati storici con pesi basati sul tempo.
        
        :param historical_data: Una lista di dizionari con chiavi 'home_team', 'away_team', 'home_goals', 'away_goals', 'date'
        :param decay_factor: Un fattore tra 0 e 1 che controlla quanto velocemente diminuisce l'importanza dei dati più vecchi
        """
        total_weighted_goals = 0
        weighted_games = {}
        weighted_goals_scored = {}
        weighted_goals_conceded = {}
        all_teams = set()

        # Ordina i dati per data
        historical_data.sort(key=lambda x: x['date'])
        most_recent_date = datetime.strptime(historical_data[-1]['date'], '%Y-%m-%d')

        for game in historical_data:
            home_team = game['home_team']
            away_team = game['away_team']
            all_teams.add(home_team)
            all_teams.add(away_team)

            for team in [home_team, away_team]:
                if team not in weighted_games:
                    weighted_games[team] = 0
                    weighted_goals_scored[team] = 0
                    weighted_goals_conceded[team] = 0

            home_goals = game['home_goals']
            away_goals = game['away_goals']
            game_date = datetime.strptime(game['date'], '%Y-%m-%d')

            # Calcola il peso basato sulla data
            days_difference = (most_recent_date - game_date).days
            weight = np.exp(-decay_factor * days_difference / 365)  # Decadimento esponenziale

            total_weighted_goals += (home_goals + away_goals) * weight
            weighted_games[home_team] += weight
            weighted_games[away_team] += weight
            weighted_goals_scored[home_team] += home_goals * weight
            weighted_goals_scored[away_team] += away_goals * weight
            weighted_goals_conceded[home_team] += away_goals * weight
            weighted_goals_conceded[away_team] += home_goals * weight

        self.average_goals = total_weighted_goals / sum(weighted_games.values())

        for team in all_teams:
            if weighted_games[team] > 0:
                self.attack_strength[team] = (weighted_goals_scored[team] / weighted_games[team]) / self.average_goals
                self.defense_strength[team] = (weighted_goals_conceded[team] / weighted_games[team]) / self.average_goals

        # Normalizza le forze
        avg_attack = np.mean(list(self.attack_strength.values()))
        avg_defense = np.mean(list(self.defense_strength.values()))
        for team in all_teams:
            self.attack_strength[team] /= avg_attack
            self.defense_strength[team] /= avg_defense

        # Gestisci le squadre attuali che non sono presenti nei dati storici
        for team in self.current_teams:
            if team not in self.attack_strength:
                self.attack_strength[team] = 1.0
                self.defense_strength[team] = 1.0

        #print(f"Squadre nei dati storici: {all_teams}")
        #print(f"Squadre attuali: {self.current_teams}")

    def predict_match(self, home_team, away_team):
        if home_team not in self.attack_strength or away_team not in self.attack_strength:
            raise ValueError(f"Una o entrambe le squadre ({home_team}, {away_team}) non sono nel modello.")

        lambda_home = self.average_goals * self.attack_strength[home_team] * self.defense_strength[away_team] * self.home_advantage
        lambda_away = self.average_goals * self.attack_strength[away_team] * self.defense_strength[home_team]

        prob_home_win = 0
        prob_draw = 0
        prob_away_win = 0

        for i in range(10):  # Consideriamo fino a 9 goal per squadra
            for j in range(10):
                p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                if i > j:
                    prob_home_win += p
                elif i == j:
                    prob_draw += p
                else:
                    prob_away_win += p

        return {
            "home_win": prob_home_win,
            "draw": prob_draw,
            "away_win": prob_away_win,
            "team_strengths": {
                home_team: {
                    "attack": self.attack_strength[home_team],
                    "defense": self.defense_strength[home_team]
                },
                away_team: {
                    "attack": self.attack_strength[away_team],
                    "defense": self.defense_strength[away_team]
                }
            },
            "expected_goals": {
                home_team: lambda_home,
                away_team: lambda_away
            }
        }

    def simulate_season(self, num_simulations=1000):
        standings = {team: 0 for team in self.current_teams}

        for _ in range(num_simulations):
            season_points = {team: 0 for team in self.current_teams}
            for home_team in self.current_teams:
                for away_team in self.current_teams:
                    if home_team != away_team:
                        prediction = self.predict_match(home_team, away_team)
                        result = np.random.choice(["home_win", "draw", "away_win"], p=[prediction["home_win"], prediction["draw"], prediction["away_win"]])
                        if result == "home_win":
                            season_points[home_team] += 3
                        elif result == "draw":
                            season_points[home_team] += 1
                            season_points[away_team] += 1
                        else:
                            season_points[away_team] += 3
            
            winner = max(season_points, key=season_points.get)
            standings[winner] += 1

        return {team: wins / num_simulations for team, wins in standings.items()}