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
        self.draw_correction_factor = 0.1  # Nuovo parametro per la correzione dei pareggi

    def initialize_with_historical_data(self, historical_data, decay_factor=0.5):
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

        # Calcolo della similarità delle forze delle squadre
        team_strength_similarity = 1 - abs(
            (self.attack_strength[home_team] * self.defense_strength[away_team]) -
            (self.attack_strength[away_team] * self.defense_strength[home_team])
        ) / ((self.attack_strength[home_team] * self.defense_strength[away_team]) +
             (self.attack_strength[away_team] * self.defense_strength[home_team]))

        # Applicazione della correzione per i pareggi
        draw_correction = self.draw_correction_factor * team_strength_similarity
        prob_draw += draw_correction
        prob_home_win *= (1 - draw_correction / 2)
        prob_away_win *= (1 - draw_correction / 2)

        # Normalizzazione delle probabilità
        total_prob = prob_home_win + prob_draw + prob_away_win
        prob_home_win /= total_prob
        prob_draw /= total_prob
        prob_away_win /= total_prob

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
            },
            "team_strength_similarity": team_strength_similarity
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

def load_historical_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Esempio di utilizzo
current_teams = ["Atalanta", "Bologna", "Cagliari", "Como", "Empoli", "Fiorentina", 
    "Genoa", "Inter", "Juventus", "Lazio", "Lecce", "Milan", "Monza", "Napoli", "Parma", "Roma", 
    "Torino", "Udinese", "Venezia", "Verona"]
model = SerieABayesModel(current_teams)

# Dati storici di esempio (incluse squadre non più in Serie A)
historical_data = load_historical_data('historical_data.json')

model.initialize_with_historical_data(historical_data, decay_factor=0.5)

# Ora puoi usare il modello come prima
matches = [
    ("Como", "Parma"),
    ("Genoa", "Bologna"),
    ("Milan", "Udinese"),
    ("Juventus", "Lazio"),
    ("Empoli", "Napoli"),
    ("Lecce", "Fiorentina"),
    ("Venezia", "Atalanta"),
    ("Cagliari", "Torino"),
    ("Roma", "Inter"),
    ("Verona", "Monza"),
]

# Ciclo per fare la predizione per ogni partita
for match in matches:
    home_team, away_team = match
    prediction = model.predict_match(home_team, away_team)
    
    print(f"\nPrevisione {home_team} vs {away_team}:")
    print(f"Vittoria {home_team}: {prediction['home_win']:.2%}")
    print(f"Pareggio: {prediction['draw']:.2%}")
    print(f"Vittoria {away_team}: {prediction['away_win']:.2%}")
    
    print("\nForze delle squadre:")
    for team, strengths in prediction['team_strengths'].items():
        print(f"{team}:")
        print(f"  Attacco: {strengths['attack']:.2f}")
        print(f"  Difesa: {strengths['defense']:.2f}")
    
    print("\nGol attesi:")
    for team, expected_goals in prediction['expected_goals'].items():
        print(f"{team}: {expected_goals:.2f}")

    print(f"\nSimilarità delle forze delle squadre: {prediction['team_strength_similarity']:.2%}")