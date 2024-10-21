from serie_a_bayes import SerieABayesModel
from  loading_historical_data import load_historical_data

# Squadre nell'attuale campionato
current_teams = ["Atalanta", "Bologna", "Cagliari", "Como", "Empoli", "Fiorentina", 
    "Genoa", "Inter", "Juventus", "Lazio", "Lecce", "Milan", "Monza", "Napoli", "Parma", "Roma", 
    "Torino", "Udinese", "Venezia", "Verona"]

# Caricamento del modello e inizializzazione della froza delel squadre
model = SerieABayesModel(current_teams)
loading_historical_data = load_historical_data('historical_data.json')
model.initialize_with_historical_data(loading_historical_data, decay_factor=0.5)

# Giornata da calcolare
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