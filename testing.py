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

# Testa il modello
test_data = loading_historical_data
test_results = model.test_model(test_data)
print(f"Risultati del test: {test_results}")

