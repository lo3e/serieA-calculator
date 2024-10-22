from typing import List, Tuple, Dict
import argparse
import os
from models.base_model import BaseModel
from models.bayes_model import SerieABayesModelWrapper
from models.mc_wrapper import SerieAMonteCarloWrapper
from models.ensemble_wrapper import SerieAEnsembleWrapper
from utils.data_loader import load_historical_data
from config.teams import CURRENT_TEAMS

class PredictionSystem:
    """Sistema principale per gestire le predizioni"""
    
    def __init__(self):
        self.available_models = {}

    def register_model(self, model_key: str, model: BaseModel):
        """Registra un nuovo modello nel sistema"""
        self.available_models[model_key] = model
        
    def get_available_models(self) -> List[str]:
        """Restituisce la lista dei modelli disponibili"""
        return list(self.available_models.keys())
    
    def predict_matches(self, model_key: str, matches: List[Tuple[str, str]], 
                       print_details: bool = True) -> List[Dict]:
        """Effettua predizioni per una lista di partite usando il modello specificato"""
        if model_key not in self.available_models:
            raise ValueError(f"Modello '{model_key}' non trovato")
            
        model = self.available_models[model_key]
        predictions = []

        print(f"\n{'='*50}")
        print(f"Predizioni usando {model.get_model_name()}")
        print(f"{'='*50}")
        
        
        for match in matches:
            home_team, away_team = match
            prediction = model.predict_match(home_team, away_team)
            predictions.append(prediction)
            
            if print_details:
                self._print_prediction(home_team, away_team, prediction, model)
                
        return predictions
    
    def _print_prediction(self, home_team: str, away_team: str, 
                         prediction: dict, model: BaseModel):
        """Stampa i dettagli della predizione"""
        print(f"\n=== {model.get_model_name()} ===")
        print(f"Previsione {home_team} vs {away_team}:")
        print(f"Vittoria {home_team}: {prediction['home_win']:.2%}")
        print(f"Pareggio: {prediction['draw']:.2%}")
        print(f"Vittoria {away_team}: {prediction['away_win']:.2%}")
        
        if 'team_strengths' in prediction:
            print("\nForze delle squadre:")
            for team, strengths in prediction['team_strengths'].items():
                print(f"{team}:")
                for key, value in strengths.items():
                    print(f"  {key}: {value:.2f}")
        
        if 'expected_goals' in prediction:
            print("\nGol attesi:")
            for team, expected_goals in prediction['expected_goals'].items():
                print(f"{team}: {expected_goals:.2f}")

def get_argument_parser():
    parser = argparse.ArgumentParser(description='Sistema di predizione Serie A')
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['bayes', 'montecarlo', 'ensemble', 'all'],
        default='bayes',
        help='Modello da utilizzare per le predizioni (default: bayes)'
    )
    parser.add_argument(
        '--bayes-type',
        type=str,
        choices=['standard', 'draw'],
        default='standard',
        help='Tipo di modello bayesiano da utilizzare (default: standard)'
    )
    parser.add_argument(
        '--simulations',
        type=int,
        default=10000,
        help='Numero di simulazioni per il modello Monte Carlo (default: 10000)'
    )

    parser.add_argument(
        '--matches-file',
        type=str,
        default=None,
        help='File CSV contenente le partite da predire (opzionale)'
    )
    return parser

def load_matches_from_file(file_path: str) -> List[Tuple[str, str]]:
    matches = []
    with open(file_path, 'r') as f:
        for line in f:
            home, away = line.strip().split(',')
            matches.append((home.strip(), away.strip()))
    return matches

def main():

    # Parse degli argomenti
    parser = get_argument_parser()
    args = parser.parse_args()

    # Inizializzazione del sistema
    prediction_system = PredictionSystem()
    
    # Definizione delle partite da predire
    if args.matches_file:
        try:
            matches = load_matches_from_file(args.matches_file)
        except Exception as e:
            print(f"Errore nel caricamento del file partite: {str(e)}")
            return
    else:
        # Partite di default se non viene specificato un file
        matches = [
            ("Udinese", "Cagliari"),
            ("Torino", "Como"),
            ("Napoli", "Lecce"),
            ("Bologna", "Milan"),
            ("Atalanta", "Verona"),
            ("Parma", "Empoli"),
            ("Monza", "Venezia"),
            ("Lazio", "Genoa"),
            ("Inter", "Juventus"),
            ("Fiorentina", "Roma"),
        ]
    
    try:
        # Caricamento dati storici
        print("Caricamento dati storici...")
        # Ottiene il percorso corrente del file main.py e costruisce il percorso completo al file JSON
        current_directory = os.path.dirname(__file__)
        file_path = os.path.join(current_directory, 'input_data', 'historical_data.json')

        historical_data = load_historical_data(file_path)
        
        # Inizializzazione e registrazione modelli
        # Registra solo i modelli richiesti
        if args.model in ['bayes', 'all']:
            bayes_model = SerieABayesModelWrapper(
                CURRENT_TEAMS,
                model_type=args.bayes_type
                )
            bayes_model.initialize_model(historical_data, decay_factor=0.5)
            prediction_system.register_model('bayes', bayes_model)

        if args.model in ['montecarlo', 'all']:  # Aggiunta gestione modello Monte Carlo
            monte_carlo_model = SerieAMonteCarloWrapper(
                CURRENT_TEAMS,
                n_simulations=args.simulations
            )
            monte_carlo_model.initialize_model(historical_data, decay_factor=0.5)
            prediction_system.register_model('montecarlo', monte_carlo_model)
        if args.model in ['ensemble', 'all']:
            ensemble_model = SerieAEnsembleWrapper(CURRENT_TEAMS)
            ensemble_model.initialize_model(historical_data, decay_factor=0.5)
            prediction_system.register_model('ensemble', ensemble_model)

        # Stampa modelli disponibili
        print("\nModelli disponibili:", prediction_system.get_available_models())
        
        # Esegui predizioni
        if args.model == 'all':
            # Usa tutti i modelli registrati
            for model_key in prediction_system.get_available_models():
                predictions = prediction_system.predict_matches(model_key, matches)
        else:
            # Usa solo il modello specificato
            predictions = prediction_system.predict_matches(args.model, matches)
            
    except FileNotFoundError:
        print("Errore: File historical_data.json non trovato!")
    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")

if __name__ == "__main__":
    main()