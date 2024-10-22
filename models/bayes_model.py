from typing import Dict, List, Tuple, Optional
from enum import Enum
from models.bayes.bayes_std import *
from models.bayes.bayes_draw import *

class BayesModelType(Enum):
    STANDARD = "standard"
    DRAW = "draw"

class SerieABayesModelWrapper:
    def __init__(self, teams: List[str], model_type: str = "standard"):
        self.teams = teams
        self.model_type = BayesModelType(model_type.lower())
        self.model = self._initialize_specific_model()

    def get_model_name(self) -> str:
        """
        Ritorna il nome del modello specifico in uso
        """
        return f"Bayes-{self.model_type.value}"
        
    def _initialize_specific_model(self):
        """
        Factory method per creare l'istanza del modello specifico
        """
        if self.model_type == BayesModelType.STANDARD:
            return SerieABayesModel(self.teams)
        elif self.model_type == BayesModelType.DRAW:
            return SerieABayesModelDraw(self.teams)
        else:
            raise ValueError(f"Tipo di modello non supportato: {self.model_type}")
            
    def initialize_model(self, historical_data: List[Dict], decay_factor: float = 0.5):
        """
        Inizializza il modello specifico con i dati storici
        """
        self.model.initialize_model(historical_data, decay_factor)
        
    def predict_match(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Utilizza il modello specifico per fare predizioni
        """
        return self.model.predict_match(home_team, away_team)
        
    @staticmethod
    def available_models() -> List[str]:
        """
        Ritorna la lista dei modelli bayesiani disponibili
        """
        return [model.value for model in BayesModelType]
