from abc import ABC, abstractmethod
from typing import List, Dict

class BaseModel(ABC):
    """Classe base astratta per i modelli di predizione"""
    
    @abstractmethod
    def initialize_model(self, historical_data: dict, **kwargs):
        pass
    
    @abstractmethod
    def predict_match(self, home_team: str, away_team: str) -> dict:
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass