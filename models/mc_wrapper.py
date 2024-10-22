from models.montecarlo.mc_model import *

class SerieAMonteCarloWrapper:
    def __init__(self, teams: List[str], n_simulations: int = 10000):
        self.model = MonteCarloSerieA(teams, n_simulations)
    
    def get_model_name(self) -> str:
        return "MonteCarlo"
    
    def initialize_model(self, historical_data: List[Dict], decay_factor: float = 0.5):
        self.model.initialize_model(historical_data, decay_factor)
    
    def predict_match(self, home_team: str, away_team: str) -> Dict[str, float]:
        return self.model.predict_match(home_team, away_team)