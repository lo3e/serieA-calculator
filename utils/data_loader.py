from typing import Dict
import json

def load_historical_data(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)