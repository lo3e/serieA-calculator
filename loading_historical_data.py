import json

def load_historical_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)