import pandas as pd
import json

# Carica il file Excel (sostituisci 'file.xlsx' con il tuo file)
df = pd.read_excel('Completo.xlsx', sheet_name='storico')

# Sostituisce 'Hellas Verona' con 'Verona' nelle colonne home_team e away_team
df['home_team'] = df['home_team'].replace('Hellas Verona', 'Verona')
df['away_team'] = df['away_team'].replace('Hellas Verona', 'Verona')

# Converte home_goals e away_goals in interi
df['home_goals'] = df['home_goals'].astype(int)
df['away_goals'] = df['away_goals'].astype(int)

# Converte la colonna date in stringa nel formato YYYY-MM-DD
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# Filtra le partite a partire dal mese di agosto 2024
df_filtered = df[df['date'] >= '2024-08-01']

# Crea la lista di dizionari con il risultato e le quote
odds = []
for _, row in df_filtered.iterrows():
    if row['home_goals'] > row['away_goals']:
        result = 'home_win'
    elif row['home_goals'] < row['away_goals']:
        result = 'away_win'
    else:
        result = 'draw'

    # Aggiungi un dizionario con home_team, away_team, risultato e quote fittizie
    match_data = {
        "home_team": row['home_team'],
        "away_team": row['away_team'],
        "result": result,
        "market_odds": {
            "home_win": 2.5,  # Quota fittizia
            "draw": 3.0,      # Quota fittizia
            "away_win": 2.8   # Quota fittizia
        }
    }
    odds.append(match_data)

# Salva i dati storici filtrati in un file JSON
with open('odds_2024.json', 'w') as json_file:
    json.dump(odds, json_file, indent=4)

# Visualizza la lista
print(odds)
