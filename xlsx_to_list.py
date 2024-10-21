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

# Crea la lista di dizionari
historical_data = df.to_dict(orient='records')

# Salva i dati storici in un file JSON
with open('historical_data.json', 'w') as json_file:
    json.dump(historical_data, json_file, indent=4)

# Visualizza la lista
print(historical_data)
