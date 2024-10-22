# sistema di analisi e scommesse

import numpy as np
from datetime import datetime, timedelta
from serie_a_bayes import SerieABayesModel
from  loading_historical_data import load_historical_data
from fpdf import FPDF

class SerieABettingSystem:
    def __init__(self, current_teams):
        self.model = SerieABayesModel(current_teams)
        self.performance_tracker = []
        self.value_threshold = 1.1  # Soglia per identificare value bets

    def initialize_model(self, historical_data):
        self.model.initialize_with_historical_data(historical_data)

    def analyze_match(self, home_team, away_team, market_odds):
        """
        Analizza una partita e identifica potenziali value bets.

        :param home_team: Nome della squadra di casa
        :param away_team: Nome della squadra ospite
        :param market_odds: Dizionario con le quote di mercato (es. {"home_win": 2.0, "draw": 3.2, "away_win": 4.0})
        :return: Dizionario con l'analisi della partita
        """
        prediction = self.model.predict_match(home_team, away_team)

        analysis = {
            "prediction": prediction,
            "market_odds": market_odds,
            "value_bets": []
        }

        for outcome in ["home_win", "draw", "away_win"]:
            model_prob = prediction[outcome]
            market_prob = 1 / market_odds[outcome]
            value = model_prob * market_odds[outcome]

            if value > self.value_threshold:
                analysis["value_bets"].append({
                    "outcome": outcome,
                    "model_probability": model_prob,
                    "market_odds": market_odds[outcome],
                    "value": value
                })

        return analysis

    def track_performance(self, date, matches):
        """
        Traccia la performance del modello per un set di partite.

        :param date: Data delle partite
        :param matches: Lista di dizionari con 'home_team', 'away_team', 'result', e 'market_odds'
        """
        correct_predictions = 0
        total_log_loss = 0
        total_value = 0

        for match in matches:
            prediction = self.model.predict_match(match['home_team'], match['away_team'])

            # Calcola log loss
            if match['result'] == 'home_win':
                actual = [1, 0, 0]
            elif match['result'] == 'away_win':
                actual = [0, 1, 0]
            else:
                actual = [0, 0, 1]

            predicted = [prediction['home_win'], prediction['draw'], prediction['away_win']]
            total_log_loss -= np.sum(np.multiply(actual, np.log(predicted)))

            # Controlla se la previsione più probabile era corretta
            valid_predictions = {outcome: prob for outcome, prob in prediction.items() if isinstance(prob, (int, float))}
            if match['result'] == max(valid_predictions, key=valid_predictions.get):
                correct_predictions += 1

            # Calcola il valore totale delle value bets
            analysis = self.analyze_match(match['home_team'], match['away_team'], match['market_odds'])
            for bet in analysis['value_bets']:
                if bet['outcome'] == match['result']:
                    total_value += bet['value'] - 1  # -1 per considerare la puntata

        accuracy = correct_predictions / len(matches)
        avg_log_loss = total_log_loss / len(matches)

        self.performance_tracker.append({
            "date": date,
            "accuracy": accuracy,
            "avg_log_loss": avg_log_loss,
            "total_value": total_value
        })

    def get_season_trends(self):
        """
        Analizza le tendenze della stagione basate sulla performance del modello.

        :return: Dizionario con le tendenze della stagione
        """
        if not self.performance_tracker:
            return "Nessun dato disponibile per l'analisi delle tendenze."

        trends = {
            "accuracy_trend": [],
            "log_loss_trend": [],
            "value_trend": []
        }

        window_size = 5  # Usiamo una finestra mobile di 5 giornate per le tendenze
        for i in range(len(self.performance_tracker) - window_size + 1):
            window = self.performance_tracker[i:i+window_size]
            trends["accuracy_trend"].append(np.mean([day["accuracy"] for day in window]))
            trends["log_loss_trend"].append(np.mean([day["avg_log_loss"] for day in window]))
            trends["value_trend"].append(np.sum([day["total_value"] for day in window]))

        return trends

    def generate_report(self, date):
        """
        Genera un report con le previsioni per la prossima giornata e l'analisi delle tendenze.

        :param date: Data della giornata per cui generare il report
        :return: Stringa con il report
        """
        # Qui dovresti implementare la logica per ottenere le partite della prossima giornata
        next_matches = [
            {"home_team": "Udinese", "away_team": "Cagliari", "market_odds": {"home_win": 2.40, "draw": 2.95, "away_win": 3.30}},
            {"home_team": "Torino", "away_team": "Como", "market_odds": {"home_win": 2.50, "draw": 3.20, "away_win": 2.85}},
            {"home_team": "Napoli", "away_team": "Lecce", "market_odds": {"home_win": 1.30, "draw": 5.40, "away_win": 9.80}},
            {"home_team": "Bologna", "away_team": "Milan", "market_odds": {"home_win": 3.50, "draw": 3.50, "away_win": 2.05}},
            {"home_team": "Atalanta", "away_team": "Verona", "market_odds": {"home_win": 1.40, "draw": 5.00, "away_win": 7.00}},
            {"home_team": "Parma", "away_team": "Empoli", "market_odds": {"home_win": 2.20, "draw": 3.35, "away_win": 3.30}},
            {"home_team": "Lazio", "away_team": "Genoa", "market_odds": {"home_win": 1.53, "draw": 4.20, "away_win": 6.00}},
            {"home_team": "Monza", "away_team": "Venezia", "market_odds": {"home_win": 2.20, "draw": 3.25, "away_win": 3.40}},
            {"home_team": "Inter", "away_team": "Juventus", "market_odds": {"home_win": 1.75, "draw": 3.50, "away_win": 4.90}},
            {"home_team": "Fiorentina", "away_team": "Roma", "market_odds": {"home_win": 2.55, "draw": 3.20, "away_win": 2.80}}
        ]

        report = f"Report per la giornata del {date}\n\n"
        report += "Previsioni per le prossime partite:\n"
        for match in next_matches:
            analysis = self.analyze_match(match['home_team'], match['away_team'], match['market_odds'])
            report += f"\n{match['home_team']} vs {match['away_team']}:\n"
            report += f"Previsione del modello: {analysis['prediction']}\n"
            report += f"Quote di mercato: {analysis['market_odds']}\n"
            if analysis['value_bets']:
                report += "Value bets suggerite:\n"
                for bet in analysis['value_bets']:
                    report += f"- {bet['outcome']} (valore: {bet['value']:.2f})\n"
            else:
                report += "Nessuna value bet identificata per questa partita.\n"

        trends = self.get_season_trends()
        report += "\nTendenze della stagione:\n"
        report += f"Trend di accuratezza: {'In aumento' if trends['accuracy_trend'][-1] > trends['accuracy_trend'][0] else 'In diminuzione'}\n"
        report += f"Trend di log loss: {'In diminuzione' if trends['log_loss_trend'][-1] < trends['log_loss_trend'][0] else 'In aumento'}\n"
        report += f"Trend di valore generato: {'Positivo' if trends['value_trend'][-1] > 0 else 'Negativo'}\n"

        return report

# Esempio di utilizzo
current_teams = ["Atalanta", "Bologna", "Cagliari", "Como", "Empoli", "Fiorentina", 
    "Genoa", "Inter", "Juventus", "Lazio", "Lecce", "Milan", "Monza", "Napoli", "Parma", "Roma", 
    "Torino", "Udinese", "Venezia", "Verona"]
betting_system = SerieABettingSystem(current_teams)

# Inizializza il sistema con dati storici
historical_data = load_historical_data('historical_data.json')
betting_system.initialize_model(historical_data)

# Traccia la performance per alcune giornate
for i in range(8):
    date = (datetime.now() - timedelta(days=7*i)).strftime("%Y-%m-%d")
    matches = load_historical_data('odds.json')
    betting_system.track_performance(date, matches)

# Genera un report per la prossima giornata
report = betting_system.generate_report(datetime.now().strftime("%Y-%m-%d"))
#print(report)
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

for line in report.split('\n'):
    pdf.cell(200, 10, txt=line, ln=True)

pdf.output("report.pdf")
print("Report salvato come report.pdf")