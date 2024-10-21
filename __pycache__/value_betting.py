# sistema di analisi e scommesse

import numpy as np
from datetime import datetime, timedelta
from SerieABayesModelAdvanced import SerieABayesModelAdvanced  # Assumiamo che questa sia la classe che abbiamo creato prima

class SerieABettingSystem:
    def __init__(self, teams):
        self.model = SerieABayesModelAdvanced(teams)
        self.performance_tracker = []
        self.value_threshold = 1.1  # Soglia per identificare value bets

    def initialize_model(self, historical_data):
        self.model.initialize_with_historical_data(historical_data)

    def update_player_impacts(self, impacts):
        for team, players in impacts.items():
            for player, impact in players.items():
                self.model.update_player_impact(team, player, impact)

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
            elif match['result'] == 'draw':
                actual = [0, 1, 0]
            else:
                actual = [0, 0, 1]
            predicted = [prediction['home_win'], prediction['draw'], prediction['away_win']]
            total_log_loss -= np.sum(np.multiply(actual, np.log(predicted)))

            # Controlla se la previsione più probabile era corretta
            if match['result'] == max(prediction, key=prediction.get):
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
            {"home_team": "Juventus", "away_team": "Inter", "market_odds": {"home_win": 2.1, "draw": 3.2, "away_win": 3.5}},
            {"home_team": "Milan", "away_team": "Roma", "market_odds": {"home_win": 1.9, "draw": 3.4, "away_win": 4.0}},
            # Aggiungi altre partite qui
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
teams = ["Juventus", "Inter", "Milan", "Roma", "Napoli", "Lazio"]
betting_system = SerieABettingSystem(teams)

# Inizializza il sistema con dati storici
historical_data = [
    {"home_team": "Juventus", "away_team": "Inter", "home_goals": 2, "away_goals": 1, "date": "2022-09-15"},
    # Aggiungi più dati storici qui
]
betting_system.initialize_model(historical_data)

# Aggiorna gli impatti dei giocatori
betting_system.update_player_impacts({
    "Juventus": {"Cristiano Ronaldo": 0.1},
    "Inter": {"Romelu Lukaku": -0.05}
})

# Traccia la performance per alcune giornate
for i in range(5):
    date = (datetime.now() - timedelta(days=7*i)).strftime("%Y-%m-%d")
    matches = [
        {"home_team": "Juventus", "away_team": "Milan", "result": "home_win", "market_odds": {"home_win": 1.8, "draw": 3.5, "away_win": 4.2}},
        {"home_team": "Inter", "away_team": "Roma", "result": "draw", "market_odds": {"home_win": 2.0, "draw": 3.2, "away_win": 3.8}},
        # Aggiungi più partite qui
    ]
    betting_system.track_performance(date, matches)

# Genera un report per la prossima giornata
report = betting_system.generate_report(datetime.now().strftime("%Y-%m-%d"))
print(report)