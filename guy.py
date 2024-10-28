import sys
import os
import json
from datetime import datetime, timedelta
import shutil
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QSpinBox, 
                           QTextEdit, QPushButton, QTabWidget, QGridLayout,
                           QTableWidget, QTableWidgetItem, QMessageBox,
                           QScrollArea, QFileDialog, QProgressBar, QGroupBox, QSizePolicy, QDateEdit, QLineEdit)
from PyQt6.QtCore import Qt, QTimer, QDate
import pandas as pd
import numpy as np
from main import PredictionSystem
from models.base_model import BaseModel
from models.bayes_model import SerieABayesModelWrapper
from models.mc_wrapper import SerieAMonteCarloWrapper
from models.ensemble_wrapper import SerieAEnsembleWrapper
from utils.data_loader import load_historical_data
from config.teams import CURRENT_TEAMS

class PredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Serie A Predictor")
        self.setMinimumSize(1200, 800)

        # Definisci il percorso del file dei dati storici
        current_directory = os.path.dirname(__file__)
        self.historical_data_path = os.path.join(current_directory, 'input_data', 'historical_data.json')
        
        # Initialize prediction system and data structures
        self.prediction_system = PredictionSystem()
        self.prediction_history = []  # Store historical predictions
        self.validation_results = []  # Store validation results
        self.historical_data = []  # Aggiunto per assicurare l'inizializzazione
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.addWidget(tab_widget)
        
        # Create tabs
        predictions_tab = QWidget()
        historical_data_tab = QWidget()
        analysis_tab = QWidget()
        comparison_tab = QWidget()
        validation_tab = QWidget()
        
        tab_widget.addTab(predictions_tab, "Predizioni")
        tab_widget.addTab(analysis_tab, "Analisi Temporale")
        tab_widget.addTab(comparison_tab, "Confronto Modelli")
        tab_widget.addTab(validation_tab, "Validazione")
        tab_widget.addTab(historical_data_tab, "Dati Storici")
        
        # Setup tabs
        self._setup_predictions_tab(predictions_tab)
        self._setup_historical_data_tab(historical_data_tab)
        self._setup_analysis_tab(analysis_tab)
        self._setup_comparison_tab(comparison_tab)
        self._setup_validation_tab(validation_tab)
        
        # Load historical data
        self._load_historical_data()

        # Configura tutte le tabelle
        self.setup_all_tables()

    def setup_all_tables(self):
        """
        Trova e configura automaticamente tutte le QTableWidget nella GUI,
        disabilitando sia la modifica che la selezione delle celle
        """
        for table in self.findChildren(QTableWidget):
            # Disabilita la modifica
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            # Disabilita la selezione
            table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
            # Disabilita il focus sulle celle
            table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            # Opzionalmente, puoi anche disabilitare l'evidenziazione al passaggio del mouse
            table.setMouseTracking(False)

    def _load_historical_data(self):
        """
        Carica i dati storici dal file JSON.
        """
        try:
            self.historical_data = load_historical_data(self.historical_data_path)
            # Aggiorna immediatamente le visualizzazioni
            self._update_historical_data()
            self._update_predictions_history()
        except Exception as e:
            QMessageBox.warning(self, "Errore", f"Errore nel caricamento dei dati storici: {str(e)}")

    def _setup_predictions_tab(self, tab):
        layout = QVBoxLayout(tab)
    
        # Sezione input giornata
        input_group = QWidget()
        input_layout = QGridLayout(input_group)
        
        # Selezione modello
        model_label = QLabel("Modello:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Ensemble", "Bayesiano", "Monte Carlo"])
        self.model_combo.currentTextChanged.connect(self._update_model_options)
        
        # Opzioni aggiuntive modello
        self.bayes_type_label = QLabel("Tipo Bayesiano:")
        self.bayes_type_combo = QComboBox()
        self.bayes_type_combo.addItems(["standard", "draw"])
        self.bayes_type_label.hide()
        self.bayes_type_combo.hide()
        
        self.mc_iterations_label = QLabel("Iterazioni:")
        self.mc_iterations_spin = QSpinBox()
        self.mc_iterations_spin.setRange(1000, 100000)
        self.mc_iterations_spin.setValue(10000)
        self.mc_iterations_spin.setSingleStep(1000)
        self.mc_iterations_label.hide()
        self.mc_iterations_spin.hide()
        
        input_layout.addWidget(model_label, 0, 0)
        input_layout.addWidget(self.model_combo, 0, 1)
        input_layout.addWidget(self.bayes_type_label, 1, 0)
        input_layout.addWidget(self.bayes_type_combo, 1, 1)
        input_layout.addWidget(self.mc_iterations_label, 2, 0)
        input_layout.addWidget(self.mc_iterations_spin, 2, 1)
        
        # Tabella partite
        matches_group = QGroupBox("Partite da Analizzare")
        matches_layout = QVBoxLayout(matches_group)
        
        # Aggiungi controlli per la gestione delle righe
        matches_controls = QHBoxLayout()
        self.matches_count_spin = QSpinBox()
        self.matches_count_spin.setRange(1, 10)
        self.matches_count_spin.setValue(10)
        self.matches_count_spin.valueChanged.connect(self._update_matches_table)
        matches_controls.addWidget(QLabel("Numero partite:"))
        matches_controls.addWidget(self.matches_count_spin)
        matches_controls.addStretch()
        matches_layout.addLayout(matches_controls)

        self.matches_table = QTableWidget()
        self.matches_table.setColumnCount(2)
        self.matches_table.setRowCount(10)  # 10 partite per giornata
        self.matches_table.setHorizontalHeaderLabels(["Squadra Casa", "Squadra Trasferta"])
        
        # Popola le celle con ComboBox
        for row in range(10):
            home_combo = QComboBox()
            away_combo = QComboBox()
            home_combo.addItems(CURRENT_TEAMS)
            away_combo.addItems(CURRENT_TEAMS)
            self.matches_table.setCellWidget(row, 0, home_combo)
            self.matches_table.setCellWidget(row, 1, away_combo)
        
        matches_layout.addWidget(self.matches_table)
        
        layout.addWidget(input_group)
        layout.addWidget(matches_group)
        
        # Pulsante predizione
        predict_btn = QPushButton("Genera Predizioni")
        predict_btn.clicked.connect(self._generate_prediction)
        layout.addWidget(predict_btn)
        
        # Sezione risultati
        results_group = QWidget()
        results_layout = QVBoxLayout(results_group)
        
        # Tabella risultati predizioni
        self.prediction_table = QTableWidget()
        self.prediction_table.setColumnCount(7)
        self.prediction_table.setHorizontalHeaderLabels([
            "Partita", "1", "X", "2", 
            "Quote Suggerite (1-X-2)", "Probabilità Più Alta",
            "Valore Atteso"
        ])
        results_layout.addWidget(self.prediction_table)
        
        # Statistiche aggiuntive
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(100)
        results_layout.addWidget(self.stats_text)
        
        layout.addWidget(results_group)

    def _update_matches_table(self):
        """
        Aggiorna la visibilità delle righe della tabella in base al numero selezionato
        """
        selected_rows = self.matches_count_spin.value()
        for row in range(10):
            self.matches_table.setRowHidden(row, row >= selected_rows)

    def _update_model_options(self):
        """Aggiorna le opzioni aggiuntive in base al modello selezionato"""
        model = self.model_combo.currentText()
        
        # Nascondi tutte le opzioni
        self.bayes_type_label.hide()
        self.bayes_type_combo.hide()
        self.mc_iterations_label.hide()
        self.mc_iterations_spin.hide()
        
        # Mostra le opzioni pertinenti
        if model == "Bayesiano":
            self.bayes_type_label.show()
            self.bayes_type_combo.show()
        elif model == "Monte Carlo":
            self.mc_iterations_label.show()
            self.mc_iterations_spin.show()

    def _setup_historical_data_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Sezione inserimento nuovi risultati
        input_group = QGroupBox("Inserisci Nuovi Risultati")
        input_layout = QGridLayout(input_group)
        
        # Controlli per l'inserimento
        self.new_match_home = QComboBox()
        self.new_match_away = QComboBox()
        self.new_match_home.addItems(CURRENT_TEAMS)
        self.new_match_away.addItems(CURRENT_TEAMS)
        
        self.home_goals_spin = QSpinBox()
        self.away_goals_spin = QSpinBox()
        self.home_goals_spin.setRange(0, 10)
        self.away_goals_spin.setRange(0, 10)
        
        self.match_date = QDateEdit()
        self.match_date.setCalendarPopup(True)
        self.match_date.setDate(QDate.currentDate())
        
        # Campo note
        self.match_notes = QLineEdit()
        
        # Layout per i controlli
        input_layout.addWidget(QLabel("Squadra Casa:"), 0, 0)
        input_layout.addWidget(self.new_match_home, 0, 1)
        input_layout.addWidget(QLabel("Gol Casa:"), 0, 2)
        input_layout.addWidget(self.home_goals_spin, 0, 3)
        
        input_layout.addWidget(QLabel("Squadra Trasferta:"), 1, 0)
        input_layout.addWidget(self.new_match_away, 1, 1)
        input_layout.addWidget(QLabel("Gol Trasferta:"), 1, 2)
        input_layout.addWidget(self.away_goals_spin, 1, 3)
        
        input_layout.addWidget(QLabel("Data:"), 2, 0)
        input_layout.addWidget(self.match_date, 2, 1)
        
        input_layout.addWidget(QLabel("Note:"), 3, 0)
        input_layout.addWidget(self.match_notes, 3, 1, 1, 3)
        
        # Pulsante per aggiungere il risultato
        add_btn = QPushButton("Aggiungi Risultato")
        add_btn.clicked.connect(self._add_new_result)
        input_layout.addWidget(add_btn, 4, 0, 1, 4)
        
        # Sezione filtri
        filter_group = QWidget()
        filter_layout = QHBoxLayout(filter_group)
        
        self.season_combo = QComboBox()
        self.season_combo.addItems(["2023-2024"])  # Aggiungi le stagioni necessarie
        
        self.historical_team_combo = QComboBox()
        self.historical_team_combo.addItems(["Tutte"] + CURRENT_TEAMS)
        
        filter_layout.addWidget(QLabel("Stagione:"))
        filter_layout.addWidget(self.season_combo)
        filter_layout.addWidget(QLabel("Squadra:"))
        filter_layout.addWidget(self.historical_team_combo)
        filter_layout.addStretch()
        
        # Tabella dati storici
        history_group = QGroupBox("Dati Storici")
        history_layout = QVBoxLayout(history_group)
        
        self.historical_table = QTableWidget()
        self.historical_table.setColumnCount(7)
        self.historical_table.setHorizontalHeaderLabels([
            "Data", "Squadra Casa", "Squadra Trasferta", 
            "Gol Casa", "Gol Trasferta", "Risultato", "Note"
        ])
        
        history_layout.addWidget(self.historical_table)
        
        # Sezione statistiche
        stats_group = QWidget()
        stats_layout = QHBoxLayout(stats_group)
        
        self.total_matches_label = QLabel("Partite Totali: 0")
        self.home_wins_label = QLabel("Vittorie Casa: 0")
        self.away_wins_label = QLabel("Vittorie Trasferta: 0")
        self.draws_label = QLabel("Pareggi: 0")
        self.avg_goals_label = QLabel("Media Goal: 0.00")
        
        stats_layout.addWidget(self.total_matches_label)
        stats_layout.addWidget(self.home_wins_label)
        stats_layout.addWidget(self.away_wins_label)
        stats_layout.addWidget(self.draws_label)
        stats_layout.addWidget(self.avg_goals_label)
        
        # Aggiungi i widget al layout principale
        layout.addWidget(input_group)
        layout.addWidget(filter_group)
        layout.addWidget(history_group)
        layout.addWidget(stats_group)
        
        # Connetti i segnali per l'aggiornamento
        self.season_combo.currentTextChanged.connect(self._update_historical_data)
        self.historical_team_combo.currentTextChanged.connect(self._update_historical_data)

    def _add_new_result(self):
        """Aggiunge un nuovo risultato ai dati storici."""
        # Validazione base
        home_team = self.new_match_home.currentText()
        away_team = self.new_match_away.currentText()
        
        if home_team == away_team:
            QMessageBox.warning(
                self, 
                "Errore", 
                "La squadra di casa e quella in trasferta non possono essere uguali."
            )
            return
        
        # Preparazione del nuovo risultato
        new_result = {
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": self.home_goals_spin.value(),
            "away_goals": self.away_goals_spin.value(),
            "date": self.match_date.date().toString("yyyy-MM-dd"),
            "notes": self.match_notes.text().strip()
        }
        
        try:
            # Carica i dati esistenti
            with open(self.historical_data_path, 'r') as f:
                data = json.load(f)
            
            # Aggiungi il nuovo risultato
            data.append(new_result)
            
            # Salva i dati aggiornati
            with open(self.historical_data_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            # Aggiorna i dati in memoria e la visualizzazione
            self.historical_data = data
            self._update_historical_data()
            
            # Reset dei controlli
            self.home_goals_spin.setValue(0)
            self.away_goals_spin.setValue(0)
            self.match_date.setDate(QDate.currentDate())
            self.match_notes.clear()
            
            QMessageBox.information(
                self, 
                "Successo", 
                "Risultato aggiunto correttamente."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Errore", 
                f"Errore durante il salvataggio dei dati: {str(e)}"
            )

    def _update_historical_data(self):
        """
        Aggiorna la visualizzazione dei dati storici nella GUI.
        """
        try:
            if not hasattr(self, 'historical_table'):
                return  # Evita errori se la tabella non è stata ancora creata
                
            season = self.season_combo.currentText()
            team_filter = self.historical_team_combo.currentText()
            
            # Verifica che historical_data sia inizializzato
            if not self.historical_data:
                self._load_historical_data()
                return
                
            # Filter data based on selection
            filtered_data = [
                match for match in self.historical_data
                if (team_filter == "Tutte" or 
                    match['home_team'] == team_filter or 
                    match['away_team'] == team_filter)
            ]
            
            # Update table
            self.historical_table.setRowCount(len(filtered_data))
            for i, match in enumerate(filtered_data):
                self.historical_table.setItem(i, 0, QTableWidgetItem(match['date']))
                self.historical_table.setItem(i, 1, QTableWidgetItem(match['home_team']))
                self.historical_table.setItem(i, 2, QTableWidgetItem(match['away_team']))
                self.historical_table.setItem(i, 3, QTableWidgetItem(str(match['home_goals'])))
                self.historical_table.setItem(i, 4, QTableWidgetItem(str(match['away_goals'])))
                
                result = "V-Casa" if match['home_goals'] > match['away_goals'] else \
                        "V-Trasferta" if match['home_goals'] < match['away_goals'] else "Pareggio"
                self.historical_table.setItem(i, 5, QTableWidgetItem(result))
                
                notes = match.get('notes', '')
                self.historical_table.setItem(i, 6, QTableWidgetItem(notes))
            
            # Update statistics
            if filtered_data:
                total_matches = len(filtered_data)
                home_wins = sum(1 for m in filtered_data if m['home_goals'] > m['away_goals'])
                away_wins = sum(1 for m in filtered_data if m['home_goals'] < m['away_goals'])
                draws = sum(1 for m in filtered_data if m['home_goals'] == m['away_goals'])
                total_goals = sum(m['home_goals'] + m['away_goals'] for m in filtered_data)
                avg_goals = total_goals / total_matches if total_matches > 0 else 0
                
                self.total_matches_label.setText(f"Partite Totali: {total_matches}")
                self.home_wins_label.setText(f"Vittorie Casa: {home_wins}")
                self.away_wins_label.setText(f"Vittorie Trasferta: {away_wins}")
                self.draws_label.setText(f"Pareggi: {draws}")
                self.avg_goals_label.setText(f"Media Goal: {avg_goals:.2f}")
                
        except Exception as e:
            QMessageBox.warning(self, "Errore", f"Errore nell'aggiornamento dei dati: {str(e)}")

    def _generate_prediction(self):
        """
        Genera predizioni per tutte le partite della giornata.
        """
        try:
            if not self.historical_data:
                raise ValueError("Dati storici non caricati")
            
            print("\n=== Debug Info ===")
            print(f"Dati storici disponibili: {len(self.historical_data)} partite")
            
            # Raccogli tutte le partite
            matches = []
            selected_rows = self.matches_count_spin.value()
            for row in range(selected_rows):
                home_combo = self.matches_table.cellWidget(row, 0)
                away_combo = self.matches_table.cellWidget(row, 1)
                if home_combo and away_combo:
                    home_team = home_combo.currentText()
                    away_team = away_combo.currentText()
                    if home_team == away_team:
                        raise ValueError(f"Riga {row+1}: Le squadre devono essere diverse")
                    matches.append((home_team, away_team))
            
            if not matches:
                raise ValueError("Inserire almeno una partita")
            
            print(f"\nPartite da analizzare: {matches}")

            # Crea il modello appropriato
            model_type = self.model_combo.currentText()
            print(f"\nTipo di modello selezionato: {model_type}")

            if model_type == "Bayesiano":
                bayes_type = self.bayes_type_combo.currentText()
                model = SerieABayesModelWrapper(CURRENT_TEAMS, model_type=bayes_type)
            elif model_type == "Monte Carlo":
                n_simulations = self.mc_iterations_spin.value()
                model = SerieAMonteCarloWrapper(CURRENT_TEAMS, n_simulations=n_simulations)
            else:  # Ensemble
                model = SerieAEnsembleWrapper(CURRENT_TEAMS)
            
            # Inizializza il modello
            model.initialize_model(self.historical_data, decay_factor=0.5)
            
            # Genera predizioni per ogni partita
            self.prediction_table.setRowCount(len(matches))
            total_confidence = 0

            print("\n=== Analisi Partite ===")
            
            for i, (home_team, away_team) in enumerate(matches):
                prediction = model.predict_match(home_team, away_team)
                print(f"Predizione raw: {prediction}")

                # Formatta la stringa della partita
                match_str = f"{home_team} vs {away_team}"
                
                # Calcola le probabilità e le quote
                probs = [
                    prediction['home_win'],
                    prediction['draw'],
                    prediction['away_win']
                ]
                print(f"Probabilità estratte: {probs}")

                 # Assicuriamoci che le probabilità siano numeri validi
                probs = [float(p) if p is not None else 0.0 for p in probs]
                print(f"Probabilità convertite: {probs}")

                # Calcolo quote con controllo per divisione per zero
                odds = []
                for p in probs:
                    if p > 0.01:  # Soglia minima per evitare quote irrealistiche
                        odds.append(1/p)
                    else:
                        odds.append(100.0)  # Quota massima per probabilità molto basse
                print(f"Quote calcolate: {odds}")

                odds_str = f"{odds[0]:.2f}-{odds[1]:.2f}-{odds[2]:.2f}"
                
                # Trova l'esito più probabile
                outcomes = ['1', 'X', '2']
                max_prob_idx = probs.index(max(probs))
                max_prob_str = f"{outcomes[max_prob_idx]} ({probs[max_prob_idx]:.1%})"
                
                evs = []
                for odd, prob in zip(odds, probs):
                    if prob > 0 and odd > 1.0:
                        ev = (odd - 1) * prob  # Formula EV corretta
                        evs.append(ev)
                    else:
                        evs.append(-1.0)  # Valore negativo per quote/prob invalide
                print(f"Valori attesi (EV): {evs}")

                max_ev = max(evs)
                print(f"EV massimo: {max_ev}")

                # Calcola la confidenza della predizione
                confidence = prediction.get('confidence')

                if confidence is None:
                    confidence = max(probs) - min(probs)
                    print(f"Confidenza calcolata dalla differenza delle probabilità: {confidence}")
                else:
                    print(f"Confidenza fornita dal modello: {confidence}")
                
                total_confidence += confidence
                print(f"Total confidence accumulata: {total_confidence}")

                # Popola la riga della tabella
                self.prediction_table.setItem(i, 0, QTableWidgetItem(match_str))
                self.prediction_table.setItem(i, 1, QTableWidgetItem(f"{probs[0]:.1%}"))
                self.prediction_table.setItem(i, 2, QTableWidgetItem(f"{probs[1]:.1%}"))
                self.prediction_table.setItem(i, 3, QTableWidgetItem(f"{probs[2]:.1%}"))
                self.prediction_table.setItem(i, 4, QTableWidgetItem(odds_str))
                self.prediction_table.setItem(i, 5, QTableWidgetItem(max_prob_str))
                self.prediction_table.setItem(i, 6, QTableWidgetItem(f"{max_ev:+.2%}"))
            
            # Calcola la confidenza media solo se abbiamo partite valide
            if matches:
                avg_confidence = total_confidence / len(matches)
                print(f"\nConfidenza media finale: {avg_confidence}")
            else:
                avg_confidence = 0.0

            # Aggiorna statistiche generali
            
            stats = (
                f"Statistiche giornata:\n"
                f"Numero partite analizzate: {len(matches)}\n"
                f"Confidenza media predizioni: {avg_confidence:.1%}"
            )
            self.stats_text.setText(stats)
            
        except Exception as e:
            print(f"\nERRORE: {str(e)}")
            print(f"Tipo di errore: {type(e)}")
            QMessageBox.warning(self, "Errore", f"Errore nella generazione delle predizioni: {str(e)}")

    def _update_predictions_history(self):
        """
        Aggiorna la visualizzazione dello storico predizioni.
        """
        try:
            if not hasattr(self, 'predictions_history_table'):
                return  # Evita errori se la tabella non è stata ancora creata
                
            # Show last 10 predictions
            recent_predictions = self.prediction_history[-10:] if self.prediction_history else []
            
            self.predictions_history_table.setRowCount(len(recent_predictions))
            for i, pred in enumerate(recent_predictions):
                self.predictions_history_table.setItem(i, 0, QTableWidgetItem(pred['date']))
                self.predictions_history_table.setItem(i, 1, 
                    QTableWidgetItem(f"{pred['home_team']} vs {pred['away_team']}")
                )
                
                # Get highest probability outcome
                outcomes = {
                    'home_win': "Vittoria Casa",
                    'draw': "Pareggio",
                    'away_win': "Vittoria Trasferta"
                }
                max_outcome = max(
                    pred['prediction'].keys(), 
                    key=lambda k: pred['prediction'].get(k, 0) if k in ['home_win', 'draw', 'away_win'] else 0
                )
                
                prob = pred['prediction'].get(max_outcome, 0)
                self.predictions_history_table.setItem(i, 2, 
                    QTableWidgetItem(f"{outcomes[max_outcome]} ({prob:.1%})")
                )
                
                # Add actual result and accuracy if available
                if 'actual_result' in pred:
                    self.predictions_history_table.setItem(i, 3, 
                        QTableWidgetItem(outcomes.get(pred['actual_result'], ''))
                    )
                    accuracy = '✓' if pred.get('accurate', False) else '✗'
                    self.predictions_history_table.setItem(i, 4, QTableWidgetItem(accuracy))
                else:
                    self.predictions_history_table.setItem(i, 3, QTableWidgetItem('In attesa'))
                    self.predictions_history_table.setItem(i, 4, QTableWidgetItem('-'))
                    
            # Adjust column widths
            self.predictions_history_table.resizeColumnsToContents()
            
        except Exception as e:
            QMessageBox.warning(self, "Errore", f"Errore nell'aggiornamento dello storico: {str(e)}")

    def _setup_analysis_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Team selection
        team_label = QLabel("Squadra:")
        self.analysis_team_combo = QComboBox()
        self.analysis_team_combo.addItem("Seleziona squadra")  # Aggiunto item iniziale
        self.analysis_team_combo.addItems(CURRENT_TEAMS)
        self.analysis_team_combo.currentTextChanged.connect(self._on_team_selected)
        
        # Time period selection
        period_label = QLabel("Periodo:")
        self.period_combo = QComboBox()
        self.period_combo.addItem("Seleziona periodo")  # Aggiunge opzione vuota iniziale
        self.period_combo.addItems(["Da inizio stagione", "Ultimi 3 mesi", "Ultimo mese"])
        self.period_combo.setEnabled(False)  # Disabilitato inizialmente
        self.period_combo.currentTextChanged.connect(self._on_period_selected)

        # Opponent selection (inizialmente nascosto)
        self.opponent_label = QLabel("Avversario:")
        self.opponent_team_combo = QComboBox()
        self.opponent_team_combo.addItem("Seleziona avversario")
        self.opponent_team_combo.addItems(CURRENT_TEAMS)
        self.opponent_team_combo.currentTextChanged.connect(self._update_h2h_graph)

        # Inizialmente nascondi i controlli per il testa a testa
        self.opponent_label.hide()
        self.opponent_team_combo.hide()
        
        # Aggiungi un pulsante per attivare la modalità testa a testa
        self.h2h_button = QPushButton("Mostra Testa a Testa")
        self.h2h_button.setCheckable(True)
        self.h2h_button.setEnabled(False)  # Disabilitato inizialmente
        self.h2h_button.clicked.connect(self._toggle_h2h_mode)
        
        controls_layout.addWidget(team_label)
        controls_layout.addWidget(self.analysis_team_combo)
        controls_layout.addWidget(period_label)
        controls_layout.addWidget(self.period_combo)
        controls_layout.addWidget(self.h2h_button)
        controls_layout.addWidget(self.opponent_label)
        controls_layout.addWidget(self.opponent_team_combo)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)

        # Create scroll area for graphs
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(0)  # Aggiungi spazio tra i widget
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Performance graph
        performance_container = QWidget()
        performance_layout = QVBoxLayout(performance_container)
        performance_layout.setContentsMargins(0, 0, 0, 0)
        self.performance_fig = Figure(figsize=(12, 8))
        self.performance_canvas = FigureCanvas(self.performance_fig)
        # Aumenta lo spazio per le etichette sull'asse x
        self.performance_fig.subplots_adjust(bottom=0.15, top=0.95)
        performance_layout.addWidget(self.performance_canvas)
        performance_container.hide()  # Nascosto inizialmente
        scroll_layout.addWidget(performance_container)
        self.performance_container = performance_container

        # Prediction graph
        prediction_container = QWidget()
        prediction_layout = QVBoxLayout(prediction_container)
        self.prediction_accuracy_fig = Figure(figsize=(12, 6))
        self.prediction_accuracy_canvas = FigureCanvas(self.prediction_accuracy_fig)
        # Aumenta lo spazio per le etichette sull'asse x
        self.prediction_accuracy_fig.subplots_adjust(bottom=0.2)
        prediction_layout.addWidget(self.prediction_accuracy_canvas)
        scroll_layout.addWidget(prediction_container)

        # Results distribution graph (Wins, Draws, Losses)
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        performance_layout.setContentsMargins(0, 0, 0, 0)
        self.results_fig = Figure(figsize=(12, 8))
        self.results_canvas = FigureCanvas(self.results_fig)
        self.results_fig.subplots_adjust(bottom=0.15, top=0.95)
        results_layout.addWidget(self.results_canvas)
        results_container.hide()  # Nascosto inizialmente
        scroll_layout.addWidget(results_container)
        self.results_container = results_container

        # Head-to-head results graph
        self.h2h_container = QWidget()
        h2h_layout = QVBoxLayout(self.h2h_container)
        performance_layout.setContentsMargins(0, 0, 0, 0)
        self.h2h_fig = Figure(figsize=(12, 10))
        self.h2h_canvas = FigureCanvas(self.h2h_fig)
        self.h2h_fig.subplots_adjust(bottom=0.15, top=0.95)
        h2h_layout.addWidget(self.h2h_canvas)
        h2h_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_layout.addWidget(self.h2h_container, alignment=Qt.AlignmentFlag.AlignCenter)
        self.h2h_container.hide()

        # Stretchers per centrare verticalmente il grafico testa a testa quando è visibile
        self.top_stretcher = QWidget()
        self.top_stretcher.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.bottom_stretcher = QWidget()
        self.bottom_stretcher.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll_layout.insertWidget(0, self.top_stretcher)
        scroll_layout.addWidget(self.bottom_stretcher)
        self.top_stretcher.hide()
        self.bottom_stretcher.hide()

        # Set up scroll area
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        #Inizializza i grafici
        self._update_analysis_graphs()

    def _on_team_selected(self, team_name):
        # Abilita il period_combo solo se è stata selezionata una squadra valida
        if team_name != "Seleziona squadra":
            self.period_combo.setEnabled(True)
            self.h2h_button.setEnabled(True)
            # Reset del period_combo quando si cambia squadra
            self.period_combo.setCurrentText("Seleziona periodo")
            # Nascondi i grafici quando si cambia squadra
            self.performance_container.hide()
            self.results_container.hide()
            self.h2h_container.hide()
        else:
            self.period_combo.setEnabled(False)
            self.h2h_button.setEnabled(False)
            self.performance_container.hide()
            self.results_container.hide()
            self.h2h_container.hide()
        
        # Aggiorna i grafici
        self._update_analysis_graphs()
        
    def _on_period_selected(self, period):
        if (self.analysis_team_combo.currentText() != "Seleziona squadra" and 
            period != "Seleziona periodo"):
            # Aggiorna e mostra i grafici solo quando sono selezionati sia squadra che periodo
            self._update_analysis_graphs()
            self.performance_container.show()
            self.results_container.show()
        else:
            self.performance_container.hide()
            self.results_container.hide()

    def _toggle_h2h_mode(self):
        h2h_enabled = self.h2h_button.isChecked()
        if h2h_enabled:
            self.opponent_label.show()
            self.opponent_team_combo.show()
            self.performance_container.hide()
            self.results_container.hide()
            # Reset della selezione avversario
            self.opponent_team_combo.setCurrentText("Seleziona avversario")
        else:
            self.opponent_label.hide()
            self.opponent_team_combo.hide()
            self.h2h_container.hide()
            # Mostra di nuovo i grafici normali se c'è una squadra selezionata
            if self.analysis_team_combo.currentText() != "Seleziona squadra":
                self.performance_container.show()
                self.results_container.show()

    def _setup_comparison_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Controls for comparison
        controls_layout = QHBoxLayout()
        
        compare_btn = QPushButton("Confronta Modelli")
        compare_btn.clicked.connect(self._compare_models)
        
        controls_layout.addWidget(compare_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Create comparison table
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(4)
        self.comparison_table.setHorizontalHeaderLabels([
            "Modello", "Accuratezza", "Precisione", "Recall"
        ])
        layout.addWidget(self.comparison_table)
        
        # Create matplotlib figure for comparison results
        comparison_container = QWidget()
        comparison_layout = QVBoxLayout(comparison_container)
        self.comparison_fig = Figure(figsize=(12, 6))
        self.comparison_canvas = FigureCanvas(self.comparison_fig)
        comparison_layout.addWidget(self.comparison_canvas)
        layout.addWidget(comparison_container)

    def _setup_validation_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Date range selection
        start_date_label = QLabel("Data Inizio:")
        self.validation_start_date = QDateEdit()
        self.validation_start_date.setCalendarPopup(True)
        self.validation_start_date.setDate(QDate.currentDate())
        
        end_date_label = QLabel("Data Fine:")
        self.validation_end_date = QDateEdit()
        self.validation_end_date.setCalendarPopup(True)
        self.validation_end_date.setDate(QDate.currentDate())
        
        validate_btn = QPushButton("Valida Predizioni")
        validate_btn.clicked.connect(self._validate_predictions)
        
        controls_layout.addWidget(start_date_label)
        controls_layout.addWidget(self.validation_start_date)
        controls_layout.addWidget(end_date_label)
        controls_layout.addWidget(self.validation_end_date)
        controls_layout.addWidget(validate_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Validation results table
        self.validation_table = QTableWidget()
        self.validation_table.setColumnCount(5)
        self.validation_table.setHorizontalHeaderLabels([
            "Data", "Partita", "Predizione", "Risultato Reale", "Accurato"
        ])
        layout.addWidget(self.validation_table)
        
        # Validation statistics
        self.validation_stats_label = QLabel()
        layout.addWidget(self.validation_stats_label)


    def _update_analysis_graphs(self):
        team = self.analysis_team_combo.currentText()
        period = self.period_combo.currentText()
        opponent = self.opponent_team_combo.currentText()  # Aggiungi questa riga
        

        # Verifica che siano state fatte le selezioni necessarie
        if team == "Seleziona squadra" or period == "Seleziona periodo":
            return

        # Calculate date range
        end_date = datetime.now()
        if period == "Ultimo mese":
            start_date = end_date - timedelta(days=30)
        elif period == "Ultimi 3 mesi":
            start_date = end_date - timedelta(days=90)
        elif period == "Da inizio stagione":
            # Calcola l'inizio della stagione corrente (assumiamo che inizi ad agosto)
            current_year = end_date.year
            if end_date.month < 8:  # Se siamo prima di agosto, usa l'anno precedente
                current_year -= 1
            start_date = datetime(current_year, 8, 1)
        else:
            return

        # Aggiorna la visibilità dei container in base alla modalità
        if not self.h2h_button.isChecked():
            self.performance_container.show()
            self.results_container.show()
            self.h2h_container.hide()
        
        # Filter historical data
        team_matches = [
            match for match in self.historical_data
            if (match['home_team'] == team or match['away_team'] == team) and
            datetime.strptime(match['date'], '%Y-%m-%d') >= start_date
        ]
        
        if not team_matches:  # Se non ci sono dati per il periodo selezionato
            self.performance_container.hide()
            self.results_container.hide()
            return

        # Update performance graph
        self.performance_fig.clear()
        ax = self.performance_fig.add_subplot(111)
        
        dates = [datetime.strptime(match['date'], '%Y-%m-%d') for match in team_matches]
        goals_scored = []
        goals_conceded = []
        
        for match in team_matches:
            if match['home_team'] == team:
                goals_scored.append(match['home_goals'])
                goals_conceded.append(match['away_goals'])
            else:
                goals_scored.append(match['away_goals'])
                goals_conceded.append(match['home_goals'])
        
        ax.plot(dates, goals_scored, label='Goal Fatti', marker='o')
        ax.plot(dates, goals_conceded, label='Goal Subiti', marker='o')
        ax.set_title(f'Andamento {team}')
        ax.set_xlabel('Data')
        ax.set_ylabel('Goal')
        ax.legend()
        
        # Formatta le date sull'asse x
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        ax.tick_params(axis='x', rotation=45)
        
        # Imposta i limiti dell'asse y per mostrare sempre da 0 al massimo + 1
        max_goals = max(max(goals_scored), max(goals_conceded)) + 1
        ax.set_ylim(0, max_goals)
            
        self.performance_fig.tight_layout()
        self.performance_canvas.draw()
        self.performance_container.show()
        
        # Update prediction accuracy graph if we have prediction history
        if self.prediction_history:
            self.prediction_accuracy_fig.clear()
            ax = self.prediction_accuracy_fig.add_subplot(111)
            
            team_predictions = [
                pred for pred in self.prediction_history
                if team in [pred['home_team'], pred['away_team']]
            ]
            
            if team_predictions:
                accuracy = [pred['accuracy'] for pred in team_predictions]
                dates = [pred['date'] for pred in team_predictions]
                
                ax.plot(dates, accuracy, label='Accuratezza Predizioni', marker='o')
                ax.set_title(f'Accuratezza Predizioni per {team}')
                ax.set_xlabel('Data')
                ax.set_ylabel('Accuratezza')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
                self.prediction_accuracy_fig.tight_layout()  # Aggiunto per migliorare la visualizzazione
                self.prediction_accuracy_canvas.draw()

        # Nuovo grafico: Distribuzione risultati
        self.results_fig.clear()
        ax = self.results_fig.add_subplot(111)
        
        wins = 0
        draws = 0
        losses = 0
        
        for match in team_matches:
            if match['home_team'] == team:
                if match['home_goals'] > match['away_goals']:
                    wins += 1
                elif match['home_goals'] == match['away_goals']:
                    draws += 1
                else:
                    losses += 1
            else:  # away team
                if match['away_goals'] > match['home_goals']:
                    wins += 1
                elif match['away_goals'] == match['home_goals']:
                    draws += 1
                else:
                    losses += 1
        
        results = [wins, draws, losses]
        labels = ['Vittorie', 'Pareggi', 'Sconfitte']
        colors = ['green', 'yellow', 'red']
        
        ax.bar(labels, results, color=colors)
        ax.set_title(f'Distribuzione Risultati {team}')
        for i, v in enumerate(results):
            ax.text(i, v, str(v), ha='center', va='bottom')
        self.results_fig.tight_layout()
        self.results_canvas.draw()
        self.results_container.show()
        
        # Nuovo grafico: Testa a testa
        if opponent and opponent != "Seleziona avversario" and team != opponent:
            h2h_matches = [
                match for match in self.historical_data
                if (match['home_team'] == team and match['away_team'] == opponent) or
                (match['home_team'] == opponent and match['away_team'] == team)
            ]
            
            if h2h_matches:  # Se ci sono partite testa a testa
                self.h2h_fig.clear()
                ax = self.h2h_fig.add_subplot(111)
                
                team_wins = draws = opponent_wins = 0
                
                for match in h2h_matches:
                    if match['home_team'] == team:
                        if match['home_goals'] > match['away_goals']:
                            team_wins += 1
                        elif match['home_goals'] == match['away_goals']:
                            draws += 1
                        else:
                            opponent_wins += 1
                    else:
                        if match['away_goals'] > match['home_goals']:
                            team_wins += 1
                        elif match['away_goals'] == match['home_goals']:
                            draws += 1
                        else:
                            opponent_wins += 1
                
                results = [team_wins, draws, opponent_wins]
                labels = [f'Vittorie {team}', 'Pareggi', f'Vittorie {opponent}']
                colors = ['blue', 'yellow', 'red']
                
                ax.bar(labels, results, color=colors)
                ax.set_title(f'Storico {team} vs {opponent}')
                for i, v in enumerate(results):
                    ax.text(i, v, str(v), ha='center', va='bottom')
                self.h2h_fig.tight_layout()
                self.h2h_canvas.draw()
                self.h2h_container.show()
            else:
                self.h2h_container.hide()

    def _update_h2h_graph(self):
        if not self.h2h_button.isChecked():
            return
            
        team = self.analysis_team_combo.currentText()
        opponent = self.opponent_team_combo.currentText()
        
        if opponent == "Seleziona avversario" or team == opponent:
            self.h2h_container.hide()
            return
            
        self.h2h_container.show()
        self.h2h_fig.clear()
        ax = self.h2h_fig.add_subplot(111)
        
        h2h_matches = [
            match for match in self.historical_data
            if (match['home_team'] == team and match['away_team'] == opponent) or
            (match['home_team'] == opponent and match['away_team'] == team)
        ]
        
        team_wins = 0
        draws = 0
        opponent_wins = 0
        
        for match in h2h_matches:
            if match['home_team'] == team:
                if match['home_goals'] > match['away_goals']:
                    team_wins += 1
                elif match['home_goals'] == match['away_goals']:
                    draws += 1
                else:
                    opponent_wins += 1
            else:
                if match['away_goals'] > match['home_goals']:
                    team_wins += 1
                elif match['away_goals'] == match['home_goals']:
                    draws += 1
                else:
                    opponent_wins += 1
        
        results = [team_wins, draws, opponent_wins]
        labels = [f'Vittorie {team}', 'Pareggi', f'Vittorie {opponent}']
        colors = ['blue', 'yellow', 'red']
        
        ax.bar(labels, results, color=colors)
        ax.set_title(f'Storico {team} vs {opponent}')
        for i, v in enumerate(results):
            ax.text(i, v, str(v), ha='center', va='bottom')
        self.h2h_fig.tight_layout()
        
        self.h2h_canvas.draw()

    def _compare_models(self):
        # Run predictions with all models on a test set
        test_matches = self._get_recent_matches(10)  # Get last 10 matches for testing
        
        models = {
            'bayes_std': SerieABayesModelWrapper(CURRENT_TEAMS, model_type='standard'),
            'bayes_draw': SerieABayesModelWrapper(CURRENT_TEAMS, model_type='draw'),
            'montecarlo': SerieAMonteCarloWrapper(CURRENT_TEAMS, n_simulations=10000),
            'ensemble': SerieAEnsembleWrapper(CURRENT_TEAMS)
        }
        
        results = {}
        for model_name, model in models.items():
            model.initialize_model(self.historical_data, decay_factor=0.5)
            predictions = []
            
            for match in test_matches:
                pred = model.predict_match(match['home_team'], match['away_team'])
                actual_result = self._get_actual_result(match)
                predicted_result = self._get_predicted_result(pred)
                
                predictions.append({
                    'predicted': predicted_result,
                    'actual': actual_result
                })
            
            results[model_name] = self._calculate_metrics(predictions)
        
        # Update comparison table
        self.comparison_table.setRowCount(len(results))
        for i, (model_name, metrics) in enumerate(results.items()):
            self.comparison_table.setItem(i, 0, QTableWidgetItem(model_name))
            self.comparison_table.setItem(i, 1, QTableWidgetItem(f"{metrics['accuracy']:.2%}"))
            self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{metrics['precision']:.2%}"))
            self.comparison_table.setItem(i, 3, QTableWidgetItem(f"{metrics['recall']:.2%}"))
        
        # Update comparison graph
        self.comparison_fig.clear()
        ax = self.comparison_fig.add_subplot(111)
        
        models = list(results.keys())
        accuracy = [results[m]['accuracy'] for m in models]
        precision = [results[m]['precision'] for m in models]
        recall = [results[m]['recall'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax.bar(x - width, accuracy, width, label='Accuratezza')
        ax.bar(x, precision, width, label='Precisione')
        ax.bar(x + width, recall, width, label='Recall')
        
        ax.set_ylabel('Punteggio')
        ax.set_title('Confronto Prestazioni Modelli')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        self.comparison_canvas.draw()

    def _validate_predictions(self):
        try:
            start_date = datetime.strptime(self.validation_start_date.date().toString("yyyy-MM-dd"), '%Y-%m-%d')
            end_date = datetime.strptime(self.validation_end_date.date().toString("yyyy-MM-dd"), '%Y-%m-%d')
            
            # Get matches in date range
            matches = [
                match for match in self.historical_data
                if start_date <= datetime.strptime(match['date'], '%Y-%m-%d') <= end_date
            ]
            
            if not matches:
                raise ValueError("Nessuna partita trovata nel periodo selezionato")
            
            # Run validation for each model
            models = {
                'bayes_std': SerieABayesModelWrapper(CURRENT_TEAMS, model_type='standard'),
                'bayes_draw': SerieABayesModelWrapper(CURRENT_TEAMS, model_type='draw'),
                'montecarlo': SerieAMonteCarloWrapper(CURRENT_TEAMS, n_simulations=10000),
                'ensemble': SerieAEnsembleWrapper(CURRENT_TEAMS)
            }
            
            validation_results = []
            
            for model_name, model in models.items():
                # Initialize model with data up to start date
                historical_data_subset = [
                    match for match in self.historical_data
                    if datetime.strptime(match['date'], '%Y-%m-%d') < start_date
                ]
                model.initialize_model(historical_data_subset, decay_factor=0.5)
                
                for match in matches:
                    pred = model.predict_match(match['home_team'], match['away_team'])
                    actual_result = self._get_actual_result(match)
                    predicted_result = self._get_predicted_result(pred)
                    
                    validation_results.append({
                        'date': match['date'],
                        'match': f"{match['home_team']} vs {match['away_team']}",
                        'model': model_name,
                        'predicted': predicted_result,
                        'actual': actual_result,
                        'accurate': predicted_result == actual_result
                    })
            
            # Update validation table
            self.validation_table.setRowCount(len(validation_results))
            for i, result in enumerate(validation_results):
                self.validation_table.setItem(i, 0, QTableWidgetItem(result['date']))
                self.validation_table.setItem(i, 1, QTableWidgetItem(result['match']))
                self.validation_table.setItem(i, 2, QTableWidgetItem(f"{result['model']}: {result['predicted']}"))
                self.validation_table.setItem(i, 3, QTableWidgetItem(result['actual']))
                self.validation_table.setItem(i, 4, QTableWidgetItem('✓' if result['accurate'] else '✗'))
            
            # Calculate and display statistics
            accuracy_by_model = {}
            for model_name in models.keys():
                model_results = [r for r in validation_results if r['model'] == model_name]
                accuracy = sum(1 for r in model_results if r['accurate']) / len(model_results)
                accuracy_by_model[model_name] = accuracy
            
            stats_text = "Statistiche di validazione:\n"
            for model_name, accuracy in accuracy_by_model.items():
                stats_text += f"{model_name}: {accuracy:.2%} accuratezza\n"
            
            self.validation_stats_label.setText(stats_text)
            
        except Exception as e:
            QMessageBox.warning(self, "Errore", str(e))

    def _get_actual_result(self, match):
        if match['home_goals'] > match['away_goals']:
            return 'home_win'
        elif match['home_goals'] < match['away_goals']:
            return 'away_win'
        else:
            return 'draw'

    def _get_predicted_result(self, prediction):
        probs = {
            'home_win': prediction['home_win'],
            'draw': prediction['draw'],
            'away_win': prediction['away_win']
        }
        return max(probs.items(), key=lambda x: x[1])[0]

    def _calculate_metrics(self, predictions):
        total = len(predictions)
        correct = sum(1 for p in predictions if p['predicted'] == p['actual'])
        
        # Calculate precision and recall for each outcome
        outcomes = ['home_win', 'draw', 'away_win']
        precision = {}
        recall = {}
        
        for outcome in outcomes:
            predicted_outcome = sum(1 for p in predictions if p['predicted'] == outcome)
            actual_outcome = sum(1 for p in predictions if p['actual'] == outcome)
            true_positive = sum(1 for p in predictions if p['predicted'] == outcome and p['actual'] == outcome)
            
            precision[outcome] = true_positive / predicted_outcome if predicted_outcome > 0 else 0
            recall[outcome] = true_positive / actual_outcome if actual_outcome > 0 else 0
        
        return {
            'accuracy': correct / total,
            'precision': sum(precision.values()) / len(precision),
            'recall': sum(recall.values()) / len(recall)
        }

    def _get_recent_matches(self, n):
        """Get the n most recent matches from historical data"""
        sorted_matches = sorted(
            self.historical_data,
            key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'),
            reverse=True
        )
        return sorted_matches[:n]

def main():
    app = QApplication(sys.argv)
    window = PredictorGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()