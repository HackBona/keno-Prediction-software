import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, timezone
import argparse
import csv
import os
import time
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem, 
                             QLabel, QVBoxLayout, QHBoxLayout, QWidget, QHeaderView, 
                             QGroupBox, QPushButton, QTextEdit, QProgressBar, QScrollArea,
                             QDialog, QMessageBox, QFormLayout, QTabWidget)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import joblib
from scipy.stats import poisson
from sklearn.multioutput import MultiOutputClassifier

# Command-line args
parser = argparse.ArgumentParser()
parser.add_argument("--uri", type=str, default="wss://keno-nrg1.atlas-v.com/gamesocket/", help="WebSocket URI")
parser.add_argument("--history", type=int, default=50, help="Historical draws to analyze")
parser.add_argument("--hot-lookback", type=int, default=15, help="Lookback for hot numbers")
parser.add_argument("--cold-threshold", type=int, default=10, help="Threshold for cold numbers")
parser.add_argument("--model", type=str, default="xgboost", choices=["rf", "gb", "mlp", "xgboost"], help="ML model to use")
args = parser.parse_args()

class KenoPredictor:
    def __init__(self):
        self.draw_history = []
        self.number_stats = defaultdict(lambda: {
            'total_count': 0,
            'hot_streak': 0,
            'cold_streak': 0,
            'last_seen': 0,
            'pair_counts': defaultdict(int),
            'position_counts': defaultdict(int),
            'recent_hit': 0,
            'exp_smooth': 0.5,
            'cluster_value': 0.0,
            'positional_value': 0.0,
            'probability_score': 0.0
        })
        self.pair_frequencies = defaultdict(int)
        self.triplet_frequencies = defaultdict(int)
        self.model = None
        self.scaler = StandardScaler()
        self.prediction_history = []
        self.position_stats = [[] for _ in range(20)]
        self.performance_stats = {
            'total_games': 0,
            'correct_4': 0,
            'correct_3': 0,
            'correct_2': 0,
            'hit_rates': defaultdict(list),
            'last_10_hits': [],
            'consecutive_hits': 0,
            'max_consecutive': 0,
            'recovery_mode': False,
            'recovery_counter': 0,
            'hot_zone_hits': 0,
            'cold_zone_hits': 0,
            'edge_hits': 0,
            'win_pattern_state': 0  # Track state in the win pattern
        }
        self.strategy_weights = {
            'hot': 1.0,
            'cold': 1.2,
            'pairs': 1.0,
            'ml': 1.0,
            'patterns': 1.0,
            'position': 1.3,
            'streak': 1.0,
            'sequence': 1.0,
            'gap': 1.4,
            'position_momentum': 1.3,
            'cluster': 1.5,
            'probability': 1.8,
            'edge': 1.2
        }
        self.model_performance = defaultdict(list)
        self.last_update_time = time.time()
        self.consistency_factor = 1.0
        self.last_hits = 0
        
        # Define hot and cold zones
        self.hot_zones = [(1, 10), (21, 30), (41, 50), (61, 70)]
        self.cold_zones = [(11, 20), (31, 40), (51, 60), (71, 80)]
        self.edge_numbers = [1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71, 80]
        
        # CSV storage setup
        self.draw_history_file = "keno_draws.csv"
        self.prediction_history_file = "keno_predictions.csv"
        self.model_file = "keno_model.pkl"
        self.initialize_csv_files()
        self.load_historical_data()
        
        # Enhanced ML tracking
        self.model_versions = []  # Track model versions for rollback
        self.model_version = 1
        self.feature_importance = {}  # Track feature importance
        self.adaptive_learning_rate = 0.01  # Dynamic learning rate
        self.pattern_analysis = {}  # Pattern recognition data
        self.number_clusters = self.initialize_clusters()  # Number grouping
        
    def initialize_clusters(self):
        """Initialize number clusters for pattern analysis"""
        return {
            'low': list(range(1, 21)),
            'mid_low': list(range(21, 41)),
            'mid_high': list(range(41, 61)),
            'high': list(range(61, 81)),
            'edges': [1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71, 80],
            'primes': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
        }
    
    def update_pattern_analysis(self, new_draw):
        """Update pattern analysis with new draw data"""
        # Cluster distribution
        for cluster_name, numbers in self.number_clusters.items():
            count = sum(1 for num in new_draw if num in numbers)
            self.pattern_analysis.setdefault(cluster_name, []).append(count)
            if len(self.pattern_analysis[cluster_name]) > 50:
                self.pattern_analysis[cluster_name].pop(0)
        
        # Number sequences
        sorted_draw = sorted(new_draw)
        for i in range(len(sorted_draw) - 1):
            diff = sorted_draw[i+1] - sorted_draw[i]
            self.pattern_analysis.setdefault('gaps', []).append(diff)
            if len(self.pattern_analysis['gaps']) > 100:
                self.pattern_analysis['gaps'].pop(0)
                
        # Odd/even ratio
        odd_count = sum(1 for num in new_draw if num % 2 == 1)
        self.pattern_analysis.setdefault('odd_even', []).append(odd_count / len(new_draw))
        if len(self.pattern_analysis['odd_even']) > 50:
            self.pattern_analysis['odd_even'].pop(0)
    
    def initialize_csv_files(self):
        """Create CSV files with headers if they don't exist"""
        if not os.path.exists(self.draw_history_file):
            with open(self.draw_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['game_id', 'timestamp', 'numbers'])
        
        if not os.path.exists(self.prediction_history_file):
            with open(self.prediction_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['game_id', 'predicted', 'actual', 'hits'])

    def save_draw(self, game_id, timestamp, numbers):
        """Save draw data to CSV"""
        with open(self.draw_history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([game_id, timestamp, json.dumps(numbers)])

    def save_prediction(self, game_id, predicted, actual, hits):
        """Save prediction data to CSV"""
        with open(self.prediction_history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([game_id, json.dumps(predicted), json.dumps(actual), hits])
            
    def save_model(self):
        """Save trained model to file"""
        if self.model:
            joblib.dump(self.model, self.model_file)
            
    def load_model(self):
        """Load trained model from file"""
        if os.path.exists(self.model_file):
            try:
                self.model = joblib.load(self.model_file)
                return True
            except:
                pass
        return False
            
    def load_historical_data(self):
        """Load historical data from CSV files"""
        try:
            # Load draw history
            if os.path.exists(self.draw_history_file):
                with open(self.draw_history_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['numbers']:
                            numbers = json.loads(row['numbers'])
                            if numbers not in self.draw_history:
                                self.update_stats(numbers, is_historical=True)
            
            # Load prediction history
            if os.path.exists(self.prediction_history_file):
                with open(self.prediction_history_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        game_id = row['game_id']
                        predicted = json.loads(row['predicted'])
                        actual = json.loads(row['actual'])
                        hits = int(row['hits'])
                        
                        # Update performance stats
                        self.performance_stats['total_games'] += 1
                        if hits == 4:
                            self.performance_stats['correct_4'] += 1
                        elif hits == 3:
                            self.performance_stats['correct_3'] += 1
                        elif hits == 2:
                            self.performance_stats['correct_2'] += 1
                            
                        # Add to prediction history
                        self.prediction_history.append((game_id, predicted, actual, hits))
                        
                        # Update consecutive hits
                        if hits == 4:
                            self.performance_stats['consecutive_hits'] += 1
                            if self.performance_stats['consecutive_hits'] > self.performance_stats['max_consecutive']:
                                self.performance_stats['max_consecutive'] = self.performance_stats['consecutive_hits']
                        else:
                            self.performance_stats['consecutive_hits'] = 0
                            
            # Update win pattern state based on last prediction
            if self.prediction_history:
                last_pred = self.prediction_history[-1][1]  # Get last prediction
                if len(last_pred) == 0:
                    self.performance_stats['win_pattern_state'] = 1
                else:
                    self.performance_stats['win_pattern_state'] = 2
                    
        except Exception as e:
            print(f"Error loading historical data: {e}")
    
    def update_stats(self, new_draw, is_historical=False):
        """Update statistics with new draw data"""
        self.draw_history.append(new_draw)
        if not is_historical:
            self.performance_stats['total_games'] += 1
        
        sorted_draw = sorted(new_draw)
        
        # Update number statistics
        for num in range(1, 81):
            # Update streaks
            if num in new_draw:
                self.number_stats[num]['hot_streak'] += 1
                self.number_stats[num]['cold_streak'] = 0
                self.number_stats[num]['last_seen'] = 0
                self.number_stats[num]['recent_hit'] += 1
                # Update exponential smoothing
                alpha = 0.7
                self.number_stats[num]['exp_smooth'] = alpha * 1 + (1 - alpha) * self.number_stats[num]['exp_smooth']
            else:
                self.number_stats[num]['hot_streak'] = 0
                self.number_stats[num]['cold_streak'] += 1
                self.number_stats[num]['last_seen'] += 1
                self.number_stats[num]['recent_hit'] = max(0, self.number_stats[num]['recent_hit'] - 0.5)
                # Update exponential smoothing
                alpha = 0.7
                self.number_stats[num]['exp_smooth'] = alpha * 0 + (1 - alpha) * self.number_stats[num]['exp_smooth']
                
            # Update total count
            if num in new_draw:
                self.number_stats[num]['total_count'] += 1
                
            # Update position stats
            if num in sorted_draw:
                position = sorted_draw.index(num)
                self.number_stats[num]['position_counts'][position] += 1
                
                # Update positional value (lower positions are better)
                self.number_stats[num]['positional_value'] = (1.0 - (position / 20.0)) * 0.5 + \
                                                           self.number_stats[num]['positional_value'] * 0.5
        
        # Update pair frequencies
        for i in range(len(sorted_draw)):
            for j in range(i+1, len(sorted_draw)):
                pair = (sorted_draw[i], sorted_draw[j])
                self.pair_frequencies[pair] += 1
                # Update individual number pair stats
                self.number_stats[sorted_draw[i]]['pair_counts'][sorted_draw[j]] += 1
                self.number_stats[sorted_draw[j]]['pair_counts'][sorted_draw[i]] += 1
                
        # Update triplet frequencies
        for i in range(len(sorted_draw)):
            for j in range(i+1, len(sorted_draw)):
                for k in range(j+1, len(sorted_draw)):
                    triplet = (sorted_draw[i], sorted_draw[j], sorted_draw[k])
                    self.triplet_frequencies[triplet] += 1
        
        # Track position patterns
        for position, number in enumerate(sorted_draw):
            if position < 20:
                self.position_stats[position].append(number)
                
        # Update cluster values
        self.update_cluster_stats(sorted_draw)
        
        # Update zone stats
        self.update_zone_stats(new_draw)
        
        # Update pattern analysis
        self.update_pattern_analysis(new_draw)
        
        # Trim history
        if len(self.draw_history) > args.history:
            old_draw = self.draw_history.pop(0)
            # Decrement counts for old draw
            for num in old_draw:
                self.number_stats[num]['total_count'] = max(0, self.number_stats[num]['total_count'] - 1)
    
    def update_cluster_stats(self, sorted_draw):
        """Update cluster statistics for numbers"""
        # Identify clusters in the current draw
        clusters = []
        current_cluster = []
        
        for i in range(1, len(sorted_draw)):
            if sorted_draw[i] - sorted_draw[i-1] <= 2:  # Numbers within 2 of each other
                if not current_cluster:
                    current_cluster.append(sorted_draw[i-1])
                current_cluster.append(sorted_draw[i])
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
        
        if current_cluster:
            clusters.append(current_cluster)
            
        # Update cluster value for numbers in clusters
        for cluster in clusters:
            if len(cluster) >= 3:  # Only consider clusters of 3 or more
                for num in cluster:
                    # Increase cluster value based on cluster size
                    self.number_stats[num]['cluster_value'] = min(1.0, self.number_stats[num]['cluster_value'] + 0.1 * len(cluster))
        
        # Decay cluster values for all numbers
        for num in range(1, 81):
            self.number_stats[num]['cluster_value'] = max(0, self.number_stats[num]['cluster_value'] * 0.9)
    
    def update_zone_stats(self, draw):
        """Update hot/cold zone statistics"""
        # Check hot zones
        in_hot_zone = False
        for start, end in self.hot_zones:
            if any(start <= num <= end for num in draw):
                in_hot_zone = True
                break
                
        # Check cold zones
        in_cold_zone = False
        for start, end in self.cold_zones:
            if any(start <= num <= end for num in draw):
                in_cold_zone = True
                break
                
        # Check edge numbers
        edge_count = sum(1 for num in draw if num in self.edge_numbers)
        
        # Update performance stats
        if in_hot_zone:
            self.performance_stats['hot_zone_hits'] += 1
        if in_cold_zone:
            self.performance_stats['cold_zone_hits'] += 1
        if edge_count >= 2:
            self.performance_stats['edge_hits'] += 1
    
    def predict_numbers(self):
        """Predict next numbers using multiple strategies"""
        if len(self.draw_history) < 5:
            return []
        
        # Calculate probability scores for all numbers
        self.calculate_probability_scores()
        
        # 1. Hot numbers (frequent in recent draws)
        hot_numbers = self.get_hot_numbers()
        
        # 2. Cold numbers (overdue to appear)
        cold_numbers = self.get_cold_numbers()
        
        # 3. Pair analysis (frequently co-occurring numbers)
        pair_numbers = self.get_pair_based_numbers()
        
        # 4. Machine learning prediction
        ml_numbers = self.enhanced_ml_predict()
        
        # 5. Pattern recognition
        pattern_numbers = self.pattern_recognition()
        
        # 6. Positional bias
        positional_numbers = self.get_positional_bias()
        
        # 7. Streak analysis
        streak_numbers = self.get_streak_sensitive_numbers()
        
        # 8. Sequence prediction
        sequence_numbers = self.sequence_prediction()
        
        # 9. Gap analysis
        gap_numbers = self.gap_analysis()
        
        # 10. Positional momentum
        momentum_numbers = self.get_positional_momentum()
        
        # 11. Cluster numbers
        cluster_numbers = self.get_cluster_numbers()
        
        # 12. Edge numbers
        edge_numbers = self.get_edge_numbers()
        
        # 13. Probability-based numbers
        probability_numbers = self.get_probability_based_numbers()
        
        # Update strategy weights
        self.calculate_strategy_weights()
        
        # Combine strategies with reinforcement learning
        candidates = Counter()
        
        # Weighted voting system with consistency factor
        for num in hot_numbers[:6]:
            candidates[num] += 2.0 * self.strategy_weights['hot'] * self.consistency_factor
        for num in cold_numbers[:3]:
            candidates[num] += 1.8 * self.strategy_weights['cold'] * self.consistency_factor
        for num in pair_numbers[:4]:
            candidates[num] += 1.2 * self.strategy_weights['pairs'] * self.consistency_factor
        for num in ml_numbers[:5]:
            candidates[num] += 1.8 * self.strategy_weights['ml'] * self.consistency_factor
        for num in pattern_numbers[:4]:
            candidates[num] += 1.5 * self.strategy_weights['patterns'] * self.consistency_factor
        for num in positional_numbers[:3]:
            candidates[num] += 1.5 * self.strategy_weights['position'] * self.consistency_factor
        for num in streak_numbers[:2]:
            candidates[num] += 1.7 * self.strategy_weights['streak'] * self.consistency_factor
        for num in sequence_numbers[:3]:
            candidates[num] += 2.0 * self.strategy_weights['sequence'] * self.consistency_factor
        for num in gap_numbers[:3]:
            candidates[num] += 1.6 * self.strategy_weights['gap'] * self.consistency_factor
        for num in momentum_numbers[:2]:
            candidates[num] += 1.2 * self.strategy_weights['position_momentum'] * self.consistency_factor
        for num in cluster_numbers[:4]:
            candidates[num] += 1.8 * self.strategy_weights['cluster'] * self.consistency_factor
        for num in edge_numbers[:2]:
            candidates[num] += 1.3 * self.strategy_weights['edge'] * self.consistency_factor
        for num in probability_numbers[:6]:
            candidates[num] += 2.5 * self.strategy_weights['probability'] * self.consistency_factor
            
        # Apply recent hit bonus
        for num in self.get_recent_hit_numbers():
            candidates[num] += 0.8 * self.consistency_factor
            
        # Get top candidates
        top_candidates = [num for num, _ in candidates.most_common(15)]
        
        # Apply advantage number selection
        final_prediction = self.select_advantage_numbers(top_candidates)
        
        # Apply win pattern strategy: |0||4|4|4|4|
        current_state = self.performance_stats['win_pattern_state']
        
        if current_state == 0:
            # State 0: predict 0 numbers
            self.performance_stats['win_pattern_state'] = 1
            return []
        elif current_state == 1:
            # State 1: predict 4 numbers
            self.performance_stats['win_pattern_state'] = 2
            return final_prediction[:4]
        else:
            # State 2: predict 4 numbers
            return final_prediction[:4]
    
    def calculate_probability_scores(self):
        """Calculate advanced probability scores for all numbers"""
        total_draws = len(self.draw_history)
        if total_draws < 10:
            return
            
        for num in range(1, 81):
            stats = self.number_stats[num]
            
            # Base probability (historical frequency)
            base_prob = stats['total_count'] / total_draws
            
            # Recent appearance probability (last 20 draws)
            recent_count = sum(1 for draw in self.draw_history[-20:] if num in draw)
            recent_prob = recent_count / 20.0
            
            # Streak-based probability
            streak_prob = 0
            if stats['hot_streak'] >= 2:
                streak_prob = min(0.5, stats['hot_streak'] * 0.1)
            elif stats['last_seen'] > args.cold_threshold:
                streak_prob = min(0.4, (stats['last_seen'] - args.cold_threshold) * 0.05)
                
            # Positional probability
            pos_prob = stats['positional_value']
            
            # Cluster probability
            cluster_prob = stats['cluster_value']
            
            # Pattern probability
            pattern_prob = self.get_pattern_adjustments().get(num, 0.5)
            
            # Combine probabilities
            combined_prob = (
                0.25 * base_prob +
                0.25 * recent_prob +
                0.1 * streak_prob +
                0.1 * pos_prob +
                0.1 * cluster_prob +
                0.2 * pattern_prob
            )
            
            # Apply Poisson distribution for expectation
            expected_count = combined_prob * total_draws
            poisson_prob = poisson.pmf(stats['last_seen'], expected_count) if expected_count > 0 else 0
            
            # Final probability score
            stats['probability_score'] = combined_prob * (1 + poisson_prob)
    
    def get_probability_based_numbers(self):
        """Get numbers based on probability scores"""
        return sorted(range(1, 81), key=lambda n: self.number_stats[n]['probability_score'], reverse=True)
    
    def get_cluster_numbers(self):
        """Get numbers with high cluster values"""
        return sorted(range(1, 81), key=lambda n: self.number_stats[n]['cluster_value'], reverse=True)
    
    def get_edge_numbers(self):
        """Get edge numbers with enhanced weighting"""
        edge_scores = {}
        for num in self.edge_numbers:
            # Base edge bonus
            score = 1.5
            
            # Add position bonus if number appears in early positions
            if self.number_stats[num]['position_counts']:
                min_pos = min(self.number_stats[num]['position_counts'].keys())
                if min_pos < 5:
                    score += 1.0 - (min_pos / 10.0)
            
            # Add recent appearance bonus
            if self.number_stats[num]['last_seen'] < 5:
                score += 0.5
                
            edge_scores[num] = score
            
        return sorted(edge_scores, key=edge_scores.get, reverse=True)
    
    def select_advantage_numbers(self, candidates):
        """Select final numbers using advantage number criteria"""
        # Sort by probability score
        candidates = sorted(candidates, key=lambda n: self.number_stats[n]['probability_score'], reverse=True)
        
        # Apply diversity filter - ensure numbers from different zones
        final_selection = []
        zones_covered = set()
        
        for num in candidates:
            # Determine which zone the number belongs to
            zone = None
            for start, end in self.hot_zones:
                if start <= num <= end:
                    zone = 'hot'
                    break
            if not zone:
                for start, end in self.cold_zones:
                    if start <= num <= end:
                        zone = 'cold'
                        break
            
            # If we don't have this zone yet, add the number
            if zone and zone not in zones_covered:
                final_selection.append(num)
                zones_covered.add(zone)
            elif not zone:  # Edge numbers
                if 'edge' not in zones_covered:
                    final_selection.append(num)
                    zones_covered.add('edge')
            
            if len(final_selection) >= 4:
                break
        
        # If we don't have 4 numbers, add top remaining candidates
        if len(final_selection) < 4:
            remaining = [n for n in candidates if n not in final_selection]
            final_selection.extend(remaining[:4 - len(final_selection)])
        
        return final_selection
    
    def get_recent_hit_numbers(self):
        """Get numbers that have hit recently in predictions"""
        recent_hits = Counter()
        lookback = min(10, len(self.prediction_history))
        
        for _, predicted, actual, hits in self.prediction_history[-lookback:]:
            if hits >= 2:
                for num in predicted:
                    if num in actual:
                        recent_hits[num] += 1
        
        return [num for num, _ in recent_hits.most_common(5)]
    
    def calculate_strategy_weights(self):
        """Calculate weights for each strategy based on historical performance"""
        weights = {
            'hot': 1.0,
            'cold': 1.2,
            'pairs': 1.0,
            'ml': 1.0,
            'patterns': 1.0,
            'position': 1.3,
            'streak': 1.0,
            'sequence': 1.0,
            'gap': 1.4,
            'position_momentum': 1.3,
            'cluster': 1.5,
            'probability': 1.8,
            'edge': 1.2
        }
        
        # Update consistency factor based on streak
        if self.performance_stats['consecutive_hits'] >= 3:
            self.consistency_factor = min(1.5, 1.0 + (self.performance_stats['consecutive_hits'] * 0.1))
        else:
            self.consistency_factor = 1.0
        
        # Recovery mode activation
        if self.last_hits == 0:
            self.performance_stats['recovery_mode'] = True
            self.performance_stats['recovery_counter'] = 2
            
        if self.performance_stats['recovery_mode']:
            weights['cold'] = 1.5
            weights['probability'] = 2.0
            weights['cluster'] = 1.8
            self.performance_stats['recovery_counter'] -= 1
            if self.performance_stats['recovery_counter'] <= 0:
                self.performance_stats['recovery_mode'] = False
        
        # Analyze recent predictions to adjust weights
        if len(self.prediction_history) >= 10:
            strategy_success = {key: [] for key in weights}
            
            for _, predicted, actual, hits in self.prediction_history[-10:]:
                if hits >= 2:
                    for num in predicted:
                        # Check which strategy would have recommended this number
                        if num in self.get_hot_numbers()[:6]:
                            strategy_success['hot'].append(1)
                        if num in self.get_cold_numbers()[:3]:
                            strategy_success['cold'].append(1)
                        if num in self.get_pair_based_numbers()[:4]:
                            strategy_success['pairs'].append(1)
                        if num in self.enhanced_ml_predict()[:5]:
                            strategy_success['ml'].append(1)
                        if num in self.pattern_recognition()[:4]:
                            strategy_success['patterns'].append(1)
                        if num in self.get_positional_bias()[:3]:
                            strategy_success['position'].append(1)
                        if num in self.get_streak_sensitive_numbers()[:2]:
                            strategy_success['streak'].append(1)
                        if num in self.sequence_prediction()[:3]:
                            strategy_success['sequence'].append(1)
                        if num in self.gap_analysis()[:3]:
                            strategy_success['gap'].append(1)
                        if num in self.get_positional_momentum()[:2]:
                            strategy_success['position_momentum'].append(1)
                        if num in self.get_cluster_numbers()[:4]:
                            strategy_success['cluster'].append(1)
                        if num in self.get_probability_based_numbers()[:6]:
                            strategy_success['probability'].append(1)
                        if num in self.get_edge_numbers()[:2]:
                            strategy_success['edge'].append(1)
            
            # Calculate success rates
            for key in weights:
                if strategy_success[key]:
                    weights[key] = sum(strategy_success[key]) / len(strategy_success[key])
                else:
                    weights[key] = self.strategy_weights[key]  # Maintain current weight
        
        # Apply minimum weights and update
        for key in weights:
            self.strategy_weights[key] = max(0.7, weights[key])
            
        return self.strategy_weights

    def pattern_recognition(self):
        """Identify patterns from historical data"""
        pattern_scores = Counter()
        
        # Analyze number sequences
        for i in range(len(self.draw_history)-1):
            prev_draw = set(self.draw_history[i])
            current_draw = set(self.draw_history[i+1])
            common = prev_draw & current_draw
            for num in common:
                pattern_scores[num] += 1
                
        # Analyze positional patterns
        position_counts = [Counter() for _ in range(20)]
        for draw in self.draw_history:
            sorted_draw = sorted(draw)
            for pos, num in enumerate(sorted_draw):
                position_counts[pos][num] += 1
        
        # Add positional frequency to scores
        for pos_counter in position_counts:
            for num, count in pos_counter.items():
                pattern_scores[num] += count / len(self.draw_history)
        
        return [num for num, _ in pattern_scores.most_common()]
    
    def sequence_prediction(self):
        """Predict numbers based on sequence patterns"""
        sequence_scores = Counter()
        
        # Analyze number sequences in recent draws
        if len(self.draw_history) > 5:
            recent_draws = self.draw_history[-5:]
            
            # Look for numbers that follow a sequence pattern
            for i in range(1, 80):
                # Check for arithmetic sequences
                if all(i in draw or i+1 in draw or i+2 in draw for draw in recent_draws[-3:]):
                    sequence_scores[i] += 1.5
                    sequence_scores[i+1] += 1.2
                    sequence_scores[i+2] += 1.0
                
                # Check for clusters
                if all(any(j in draw for j in range(i, i+5)) for draw in recent_draws[-2:]):
                    for j in range(i, i+5):
                        sequence_scores[j] += 1.3
        
        return [num for num, _ in sequence_scores.most_common()]
    
    def get_positional_bias(self):
        """Find numbers that appear frequently in specific positions"""
        position_scores = Counter()
        
        # Analyze each position (1st number drawn, 2nd, etc.)
        for position in range(10):  # Focus on first 10 positions
            if len(self.position_stats[position]) > 3:
                # Find most common numbers in this position
                counter = Counter(self.position_stats[position])
                for num, count in counter.items():
                    # Higher weight for consistent positioning
                    position_scores[num] += count * (1 + 0.3 * (10 - position))
                    
        return [num for num, _ in position_scores.most_common()]
    
    def get_streak_sensitive_numbers(self):
        """Identify numbers where streaks are likely to end"""
        streak_scores = {}
        
        for num in range(1, 81):
            stats = self.number_stats[num]
            # Hot numbers with long streaks
            if stats['hot_streak'] >= 3:
                streak_scores[num] = 1.0 + min(3.0, stats['hot_streak'] * 0.3)
            # Cold numbers with long absences
            elif stats['last_seen'] >= args.cold_threshold + 3:
                streak_scores[num] = 1.7 + min(2.0, (stats['last_seen'] - args.cold_threshold) * 0.15)
            # Numbers with medium absence
            elif args.cold_threshold - 3 <= stats['last_seen'] <= args.cold_threshold + 3:
                streak_scores[num] = 2.0
        
        return sorted(streak_scores, key=streak_scores.get, reverse=True)
    
    def get_hot_numbers(self):
        """Get hot numbers based on recent frequency and streaks"""
        # Calculate recent frequency (last N draws)
        recent_counts = Counter()
        lookback = min(args.hot_lookback, len(self.draw_history))
        for draw in self.draw_history[-lookback:]:
            for num in draw:
                recent_counts[num] += 1
        
        # Combine with hot streaks
        hot_scores = {}
        for num in range(1, 81):
            streak_bonus = min(5, self.number_stats[num]['hot_streak']) * 0.3
            hot_scores[num] = recent_counts[num] * (1 + streak_bonus)
        
        return sorted(hot_scores, key=hot_scores.get, reverse=True)
    
    def get_cold_numbers(self):
        """Enhanced cold number detection with probability modeling"""
        cold_scores = {}
        max_absence = max(self.number_stats[num]['last_seen'] for num in range(1, 81))
        
        for num in range(1, 81):
            absence = self.number_stats[num]['last_seen']
            if absence > args.cold_threshold:
                # Base score based on absence duration
                base_score = absence / max_absence
                
                # Frequency adjustment
                freq_weight = min(1.0, self.number_stats[num]['total_count'] / len(self.draw_history))
                
                # Streak factor
                streak_factor = 1 + min(2.0, (self.number_stats[num]['cold_streak'] - args.cold_threshold) * 0.1)
                
                # Positional bias
                pos_bonus = 0
                if self.number_stats[num]['position_counts']:
                    common_pos = min(self.number_stats[num]['position_counts'].keys())
                    if common_pos < 5:
                        pos_bonus = 0.5 * (1 - common_pos/10)
                
                # Probability boost
                prob_boost = self.number_stats[num]['probability_score'] * 0.5
                
                cold_scores[num] = (base_score * (0.7 + 0.3 * freq_weight) * streak_factor + 
                                   pos_bonus + prob_boost)
        
        return sorted(cold_scores, key=cold_scores.get, reverse=True)
    
    def get_pair_based_numbers(self):
        """Get numbers based on pair frequencies"""
        # Get most common pairs
        top_pairs = Counter(self.pair_frequencies).most_common(15)
        
        # Extract numbers from top pairs
        pair_numbers = Counter()
        for pair, count in top_pairs:
            pair_numbers[pair[0]] += count
            pair_numbers[pair[1]] += count
        
        return [num for num, _ in pair_numbers.most_common()]
    
    def gap_analysis(self):
        """Identify numbers due to appear based on gap patterns"""
        gap_scores = {}
        
        for num in range(1, 81):
            # Calculate average gap between appearances
            if self.number_stats[num]['total_count'] > 3:
                gaps = []
                last_seen = -1
                for i, draw in enumerate(self.draw_history):
                    if num in draw:
                        if last_seen >= 0:
                            gaps.append(i - last_seen)
                        last_seen = i
                
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    current_gap = self.number_stats[num]['last_seen']
                    
                    # Score based on how much current gap exceeds average
                    gap_scores[num] = min(5.0, current_gap / max(1, avg_gap))
        
        return sorted(gap_scores, key=gap_scores.get, reverse=True)
    
    def get_positional_momentum(self):
        """Numbers moving toward favorable positions"""
        momentum_scores = Counter()
        
        # Track position changes
        for num in range(1, 81):
            if len(self.position_stats[0]) > 10:
                # Get last 5 positions where number appeared
                positions = [i for i, pos_list in enumerate(self.position_stats) 
                            if num in pos_list[-5:]]
                if len(positions) > 2:
                    # Calculate position trend (lower positions are better)
                    trend = np.polyfit(range(len(positions)), positions, 1)[0]
                    momentum_scores[num] = -trend * 5
                    
        return [num for num, _ in momentum_scores.most_common()]
    
    def prepare_ml_data(self):
        """Prepare data for machine learning model with enhanced features"""
        if len(self.draw_history) < 10:
            return None, None
        
        # Create features: for each number, its recent frequency, last seen, etc.
        X = []
        y = []
        
        # We'll predict if a number will appear in the next draw
        for i in range(10, len(self.draw_history)):
            features = []
            for num in range(1, 81):
                # Recent frequency (last 5, 10, 20 draws)
                recent_5 = sum(1 for draw in self.draw_history[i-5:i] if num in draw)
                recent_10 = sum(1 for draw in self.draw_history[i-10:i] if num in draw)
                recent_20 = sum(1 for draw in self.draw_history[i-20:i] if num in draw)
                
                # Last seen
                last_seen = next((j for j in range(1, 11) if num in self.draw_history[i-j]), 10)
                
                # Pair frequency with last draw's numbers
                pair_freq = sum(self.number_stats[num]['pair_counts'][n] for n in self.draw_history[i-1])
                
                # Position stability
                pos_stability = 0
                if self.number_stats[num]['position_counts']:
                    max_pos = max(self.number_stats[num]['position_counts'].values())
                    pos_stability = max_pos / sum(self.number_stats[num]['position_counts'].values())
                
                # Recent hit bonus
                recent_hit = self.number_stats[num]['recent_hit']
                
                # Moving averages
                ma_3 = sum(1 for draw in self.draw_history[i-3:i] if num in draw) / 3
                ma_5 = recent_5 / 5
                ma_10 = recent_10 / 10
                
                # Exponential smoothing
                exp_smooth = self.number_stats[num]['exp_smooth']
                
                # Positional momentum
                pos_momentum = 0
                if len(self.position_stats[0]) > 5 and num in self.position_stats[0][-5:]:
                    pos_momentum = 1 - (self.position_stats[0][-5:].index(num) / 5)
                
                # Cluster value
                cluster_value = self.number_stats[num]['cluster_value']
                
                # Positional value
                positional_value = self.number_stats[num]['positional_value']
                
                features.extend([
                    recent_5, recent_10, recent_20, last_seen, pair_freq, 
                    pos_stability, recent_hit, ma_3, ma_5, ma_10, 
                    exp_smooth, pos_momentum, cluster_value, positional_value
                ])
            
            # Target: if the number is in the next draw
            target = [1 if num in self.draw_history[i] else 0 for num in range(1, 81)]
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train_model(self, incremental=False):
        """Enhanced model training with adaptive learning"""
        X, y = self.prepare_ml_data()
        if X is None or len(X) < 10:
            return
        
        # Create base model
        if args.model == "rf":
            base_model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
        elif args.model == "gb":
            base_model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=self.adaptive_learning_rate,
                max_depth=5,
                random_state=42
            )
        elif args.model == "mlp":
            base_model = MLPClassifier(
                hidden_layer_sizes=(200, 100),
                max_iter=1000,
                learning_rate_init=self.adaptive_learning_rate,
                early_stopping=True,
                random_state=42
            )
        elif args.model == "xgboost":
            base_model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=self.adaptive_learning_rate,
                max_depth=6,
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            )
        
        # Wrap base model in MultiOutputClassifier
        new_model = make_pipeline(
            StandardScaler(),
            MultiOutputClassifier(base_model)
        )
        
        # Train the model
        if incremental and hasattr(self.model, 'partial_fit'):
            try:
                self.model.partial_fit(X, y)
                print("Incremental training completed")
                return
            except Exception as e:
                print(f"Incremental training failed: {e}. Performing full training.")
                
        # Full training
        new_model.fit(X, y)
        
        # Save previous model for rollback
        if self.model:
            self.model_versions.append((self.model, self.model_version))
            if len(self.model_versions) > 3:  # Keep only last 3 versions
                self.model_versions.pop(0)
        
        # Update model and version
        self.model = new_model
        self.model_version += 1
        
        # Extract feature importance if available
        if hasattr(base_model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.get_feature_names(),
                base_model.feature_importances_
            ))
        elif hasattr(base_model, 'coef_'):
            # For linear models, use absolute coefficients
            self.feature_importance = dict(zip(
                self.get_feature_names(),
                np.mean(np.abs(base_model.coef_), axis=0)
            ))
        
        self.save_model()
        print(f"Model updated to version {self.model_version}")
        
        # Adjust learning rate based on recent performance
        self.adjust_learning_rate()
    
    def adjust_learning_rate(self):
        """Adjust learning rate based on recent performance"""
        if len(self.prediction_history) < 10:
            return
            
        # Calculate average hits in last 10 predictions
        avg_hits = sum(hits for _, _, _, hits in self.prediction_history[-10:]) / 10
        
        if avg_hits < 2.0:  # Underperforming
            self.adaptive_learning_rate = min(0.1, self.adaptive_learning_rate * 1.3)
            print(f"Increasing learning rate to {self.adaptive_learning_rate:.4f}")
        elif avg_hits > 2.5:  # Good performance
            self.adaptive_learning_rate = max(0.001, self.adaptive_learning_rate * 0.8)
            print(f"Decreasing learning rate to {self.adaptive_learning_rate:.4f}")
    
    def get_feature_names(self):
        """Get feature names for importance tracking"""
        return [
            f"Num_{num}_" + feat 
            for num in range(1, 81) 
            for feat in [
                'recent_5', 'recent_10', 'recent_20', 'last_seen', 'pair_freq',
                'pos_stability', 'recent_hit', 'ma_3', 'ma_5', 'ma_10',
                'exp_smooth', 'pos_momentum', 'cluster_value', 'positional_value'
            ]
        ]
    
    def enhanced_ml_predict(self):
        """Enhanced prediction with ensemble and pattern analysis"""
        # Get base ML prediction
        ml_prediction = self.ml_predict()
        
        # Apply pattern-based adjustments
        pattern_adjustments = self.get_pattern_adjustments()
        
        # Combine predictions
        combined_scores = {}
        for num in range(1, 81):
            ml_rank = ml_prediction.index(num) if num in ml_prediction else 80
            pattern_score = pattern_adjustments.get(num, 0.5)
            
            # Combine scores (weighted average)
            combined_scores[num] = (0.7 * (80 - ml_rank)/80) + (0.3 * pattern_score)
        
        return sorted(combined_scores, key=combined_scores.get, reverse=True)
    
    def get_pattern_adjustments(self):
        """Get pattern-based probability adjustments"""
        adjustments = {}
        
        # Analyze cluster trends
        for cluster_name, counts in self.pattern_analysis.items():
            if len(counts) < 10 or cluster_name in ['gaps', 'odd_even']:
                continue
                
            # Simple moving average
            window = min(10, len(counts))
            sma = sum(counts[-window:]) / window
            
            # Apply adjustment to cluster numbers
            for num in self.number_clusters[cluster_name]:
                adjustments[num] = adjustments.get(num, 0) + (sma / 10)
        
        # Gap analysis
        if 'gaps' in self.pattern_analysis and len(self.pattern_analysis['gaps']) > 20:
            gap_counts = Counter(self.pattern_analysis['gaps'][-20:])
            common_gap = gap_counts.most_common(1)[0][0]
            
            # Boost numbers that could complete common gaps
            for draw in self.draw_history[-5:]:
                sorted_draw = sorted(draw)
                for num in sorted_draw:
                    potential_pairs = [num + common_gap, num - common_gap]
                    for pair in potential_pairs:
                        if 1 <= pair <= 80:
                            adjustments[pair] = adjustments.get(pair, 0) + 0.2
        
        # Odd/even balance
        if 'odd_even' in self.pattern_analysis and len(self.pattern_analysis['odd_even']) > 10:
            odd_ratio = sum(self.pattern_analysis['odd_even'][-10:]) / 10
            # If consistently unbalanced, predict correction
            if odd_ratio > 0.65:  # Too many odds
                for num in range(1, 81):
                    if num % 2 == 0:  # Even numbers
                        adjustments[num] = adjustments.get(num, 0) + 0.3
            elif odd_ratio < 0.35:  # Too many evens
                for num in range(1, 81):
                    if num % 2 == 1:  # Odd numbers
                        adjustments[num] = adjustments.get(num, 0) + 0.3
        
        # Normalize adjustments
        max_adj = max(adjustments.values()) if adjustments else 1
        for num in adjustments:
            adjustments[num] /= max_adj
            
        return adjustments
    
    def ml_predict(self):
        """Predict using machine learning model"""
        if self.model is None:
            if not self.load_model():
                return []
        
        if len(self.draw_history) < 10:
            return []
        
        # Create features for current state
        features = []
        for num in range(1, 81):
            # Recent frequency
            recent_5 = sum(1 for draw in self.draw_history[-5:] if num in draw)
            recent_10 = sum(1 for draw in self.draw_history[-10:] if num in draw)
            recent_20 = sum(1 for draw in self.draw_history[-20:] if num in draw)
            
            # Last seen
            last_seen = self.number_stats[num]['last_seen']
            
            # Pair frequency
            pair_freq = sum(self.number_stats[num]['pair_counts'][n] for n in self.draw_history[-1])
            
            # Position stability
            pos_stability = 0
            if self.number_stats[num]['position_counts']:
                max_pos = max(self.number_stats[num]['position_counts'].values())
                pos_stability = max_pos / sum(self.number_stats[num]['position_counts'].values())
            
            # Recent hit bonus
            recent_hit = self.number_stats[num]['recent_hit']
            
            # Moving averages
            ma_3 = sum(1 for draw in self.draw_history[-3:] if num in draw) / 3
            ma_5 = recent_5 / 5
            ma_10 = recent_10 / 10
            
            # Exponential smoothing
            exp_smooth = self.number_stats[num]['exp_smooth']
            
            # Positional momentum
            pos_momentum = 0
            if len(self.position_stats[0]) > 5 and num in self.position_stats[0][-5:]:
                pos_momentum = 1 - (self.position_stats[0][-5:].index(num) / 5)
            
            # Cluster value
            cluster_value = self.number_stats[num]['cluster_value']
            
            # Positional value
            positional_value = self.number_stats[num]['positional_value']
            
            features.extend([
                recent_5, recent_10, recent_20, last_seen, pair_freq, 
                pos_stability, recent_hit, ma_3, ma_5, ma_10, 
                exp_smooth, pos_momentum, cluster_value, positional_value
            ])
        
        # Get predictions
        probabilities = self.model.predict_proba([features])
        
        # Properly extract probabilities for each number
        num_probs = []
        for i in range(80):
            # Each probabilities[i] is an array of shape (1, 2)
            # Extract probability for class 1 (appearing in next draw)
            num_probs.append(probabilities[i][0, 1])
        
        # Get numbers with highest probability
        return sorted(range(1, 81), key=lambda i: num_probs[i-1], reverse=True)
    
    def evaluate_prediction(self, prediction, actual):
        """Evaluate prediction against actual results"""
        hits = set(prediction) & set(actual)
        num_hits = len(hits)
        self.last_hits = num_hits
        
        # Update performance stats
        if num_hits == 4:
            self.performance_stats['correct_4'] += 1
            self.performance_stats['consecutive_hits'] += 1
            if self.performance_stats['consecutive_hits'] > self.performance_stats['max_consecutive']:
                self.performance_stats['max_consecutive'] = self.performance_stats['consecutive_hits']
        else:
            self.performance_stats['consecutive_hits'] = 0
            
        if num_hits == 3:
            self.performance_stats['correct_3'] += 1
        elif num_hits == 2:
            self.performance_stats['correct_2'] += 1
            
        # Update hit rates
        for num in prediction:
            was_hit = 1 if num in actual else 0
            self.performance_stats['hit_rates'][num].append(was_hit)
            if len(self.performance_stats['hit_rates'][num]) > 50:
                self.performance_stats['hit_rates'][num].pop(0)
        
        # Update last 10 hits
        self.performance_stats['last_10_hits'].append(num_hits)
        if len(self.performance_stats['last_10_hits']) > 10:
            self.performance_stats['last_10_hits'].pop(0)
        
        # Track model performance
        current_time = time.time()
        if current_time - self.last_update_time > 3600:
            self.model_performance['time'].append(current_time)
            self.model_performance['accuracy'].append(num_hits / 4)
            self.last_update_time = current_time
        
        return num_hits, hits


class WebSocketWorker(QThread):
    data_received = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    connection_active = pyqtSignal(bool)
    
    def __init__(self, uri):
        super().__init__()
        self.uri = uri
        self.running = True
        self.reconnect_delay = 5
        self.max_reconnect_delay = 60
        
    def run(self):
        asyncio.run(self.websocket_loop())
        
    async def websocket_loop(self):
        while self.running:
            try:
                self.status_changed.emit("Connecting to WebSocket...")
                self.connection_active.emit(False)
                async with websockets.connect(self.uri, ping_interval=10, ping_timeout=5) as websocket:
                    self.status_changed.emit("Connected! Waiting for data...")
                    self.connection_active.emit(True)
                    self.reconnect_delay = 5  # Reset delay on successful connection
                    
                    while self.running:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=15)
                            try:
                                data = json.loads(message)
                                self.data_received.emit(data)
                            except json.JSONDecodeError:
                                self.error_occurred.emit("Invalid JSON format received")
                            except Exception as e:
                                self.error_occurred.emit(f"Data processing error: {str(e)}")
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            await websocket.ping()
                            continue
                        except websockets.ConnectionClosed:
                            self.error_occurred.emit("Connection closed unexpectedly")
                            break
            except Exception as e:
                self.connection_active.emit(False)
                self.error_occurred.emit(f"Connection error: {str(e)}. Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
                # Exponential backoff with max limit
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                
    def stop(self):
        self.running = False
        self.wait(2000)  # Wait up to 2 seconds for thread to finish


class KenoGUI(QMainWindow):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.setWindowTitle("Enhanced Keno Predictor Pro")
        self.setGeometry(50, 50, 1000, 700)  # Smaller initial size
        
        # Central widget in scroll area for small screens
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs for different sections
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Tab 1: Overview
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        self.create_pattern_banner(overview_layout)
        self.create_prediction_display(overview_layout)
        self.create_performance_stats(overview_layout)
        self.tab_widget.addTab(overview_tab, "Overview")
        
        # Tab 2: Number Stats
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.create_number_stats_table(stats_layout)
        self.tab_widget.addTab(stats_tab, "Number Stats")
        
        # Tab 3: History & Patterns
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.create_last_10_predictions(history_layout)
        self.create_history_table(history_layout)
        self.tab_widget.addTab(history_tab, "History & Patterns")
        
        # Tab 4: Model Info
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        self.create_model_info(model_layout)
        self.tab_widget.addTab(model_tab, "Model Info")
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Connecting to WebSocket...")
        self.status_bar.addWidget(self.status_label)
        
        # Connection indicator
        self.connection_indicator = QLabel()
        self.connection_indicator.setFixedWidth(20)
        self.connection_indicator.setFixedHeight(20)
        self.connection_indicator.setStyleSheet("background-color: gray; border-radius: 10px;")
        self.status_bar.addPermanentWidget(self.connection_indicator)
        
        # Model version label
        self.model_version_label = QLabel(f"Model: v{predictor.model_version}")
        self.status_bar.addPermanentWidget(self.model_version_label)
        
        # Feature importance button
        self.feature_button = QPushButton("Feature Importance")
        self.feature_button.clicked.connect(self.show_feature_importance)
        self.status_bar.addPermanentWidget(self.feature_button)
        
        # WebSocket setup
        self.websocket_worker = WebSocketWorker(args.uri)
        self.websocket_worker.data_received.connect(self.handle_websocket_data)
        self.websocket_worker.status_changed.connect(self.status_label.setText)
        self.websocket_worker.error_occurred.connect(self.status_label.setText)
        self.websocket_worker.connection_active.connect(self.update_connection_indicator)
        self.websocket_worker.start()
        
        # Setup timer for periodic updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_gui)
        self.update_timer.start(1000)
        
        # Initialize prediction tracking
        self.last_prediction = None
        self.next_prediction = []
    
    def create_model_info(self, layout):
        """Create model information display"""
        group = QGroupBox("Model Information")
        form_layout = QFormLayout()
        group.setLayout(form_layout)
        
        self.model_type_label = QLabel(args.model.upper())
        self.model_version_label = QLabel(str(self.predictor.model_version))
        self.learning_rate_label = QLabel(f"{self.predictor.adaptive_learning_rate:.6f}")
        self.last_train_label = QLabel("Never")
        
        form_layout.addRow("Model Type:", self.model_type_label)
        form_layout.addRow("Model Version:", self.model_version_label)
        form_layout.addRow("Learning Rate:", self.learning_rate_label)
        form_layout.addRow("Last Trained:", self.last_train_label)
        
        # Feature importance button
        self.importance_button = QPushButton("Show Feature Importance")
        self.importance_button.clicked.connect(self.show_feature_importance)
        form_layout.addRow(self.importance_button)
        
        layout.addWidget(group)
        
        # Pattern analysis display
        pattern_group = QGroupBox("Pattern Analysis")
        pattern_layout = QVBoxLayout()
        pattern_group.setLayout(pattern_layout)
        
        self.pattern_text = QTextEdit()
        self.pattern_text.setReadOnly(True)
        pattern_layout.addWidget(self.pattern_text)
        
        layout.addWidget(pattern_group)
    
    def show_feature_importance(self):
        """Show feature importance dialog"""
        if not self.predictor.feature_importance:
            QMessageBox.information(self, "Feature Importance", "No feature importance data available")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Feature Importance")
        dialog.setGeometry(100, 100, 600, 400)
        
        layout = QVBoxLayout()
        
        # Create table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Feature", "Importance"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Populate table
        features = sorted(
            self.predictor.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:100]  # Show top 100 features
        
        table.setRowCount(len(features))
        for i, (feature, importance) in enumerate(features):
            table.setItem(i, 0, QTableWidgetItem(feature))
            table.setItem(i, 1, QTableWidgetItem(f"{importance:.6f}"))
        
        layout.addWidget(table)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def create_history_table(self, layout):
        """Create compact history table"""
        group = QGroupBox("Prediction History")
        history_layout = QVBoxLayout()
        group.setLayout(history_layout)
        
        self.history_table = QTableWidget(5, 4)  # Smaller table
        self.history_table.setHorizontalHeaderLabels(["Game ID", "Predicted", "Actual", "Hits"])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        history_layout.addWidget(self.history_table)
        layout.addWidget(group)
    
    def create_pattern_banner(self, layout):
        """Create a prominent banner for pattern state"""
        self.pattern_banner = QGroupBox("WIN PATTERN STATE")
        pattern_layout = QHBoxLayout()
        self.pattern_banner.setLayout(pattern_layout)
        
        # Pattern indicator
        self.pattern_label = QLabel("|0||4|4|4|4|")
        self.pattern_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.pattern_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Current state indicator
        self.state_label = QLabel("Current State: Initializing")
        self.state_label.setFont(QFont("Arial", 14))
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Next action indicator
        self.action_label = QLabel("Next Action: Waiting")
        self.action_label.setFont(QFont("Arial", 14))
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        pattern_layout.addWidget(self.pattern_label)
        pattern_layout.addWidget(self.state_label)
        pattern_layout.addWidget(self.action_label)
        
        layout.addWidget(self.pattern_banner)
        
        # Set initial styles
        self.update_pattern_display()
    
    def update_pattern_display(self):
        """Update pattern display based on current state"""
        state = self.predictor.performance_stats['win_pattern_state']
        state_text = {
            0: "State 0: Skip Prediction",
            1: "State 1: Start Pattern",
            2: "State 2: Active Pattern"
        }.get(state, "Unknown State")
        
        action_text = {
            0: "Predict 0 Numbers",
            1: "Predict 4 Numbers",
            2: "Predict 4 Numbers"
        }.get(state, "Unknown Action")
        
        self.state_label.setText(f"Current State: {state_text}")
        self.action_label.setText(f"Next Action: {action_text}")
        
        # Set background color based on state
        color = {
            0: "#FF9800",  # Orange
            1: "#2196F3",  # Blue
            2: "#4CAF50"   # Green
        }.get(state, "#9E9E9E")  # Gray for unknown
        
        self.pattern_banner.setStyleSheet(f"""
            QGroupBox {{
                background-color: {color};
                border: 2px solid #000000;
                border-radius: 10px;
                font-weight: bold;
            }}
            QLabel {{
                background-color: transparent;
            }}
        """)
    
    def update_connection_indicator(self, active):
        """Update connection indicator based on status"""
        if active:
            self.connection_indicator.setStyleSheet("background-color: green; border-radius: 10px;")
        else:
            self.connection_indicator.setStyleSheet("background-color: red; border-radius: 10px;")
        
    def create_number_stats_table(self, layout):
        """Create table for number statistics"""
        group = QGroupBox("Number Statistics")
        layout.addWidget(group)
        
        vbox = QVBoxLayout()
        group.setLayout(vbox)
        
        self.stats_table = QTableWidget(80, 12)
        self.stats_table.setHorizontalHeaderLabels([
            "Number", "Total", "Hot Str", "Cold Str", 
            "Last Seen", "Hit Rate", "Prob Score", "Hot Score", 
            "Cold Score", "Position", "Pair Freq", "Cluster"
        ])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        vbox.addWidget(self.stats_table)
        
    def create_prediction_display(self, layout):
        """Create prediction display area"""
        hbox = QHBoxLayout()
        layout.addLayout(hbox)
        
        # Current prediction
        prediction_group = QGroupBox("Current Prediction")
        prediction_layout = QVBoxLayout()
        prediction_group.setLayout(prediction_layout)
        
        self.prediction_label = QLabel("Waiting for data...")
        self.prediction_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        prediction_layout.addWidget(self.prediction_label)
        
        # Next prediction
        self.next_prediction_label = QLabel("Next Prediction: Processing...")
        self.next_prediction_label.setFont(QFont("Arial", 16))
        prediction_layout.addWidget(self.next_prediction_label)
        
        # History table
        history_group = QGroupBox("Prediction History")
        history_layout = QVBoxLayout()
        history_group.setLayout(history_layout)
        
        self.history_table = QTableWidget(5, 4)  # Smaller table for overview
        self.history_table.setHorizontalHeaderLabels(["Game ID", "Predicted", "Actual", "Hits"])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        history_layout.addWidget(self.history_table)
        
        hbox.addWidget(prediction_group, 1)
        hbox.addWidget(history_group, 1)
        
    def create_performance_stats(self, layout):
        """Create performance statistics display"""
        group = QGroupBox("Performance Statistics")
        layout.addWidget(group)
        
        vbox = QVBoxLayout()
        group.setLayout(vbox)
        
        # Stats labels
        stats_layout = QHBoxLayout()
        
        self.total_games_label = QLabel("Total Games: 0")
        self.hit_rate_label = QLabel("4-Hit Rate: 0.00%")
        self.avg_hits_label = QLabel("Avg Hits: 0.00")
        self.consecutive_label = QLabel("Consecutive 4-Hits: 0")
        self.max_consecutive_label = QLabel("Max Consecutive: 0")
        
        stats_layout.addWidget(self.total_games_label)
        stats_layout.addWidget(self.hit_rate_label)
        stats_layout.addWidget(self.avg_hits_label)
        stats_layout.addWidget(self.consecutive_label)
        stats_layout.addWidget(self.max_consecutive_label)
        
        vbox.addLayout(stats_layout)
        
        # Zone stats
        self.zone_label = QLabel("Hot Zone: 0.0% | Cold Zone: 0.0% | Edge: 0.0%")
        vbox.addWidget(self.zone_label)
        
        # Last 10 hits
        self.last_10_label = QLabel("Last 10 Predictions: ")
        vbox.addWidget(self.last_10_label)
        
        # Strategy weights display
        weights_group = QGroupBox("Strategy Weights & Consistency")
        weights_layout = QVBoxLayout()
        weights_group.setLayout(weights_layout)
        
        self.weights_label = QLabel("Weights: Loading...")
        weights_layout.addWidget(self.weights_label)
        
        self.consistency_label = QLabel("Consistency Factor: 1.00")
        weights_layout.addWidget(self.consistency_label)
        
        self.consistency_bar = QProgressBar()
        self.consistency_bar.setRange(0, 150)
        self.consistency_bar.setValue(100)
        self.consistency_bar.setFormat("Consistency: %v%")
        weights_layout.addWidget(self.consistency_bar)
        
        vbox.addWidget(weights_group)
        
    def create_last_10_predictions(self, layout):
        """Create visual display for last 10 predictions"""
        group = QGroupBox("Last 10 Predictions Performance")
        layout.addWidget(group)
        
        vbox = QVBoxLayout()
        group.setLayout(vbox)
        
        # Create container for prediction bars
        self.prediction_bars_layout = QVBoxLayout()
        vbox.addLayout(self.prediction_bars_layout)
        
        # Add summary labels
        summary_layout = QHBoxLayout()
        
        self.last_10_avg_label = QLabel("Average Hits: 0.00")
        self.last_10_4hit_label = QLabel("4-Hit Predictions: 0")
        self.last_10_streak_label = QLabel("Current Streak: 0")
        
        summary_layout.addWidget(self.last_10_avg_label)
        summary_layout.addWidget(self.last_10_4hit_label)
        summary_layout.addWidget(self.last_10_streak_label)
        
        vbox.addLayout(summary_layout)
        
        # Initialize bar display
        self.prediction_bars = []
        for i in range(10):
            bar_layout = QHBoxLayout()
            
            # Prediction number label
            pred_label = QLabel(f"#{i+1}")
            pred_label.setFixedWidth(30)
            
            # Bar with hit count
            bar = QProgressBar()
            bar.setRange(0, 4)
            bar.setFixedHeight(30)
            bar.setFormat("")
            
            # Value label
            value_label = QLabel("0")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setFixedWidth(30)
            
            bar_layout.addWidget(pred_label)
            bar_layout.addWidget(bar)
            bar_layout.addWidget(value_label)
            
            self.prediction_bars_layout.addLayout(bar_layout)
            self.prediction_bars.append((bar, value_label))
    
    def update_last_10_predictions(self):
        """Update the last 10 predictions display"""
        last_10 = self.predictor.performance_stats['last_10_hits']
        
        # Calculate stats
        total_hits = sum(last_10)
        avg_hits = total_hits / len(last_10) if last_10 else 0
        hit_4_count = sum(1 for hits in last_10 if hits == 4)
        current_streak = self.predictor.performance_stats['consecutive_hits']
        
        self.last_10_avg_label.setText(f"Average Hits: {avg_hits:.2f}")
        self.last_10_4hit_label.setText(f"4-Hit Predictions: {hit_4_count}/10")
        self.last_10_streak_label.setText(f"Current Streak: {current_streak}")
        
        # Update bars
        for i in range(10):
            bar, value_label = self.prediction_bars[i]
            if i < len(last_10):
                hits = last_10[i]
                bar.setValue(hits)
                value_label.setText(str(hits))
                
                # Set color based on hits
                if hits == 4:
                    bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
                elif hits == 3:
                    bar.setStyleSheet("QProgressBar::chunk { background-color: #8BC34A; }")
                elif hits == 2:
                    bar.setStyleSheet("QProgressBar::chunk { background-color: #FFC107; }")
                elif hits == 1:
                    bar.setStyleSheet("QProgressBar::chunk { background-color: #FF9800; }")
                else:
                    bar.setStyleSheet("QProgressBar::chunk { background-color: #F44336; }")
            else:
                bar.setValue(0)
                value_label.setText("")
                bar.setStyleSheet("")

    def update_gui(self):
        """Update GUI with latest data"""
        # Update number stats table
        self.stats_table.setRowCount(80)
        for num in range(1, 81):
            stats = self.predictor.number_stats[num]
            
            # Calculate hit rate
            hit_rates = self.predictor.performance_stats['hit_rates'][num]
            hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0
            
            # Calculate hot/cold scores
            hot_numbers = self.predictor.get_hot_numbers()
            cold_numbers = self.predictor.get_cold_numbers()
            
            hot_score = hot_numbers.index(num) + 1 if num in hot_numbers else 999
            cold_score = cold_numbers.index(num) + 1 if num in cold_numbers else 999
            
            # Position stats
            position_stats = ""
            if stats['position_counts']:
                common_pos = max(stats['position_counts'], key=stats['position_counts'].get)
                position_stats = f"P{common_pos+1}:{stats['position_counts'][common_pos]}"
                
            # Pair stats
            pair_stats = ""
            if stats['pair_counts']:
                common_pair = max(stats['pair_counts'], key=stats['pair_counts'].get)
                pair_stats = f"{common_pair}:{stats['pair_counts'][common_pair]}"
            
            self.stats_table.setItem(num-1, 0, QTableWidgetItem(str(num)))
            self.stats_table.setItem(num-1, 1, QTableWidgetItem(str(stats['total_count'])))
            self.stats_table.setItem(num-1, 2, QTableWidgetItem(str(stats['hot_streak'])))
            self.stats_table.setItem(num-1, 3, QTableWidgetItem(str(stats['cold_streak'])))
            self.stats_table.setItem(num-1, 4, QTableWidgetItem(str(stats['last_seen'])))
            self.stats_table.setItem(num-1, 5, QTableWidgetItem(f"{hit_rate:.2%}"))
            self.stats_table.setItem(num-1, 6, QTableWidgetItem(f"{stats['probability_score']:.3f}"))
            self.stats_table.setItem(num-1, 7, QTableWidgetItem(str(hot_score)))
            self.stats_table.setItem(num-1, 8, QTableWidgetItem(str(cold_score)))
            self.stats_table.setItem(num-1, 9, QTableWidgetItem(position_stats))
            self.stats_table.setItem(num-1, 10, QTableWidgetItem(pair_stats))
            self.stats_table.setItem(num-1, 11, QTableWidgetItem(f"{stats['cluster_value']:.2f}"))
            
            # Highlight advantage numbers based on probability score
            if stats['probability_score'] > 0.3:
                for col in range(12):
                    item = self.stats_table.item(num-1, col)
                    if item:
                        item.setBackground(QColor(255, 255, 200))
            
            # Highlight hot numbers
            if stats['hot_streak'] >= 3:
                for col in range(12):
                    item = self.stats_table.item(num-1, col)
                    if item:
                        item.setBackground(QColor(255, 200, 200))
            
            # Highlight cold numbers
            if stats['last_seen'] >= args.cold_threshold:
                for col in range(12):
                    item = self.stats_table.item(num-1, col)
                    if item:
                        item.setBackground(QColor(200, 200, 255))
        
        # Update performance stats
        total_games = self.predictor.performance_stats['total_games']
        correct_4 = self.predictor.performance_stats['correct_4']
        correct_3 = self.predictor.performance_stats['correct_3']
        correct_2 = self.predictor.performance_stats['correct_2']
        consecutive = self.predictor.performance_stats['consecutive_hits']
        max_consecutive = self.predictor.performance_stats['max_consecutive']
        
        hit_rate = correct_4 / total_games if total_games > 0 else 0
        avg_hits = (correct_4*4 + correct_3*3 + correct_2*2) / total_games if total_games > 0 else 0
        
        # Zone stats
        hot_zone_rate = self.predictor.performance_stats['hot_zone_hits'] / total_games if total_games > 0 else 0
        cold_zone_rate = self.predictor.performance_stats['cold_zone_hits'] / total_games if total_games > 0 else 0
        edge_rate = self.predictor.performance_stats['edge_hits'] / total_games if total_games > 0 else 0
        
        self.total_games_label.setText(f"Total Games: {total_games}")
        self.hit_rate_label.setText(f"4-Hit Rate: {hit_rate:.2%}")
        self.avg_hits_label.setText(f"Avg Hits: {avg_hits:.2f}")
        self.consecutive_label.setText(f"Consecutive 4-Hits: {consecutive}")
        self.max_consecutive_label.setText(f"Max Consecutive: {max_consecutive}")
        self.zone_label.setText(f"Hot Zone: {hot_zone_rate:.1%} | Cold Zone: {cold_zone_rate:.1%} | Edge: {edge_rate:.1%}")
        
        # Update last 10 hits display
        last_10 = self.predictor.performance_stats['last_10_hits']
        last_10_text = " | ".join([
            f"<b style='color:{'green' if h==4 else 'blue' if h==3 else 'orange' if h==2 else 'red'}'>{h}</b>" 
            for h in last_10
        ])
        self.last_10_label.setText(f"Last 10 Predictions: {last_10_text}")
        
        # Update prediction history table
        history = self.predictor.prediction_history[-5:]  # Only show last 5 for compact view
        self.history_table.setRowCount(len(history))
        for i, (game_id, prediction, actual, hits) in enumerate(history):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(game_id)))
            self.history_table.setItem(i, 1, QTableWidgetItem(", ".join(map(str, prediction))))
            self.history_table.setItem(i, 2, QTableWidgetItem(", ".join(map(str, actual))))
            self.history_table.setItem(i, 3, QTableWidgetItem(f"{hits} hits"))
            
            # Highlight row based on hits
            color = QColor()
            if hits == 4:
                color = QColor(200, 255, 200)
            elif hits == 3:
                color = QColor(200, 200, 255)
            elif hits == 2:
                color = QColor(255, 255, 200)
            
            for col in range(4):
                item = self.history_table.item(i, col)
                if item:
                    item.setBackground(color)
        
        # Update strategy weights display
        weights = self.predictor.strategy_weights
        weight_text = " | ".join([f"{k}: {v:.2f}" for k, v in weights.items()])
        self.weights_label.setText(f"Strategy Weights: {weight_text}")
        
        # Update consistency factor
        consistency = int(self.predictor.consistency_factor * 100)
        self.consistency_label.setText(f"Consistency Factor: {self.predictor.consistency_factor:.2f}")
        self.consistency_bar.setValue(consistency)
        
        # Update next prediction
        if hasattr(self, 'next_prediction'):
            self.next_prediction_label.setText(f"Next Prediction: {', '.join(map(str, self.next_prediction))}")
            
        # Update pattern display
        self.update_pattern_display()
        
        # Update last 10 predictions display
        self.update_last_10_predictions()
        
        # Update model info
        self.model_version_label.setText(f"Model: v{self.predictor.model_version}")
        self.learning_rate_label.setText(f"{self.predictor.adaptive_learning_rate:.6f}")
        
        # Update pattern analysis display
        pattern_text = "Pattern Analysis:\n\n"
        for pattern, data in self.predictor.pattern_analysis.items():
            if len(data) > 5:
                last_5 = data[-5:]
                avg = sum(last_5) / len(last_5)
                pattern_text += f"{pattern.capitalize()} (last 5 avg: {avg:.2f})\n"
        self.pattern_text.setText(pattern_text)
    
    def handle_websocket_data(self, data):
        """Handle incoming WebSocket data"""
        if isinstance(data, dict) and data.get("status") == "roomevent":
            event = data.get("data", {})
            
            if event.get("status") == "play" and "numbers" in event:
                result = event["numbers"]
                game_id = event.get("game_id", "N/A")
                timestamp = event.get("timestamp", datetime.now(timezone.utc).isoformat())
                
                # Save draw to CSV
                self.predictor.save_draw(game_id, timestamp, result)
                
                # Update predictor with new draw
                self.predictor.update_stats(result)
                
                # Make prediction for next draw
                self.next_prediction = self.predictor.predict_numbers()
                
                # Update prediction display
                if self.next_prediction:
                    self.prediction_label.setText(
                        f"Current Prediction: {', '.join(map(str, self.next_prediction))}"
                    )
                else:
                    self.prediction_label.setText("Current Prediction: 0 numbers (Pattern State)")
                
                # Train ML model periodically
                if len(self.predictor.draw_history) % 10 == 0:
                    self.predictor.train_model()
                
                # Save prediction history and evaluate
                if self.last_prediction is not None:
                    # Evaluate last prediction against current result
                    num_hits, hits = self.predictor.evaluate_prediction(
                        self.last_prediction['prediction'], result
                    )
                    # Save prediction to CSV
                    self.predictor.save_prediction(
                        game_id,
                        self.last_prediction['prediction'],
                        result,
                        num_hits
                    )
                    self.predictor.prediction_history.append((
                        game_id, 
                        self.last_prediction['prediction'], 
                        result, 
                        num_hits
                    ))
                
                # Save current prediction for next evaluation
                self.last_prediction = {
                    'game_id': game_id,
                    'timestamp': timestamp,
                    'prediction': self.next_prediction
                }
                
                # Update status
                self.status_label.setText(
                    f"Game {game_id} processed | Hits: {num_hits if 'num_hits' in locals() else 'N/A'}/4 | Consecutive: {self.predictor.performance_stats['consecutive_hits']} | Pattern: {self.predictor.performance_stats['win_pattern_state']}"
                )
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        if hasattr(self, 'websocket_worker'):
            self.websocket_worker.stop()
            self.websocket_worker.wait(2000)
        event.accept()


if __name__ == "__main__":
    # Create predictor and application
    predictor = KenoPredictor()
    app = QApplication([])
    
    # Create and show GUI
    gui = KenoGUI(predictor)
    gui.show()
    
    # Run event loop
    app.exec()
