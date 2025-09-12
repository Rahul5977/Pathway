"""Data processing and anomaly detection algorithms."""

import pathway as pw
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

from config.settings import Settings
from utils.logger import setup_logger, log_anomaly_detected

logger = setup_logger(__name__)


class DataProcessor:
    """Process raw market data for analysis and anomaly detection."""
    
    def __init__(self):
        self.price_history = {}  # Symbol -> price history
        self.volume_history = {}  # Symbol -> volume history
    
    def process_market_data(self, data: pw.Pointer) -> dict:
        """
        Process market data and calculate technical indicators.
        
        Args:
            data: Pathway data pointer with market information
            
        Returns:
            Processed data with technical indicators
        """
        symbol = data.symbol
        price = data.price
        volume = data.volume
        timestamp = data.timestamp
        
        # Initialize history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
        
        # Add current data to history
        self.price_history[symbol].append({
            'timestamp': timestamp,
            'price': price,
            'volume': volume
        })
        
        # Keep only recent history (last 100 points)
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
            self.volume_history[symbol] = self.volume_history[symbol][-100:]
        
        # Calculate technical indicators
        indicators = self._calculate_indicators(symbol, price, volume)
        
        return {
            'symbol': symbol,
            'timestamp': timestamp,
            'price': price,
            'volume': volume,
            'indicators': indicators,
            'processed_at': datetime.now().isoformat()
        }
    
    def _calculate_indicators(self, symbol: str, current_price: float, current_volume: int) -> dict:
        """Calculate technical indicators for anomaly detection."""
        history = self.price_history.get(symbol, [])
        
        if len(history) < 2:
            return {
                'sma_5': current_price,
                'sma_20': current_price,
                'rsi': 50.0,
                'volatility': 0.0,
                'volume_sma': current_volume,
                'price_change': 0.0,
                'volume_ratio': 1.0
            }
        
        prices = [h['price'] for h in history]
        volumes = [h['volume'] for h in history]
        
        # Simple Moving Averages
        sma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else current_price
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
        
        # RSI calculation
        rsi = self._calculate_rsi(prices) if len(prices) >= 14 else 50.0
        
        # Volatility (standard deviation of recent price changes)
        if len(prices) >= 2:
            price_changes = np.diff(prices)
            volatility = np.std(price_changes[-20:]) if len(price_changes) >= 20 else np.std(price_changes)
        else:
            volatility = 0.0
        
        # Volume indicators
        volume_sma = np.mean(volumes[-10:]) if len(volumes) >= 10 else current_volume
        volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1.0
        
        # Price change percentage
        if len(prices) >= 2:
            price_change = ((current_price - prices[-2]) / prices[-2]) * 100
        else:
            price_change = 0.0
        
        return {
            'sma_5': float(sma_5),
            'sma_20': float(sma_20),
            'rsi': float(rsi),
            'volatility': float(volatility),
            'volume_sma': float(volume_sma),
            'price_change': float(price_change),
            'volume_ratio': float(volume_ratio)
        }
    
    def _calculate_rsi(self, prices: List[float], window: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < window + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-window:])
        avg_loss = np.mean(losses[-window:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)


class AnomalyDetector:
    """Advanced anomaly detection using multiple algorithms."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.threshold = settings.ANOMALY_THRESHOLD
        self.lookback_window = settings.LOOKBACK_WINDOW
        
        # Initialize ML models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Historical data for model training
        self.feature_history = {}
        self.model_trained = {}
    
    def detect_anomaly(self, data: pw.Pointer) -> float:
        """
        Detect anomalies in market data using multiple methods.
        
        Args:
            data: Processed market data
            
        Returns:
            Anomaly score (higher = more anomalous)
        """
        symbol = data.symbol
        processed_data = data.processed_data
        
        if not processed_data or 'indicators' not in processed_data:
            return 0.0
        
        indicators = processed_data['indicators']
        
        # Calculate multiple anomaly scores
        statistical_score = self._statistical_anomaly_score(symbol, indicators)
        ml_score = self._ml_anomaly_score(symbol, indicators)
        rule_based_score = self._rule_based_anomaly_score(indicators)
        
        # Combine scores (weighted average)
        combined_score = (
            0.4 * statistical_score +
            0.4 * ml_score +
            0.2 * rule_based_score
        )
        
        # Log significant anomalies
        if combined_score > self.threshold:
            log_anomaly_detected(
                symbol, 
                combined_score, 
                {
                    'statistical': statistical_score,
                    'ml': ml_score,
                    'rule_based': rule_based_score,
                    'indicators': indicators
                }
            )
        
        return float(combined_score)
    
    def is_anomaly(self, anomaly_score: float) -> bool:
        """Check if the anomaly score indicates an anomaly."""
        return anomaly_score > self.threshold
    
    def _statistical_anomaly_score(self, symbol: str, indicators: dict) -> float:
        """Calculate anomaly score using statistical methods."""
        if symbol not in self.feature_history:
            self.feature_history[symbol] = []
        
        # Extract key features for statistical analysis
        features = [
            indicators.get('price_change', 0),
            indicators.get('volume_ratio', 1),
            indicators.get('volatility', 0),
            indicators.get('rsi', 50)
        ]
        
        self.feature_history[symbol].append(features)
        
        # Keep only recent history
        if len(self.feature_history[symbol]) > self.lookback_window:
            self.feature_history[symbol] = self.feature_history[symbol][-self.lookback_window:]
        
        if len(self.feature_history[symbol]) < 10:
            return 0.0
        
        # Calculate z-scores for each feature
        history_array = np.array(self.feature_history[symbol][:-1])  # Exclude current
        current_features = np.array(features)
        
        means = np.mean(history_array, axis=0)
        stds = np.std(history_array, axis=0)
        
        # Avoid division by zero
        stds = np.where(stds == 0, 1, stds)
        
        z_scores = np.abs((current_features - means) / stds)
        
        # Return the maximum z-score as anomaly indicator
        return float(np.max(z_scores))
    
    def _ml_anomaly_score(self, symbol: str, indicators: dict) -> float:
        """Calculate anomaly score using machine learning (Isolation Forest)."""
        if symbol not in self.feature_history or len(self.feature_history[symbol]) < 20:
            return 0.0
        
        # Prepare features for ML model
        features = np.array([
            indicators.get('price_change', 0),
            indicators.get('volume_ratio', 1),
            indicators.get('volatility', 0),
            indicators.get('rsi', 50),
            indicators.get('sma_5', 0) - indicators.get('sma_20', 0)  # SMA divergence
        ]).reshape(1, -1)
        
        # Train model periodically
        if symbol not in self.model_trained or len(self.feature_history[symbol]) % 50 == 0:
            try:
                # Prepare training data
                history_features = []
                for hist_indicators in self.feature_history[symbol]:
                    if len(hist_indicators) >= 4:
                        hist_features = hist_indicators + [hist_indicators[0] - hist_indicators[1]]  # Add SMA divergence approximation
                        history_features.append(hist_features)
                
                if len(history_features) >= 20:
                    X_train = np.array(history_features)
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    self.isolation_forest.fit(X_train_scaled)
                    self.model_trained[symbol] = True
                    
            except Exception as e:
                logger.warning(f"ML model training failed for {symbol}: {str(e)}")
                return 0.0
        
        if symbol not in self.model_trained:
            return 0.0
        
        try:
            # Scale features and predict
            features_scaled = self.scaler.transform(features)
            anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
            
            # Convert to positive score (more negative = more anomalous)
            return float(max(0, -anomaly_score * 5))  # Scale and invert
            
        except Exception as e:
            logger.warning(f"ML anomaly detection failed for {symbol}: {str(e)}")
            return 0.0
    
    def _rule_based_anomaly_score(self, indicators: dict) -> float:
        """Calculate anomaly score using rule-based methods."""
        score = 0.0
        
        # Rule 1: Extreme price movements
        price_change = abs(indicators.get('price_change', 0))
        if price_change > 5:  # >5% change
            score += 1.0
        elif price_change > 3:  # >3% change
            score += 0.5
        
        # Rule 2: Unusual volume
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 3:  # 3x normal volume
            score += 1.0
        elif volume_ratio > 2:  # 2x normal volume
            score += 0.5
        
        # Rule 3: Extreme RSI
        rsi = indicators.get('rsi', 50)
        if rsi < 20 or rsi > 80:
            score += 0.5
        
        # Rule 4: High volatility
        volatility = indicators.get('volatility', 0)
        if volatility > 2:  # High volatility threshold
            score += 0.5
        
        return float(score)


class EventProcessor:
    """Process and correlate different types of market events."""
    
    def __init__(self):
        self.event_history = []
        self.correlation_threshold = 0.7
    
    def correlate_events(self, market_data: dict, news_data: dict, sentiment_data: dict) -> dict:
        """Correlate market events with news and sentiment."""
        correlation_score = 0.0
        related_events = []
        
        # Simple correlation based on timing and symbol overlap
        if market_data and news_data:
            # Check if news mentions the same symbols
            market_symbols = {market_data.get('symbol', '')}
            news_symbols = set(news_data.get('symbols_mentioned', []))
            
            if market_symbols.intersection(news_symbols):
                correlation_score += 0.5
                related_events.append({
                    'type': 'news_correlation',
                    'title': news_data.get('title', ''),
                    'sentiment': news_data.get('sentiment_score', 0)
                })
        
        if market_data and sentiment_data:
            # Check sentiment correlation
            sentiment_symbols = set(sentiment_data.get('symbols_mentioned', []))
            if market_symbols.intersection(sentiment_symbols):
                correlation_score += 0.3
                related_events.append({
                    'type': 'sentiment_correlation',
                    'sentiment': sentiment_data.get('sentiment_score', 0),
                    'influence': sentiment_data.get('influence_score', 0)
                })
        
        return {
            'correlation_score': correlation_score,
            'related_events': related_events,
            'timestamp': datetime.now().isoformat()
        }
