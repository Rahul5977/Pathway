"""
Real-Time Predictive Sentiment AI Agent
=======================================

Advanced sentiment analysis and prediction system that forecasts market movements
based on multi-source sentiment analysis and temporal modeling.

Features:
- Multi-source sentiment extraction (Twitter, Reddit, News, Options Flow)
- LSTM/Transformer-based sentiment trajectory prediction
- Real-time sentiment-price correlation analysis
- Proactive alerting before major market moves
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import pickle

# ML and NLP imports
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf

# Sentiment analysis models
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Social media APIs (mock implementations for demo)
import tweepy
import praw

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Structured sentiment data point"""
    timestamp: datetime
    symbol: str
    source: str  # 'twitter', 'reddit', 'news', 'options'
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    volume_indicator: float  # mentions/volume
    text_sample: str
    metadata: Dict[str, Any]

@dataclass
class SentimentPrediction:
    """Sentiment prediction with confidence intervals"""
    symbol: str
    timestamp: datetime
    predicted_sentiment: float
    confidence_interval: Tuple[float, float]
    price_impact_prediction: float
    time_horizon_minutes: int
    confidence_score: float
    contributing_factors: List[str]

class AdvancedSentimentAnalyzer:
    """Multi-source sentiment analysis with financial domain specificity"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.financial_keywords = self._load_financial_keywords()
        self.sentiment_weights = {
            'twitter': 0.3,
            'reddit': 0.25,
            'news': 0.35,
            'options': 0.1
        }
        
        # Initialize financial sentiment model (FinBERT-like)
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        except:
            logger.warning("FinBERT model not available, using VADER as fallback")
            self.finbert_tokenizer = None
            self.finbert_model = None
    
    def _load_financial_keywords(self) -> Dict[str, float]:
        """Load financial sentiment keywords with weights"""
        return {
            # Positive indicators
            'bullish': 1.5, 'moon': 1.2, 'pump': 1.0, 'buy': 0.8, 'rally': 1.0,
            'breakout': 1.2, 'surge': 1.3, 'rocket': 1.4, 'diamond hands': 1.1,
            'hodl': 0.9, 'calls': 0.7, 'long': 0.6, 'uptrend': 1.0,
            
            # Negative indicators
            'bearish': -1.5, 'crash': -1.8, 'dump': -1.2, 'sell': -0.8, 'drop': -1.0,
            'breakdown': -1.2, 'plunge': -1.4, 'puts': -0.7, 'short': -0.6,
            'downtrend': -1.0, 'bear trap': -0.9, 'panic': -1.6,
            
            # Neutral/uncertainty
            'sideways': 0.0, 'consolidation': 0.0, 'uncertainty': -0.2, 'volatile': -0.1
        }
    
    async def analyze_text_sentiment(self, text: str, source: str) -> float:
        """Analyze sentiment of text using multiple methods"""
        try:
            # VADER sentiment
            vader_score = self.vader_analyzer.polarity_scores(text)['compound']
            
            # TextBlob sentiment
            blob_score = TextBlob(text).sentiment.polarity
            
            # Financial keyword analysis
            keyword_score = self._analyze_financial_keywords(text)
            
            # FinBERT analysis (if available)
            finbert_score = 0.0
            if self.finbert_model and self.finbert_tokenizer:
                finbert_score = await self._get_finbert_sentiment(text)
            
            # Weighted combination
            weights = [0.3, 0.2, 0.3, 0.2] if self.finbert_model else [0.4, 0.3, 0.3, 0.0]
            combined_score = (
                vader_score * weights[0] +
                blob_score * weights[1] +
                keyword_score * weights[2] +
                finbert_score * weights[3]
            )
            
            return np.clip(combined_score, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def _analyze_financial_keywords(self, text: str) -> float:
        """Analyze text for financial keywords"""
        text_lower = text.lower()
        total_weight = 0.0
        keyword_count = 0
        
        for keyword, weight in self.financial_keywords.items():
            if keyword in text_lower:
                total_weight += weight
                keyword_count += 1
        
        if keyword_count == 0:
            return 0.0
        
        return np.clip(total_weight / keyword_count, -1.0, 1.0)
    
    async def _get_finbert_sentiment(self, text: str) -> float:
        """Get sentiment from FinBERT model"""
        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                # Convert to sentiment score: positive - negative
                sentiment = predictions[0][0].item() - predictions[0][2].item()  # positive - negative
                return sentiment
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return 0.0

class SentimentLSTMPredictor(nn.Module):
    """LSTM model for sentiment trajectory prediction"""
    
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Use last time step
        out = self.fc(out)
        return out

class PredictiveSentimentAgent:
    """Main AI agent for predictive sentiment analysis"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'BTC-USD', 'ETH-USD']
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        
        # Data storage
        self.sentiment_history = {symbol: deque(maxlen=1000) for symbol in self.symbols}
        self.price_history = {symbol: deque(maxlen=1000) for symbol in self.symbols}
        self.prediction_cache = {}
        
        # ML models
        self.sentiment_predictors = {}
        self.scalers = {}
        self._initialize_models()
        
        # Correlation tracking
        self.sentiment_price_correlations = {symbol: deque(maxlen=100) for symbol in self.symbols}
        
        # Mock API clients (replace with real implementations)
        self.twitter_client = None  # Initialize with real API keys
        self.reddit_client = None   # Initialize with real API keys
        
        logger.info(f"Initialized Predictive Sentiment Agent for {len(self.symbols)} symbols")
    
    def _initialize_models(self):
        """Initialize LSTM models for each symbol"""
        for symbol in self.symbols:
            # Create LSTM model
            model = SentimentLSTMPredictor()
            self.sentiment_predictors[symbol] = model
            self.scalers[symbol] = MinMaxScaler()
            
            # Load pre-trained model if available
            try:
                model_path = f"models/sentiment_lstm_{symbol}.pt"
                model.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded pre-trained model for {symbol}")
            except:
                logger.info(f"No pre-trained model found for {symbol}, will train from scratch")
    
    async def collect_multi_source_sentiment(self, symbol: str) -> List[SentimentData]:
        """Collect sentiment data from multiple sources"""
        sentiment_data = []
        
        try:
            # Twitter sentiment
            twitter_sentiment = await self._get_twitter_sentiment(symbol)
            if twitter_sentiment:
                sentiment_data.extend(twitter_sentiment)
            
            # Reddit sentiment
            reddit_sentiment = await self._get_reddit_sentiment(symbol)
            if reddit_sentiment:
                sentiment_data.extend(reddit_sentiment)
            
            # News sentiment
            news_sentiment = await self._get_news_sentiment(symbol)
            if news_sentiment:
                sentiment_data.extend(news_sentiment)
            
            # Options flow sentiment
            options_sentiment = await self._get_options_sentiment(symbol)
            if options_sentiment:
                sentiment_data.append(options_sentiment)
            
        except Exception as e:
            logger.error(f"Error collecting sentiment for {symbol}: {e}")
        
        return sentiment_data
    
    async def _get_twitter_sentiment(self, symbol: str) -> List[SentimentData]:
        """Get Twitter sentiment (mock implementation)"""
        # In real implementation, use Twitter API v2
        mock_tweets = [
            f"${symbol} looking bullish! Strong momentum building 🚀",
            f"Technical analysis shows {symbol} breaking resistance",
            f"Bearish divergence on {symbol} charts, expecting pullback",
            f"Volume spike on {symbol} - something big coming?",
            f"Institutional buying pressure on {symbol} increasing"
        ]
        
        sentiment_data = []
        for i, tweet in enumerate(mock_tweets):
            sentiment_score = await self.sentiment_analyzer.analyze_text_sentiment(tweet, 'twitter')
            sentiment_data.append(SentimentData(
                timestamp=datetime.now() - timedelta(minutes=i*5),
                symbol=symbol,
                source='twitter',
                sentiment_score=sentiment_score,
                confidence=0.6 + np.random.random() * 0.3,
                volume_indicator=np.random.randint(50, 500),
                text_sample=tweet,
                metadata={'retweets': np.random.randint(10, 100), 'likes': np.random.randint(50, 1000)}
            ))
        
        return sentiment_data
    
    async def _get_reddit_sentiment(self, symbol: str) -> List[SentimentData]:
        """Get Reddit sentiment (mock implementation)"""
        # In real implementation, use PRAW (Reddit API)
        mock_posts = [
            f"DD: Why {symbol} is undervalued - comprehensive analysis",
            f"{symbol} earnings call highlights - better than expected",
            f"Unusual options activity on {symbol} - big move coming?",
            f"Technical breakdown: {symbol} showing weakness",
            f"Institutional flows suggest {symbol} accumulation"
        ]
        
        sentiment_data = []
        for i, post in enumerate(mock_posts):
            sentiment_score = await self.sentiment_analyzer.analyze_text_sentiment(post, 'reddit')
            sentiment_data.append(SentimentData(
                timestamp=datetime.now() - timedelta(minutes=i*10),
                symbol=symbol,
                source='reddit',
                sentiment_score=sentiment_score,
                confidence=0.7 + np.random.random() * 0.2,
                volume_indicator=np.random.randint(20, 200),
                text_sample=post,
                metadata={'upvotes': np.random.randint(100, 2000), 'comments': np.random.randint(20, 300)}
            ))
        
        return sentiment_data
    
    async def _get_news_sentiment(self, symbol: str) -> List[SentimentData]:
        """Get financial news sentiment"""
        # Mock news headlines
        mock_headlines = [
            f"{symbol} reports strong quarterly earnings, beats expectations",
            f"Analysts upgrade {symbol} price target following strong performance",
            f"Market volatility affects {symbol} trading volumes",
            f"{symbol} announces strategic partnership deal",
            f"Regulatory concerns impact {symbol} sector outlook"
        ]
        
        sentiment_data = []
        for i, headline in enumerate(mock_headlines):
            sentiment_score = await self.sentiment_analyzer.analyze_text_sentiment(headline, 'news')
            sentiment_data.append(SentimentData(
                timestamp=datetime.now() - timedelta(hours=i),
                symbol=symbol,
                source='news',
                sentiment_score=sentiment_score,
                confidence=0.8 + np.random.random() * 0.15,
                volume_indicator=1.0,  # News has consistent weight
                text_sample=headline,
                metadata={'source': 'Financial News API', 'category': 'earnings'}
            ))
        
        return sentiment_data
    
    async def _get_options_sentiment(self, symbol: str) -> SentimentData:
        """Get options flow sentiment"""
        # Mock options data (put/call ratio analysis)
        put_call_ratio = 0.5 + np.random.random() * 1.0  # 0.5 to 1.5
        
        # Convert put/call ratio to sentiment score
        # Lower ratio (more calls) = bullish, Higher ratio (more puts) = bearish
        sentiment_score = (1.0 - put_call_ratio) * 2 - 1  # Convert to -1 to 1 scale
        sentiment_score = np.clip(sentiment_score, -1.0, 1.0)
        
        return SentimentData(
            timestamp=datetime.now(),
            symbol=symbol,
            source='options',
            sentiment_score=sentiment_score,
            confidence=0.75,
            volume_indicator=put_call_ratio,
            text_sample=f"Put/Call ratio: {put_call_ratio:.2f}",
            metadata={'put_call_ratio': put_call_ratio, 'options_volume': np.random.randint(1000, 10000)}
        )
    
    def _create_feature_vector(self, sentiment_data: List[SentimentData]) -> np.ndarray:
        """Create feature vector for ML model"""
        if not sentiment_data:
            return np.zeros(5)
        
        # Aggregate sentiment by source
        source_sentiments = {source: [] for source in ['twitter', 'reddit', 'news', 'options']}
        for data in sentiment_data:
            source_sentiments[data.source].append(data.sentiment_score)
        
        # Calculate features
        features = []
        for source in ['twitter', 'reddit', 'news', 'options']:
            if source_sentiments[source]:
                features.append(np.mean(source_sentiments[source]))
            else:
                features.append(0.0)
        
        # Add overall sentiment momentum
        overall_sentiment = np.mean([data.sentiment_score for data in sentiment_data])
        features.append(overall_sentiment)
        
        return np.array(features)
    
    async def update_sentiment_history(self, symbol: str):
        """Update sentiment history for a symbol"""
        try:
            # Collect new sentiment data
            sentiment_data = await self.collect_multi_source_sentiment(symbol)
            
            if sentiment_data:
                # Create feature vector
                features = self._create_feature_vector(sentiment_data)
                
                # Store in history
                timestamp = datetime.now()
                self.sentiment_history[symbol].append({
                    'timestamp': timestamp,
                    'features': features,
                    'raw_data': sentiment_data
                })
                
                # Update correlation analysis
                await self._update_correlation_analysis(symbol)
                
                logger.debug(f"Updated sentiment history for {symbol}")
        
        except Exception as e:
            logger.error(f"Error updating sentiment history for {symbol}: {e}")
    
    async def _update_correlation_analysis(self, symbol: str):
        """Update sentiment-price correlation analysis"""
        try:
            # Get recent price data
            if symbol.endswith('-USD'):  # Crypto
                # Mock crypto price data
                price = 40000 + np.random.randn() * 2000  # Mock BTC price
            else:
                # Mock stock price data
                price = 150 + np.random.randn() * 10
            
            self.price_history[symbol].append({
                'timestamp': datetime.now(),
                'price': price
            })
            
            # Calculate correlation if we have enough data
            if len(self.sentiment_history[symbol]) >= 20 and len(self.price_history[symbol]) >= 20:
                sentiment_values = [entry['features'][-1] for entry in list(self.sentiment_history[symbol])[-20:]]
                price_values = [entry['price'] for entry in list(self.price_history[symbol])[-20:]]
                
                correlation = np.corrcoef(sentiment_values, price_values)[0, 1]
                if not np.isnan(correlation):
                    self.sentiment_price_correlations[symbol].append({
                        'timestamp': datetime.now(),
                        'correlation': correlation
                    })
        
        except Exception as e:
            logger.error(f"Error updating correlation for {symbol}: {e}")
    
    async def predict_sentiment_trajectory(self, symbol: str, horizon_minutes: int = 15) -> Optional[SentimentPrediction]:
        """Predict sentiment trajectory and price impact"""
        try:
            if len(self.sentiment_history[symbol]) < 20:
                logger.warning(f"Insufficient data for prediction: {symbol}")
                return None
            
            # Prepare data for prediction
            recent_data = list(self.sentiment_history[symbol])[-20:]
            feature_sequence = np.array([entry['features'] for entry in recent_data])
            
            # Scale features
            scaled_features = self.scalers[symbol].fit_transform(feature_sequence)
            
            # Prepare input for LSTM (batch_size=1, sequence_length=20, features=5)
            lstm_input = torch.FloatTensor(scaled_features).unsqueeze(0)
            
            # Make prediction
            model = self.sentiment_predictors[symbol]
            model.eval()
            with torch.no_grad():
                predicted_sentiment = model(lstm_input).item()
            
            # Calculate confidence intervals (simplified)
            confidence_interval = (predicted_sentiment - 0.2, predicted_sentiment + 0.2)
            
            # Predict price impact based on correlation
            correlation = 0.0
            if self.sentiment_price_correlations[symbol]:
                recent_correlations = list(self.sentiment_price_correlations[symbol])[-5:]
                correlation = np.mean([entry['correlation'] for entry in recent_correlations])
            
            price_impact_prediction = predicted_sentiment * correlation * 0.05  # 5% max impact
            
            # Calculate confidence score
            data_quality = min(len(self.sentiment_history[symbol]) / 100, 1.0)
            model_confidence = 1.0 - abs(predicted_sentiment)  # Lower confidence for extreme predictions
            confidence_score = (data_quality + model_confidence) / 2
            
            # Identify contributing factors
            contributing_factors = []
            recent_sentiment = recent_data[-1]['features']
            if recent_sentiment[0] > 0.5:  # Twitter positive
                contributing_factors.append("Strong positive Twitter sentiment")
            if recent_sentiment[1] > 0.5:  # Reddit positive
                contributing_factors.append("Bullish Reddit discussions")
            if recent_sentiment[2] > 0.5:  # News positive
                contributing_factors.append("Positive news coverage")
            if recent_sentiment[3] > 0.5:  # Options bullish
                contributing_factors.append("Bullish options flow")
            
            return SentimentPrediction(
                symbol=symbol,
                timestamp=datetime.now(),
                predicted_sentiment=predicted_sentiment,
                confidence_interval=confidence_interval,
                price_impact_prediction=price_impact_prediction,
                time_horizon_minutes=horizon_minutes,
                confidence_score=confidence_score,
                contributing_factors=contributing_factors
            )
        
        except Exception as e:
            logger.error(f"Error predicting sentiment for {symbol}: {e}")
            return None
    
    async def generate_sentiment_alerts(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate proactive sentiment alerts"""
        alerts = []
        
        try:
            # Get recent prediction
            prediction = await self.predict_sentiment_trajectory(symbol)
            
            if not prediction:
                return alerts
            
            # Check for significant sentiment shifts
            if abs(prediction.predicted_sentiment) > 0.7 and prediction.confidence_score > 0.6:
                alert_type = "BULLISH" if prediction.predicted_sentiment > 0 else "BEARISH"
                
                alerts.append({
                    'type': 'sentiment_shift',
                    'symbol': symbol,
                    'priority': 'HIGH' if abs(prediction.predicted_sentiment) > 0.8 else 'MEDIUM',
                    'message': f"{alert_type} sentiment shift detected for {symbol}",
                    'predicted_impact': f"{prediction.price_impact_prediction:.2%}",
                    'time_horizon': f"{prediction.time_horizon_minutes} minutes",
                    'confidence': f"{prediction.confidence_score:.1%}",
                    'factors': prediction.contributing_factors,
                    'timestamp': prediction.timestamp.isoformat()
                })
            
            # Check for sentiment-price divergence
            if len(self.sentiment_price_correlations[symbol]) > 5:
                recent_correlation = list(self.sentiment_price_correlations[symbol])[-1]['correlation']
                if abs(recent_correlation) < 0.3:  # Weak correlation might indicate divergence
                    alerts.append({
                        'type': 'sentiment_divergence',
                        'symbol': symbol,
                        'priority': 'MEDIUM',
                        'message': f"Sentiment-price divergence detected for {symbol}",
                        'correlation': f"{recent_correlation:.2f}",
                        'implication': "Price may be due for correction",
                        'timestamp': datetime.now().isoformat()
                    })
        
        except Exception as e:
            logger.error(f"Error generating alerts for {symbol}: {e}")
        
        return alerts
    
    async def get_sentiment_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive sentiment summary for a symbol"""
        try:
            if not self.sentiment_history[symbol]:
                return {'error': 'No sentiment data available'}
            
            recent_data = list(self.sentiment_history[symbol])[-10:]
            latest_features = recent_data[-1]['features']
            
            # Calculate trends
            sentiment_trend = np.mean([entry['features'][-1] for entry in recent_data])
            sentiment_momentum = latest_features[-1] - recent_data[-5]['features'][-1] if len(recent_data) >= 5 else 0
            
            # Get prediction
            prediction = await self.predict_sentiment_trajectory(symbol)
            
            # Get correlation info
            correlation = 0.0
            if self.sentiment_price_correlations[symbol]:
                correlation = list(self.sentiment_price_correlations[symbol])[-1]['correlation']
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_sentiment': {
                    'twitter': latest_features[0],
                    'reddit': latest_features[1],
                    'news': latest_features[2],
                    'options': latest_features[3],
                    'overall': latest_features[4]
                },
                'sentiment_trend': sentiment_trend,
                'sentiment_momentum': sentiment_momentum,
                'prediction': {
                    'predicted_sentiment': prediction.predicted_sentiment if prediction else None,
                    'price_impact': prediction.price_impact_prediction if prediction else None,
                    'confidence': prediction.confidence_score if prediction else None
                } if prediction else None,
                'correlation': {
                    'sentiment_price_correlation': correlation,
                    'interpretation': self._interpret_correlation(correlation)
                },
                'data_quality': {
                    'data_points': len(self.sentiment_history[symbol]),
                    'last_update': recent_data[-1]['timestamp'].isoformat()
                }
            }
        
        except Exception as e:
            logger.error(f"Error getting sentiment summary for {symbol}: {e}")
            return {'error': str(e)}
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation value"""
        if abs(correlation) > 0.7:
            return "Strong correlation" if correlation > 0 else "Strong negative correlation"
        elif abs(correlation) > 0.4:
            return "Moderate correlation" if correlation > 0 else "Moderate negative correlation"
        elif abs(correlation) > 0.2:
            return "Weak correlation" if correlation > 0 else "Weak negative correlation"
        else:
            return "No significant correlation"
    
    async def save_model_state(self, symbol: str):
        """Save trained model state"""
        try:
            model_path = f"models/sentiment_lstm_{symbol}.pt"
            torch.save(self.sentiment_predictors[symbol].state_dict(), model_path)
            
            # Save scaler
            scaler_path = f"models/sentiment_scaler_{symbol}.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[symbol], f)
            
            logger.info(f"Saved model state for {symbol}")
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {e}")

# Global instance
sentiment_agent = PredictiveSentimentAgent()

async def main():
    """Test the sentiment prediction system"""
    logger.info("Starting Predictive Sentiment Agent test...")
    
    # Test sentiment analysis for a few symbols
    test_symbols = ['AAPL', 'TSLA', 'BTC-USD']
    
    for symbol in test_symbols:
        print(f"\n=== Testing {symbol} ===")
        
        # Update sentiment data
        await sentiment_agent.update_sentiment_history(symbol)
        
        # Get sentiment summary
        summary = await sentiment_agent.get_sentiment_summary(symbol)
        print(f"Sentiment Summary: {json.dumps(summary, indent=2)}")
        
        # Generate alerts
        alerts = await sentiment_agent.generate_sentiment_alerts(symbol)
        if alerts:
            print(f"Alerts: {json.dumps(alerts, indent=2)}")
        
        # Wait between symbols
        await asyncio.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
