"""Data connectors for real-time financial data ingestion using Pathway."""

import pathway as pw
import yfinance as yf
import requests
import json
import threading
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
from abc import ABC, abstractmethod

from config.settings import Settings
from utils.logger import setup_logger, log_data_ingestion

logger = setup_logger(__name__)


class BaseConnector(ABC):
    """Base class for all data connectors."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    @abstractmethod
    def create_stream(self) -> pw.Table:
        """Create a Pathway stream for this data source."""
        pass


class MarketDataConnector(BaseConnector):
    """Real-time market data connector using multiple financial APIs."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.symbols = settings.SYMBOLS_TO_MONITOR
        self.update_interval = settings.DATA_UPDATE_INTERVAL
    
    def create_stream(self) -> pw.Table:
        """Create a market data stream using Pathway's REST connector."""
        
        # For demo purposes, we'll use a file-based connector that simulates real-time data
        # In production, you'd use REST/WebSocket connectors to actual financial APIs
        
        class MarketDataSchema(pw.Schema):
            timestamp: str
            symbol: str
            price: float
            volume: int
            change: float
            change_percent: float
            high: float
            low: float
            open: float
            previous_close: float
        
        # Simulate real-time market data updates
        def generate_market_data():
            """Generate simulated market data for demo purposes."""
            from pathlib import Path
            
            output_dir = Path("./data/input")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_prices = {
                "AAPL": 180.00, "GOOGL": 135.00, "MSFT": 370.00, 
                "TSLA": 250.00, "AMZN": 145.00, "META": 315.00,
                "NVDA": 480.00, "BTC-USD": 43000.00, "ETH-USD": 2600.00
            }
            
            while True:
                timestamp = datetime.now().isoformat()
                
                for symbol in self.symbols:
                    # Simulate price movements
                    base_price = base_prices.get(symbol, 100.0)
                    change_percent = random.uniform(-5.0, 5.0)  # ±5% movement
                    new_price = base_price * (1 + change_percent / 100)
                    
                    # Occasionally create anomalies for demo
                    if random.random() < 0.05:  # 5% chance of anomaly
                        anomaly_multiplier = random.choice([0.85, 1.15])  # ±15% spike
                        new_price *= anomaly_multiplier
                        logger.info(f"🎯 Simulating anomaly for {symbol}: {anomaly_multiplier:.2%} change")
                    
                    data = {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "price": round(new_price, 2),
                        "volume": random.randint(1000000, 10000000),
                        "change": round(new_price - base_price, 2),
                        "change_percent": round(change_percent, 2),
                        "high": round(new_price * 1.02, 2),
                        "low": round(new_price * 0.98, 2),
                        "open": round(base_price, 2),
                        "previous_close": round(base_price, 2)
                    }
                    
                    # Write to file for Pathway to pick up
                    with open(output_dir / f"market_{symbol}_{int(time.time())}.json", "w") as f:
                        json.dump(data, f)
                
                time.sleep(self.update_interval)
        
        # Start data generation in background
        data_thread = threading.Thread(target=generate_market_data, daemon=True)
        data_thread.start()
        
        # Create Pathway stream from JSON files
        return pw.io.fs.read(
            "./data/input/market_*.json",
            format="json",
            schema=MarketDataSchema,
            mode="streaming"
        )
    
    def get_real_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get real market data using yfinance (for production use)."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period="1d", interval="1m")
            
            if not history.empty:
                latest = history.iloc[-1]
                return {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "price": float(latest['Close']),
                    "volume": int(latest['Volume']),
                    "high": float(latest['High']),
                    "low": float(latest['Low']),
                    "open": float(latest['Open']),
                    "previous_close": float(info.get('previousClose', latest['Close']))
                }
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None


class NewsConnector(BaseConnector):
    """Real-time news and sentiment data connector."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.news_api_key = settings.NEWS_API_KEY
        self.sources = settings.NEWS_SOURCES
        self.update_interval = settings.SENTIMENT_UPDATE_INTERVAL
    
    def create_stream(self) -> pw.Table:
        """Create a news and sentiment stream."""
        
        class NewsSchema(pw.Schema):
            timestamp: str
            title: str
            content: str
            source: str
            url: str
            sentiment_score: float
            relevance_score: float
            symbols_mentioned: str  # JSON string of symbols
        
        def generate_news_data():
            """Generate simulated news data for demo purposes."""
            from pathlib import Path
            
            output_dir = Path("./data/input/news")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            sample_news = [
                {
                    "title": "Tech stocks rally on strong earnings outlook",
                    "content": "Major technology companies reported better than expected earnings, driving market optimism.",
                    "symbols": ["AAPL", "GOOGL", "MSFT"],
                    "sentiment": 0.8
                },
                {
                    "title": "Federal Reserve signals potential rate cuts",
                    "content": "The Federal Reserve indicated a more dovish stance in their latest meeting minutes.",
                    "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
                    "sentiment": 0.6
                },
                {
                    "title": "Cryptocurrency market shows volatility",
                    "content": "Bitcoin and Ethereum prices fluctuate amid regulatory uncertainty.",
                    "symbols": ["BTC-USD", "ETH-USD"],
                    "sentiment": -0.2
                },
                {
                    "title": "EV market expansion continues",
                    "content": "Electric vehicle companies announce new manufacturing facilities.",
                    "symbols": ["TSLA"],
                    "sentiment": 0.7
                }
            ]
            
            while True:
                # Randomly select and modify news
                news_item = random.choice(sample_news)
                timestamp = datetime.now().isoformat()
                
                data = {
                    "timestamp": timestamp,
                    "title": news_item["title"],
                    "content": news_item["content"] + f" Updated at {timestamp}",
                    "source": random.choice(self.sources),
                    "url": f"https://example.com/news/{int(time.time())}",
                    "sentiment_score": news_item["sentiment"] + random.uniform(-0.2, 0.2),
                    "relevance_score": random.uniform(0.5, 1.0),
                    "symbols_mentioned": json.dumps(news_item["symbols"])
                }
                
                # Occasionally create breaking news for anomalies
                if random.random() < 0.1:  # 10% chance
                    data["title"] = "🚨 BREAKING: " + data["title"]
                    data["sentiment_score"] = random.choice([-0.8, 0.8])  # Strong sentiment
                    logger.info(f"📰 Simulating breaking news: {data['title']}")
                
                with open(output_dir / f"news_{int(time.time())}.json", "w") as f:
                    json.dump(data, f)
                
                time.sleep(self.update_interval)
        
        # Start news generation
        news_thread = threading.Thread(target=generate_news_data, daemon=True)
        news_thread.start()
        
        return pw.io.fs.read(
            "./data/input/news",
            format="json",
            schema=NewsSchema,
            mode="streaming"
        )
    
    def fetch_real_news(self) -> List[Dict[str, Any]]:
        """Fetch real news using News API (for production use)."""
        if not self.news_api_key:
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": "stocks OR finance OR market OR economy",
                "sources": ",".join(self.sources),
                "sortBy": "publishedAt",
                "apiKey": self.news_api_key,
                "pageSize": 20
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            articles = response.json().get("articles", [])
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []


class SocialSentimentConnector(BaseConnector):
    """Social media sentiment data connector."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.twitter_token = settings.TWITTER_BEARER_TOKEN
    
    def create_stream(self) -> pw.Table:
        """Create a social sentiment stream."""
        
        class SentimentSchema(pw.Schema):
            timestamp: str
            platform: str
            content: str
            sentiment_score: float
            influence_score: float
            symbols_mentioned: str
        
        # For demo, generate simulated social sentiment data
        def generate_sentiment_data():
            """Generate simulated social sentiment data."""
            from pathlib import Path
            
            output_dir = Path("./data/input/sentiment")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            sample_posts = [
                "Bullish on $AAPL after latest iPhone announcement!",
                "$TSLA looking strong this quarter #ElectricVehicles",
                "Market volatility increasing, time to be cautious",
                "$NVDA AI revolution continues to drive growth",
                "Fed policy changes could impact $GOOGL and tech sector"
            ]
            
            while True:
                timestamp = datetime.now().isoformat()
                content = random.choice(sample_posts)
                
                # Extract symbols mentioned
                import re
                symbols = re.findall(r'\$([A-Z]{1,5})', content)
                
                data = {
                    "timestamp": timestamp,
                    "platform": random.choice(["twitter", "reddit", "discord"]),
                    "content": content,
                    "sentiment_score": random.uniform(-1.0, 1.0),
                    "influence_score": random.uniform(0.1, 1.0),
                    "symbols_mentioned": json.dumps(symbols)
                }
                
                with open(output_dir / f"sentiment_{int(time.time())}.json", "w") as f:
                    json.dump(data, f)
                
                time.sleep(30)  # Update every 30 seconds
        
        import threading
        sentiment_thread = threading.Thread(target=generate_sentiment_data, daemon=True)
        sentiment_thread.start()
        
        return pw.io.fs.read(
            "./data/input/sentiment",
            format="json",
            schema=SentimentSchema,
            mode="streaming"
        )
