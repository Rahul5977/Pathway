"""Demo script to showcase the Market Anomaly Detector capabilities."""

import os
import sys
import json
import time
import random
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DemoScenario:
    """Demo scenario generator for showcasing the system."""
    
    def __init__(self):
        self.settings = Settings()
        self.demo_data_dir = Path("./data/demo")
        self.demo_data_dir.mkdir(parents=True, exist_ok=True)
        
    def run_demo_sequence(self):
        """Run a complete demo sequence showcasing all features."""
        logger.info("🎬 Starting Market Anomaly Detector Demo")
        logger.info("=" * 50)
        
        print("\n🚀 MARKET ANOMALY DETECTOR DEMO")
        print("=" * 50)
        print("This demo will showcase:")
        print("1. 📊 Real-time data ingestion")
        print("2. 🔍 Anomaly detection in action")
        print("3. 🤖 AI explanations via RAG")
        print("4. 🚨 Alert generation")
        print("5. 📈 Live dashboard updates")
        print("\nPress Ctrl+C at any time to stop the demo.\n")
        
        try:
            self.scenario_1_normal_operation()
            time.sleep(2)
            
            self.scenario_2_market_volatility()
            time.sleep(2)
            
            self.scenario_3_breaking_news_impact()
            time.sleep(2)
            
            self.scenario_4_crypto_flash_crash()
            time.sleep(2)
            
            self.scenario_5_earnings_surprise()
            
            print("\n🎉 Demo completed successfully!")
            print("Check the dashboard at http://localhost:8501 to see the results")
            
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        except Exception as e:
            print(f"\n❌ Demo error: {str(e)}")
    
    def scenario_1_normal_operation(self):
        """Scenario 1: Normal market operation with steady data flow."""
        print("\n📊 Scenario 1: Normal Market Operation")
        print("-" * 40)
        print("Showing steady data flow with normal market movements...")
        
        # Generate normal market data
        symbols = ["AAPL", "GOOGL", "MSFT"]
        base_prices = {"AAPL": 180.0, "GOOGL": 135.0, "MSFT": 370.0}
        
        for i in range(10):
            for symbol in symbols:
                # Normal price movements (±1%)
                price_change = random.uniform(-1.0, 1.0)
                new_price = base_prices[symbol] * (1 + price_change / 100)
                
                market_data = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "price": round(new_price, 2),
                    "volume": random.randint(1000000, 3000000),
                    "change": round(price_change, 2),
                    "processed_data": {
                        "indicators": {
                            "price_change": price_change,
                            "volume_ratio": random.uniform(0.8, 1.2),
                            "rsi": random.uniform(45, 55),
                            "volatility": random.uniform(0.01, 0.03)
                        }
                    },
                    "anomaly_score": random.uniform(0.1, 0.5)  # Low scores
                }
                
                self._save_demo_data("market", market_data)
            
            time.sleep(0.5)
        
        print("✅ Normal operation scenario completed")
    
    def scenario_2_market_volatility(self):
        """Scenario 2: High volatility period with multiple anomalies."""
        print("\n🌪️  Scenario 2: Market Volatility")
        print("-" * 40)
        print("Simulating high volatility period with anomalies...")
        
        symbols = ["AAPL", "TSLA", "NVDA"]
        base_prices = {"AAPL": 180.0, "TSLA": 250.0, "NVDA": 480.0}
        
        # Create volatility event
        news_event = {
            "timestamp": datetime.now().isoformat(),
            "title": "Federal Reserve announces unexpected rate decision",
            "content": "The Federal Reserve made an unexpected announcement regarding interest rates, causing market volatility across technology stocks.",
            "sentiment_score": -0.6,
            "symbols_mentioned": json.dumps(symbols),
            "scenario": "volatility"
        }
        self._save_demo_data("news", news_event)
        
        for i in range(8):
            for symbol in symbols:
                # High volatility movements (±5-10%)
                price_change = random.uniform(-10.0, 10.0)
                new_price = base_prices[symbol] * (1 + price_change / 100)
                anomaly_score = abs(price_change) / 2  # Higher scores for bigger moves
                
                market_data = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "price": round(new_price, 2),
                    "volume": random.randint(3000000, 8000000),  # Higher volume
                    "anomaly_score": anomaly_score,
                    "processed_data": {
                        "indicators": {
                            "price_change": price_change,
                            "volume_ratio": random.uniform(2.0, 4.0),  # High volume
                            "rsi": random.uniform(20, 80),
                            "volatility": random.uniform(0.05, 0.15)
                        }
                    }
                }
                
                # Create alerts for high anomalies
                if anomaly_score > 3.0:
                    alert_data = {
                        **market_data,
                        "alert_sent": {
                            "priority": "HIGH",
                            "message": f"High volatility detected for {symbol}: {price_change:.1f}% change",
                            "timestamp": datetime.now().isoformat()
                        },
                        "explanation": f"Unusual {abs(price_change):.1f}% price movement detected for {symbol}, likely related to Federal Reserve announcement."
                    }
                    self._save_demo_data("alerts", alert_data)
                
                self._save_demo_data("market", market_data)
            
            time.sleep(0.3)
        
        print("✅ Market volatility scenario completed")
    
    def scenario_3_breaking_news_impact(self):
        """Scenario 3: Breaking news with immediate market impact."""
        print("\n📰 Scenario 3: Breaking News Impact")
        print("-" * 40)
        print("Simulating breaking news with immediate market reaction...")
        
        # Breaking news event
        breaking_news = {
            "timestamp": datetime.now().isoformat(),
            "title": "🚨 BREAKING: Major Tech Company Announces Revolutionary AI Breakthrough",
            "content": "A major technology company has announced a breakthrough in artificial intelligence that could revolutionize the industry. The announcement has immediate implications for AI and semiconductor stocks.",
            "sentiment_score": 0.85,
            "symbols_mentioned": json.dumps(["NVDA", "GOOGL", "MSFT"]),
            "scenario": "breaking_news"
        }
        self._save_demo_data("news", breaking_news)
        
        # Immediate market reaction
        affected_symbols = ["NVDA", "GOOGL", "MSFT"]
        base_prices = {"NVDA": 480.0, "GOOGL": 135.0, "MSFT": 370.0}
        
        for symbol in affected_symbols:
            # Strong positive reaction
            price_change = random.uniform(5.0, 15.0)  # Positive spike
            new_price = base_prices[symbol] * (1 + price_change / 100)
            
            market_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "price": round(new_price, 2),
                "volume": random.randint(5000000, 15000000),  # Very high volume
                "anomaly_score": price_change / 2,
                "processed_data": {
                    "indicators": {
                        "price_change": price_change,
                        "volume_ratio": random.uniform(3.0, 6.0),
                        "rsi": random.uniform(70, 85),
                        "volatility": random.uniform(0.08, 0.20)
                    }
                }
            }
            
            # Generate alert
            alert_data = {
                **market_data,
                "alert_sent": {
                    "priority": "HIGH",
                    "message": f"Breaking news impact: {symbol} surged {price_change:.1f}% on AI breakthrough announcement",
                    "timestamp": datetime.now().isoformat()
                },
                "explanation": f"Strong positive reaction in {symbol} following major AI breakthrough announcement. {price_change:.1f}% price increase with {market_data['processed_data']['indicators']['volume_ratio']:.1f}x normal volume."
            }
            self._save_demo_data("alerts", alert_data)
            self._save_demo_data("market", market_data)
        
        print("✅ Breaking news scenario completed")
    
    def scenario_4_crypto_flash_crash(self):
        """Scenario 4: Cryptocurrency flash crash simulation."""
        print("\n₿ Scenario 4: Crypto Flash Crash")
        print("-" * 40)
        print("Simulating cryptocurrency flash crash...")
        
        crypto_symbols = ["BTC-USD", "ETH-USD"]
        base_prices = {"BTC-USD": 43000.0, "ETH-USD": 2600.0}
        
        # Crash event
        crash_news = {
            "timestamp": datetime.now().isoformat(),
            "title": "🚨 FLASH: Major Cryptocurrency Exchange Reports Security Incident",
            "content": "A major cryptocurrency exchange has reported a security incident, causing immediate sell-off in Bitcoin and Ethereum markets.",
            "sentiment_score": -0.9,
            "symbols_mentioned": json.dumps(crypto_symbols),
            "scenario": "crypto_crash"
        }
        self._save_demo_data("news", crash_news)
        
        for symbol in crypto_symbols:
            # Sharp decline
            price_change = random.uniform(-20.0, -10.0)  # Severe drop
            new_price = base_prices[symbol] * (1 + price_change / 100)
            
            market_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "price": round(new_price, 2),
                "volume": random.randint(8000000, 20000000),  # Panic selling volume
                "anomaly_score": abs(price_change) / 2,
                "processed_data": {
                    "indicators": {
                        "price_change": price_change,
                        "volume_ratio": random.uniform(5.0, 10.0),
                        "rsi": random.uniform(10, 25),  # Oversold
                        "volatility": random.uniform(0.15, 0.30)
                    }
                }
            }
            
            # Critical alert
            alert_data = {
                **market_data,
                "alert_sent": {
                    "priority": "HIGH",
                    "message": f"CRITICAL: {symbol} flash crash {abs(price_change):.1f}% on exchange security incident",
                    "timestamp": datetime.now().isoformat()
                },
                "explanation": f"Severe {abs(price_change):.1f}% decline in {symbol} following cryptocurrency exchange security incident. Extreme volume and volatility indicate market panic."
            }
            self._save_demo_data("alerts", alert_data)
            self._save_demo_data("market", market_data)
        
        print("✅ Crypto flash crash scenario completed")
    
    def scenario_5_earnings_surprise(self):
        """Scenario 5: Earnings surprise reaction."""
        print("\n📈 Scenario 5: Earnings Surprise")
        print("-" * 40)
        print("Simulating positive earnings surprise...")
        
        # Earnings announcement
        earnings_news = {
            "timestamp": datetime.now().isoformat(),
            "title": "Apple Reports Record Q4 Earnings, Beats All Estimates",
            "content": "Apple Inc. reported record fourth quarter earnings, significantly beating analyst estimates on both revenue and profit margins. iPhone sales exceeded expectations.",
            "sentiment_score": 0.9,
            "symbols_mentioned": json.dumps(["AAPL"]),
            "scenario": "earnings_surprise"
        }
        self._save_demo_data("news", earnings_news)
        
        # Market reaction
        price_change = random.uniform(8.0, 12.0)  # Strong positive reaction
        new_price = 180.0 * (1 + price_change / 100)
        
        market_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "AAPL",
            "price": round(new_price, 2),
            "volume": random.randint(10000000, 25000000),  # Massive volume
            "anomaly_score": price_change / 2,
            "processed_data": {
                "indicators": {
                    "price_change": price_change,
                    "volume_ratio": random.uniform(4.0, 8.0),
                    "rsi": random.uniform(75, 85),
                    "volatility": random.uniform(0.06, 0.12)
                }
            }
        }
        
        # Earnings alert
        alert_data = {
            **market_data,
            "alert_sent": {
                "priority": "MEDIUM",
                "message": f"Earnings surprise: AAPL surges {price_change:.1f}% on record Q4 results",
                "timestamp": datetime.now().isoformat()
            },
            "explanation": f"Strong {price_change:.1f}% rally in AAPL following record Q4 earnings beat. Volume {market_data['processed_data']['indicators']['volume_ratio']:.1f}x normal indicates strong institutional buying."
        }
        self._save_demo_data("alerts", alert_data)
        self._save_demo_data("market", market_data)
        
        print("✅ Earnings surprise scenario completed")
    
    def _save_demo_data(self, data_type: str, data: dict):
        """Save demo data to appropriate files."""
        try:
            if data_type == "market":
                output_dir = Path("./data/output")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(output_dir / "market_data.jsonl", "a") as f:
                    f.write(json.dumps(data) + "\n")
                    
            elif data_type == "alerts":
                output_dir = Path("./data/output")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(output_dir / "alerts.jsonl", "a") as f:
                    f.write(json.dumps(data) + "\n")
                    
            elif data_type == "news":
                input_dir = Path("./data/input/news")
                input_dir.mkdir(parents=True, exist_ok=True)
                
                with open(input_dir / f"news_{int(time.time())}.json", "w") as f:
                    f.write(json.dumps(data))
            
            # Also save to demo directory
            demo_file = self.demo_data_dir / f"{data_type}_demo.jsonl"
            with open(demo_file, "a") as f:
                f.write(json.dumps(data) + "\n")
                
        except Exception as e:
            logger.error(f"Error saving demo data: {str(e)}")


if __name__ == "__main__":
    demo = DemoScenario()
    demo.run_demo_sequence()
