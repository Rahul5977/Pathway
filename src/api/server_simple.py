"""
Simplified FastAPI server for the Market Anomaly Detector.
This version avoids complex imports and focuses on essential functionality.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Create the FastAPI app
app = FastAPI(
    title="Market Anomaly Detector API",
    description="Real-time market anomaly detection with AI explanations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    symbol: Optional[str] = None
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    timestamp: str

# Helper functions
def load_data_from_files():
    """Load data from the output files."""
    market_data = []
    anomalies_data = []
    
    try:
        # Load market data
        market_file = project_root / "data" / "output" / "market_data.jsonl"
        if market_file.exists():
            with open(market_file, 'r') as f:
                for line in f:
                    try:
                        market_data.append(json.loads(line.strip()))
                    except:
                        continue
        
        # Load anomalies/alerts
        alerts_file = project_root / "data" / "output" / "alerts.jsonl"
        if alerts_file.exists():
            with open(alerts_file, 'r') as f:
                for line in f:
                    try:
                        anomalies_data.append(json.loads(line.strip()))
                    except:
                        continue
    except Exception as e:
        print(f"Error loading data: {e}")
    
    return market_data, anomalies_data

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Market Anomaly Detector API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "market_data": "/market-data",
            "anomalies": "/anomalies", 
            "alerts": "/alerts",
            "query": "/query",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running"
    }

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base with a question."""
    try:
        # Simple response for demo
        answer = f"AI Analysis: Based on current market data, regarding '{request.query}'"
        if request.symbol:
            answer += f" for {request.symbol}"
        answer += ", the system is monitoring for anomalies and trends."
        
        return QueryResponse(
            answer=answer,
            context=[{"source": "knowledge_base", "relevance": 0.8}],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/market-data")
async def get_market_data(limit: int = 50):
    """Get recent market data."""
    try:
        market_data, _ = load_data_from_files()
        
        # Return the most recent data
        recent_data = market_data[-limit:] if market_data else []
        
        return {
            "data": recent_data,
            "count": len(recent_data),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get market data: {str(e)}")

@app.get("/anomalies")
async def get_anomalies(limit: int = 20):
    """Get recent anomalies."""
    try:
        _, anomalies_data = load_data_from_files()
        
        # Return the most recent anomalies
        recent_anomalies = anomalies_data[-limit:] if anomalies_data else []
        
        return {
            "anomalies": recent_anomalies,
            "count": len(recent_anomalies),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get anomalies: {str(e)}")

@app.get("/alerts")
async def get_alerts(limit: int = 20):
    """Get recent alerts."""
    try:
        _, alerts_data = load_data_from_files()
        
        # Return the most recent alerts
        recent_alerts = alerts_data[-limit:] if alerts_data else []
        
        return {
            "alerts": recent_alerts,
            "total_count": len(recent_alerts),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@app.get("/symbols")
async def get_symbols():
    """Get list of monitored symbols."""
    return {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "BTC-USD", "ETH-USD"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/statistics")
async def get_statistics():
    """Get system statistics."""
    try:
        market_data, anomalies_data = load_data_from_files()
        
        return {
            "total_data_points": len(market_data),
            "total_anomalies": len(anomalies_data),
            "last_updated": market_data[-1]["timestamp"] if market_data else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "total_data_points": 0,
            "total_anomalies": 0,
            "last_updated": None,
            "timestamp": datetime.now().isoformat()
        }

@app.post("/simulate-anomaly")
async def simulate_anomaly(symbol: str = "AAPL", severity: str = "HIGH"):
    """Simulate an anomaly for testing."""
    try:
        # Create a simulated anomaly
        anomaly = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "anomaly_score": 3.5 if severity == "HIGH" else 2.0,
            "risk_level": severity,
            "explanation": f"Simulated {severity.lower()} anomaly for {symbol}",
            "type": "simulated"
        }
        
        # Optionally save to file
        alerts_file = project_root / "data" / "output" / "alerts.jsonl"
        alerts_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(alerts_file, 'a') as f:
            f.write(json.dumps(anomaly) + '\n')
        
        return {
            "message": f"Simulated anomaly created for {symbol}",
            "anomaly": anomaly,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to simulate anomaly: {str(e)}")

# Sentiment prediction endpoints (simplified)
class SentimentPredictionRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = 15

@app.post("/predict-sentiment")
async def predict_sentiment(request: SentimentPredictionRequest):
    """Predict sentiment impact (simplified version)."""
    try:
        # Simplified sentiment prediction
        import random
        predicted_sentiment = random.uniform(0.3, 0.8)
        
        return {
            "symbol": request.symbol,
            "predicted_sentiment": predicted_sentiment,
            "price_impact_prediction": predicted_sentiment * 0.1,  # Simple mapping
            "confidence_score": random.uniform(0.6, 0.9),
            "time_horizon_minutes": request.horizon,
            "contributing_factors": ["market_sentiment", "news_analysis"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment prediction failed: {str(e)}")

@app.get("/sentiment-summary/{symbol}")
async def get_sentiment_summary(symbol: str):
    """Get sentiment summary for a symbol."""
    try:
        import random
        
        return {
            "symbol": symbol,
            "current_sentiment": {
                "overall": random.uniform(0.3, 0.8),
                "twitter": random.uniform(0.2, 0.9),
                "news": random.uniform(0.4, 0.7),
                "reddit": random.uniform(0.3, 0.8)
            },
            "sentiment_trend": random.uniform(-0.1, 0.1),
            "sentiment_momentum": random.uniform(-0.05, 0.05),
            "prediction": {
                "next_15_min": random.uniform(0.3, 0.8),
                "confidence": random.uniform(0.6, 0.9)
            },
            "correlation": {
                "sentiment_price_correlation": random.uniform(0.3, 0.7)
            },
            "data_quality": {
                "data_points": random.randint(50, 200),
                "freshness_minutes": random.randint(1, 5)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sentiment summary: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
