from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import uvicorn

from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    symbol: Optional[str] = None
    max_results: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    timestamp: str


class AnomalyResponse(BaseModel):
    symbol: str
    anomaly_score: float
    risk_level: str
    explanation: str
    timestamp: str
    indicators: Dict[str, Any]


class AlertResponse(BaseModel):
    alerts: List[Dict[str, Any]]
    total_count: int
    timestamp: str


def create_fastapi_app(settings: Settings) -> FastAPI:
    """Create and configure the FastAPI application."""
    
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
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Market Anomaly Detector API",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "query": "/query - Ask questions about market data",
                "anomalies": "/anomalies - Get recent anomalies",
                "alerts": "/alerts - Get active alerts",
                "market_data": "/market-data - Get latest market data",
                "health": "/health - Health check"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "api_server": "active",
                "data_pipeline": "running"
            }
        }
    
    @app.post("/query", response_model=QueryResponse)
    async def query_knowledge_base(request: QueryRequest):
        """Query the knowledge base with natural language."""
        try:
            # Simple response for demo
            answer = f"Based on current market data for {request.symbol or 'the market'}, here's the analysis: {request.query}"
            context = []
            
            return QueryResponse(
                answer=answer,
                context=context,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/anomalies")
    async def get_recent_anomalies(
        limit: int = 10,
        symbol: Optional[str] = None,
        hours: int = 24
    ):
        """Get recent anomalies from the system."""
        try:
            # Read from output files
            anomalies_file = Path("./data/output/alerts.jsonl")
            if not anomalies_file.exists():
                return []
            
            anomalies = []
            try:
                with open(anomalies_file, 'r') as f:
                    for line in f.readlines()[-limit:]:
                        try:
                            data = json.loads(line)
                            if symbol and data.get('symbol') != symbol:
                                continue
                            
                            anomalies.append({
                                "symbol": data.get('symbol', 'Unknown'),
                                "anomaly_score": data.get('anomaly_score', 0),
                                "risk_level": data.get('risk_assessment', {}).get('risk_level', 'LOW'),
                                "explanation": data.get('explanation', 'No explanation available'),
                                "timestamp": data.get('timestamp', datetime.now().isoformat()),
                                "indicators": data.get('processed_data', {}).get('indicators', {})
                            })
                        except:
                            continue
            except:
                pass
            
            return anomalies[-limit:]
        except Exception as e:
            logger.error(f"Error getting anomalies: {str(e)}")
            return []
    
    @app.get("/alerts")
    async def get_active_alerts(
        limit: int = 20,
        priority: Optional[str] = None
    ):
        """Get active alerts from the system."""
        try:
            alerts_file = Path("./data/output/alerts.jsonl").resolve()
            if not alerts_file.exists():
                return {"alerts": [], "total_count": 0, "timestamp": datetime.now().isoformat()}
            
            alerts = []
            try:
                with open(alerts_file, 'r') as f:
                    for line in f.readlines()[-limit:]:
                        try:
                            data = json.loads(line)
                            # Check if this line has alert data
                            if 'alert_sent' in data:
                                alert_data = data.get('alert_sent', {})
                                if priority and alert_data.get('priority') != priority:
                                    continue
                                
                                alerts.append({
                                    'symbol': data.get('symbol'),
                                    'message': alert_data.get('message', ''),
                                    'priority': alert_data.get('priority', 'LOW'),
                                    'timestamp': data.get('timestamp'),
                                    'anomaly_score': data.get('anomaly_score', 0)
                                })
                        except:
                            continue
            except:
                pass
            
            return {
                "alerts": alerts,
                "total_count": len(alerts),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting alerts: {str(e)}")
            return {"alerts": [], "total_count": 0, "timestamp": datetime.now().isoformat()}
    
    @app.get("/market-data")
    async def get_market_data(
        symbol: Optional[str] = None,
        limit: int = 50
    ):
        """Get latest market data."""
        try:
            market_file = Path("./data/output/market_data.jsonl").resolve()
            if not market_file.exists():
                return {"data": [], "count": 0}
            
            data = []
            try:
                with open(market_file, 'r') as f:
                    for line in f.readlines()[-limit:]:
                        try:
                            item = json.loads(line)
                            if symbol and item.get('symbol') != symbol:
                                continue
                            data.append(item)
                        except:
                            continue
            except:
                pass
            
            return {"data": data, "count": len(data), "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return {"data": [], "count": 0, "error": str(e)}
    
    @app.get("/symbols")
    async def get_monitored_symbols():
        """Get list of monitored symbols."""
        return {
            "symbols": settings.SYMBOLS_TO_MONITOR,
            "count": len(settings.SYMBOLS_TO_MONITOR),
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/statistics")
    async def get_system_statistics():
        """Get system statistics."""
        return {
            "active": True,
            "start_time": datetime.now().isoformat(),
            "anomalies_detected": 0,
            "alerts_sent": 0,
            "monitored_symbols": len(settings.SYMBOLS_TO_MONITOR),
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/simulate-anomaly")
    async def simulate_anomaly(
        symbol: str = "AAPL",
        severity: float = 3.0
    ):
        """Simulate an anomaly for demo purposes."""
        try:
            # Create simulated anomaly data
            anomaly_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price": 180.0,
                "volume": 5000000,
                "anomaly_score": severity,
                "processed_data": {
                    "indicators": {
                        "price_change": severity * 2,
                        "volume_ratio": severity,
                        "rsi": 85 if severity > 0 else 15,
                        "volatility": severity * 0.1
                    }
                },
                "alert_sent": {
                    "priority": "HIGH" if severity > 3 else "MEDIUM",
                    "message": f"Anomaly detected for {symbol} with score {severity}"
                }
            }
            
            # Save to alerts file for demo
            alerts_file = Path("./data/output/alerts.jsonl").resolve()
            alerts_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(alerts_file, 'a') as f:
                f.write(json.dumps(anomaly_data) + '\n')
            
            return {
                "message": f"Simulated {severity:.1f} severity anomaly for {symbol}",
                "anomaly_data": anomaly_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# For running with uvicorn directly
if __name__ == "__main__":
    from config.settings import settings
    
    app = create_fastapi_app(settings)
    uvicorn.run(
        app,
        host=settings.FASTAPI_HOST,
        port=settings.FASTAPI_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
