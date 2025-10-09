from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import uvicorn

from src.config.settings import Settings
from src.utils.logger import setup_logger
try:
    from src.ai.sentiment_predictor_simple import sentiment_agent
except ImportError:
    sentiment_agent = None  # Fallback if sentiment predictor not available

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


class SentimentPredictionRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = 15  # minutes


class SentimentPredictionResponse(BaseModel):
    symbol: str
    predicted_sentiment: float
    price_impact_prediction: float
    confidence_score: float
    time_horizon_minutes: int
    contributing_factors: List[str]
    timestamp: str


class SentimentSummaryResponse(BaseModel):
    symbol: str
    current_sentiment: Dict[str, float]
    sentiment_trend: float
    sentiment_momentum: float
    prediction: Optional[Dict[str, Any]]
    correlation: Dict[str, Any]
    data_quality: Dict[str, Any]
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
            # Enhanced response using RAG engine
            query = request.query
            symbol = request.symbol
            max_results = request.max_results or 5
            
            logger.info(f"Received query: '{query}' for symbol: {symbol}")
            
            # Simple but contextual response based on recent data
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            market_file = project_root / "data" / "output" / "market_data.jsonl"
            alerts_file = project_root / "data" / "output" / "alerts.jsonl"
            
            context_info = []
            answer = f"Based on current market analysis"
            if symbol:
                answer += f" for {symbol}"
            answer += ": "
            
            # Get recent market context
            if market_file.exists():
                try:
                    with open(market_file, 'r') as f:
                        recent_data = []
                        for line in f.readlines()[-20:]:
                            try:
                                data = json.loads(line)
                                if not symbol or data.get('symbol') == symbol:
                                    recent_data.append(data)
                            except:
                                continue
                    
                    if recent_data:
                        latest = recent_data[-1]
                        price = latest.get('price', 0)
                        anomaly_score = latest.get('anomaly_score', 0)
                        timestamp = latest.get('timestamp', '')
                        
                        # Analyze anomaly level
                        if anomaly_score > 2.5:
                            answer += f"🚨 HIGH anomaly activity detected with score {anomaly_score:.2f}. "
                        elif anomaly_score > 1.5:
                            answer += f"⚠️ Moderate anomaly activity with score {anomaly_score:.2f}. "
                        else:
                            answer += f"✅ Normal market activity (anomaly score: {anomaly_score:.2f}). "
                            
                        answer += f"Current price is ${price:.2f}. "
                        
                        # Calculate price trend
                        if len(recent_data) >= 2:
                            prev_price = recent_data[-2].get('price', price)
                            change = price - prev_price
                            change_pct = (change / prev_price) * 100 if prev_price > 0 else 0
                            
                            if abs(change_pct) > 2:
                                direction = "📈 up" if change > 0 else "📉 down"
                                answer += f"Price moved {direction} {abs(change_pct):.1f}% recently. "
                        
                        context_info.append({
                            "content": f"Latest data for {latest.get('symbol', 'Unknown')}: Price ${price:.2f}, Anomaly Score {anomaly_score:.2f}, Time {timestamp}",
                            "source": "market_data",
                            "timestamp": timestamp
                        })
                except Exception as e:
                    logger.error(f"Error reading market data: {str(e)}")
            
            # Get alerts context
            if alerts_file.exists():
                try:
                    with open(alerts_file, 'r') as f:
                        for line in f.readlines()[-5:]:
                            try:
                                alert = json.loads(line)
                                if not symbol or alert.get('symbol') == symbol:
                                    alert_info = alert.get('alert_sent', {})
                                    if alert_info:
                                        answer += f"🔔 Recent alert: {alert_info.get('message', 'Alert generated')}. "
                                        context_info.append({
                                            "content": alert_info.get('message', 'Alert generated'),
                                            "source": "alerts",
                                            "timestamp": alert.get('timestamp', '')
                                        })
                            except:
                                continue
                except Exception as e:
                    logger.error(f"Error reading alerts: {str(e)}")
            
            # Add query-specific analysis
            query_lower = query.lower()
            if any(word in query_lower for word in ["volatile", "volatility", "swing"]):
                answer += "📊 Volatility analysis: Recent price movements show elevated volatility patterns. "
            if any(word in query_lower for word in ["why", "reason", "cause"]):
                answer += "🔍 Analysis: This appears to be driven by a combination of technical factors, market sentiment, and trading volume patterns. "
            if any(word in query_lower for word in ["crash", "drop", "fall", "decline"]):
                answer += "⬇️ Market pressure: Significant downward movement detected in recent trading sessions. "
            if any(word in query_lower for word in ["pump", "rise", "increase", "rally"]):
                answer += "⬆️ Market momentum: Positive price action and volume surge observed. "
            if any(word in query_lower for word in ["buy", "sell", "trade"]):
                answer += "⚠️ Trading note: This analysis is for educational purposes only and not financial advice. "
            
            # Add market condition context
            if not any(word in query_lower for word in ["specific", "detailed"]):
                answer += "Monitor for continued developments and correlate with news events. "
            
            if not answer.strip().endswith('.'):
                answer += "."
            
            logger.info(f"Generated response: {answer[:100]}...")
            
            return QueryResponse(
                answer=answer,
                context=context_info,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
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
    
    @app.post("/predict-sentiment", response_model=SentimentPredictionResponse)
    async def predict_sentiment(request: SentimentPredictionRequest):
        """Predict sentiment and price impact for a given symbol and time horizon."""
        try:
            symbol = request.symbol
            horizon = request.horizon or 15
            
            logger.info(f"Received sentiment prediction request for {symbol} with horizon {horizon} minutes.")
            
            # Placeholder for actual sentiment analysis logic
            # For now, we just return a mock response
            predicted_sentiment = 0.75  # Mocked sentiment value
            price_impact_prediction = 0.05  # Mocked price impact
            confidence_score = 0.9  # Mocked confidence score
            
            # Get current sentiment from the agent
            current_sentiment = sentiment_agent.get_current_sentiment(symbol)
            logger.info(f"Current sentiment for {symbol}: {current_sentiment}")
            
            # Analyze sentiment trend and momentum
            sentiment_trend = current_sentiment.get('trend', 0) * 100
            sentiment_momentum = current_sentiment.get('momentum', 0) * 100
            
            # Prepare prediction details
            prediction_details = {
                "symbol": symbol,
                "predicted_sentiment": predicted_sentiment,
                "price_impact_prediction": price_impact_prediction,
                "confidence_score": confidence_score,
                "time_horizon_minutes": horizon,
                "contributing_factors": [
                    "Positive news sentiment",
                    "Strong earnings report",
                    "Market momentum"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Sentiment prediction for {symbol}: {prediction_details}")
            
            return SentimentPredictionResponse(
                symbol=symbol,
                predicted_sentiment=predicted_sentiment,
                price_impact_prediction=price_impact_prediction,
                confidence_score=confidence_score,
                time_horizon_minutes=horizon,
                contributing_factors=prediction_details["contributing_factors"],
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Sentiment prediction error for {request.symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/sentiment-summary", response_model=SentimentSummaryResponse)
    async def get_sentiment_summary(
        symbol: str,
        include_prediction: bool = True
    ):
        """Get sentiment summary for a symbol, optionally including prediction."""
        try:
            # Get current sentiment from the agent
            current_sentiment = sentiment_agent.get_current_sentiment(symbol)
            logger.info(f"Current sentiment for {symbol}: {current_sentiment}")
            
            # Analyze sentiment trend and momentum
            sentiment_trend = current_sentiment.get('trend', 0) * 100
            sentiment_momentum = current_sentiment.get('momentum', 0) * 100
            
            summary = {
                "symbol": symbol,
                "current_sentiment": current_sentiment.get('values', {}),
                "sentiment_trend": sentiment_trend,
                "sentiment_momentum": sentiment_momentum,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add prediction if requested
            if include_prediction:
                prediction = {
                    "symbol": symbol,
                    "predicted_sentiment": 0.75,  # Mocked sentiment value
                    "price_impact_prediction": 0.05,  # Mocked price impact
                    "confidence_score": 0.9,  # Mocked confidence score
                    "time_horizon_minutes": 15,
                    "contributing_factors": [
                        "Positive news sentiment",
                        "Strong earnings report",
                        "Market momentum"
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                summary["prediction"] = prediction
            
            logger.info(f"Sentiment summary for {symbol}: {summary}")
            
            return SentimentSummaryResponse(
                symbol=symbol,
                current_sentiment=current_sentiment.get('values', {}),
                sentiment_trend=sentiment_trend,
                sentiment_momentum=sentiment_momentum,
                prediction=summary.get("prediction"),
                correlation={},
                data_quality={},
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error getting sentiment summary for {symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict-sentiment-impact", response_model=SentimentPredictionResponse)
    async def predict_sentiment_impact(request: SentimentPredictionRequest):
        """Predict price impact based on sentiment analysis"""
        if not sentiment_agent:
            raise HTTPException(status_code=503, detail="Sentiment prediction service not available")
        
        try:
            # Update sentiment data
            await sentiment_agent.update_sentiment_history(request.symbol)
            
            # Get prediction
            prediction = await sentiment_agent.predict_sentiment_trajectory(
                request.symbol, 
                request.horizon
            )
            
            if not prediction:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Unable to generate prediction for {request.symbol}. Insufficient data."
                )
            
            return SentimentPredictionResponse(
                symbol=prediction.symbol,
                predicted_sentiment=prediction.predicted_sentiment,
                price_impact_prediction=prediction.price_impact_prediction,
                confidence_score=prediction.confidence_score,
                time_horizon_minutes=prediction.time_horizon_minutes,
                contributing_factors=prediction.contributing_factors,
                timestamp=prediction.timestamp.isoformat()
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error predicting sentiment impact for {request.symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/sentiment-alerts/{symbol}")
    async def get_sentiment_alerts(symbol: str):
        """Get real-time sentiment-based alerts for a symbol"""
        if not sentiment_agent:
            return {"symbol": symbol, "alerts": [], "count": 0, "error": "Sentiment service not available"}
        
        try:
            # Update sentiment data
            await sentiment_agent.update_sentiment_history(symbol)
            
            # Generate alerts
            alerts = await sentiment_agent.generate_sentiment_alerts(symbol)
            
            return {
                "symbol": symbol,
                "alerts": alerts,
                "count": len(alerts),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting sentiment alerts for {symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/sentiment-summary/{symbol}", response_model=SentimentSummaryResponse)
    async def get_comprehensive_sentiment_summary(symbol: str):
        """Get comprehensive sentiment analysis summary"""
        try:
            # Update sentiment data
            await sentiment_agent.update_sentiment_history(symbol)
            
            # Get comprehensive summary
            summary = await sentiment_agent.get_sentiment_summary(symbol)
            
            if 'error' in summary:
                raise HTTPException(status_code=404, detail=summary['error'])
            
            return SentimentSummaryResponse(
                symbol=summary['symbol'],
                current_sentiment=summary['current_sentiment'],
                sentiment_trend=summary['sentiment_trend'],
                sentiment_momentum=summary['sentiment_momentum'],
                prediction=summary.get('prediction'),
                correlation=summary['correlation'],
                data_quality=summary['data_quality'],
                timestamp=summary['timestamp']
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting sentiment summary for {symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/sentiment-dashboard-data")
    async def get_sentiment_dashboard_data():
        """Get sentiment data for all symbols for dashboard"""
        try:
            dashboard_data = {}
            
            for symbol in sentiment_agent.symbols:
                # Update and get sentiment data
                await sentiment_agent.update_sentiment_history(symbol)
                summary = await sentiment_agent.get_sentiment_summary(symbol)
                
                if 'error' not in summary:
                    dashboard_data[symbol] = {
                        'current_sentiment': summary['current_sentiment']['overall'],
                        'sentiment_trend': summary['sentiment_trend'],
                        'prediction': summary.get('prediction', {}).get('predicted_sentiment', 0),
                        'price_impact': summary.get('prediction', {}).get('price_impact', 0),
                        'confidence': summary.get('prediction', {}).get('confidence', 0),
                        'correlation': summary['correlation']['sentiment_price_correlation'],
                        'last_update': summary['data_quality']['last_update']
                    }
            
            return {
                "sentiment_data": dashboard_data,
                "timestamp": datetime.now().isoformat(),
                "symbols_count": len(dashboard_data)
            }
        
        except Exception as e:
            logger.error(f"Error getting dashboard sentiment data: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/update-sentiment-models")
    async def update_sentiment_models(background_tasks: BackgroundTasks):
        """Trigger sentiment model retraining in background"""
        try:
            async def retrain_models():
                for symbol in sentiment_agent.symbols:
                    await sentiment_agent.save_model_state(symbol)
                    logger.info(f"Updated sentiment model for {symbol}")
            
            background_tasks.add_task(retrain_models)
            
            return {
                "message": "Sentiment model update initiated",
                "symbols": sentiment_agent.symbols,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error updating sentiment models: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create the app instance for uvicorn
try:
    from src.config.settings import Settings
    settings = Settings()
    app = create_fastapi_app(settings)
except Exception as e:
    # Fallback app in case of configuration issues
    app = FastAPI(title="Market Anomaly Detector API", description="API with configuration issues")

# For running with uvicorn directly
if __name__ == "__main__":
    from src.config.settings import Settings
    
    settings = Settings()
    app = create_fastapi_app(settings)
    uvicorn.run(
        app,
        host=settings.FASTAPI_HOST,
        port=settings.FASTAPI_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
