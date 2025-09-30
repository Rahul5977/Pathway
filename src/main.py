import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
import pathway as pw
from datetime import datetime

from config.settings import Settings
from data.connectors import MarketDataConnector, NewsConnector
from data.processors_simple import AnomalyDetector, DataProcessor
from ai.rag_simple import RAGEngine
from ai.agents_simple import AnomalyAnalystAgent, AlertManagerAgent
from api.server import create_fastapi_app
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)


class MarketAnomalyDetectorPipeline:
    """Main Pathway pipeline for real-time market anomaly detection."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rag_engine = RAGEngine(settings)
        self.anomaly_agent = AnomalyAnalystAgent(settings)
        self.alert_agent = AlertManagerAgent(settings)
        
    def create_pipeline(self):
        """Create the main Pathway processing pipeline."""
        
        # 1. Data Ingestion - Multiple sources in real-time
        market_connector = MarketDataConnector(self.settings)
        news_connector = NewsConnector(self.settings)
        
        # Market data stream (prices, volumes, etc.)
        market_stream = market_connector.create_stream()
        
        # News and sentiment stream
        news_stream = news_connector.create_stream()
        
        # 2. Real-time data processing and anomaly detection
        processor = DataProcessor()
        anomaly_detector = AnomalyDetector(self.settings)
        
        # Process market data for anomalies
        processed_market = market_stream.select(
            timestamp=pw.this.timestamp,
            symbol=pw.this.symbol,
            price=pw.this.price,
            volume=pw.this.volume,
            processed_data=processor.process_market_data(pw.this)
        )
        
        # Detect anomalies in real-time
        market_with_anomalies = processed_market.select(
            **pw.this,
            anomaly_score=anomaly_detector.detect_anomaly(pw.this)
        )
        
        anomalies = market_with_anomalies.select(
            **pw.this,
            is_anomaly=anomaly_detector.is_anomaly(pw.this.anomaly_score)
        ).filter(pw.this.is_anomaly)
        
        # 3. Live RAG indexing - Index news and market data for context
        indexed_news = self.rag_engine.index_documents(news_stream)
        indexed_market = self.rag_engine.index_market_data(market_with_anomalies)
        
        # 4. AI Agent processing for anomalies
        analyzed_anomalies = anomalies.select(
            **pw.this,
            ai_analysis=self.anomaly_agent.analyze_anomaly(
                pw.this, 
                indexed_news, 
                indexed_market
            ),
            risk_assessment=self.anomaly_agent.assess_risk(pw.this),
            explanation=self.rag_engine.explain_anomaly(pw.this)
        )
        
        # 5. Alert generation and notification
        alerts = analyzed_anomalies.select(
            **pw.this,
            alert_sent=self.alert_agent.send_alert(pw.this)
        )
        
        # 6. Output to various sinks for dashboard consumption
        pw.io.jsonlines.write(
            alerts,
            "./data/output/alerts.jsonl"
        )
        
        pw.io.jsonlines.write(
            market_with_anomalies,
            "./data/output/market_data.jsonl"
        )
        
        return {
            'market_stream': market_stream,
            'market_with_anomalies': market_with_anomalies,
            'anomalies': anomalies,
            'analyzed_anomalies': analyzed_anomalies,
            'alerts': alerts
        }


async def start_api_server(settings: Settings):
    """Start the FastAPI server for external queries."""
    import uvicorn
    
    app = create_fastapi_app(settings)
    config = uvicorn.Config(
        app, 
        host=settings.FASTAPI_HOST, 
        port=settings.FASTAPI_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
    server = uvicorn.Server(config)
    await server.serve()


def main():
    """Main entry point for the application."""
    logger.info("🚀 Starting Real-Time Market Anomaly Detector")
    
    # Load configuration
    settings = Settings()
    
    # Create output directories
    Path("./data/output").mkdir(parents=True, exist_ok=True)
    Path("./data/persistence").mkdir(parents=True, exist_ok=True)
    
    # Initialize and run the Pathway pipeline
    pipeline = MarketAnomalyDetectorPipeline(settings)
    streams = pipeline.create_pipeline()
    
    logger.info("✅ Pathway pipeline created successfully")
    logger.info("📊 Starting real-time data processing...")
    
    # Run the pipeline
    try:
        # Start API server in background
        if settings.ENABLE_API:
            import threading
            api_thread = threading.Thread(
                target=lambda: asyncio.run(start_api_server(settings))
            )
            api_thread.daemon = True
            api_thread.start()
            logger.info(f"🌐 API server starting on {settings.FASTAPI_HOST}:{settings.FASTAPI_PORT}")
        
        # Run Pathway pipeline (blocking)
        pw.run(
            monitoring_level=pw.MonitoringLevel.AUTO,
            persistence_config=pw.persistence.Config.simple_config(
                pw.persistence.Backend.filesystem(settings.PATHWAY_PERSISTENCE_DIR)
            )
        )
         
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Pipeline error: {str(e)}")
        raise
    finally:
        logger.info("🏁 Application stopped")


if __name__ == "__main__":
    main()
