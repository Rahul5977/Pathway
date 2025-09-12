"""Logging utility for the Market Anomaly Detector."""

import logging
import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    name: str, 
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both console and file output using loguru.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    # Remove default loguru logger
    logger.remove()
    
    # Console handler with colorful output
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=log_level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="7 days",
            compression="gz"
        )
    
    # Default log file
    else:
        logger.add(
            "./logs/anomaly_detector.log",
            level=log_level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="7 days",
            compression="gz"
        )
    
    # Create a standard logging.Logger that forwards to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Replace standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    return logging.getLogger(name)


# Convenience function for common logging patterns
def log_anomaly_detected(symbol: str, score: float, details: dict):
    """Log anomaly detection with structured format."""
    logger.warning(
        f"🚨 ANOMALY DETECTED: {symbol} | Score: {score:.2f} | Details: {details}"
    )


def log_pipeline_status(component: str, status: str, details: Optional[dict] = None):
    """Log pipeline component status."""
    if details:
        logger.info(f"📊 {component}: {status} | {details}")
    else:
        logger.info(f"📊 {component}: {status}")


def log_data_ingestion(source: str, count: int, timestamp: str):
    """Log data ingestion statistics."""
    logger.debug(f"📥 Data ingested from {source}: {count} records at {timestamp}")


def log_ai_response(agent: str, query: str, response_time: float):
    """Log AI agent responses."""
    logger.info(f"🤖 {agent} responded to query in {response_time:.2f}s: {query[:100]}...")


def log_alert_sent(alert_type: str, recipient: str, details: dict):
    """Log alert notifications."""
    logger.info(f"📢 Alert sent: {alert_type} to {recipient} | {details}")
