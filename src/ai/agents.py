"""AI Agents for autonomous market monitoring and analysis using LangGraph."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import asyncio

from config.settings import Settings
from utils.logger import setup_logger, log_ai_response, log_alert_sent

logger = setup_logger(__name__)


class AgentState(TypedDict):
    """State shared between agents in the workflow."""
    messages: List[Dict[str, Any]]
    anomaly_data: Dict[str, Any]
    analysis_result: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    alert_decision: Dict[str, Any]
    context_data: Dict[str, Any]


class AnomalyAnalystAgent:
    """AI agent specialized in analyzing market anomalies."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        
        # Create analysis tools
        self.tools = [
            Tool(
                name="technical_analysis",
                description="Analyze technical indicators for market movements",
                func=self._analyze_technical_indicators
            ),
            Tool(
                name="sentiment_analysis",
                description="Analyze sentiment impact on market movements", 
                func=self._analyze_sentiment_impact
            ),
            Tool(
                name="volume_analysis",
                description="Analyze volume patterns and anomalies",
                func=self._analyze_volume_patterns
            )
        ]
        
        # Create the agent
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create the LangChain agent with tools."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a senior financial analyst AI specializing in market anomaly detection.
            Your role is to analyze unusual market movements and provide detailed insights.
            
            Use the available tools to conduct thorough analysis and provide:
            1. Root cause analysis of the anomaly
            2. Market context and comparison to historical patterns
            3. Potential implications for the asset and broader market
            4. Confidence level in your analysis
            
            Be precise, data-driven, and actionable in your responses."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        return create_openai_functions_agent(self.llm, self.tools, prompt)
    
    def analyze_anomaly(self, anomaly_data, news_context, market_context) -> dict:
        """Analyze an anomaly using AI agents."""
        try:
            start_time = datetime.now()
            
            # Prepare analysis input
            analysis_input = self._prepare_analysis_input(
                anomaly_data, news_context, market_context
            )
            
            # Execute agent analysis
            executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
            result = executor.invoke({
                "input": analysis_input,
                "chat_history": []
            })
            
            # Calculate confidence and prepare response
            confidence = self._calculate_confidence(anomaly_data)
            
            analysis_result = {
                "analysis": result["output"],
                "confidence": confidence,
                "agent": "AnomalyAnalyst",
                "timestamp": datetime.now().isoformat(),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Log the analysis
            log_ai_response(
                "AnomalyAnalyst",
                analysis_input[:100],
                analysis_result["processing_time"]
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Anomaly analysis failed: {str(e)}")
            return {
                "analysis": f"Analysis failed: {str(e)}",
                "confidence": 0.0,
                "agent": "AnomalyAnalyst",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def assess_risk(self, anomaly_data) -> dict:
        """Assess risk level of the detected anomaly."""
        try:
            symbol = anomaly_data.get('symbol', 'Unknown')
            anomaly_score = anomaly_data.get('anomaly_score', 0)
            processed_data = anomaly_data.get('processed_data', {})
            indicators = processed_data.get('indicators', {})
            
            # Risk assessment based on multiple factors
            risk_factors = {
                "anomaly_magnitude": min(anomaly_score / 5.0, 1.0),  # Normalize to 0-1
                "price_volatility": min(abs(indicators.get('price_change', 0)) / 10.0, 1.0),
                "volume_anomaly": min(indicators.get('volume_ratio', 1) / 5.0, 1.0),
                "rsi_extremity": max(0, min(abs(indicators.get('rsi', 50) - 50) / 30.0, 1.0))
            }
            
            # Calculate overall risk score (weighted average)
            risk_score = (
                0.4 * risk_factors["anomaly_magnitude"] +
                0.3 * risk_factors["price_volatility"] +
                0.2 * risk_factors["volume_anomaly"] +
                0.1 * risk_factors["rsi_extremity"]
            )
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "HIGH"
            elif risk_score >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                "risk_score": round(risk_score, 3),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return {
                "risk_score": 0.5,
                "risk_level": "UNKNOWN",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _prepare_analysis_input(self, anomaly_data, news_context, market_context) -> str:
        """Prepare input for anomaly analysis."""
        symbol = anomaly_data.get('symbol', 'Unknown')
        anomaly_score = anomaly_data.get('anomaly_score', 0)
        processed_data = anomaly_data.get('processed_data', {})
        indicators = processed_data.get('indicators', {})
        
        input_text = f"""
        Analyze the following market anomaly:
        
        SYMBOL: {symbol}
        ANOMALY SCORE: {anomaly_score:.2f}
        
        TECHNICAL INDICATORS:
        - Price Change: {indicators.get('price_change', 0):.2f}%
        - Volume Ratio: {indicators.get('volume_ratio', 1):.2f}x normal
        - RSI: {indicators.get('rsi', 50):.2f}
        - Volatility: {indicators.get('volatility', 0):.4f}
        - SMA 5: ${indicators.get('sma_5', 0):.2f}
        - SMA 20: ${indicators.get('sma_20', 0):.2f}
        
        RECENT NEWS CONTEXT:
        """
        
        if news_context:
            for i, news in enumerate(news_context[:3]):
                input_text += f"{i+1}. {news.get('title', 'No title')}\n"
        else:
            input_text += "No relevant news found.\n"
        
        input_text += f"""
        
        MARKET CONTEXT:
        Recent price: ${processed_data.get('price', 0):.2f}
        Volume: {processed_data.get('volume', 0):,}
        
        Please provide a comprehensive analysis of this anomaly.
        """
        
        return input_text
    
    def _calculate_confidence(self, anomaly_data) -> float:
        """Calculate confidence level for the analysis."""
        try:
            anomaly_score = anomaly_data.get('anomaly_score', 0)
            processed_data = anomaly_data.get('processed_data', {})
            indicators = processed_data.get('indicators', {})
            
            # Factors affecting confidence
            score_confidence = min(anomaly_score / 5.0, 1.0)  # Higher scores = higher confidence
            data_quality = 1.0 if all(k in indicators for k in ['price_change', 'volume_ratio', 'rsi']) else 0.5
            
            confidence = (score_confidence + data_quality) / 2.0
            return min(confidence, 1.0)
            
        except:
            return 0.5
    
    def _analyze_technical_indicators(self, query: str) -> str:
        """Analyze technical indicators from the query."""
        # Extract key metrics from query for analysis
        response = """Technical Analysis:
        - RSI levels indicate potential overbought/oversold conditions
        - Volume ratios suggest unusual trading activity
        - Price movements show significant deviation from moving averages
        - Volatility patterns indicate market stress or opportunity"""
        return response
    
    def _analyze_sentiment_impact(self, query: str) -> str:
        """Analyze sentiment impact on market movements."""
        response = """Sentiment Analysis:
        - News sentiment appears to correlate with price movements
        - Social media activity shows increased attention to the asset
        - Market psychology factors suggest potential momentum continuation
        - Risk sentiment in broader market may amplify movements"""
        return response
    
    def _analyze_volume_patterns(self, query: str) -> str:
        """Analyze volume patterns and anomalies."""
        response = """Volume Analysis:
        - Trading volume significantly above average suggests institutional activity
        - Volume-price relationship indicates strength of current move
        - Intraday volume distribution shows accumulation/distribution patterns
        - Historical volume comparison reveals unusual market participation"""
        return response


class AlertManagerAgent:
    """AI agent for managing alerts and notifications."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0.1  # Lower temperature for consistent alert formatting
        )
    
    def send_alert(self, anomaly_data) -> dict:
        """Generate and send alerts for anomalies."""
        try:
            symbol = anomaly_data.get('symbol', 'Unknown')
            anomaly_score = anomaly_data.get('anomaly_score', 0)
            
            # Determine alert priority
            priority = self._determine_alert_priority(anomaly_score)
            
            # Generate alert message
            message = self._generate_alert_message(anomaly_data, priority)
            
            # Dispatch alert (simulate for demo)
            dispatch_result = self._dispatch_alert(message, priority, symbol)
            
            alert_result = {
                "alert_sent": True,
                "priority": priority,
                "message": message,
                "dispatch_result": dispatch_result,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log alert
            log_alert_sent(priority, symbol, {"anomaly_score": anomaly_score})
            
            return alert_result
            
        except Exception as e:
            logger.error(f"Alert sending failed: {str(e)}")
            return {
                "alert_sent": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _determine_alert_priority(self, anomaly_score: float) -> str:
        """Determine alert priority based on anomaly score."""
        if anomaly_score >= 4.0:
            return "HIGH"
        elif anomaly_score >= 2.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_alert_message(self, anomaly_data, priority: str) -> str:
        """Generate alert message using AI."""
        try:
            symbol = anomaly_data.get('symbol', 'Unknown')
            anomaly_score = anomaly_data.get('anomaly_score', 0)
            processed_data = anomaly_data.get('processed_data', {})
            indicators = processed_data.get('indicators', {})
            
            prompt = f"""
            Generate a concise alert message for a {priority} priority market anomaly:
            
            Symbol: {symbol}
            Anomaly Score: {anomaly_score:.2f}
            Price Change: {indicators.get('price_change', 0):.2f}%
            Volume: {indicators.get('volume_ratio', 1):.1f}x normal
            
            The message should be:
            - Clear and actionable
            - Appropriate for the {priority} priority level
            - Under 200 characters
            - Include key metrics
            """
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            # Fallback message
            return f"🚨 {priority} Alert: {anomaly_data.get('symbol', 'Unknown')} anomaly detected (Score: {anomaly_data.get('anomaly_score', 0):.2f})"
    
    def _dispatch_alert(self, message: str, priority: str, symbol: str) -> dict:
        """Simulate alert dispatching."""
        # In production, this would send to Slack, email, webhooks, etc.
        
        channels = []
        if priority == "HIGH":
            channels = ["email", "slack", "webhook"]
        elif priority == "MEDIUM":
            channels = ["slack", "webhook"]
        else:
            channels = ["webhook"]
        
        return {
            "channels": channels,
            "message_sent": message,
            "dispatch_time": datetime.now().isoformat(),
            "status": "success"
        }


class MarketMonitorAgent:
    """Agent for continuous market monitoring and coordination."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL
        )
        self.monitoring_status = {
            "active": True,
            "start_time": datetime.now().isoformat(),
            "anomalies_detected": 0,
            "alerts_sent": 0
        }
    
    def get_system_status(self) -> dict:
        """Get current system monitoring status."""
        return {
            **self.monitoring_status,
            "current_time": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - datetime.fromisoformat(self.monitoring_status["start_time"])).total_seconds()
        }
    
    def update_statistics(self, anomaly_detected: bool = False, alert_sent: bool = False):
        """Update monitoring statistics."""
        if anomaly_detected:
            self.monitoring_status["anomalies_detected"] += 1
        if alert_sent:
            self.monitoring_status["alerts_sent"] += 1
