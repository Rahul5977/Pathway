"""
Streamlit Dashboard for Real-Time Market Anomaly Detection
Interactive dashboard showing live market data, anomalies, and AI insights.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Market Anomaly Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #3b82f6;
}

.anomaly-high {
    border-left-color: #ef4444 !important;
    background-color: #fef2f2;
}

.anomaly-medium {
    border-left-color: #f59e0b !important;
    background-color: #fffbeb;
}

.anomaly-low {
    border-left-color: #10b981 !important;
    background-color: #f0fdf4;
}

.alert-container {
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.alert-high {
    background-color: #fef2f2;
    border: 1px solid #fecaca;
}

.alert-medium {
    background-color: #fffbeb;
    border: 1px solid #fed7aa;
}

.alert-low {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
}
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 5  # seconds

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'anomalies_data' not in st.session_state:
    st.session_state.anomalies_data = []
if 'market_data' not in st.session_state:
    st.session_state.market_data = []


def load_data_from_files():
    """Load data from local files if API is not available."""
    market_data = []
    anomalies_data = []
    
    # Load market data
    market_file = Path("./data/output/market_data.jsonl")
    if market_file.exists():
        try:
            with open(market_file, 'r') as f:
                for line in f.readlines()[-50:]:  # Last 50 entries
                    try:
                        data = json.loads(line)
                        market_data.append(data)
                    except:
                        continue
        except:
            pass
    
    # Load anomalies
    alerts_file = Path("./data/output/alerts.jsonl")
    if alerts_file.exists():
        try:
            with open(alerts_file, 'r') as f:
                for line in f.readlines()[-20:]:  # Last 20 entries
                    try:
                        data = json.loads(line)
                        anomalies_data.append(data)
                    except:
                        continue
        except:
            pass
    
    return market_data, anomalies_data


def fetch_api_data(endpoint: str) -> Dict:
    """Fetch data from API with fallback to file loading."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return {"error": "API not available", "data": []}


def query_knowledge_base(query: str, symbol: str = None) -> Dict:
    """Query the knowledge base."""
    try:
        payload = {"query": query}
        if symbol:
            payload["symbol"] = symbol
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"Unable to process query: {str(e)}", "context": []}


def create_price_chart(market_data: List[Dict]) -> go.Figure:
    """Create real-time price chart."""
    if not market_data:
        fig = go.Figure()
        fig.add_annotation(text="No market data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Group by symbol
    symbols = {}
    for item in market_data:
        symbol = item.get('symbol', 'Unknown')
        if symbol not in symbols:
            symbols[symbol] = {'timestamps': [], 'prices': [], 'anomaly_scores': []}
        
        symbols[symbol]['timestamps'].append(item.get('timestamp', ''))
        symbols[symbol]['prices'].append(item.get('price', 0))
        symbols[symbol]['anomaly_scores'].append(item.get('anomaly_score', 0))
    
    fig = go.Figure()
    
    # Add price traces
    for symbol, data in symbols.items():
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=data['prices'],
                mode='lines+markers',
                name=symbol,
                line=dict(width=2),
                hovertemplate=f"<b>{symbol}</b><br>Price: $%{{y:.2f}}<br>Time: %{{x}}<extra></extra>"
            )
        )
    
    fig.update_layout(
        title="Real-Time Market Prices",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=400,
        showlegend=True
    )
    
    return fig


def create_anomaly_chart(anomalies_data: List[Dict]) -> go.Figure:
    """Create anomaly detection chart."""
    if not anomalies_data:
        fig = go.Figure()
        fig.add_annotation(text="No anomalies detected", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    symbols = []
    scores = []
    timestamps = []
    colors = []
    
    for item in anomalies_data:
        symbols.append(item.get('symbol', 'Unknown'))
        score = item.get('anomaly_score', 0)
        scores.append(score)
        timestamps.append(item.get('timestamp', ''))
        
        # Color based on severity
        if score > 4:
            colors.append('red')
        elif score > 2:
            colors.append('orange')
        else:
            colors.append('yellow')
    
    fig = go.Figure(data=go.Scatter(
        x=timestamps,
        y=scores,
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            opacity=0.7
        ),
        text=symbols,
        hovertemplate="<b>%{text}</b><br>Anomaly Score: %{y:.2f}<br>Time: %{x}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Anomaly Detection Timeline",
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        height=300
    )
    
    return fig


def display_alerts(alerts_data: List[Dict]):
    """Display alert cards."""
    if not alerts_data:
        st.info("No active alerts")
        return
    
    for alert in alerts_data[-10:]:  # Show last 10 alerts
        priority = alert.get('priority', 'LOW')
        symbol = alert.get('symbol', 'Unknown')
        message = alert.get('message', 'No message')
        timestamp = alert.get('timestamp', '')
        
        # Color based on priority
        if priority == 'HIGH':
            st.error(f"🚨 **{priority}** - {symbol}: {message}")
        elif priority == 'MEDIUM':
            st.warning(f"⚠️ **{priority}** - {symbol}: {message}")
        else:
            st.info(f"ℹ️ **{priority}** - {symbol}: {message}")
        
        if timestamp:
            st.caption(f"Time: {timestamp}")


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header"><h1>🚀 Real-Time Market Anomaly Detector</h1><p>AI-Powered Financial Market Monitoring</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Demo controls
    st.sidebar.title("🎬 Demo Controls")
    if st.sidebar.button("🎯 Simulate Anomaly"):
        try:
            symbol = st.sidebar.selectbox("Symbol", ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
            severity = st.sidebar.slider("Severity", 1.0, 5.0, 3.0)
            
            response = requests.post(
                f"{API_BASE_URL}/simulate-anomaly",
                params={"symbol": symbol, "severity": severity},
                timeout=5
            )
            if response.status_code == 200:
                st.sidebar.success(f"Simulated anomaly for {symbol}")
            else:
                st.sidebar.error("Failed to simulate anomaly")
        except:
            st.sidebar.error("API not available")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Load data
    try:
        # Try API first, fallback to files
        market_response = fetch_api_data("/market-data")
        alerts_response = fetch_api_data("/alerts")
        
        if "error" in market_response:
            market_data, anomalies_data = load_data_from_files()
            alerts_data = anomalies_data
        else:
            market_data = market_response.get('data', [])
            alerts_data = alerts_response.get('alerts', [])
            anomalies_data = alerts_data
        
        # Metrics
        with col1:
            st.metric("📊 Symbols Monitored", len(set(item.get('symbol', '') for item in market_data)))
        
        with col2:
            active_alerts = len([a for a in alerts_data if a.get('priority') in ['HIGH', 'MEDIUM']])
            st.metric("🚨 Active Alerts", active_alerts)
        
        with col3:
            total_anomalies = len(anomalies_data)
            st.metric("🔍 Anomalies Detected", total_anomalies)
        
        with col4:
            st.metric("⏰ Last Update", datetime.now().strftime("%H:%M:%S"))
        
        # Charts
        st.plotly_chart(create_price_chart(market_data), use_container_width=True)
        st.plotly_chart(create_anomaly_chart(anomalies_data), use_container_width=True)
        
        # Two columns for content
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("🚨 Recent Alerts")
            display_alerts(alerts_data)
        
        with col_right:
            st.subheader("🤖 AI Assistant")
            
            # Query interface
            with st.form("query_form"):
                query = st.text_input("Ask about market conditions:", placeholder="Why is AAPL volatile today?")
                symbol_filter = st.selectbox("Focus on symbol (optional):", [""] + ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
                submitted = st.form_submit_button("Ask AI")
                
                if submitted and query:
                    with st.spinner("Analyzing..."):
                        result = query_knowledge_base(query, symbol_filter if symbol_filter else None)
                        
                        if "error" not in result:
                            st.success("✅ Analysis Complete")
                            st.write(result.get('answer', 'No response'))
                            
                            # Show context if available
                            if result.get('context'):
                                with st.expander("📄 Sources Used"):
                                    for ctx in result['context'][:3]:
                                        st.write(f"• {ctx.get('content', '')[:200]}...")
                        else:
                            st.error(f"❌ {result['error']}")
        
        # Data tables
        if st.checkbox("Show Raw Data"):
            st.subheader("📋 Market Data")
            if market_data:
                df = pd.DataFrame(market_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No market data available")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(REFRESH_INTERVAL)
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Dashboard error: {str(e)}")
        st.info("💡 Make sure the system is running with `./start_system.sh`")


if __name__ == "__main__":
    main()
