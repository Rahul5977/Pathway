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
    
    # Use absolute paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    market_file = project_root / "data" / "output" / "market_data.jsonl"
    alerts_file = project_root / "data" / "output" / "alerts.jsonl"
    
    # Load market data
    if market_file.exists():
        try:
            with open(market_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-100:]:  # Last 100 entries for better visualization
                    try:
                        data = json.loads(line.strip())
                        market_data.append(data)
                    except:
                        continue
        except Exception as e:
            st.error(f"Error loading market data: {str(e)}")
    
    # Load anomalies/alerts
    if alerts_file.exists():
        try:
            with open(alerts_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-50:]:  # Last 50 entries
                    try:
                        data = json.loads(line.strip())
                        # Handle the new alert structure
                        if data.get('alert_sent') and data.get('is_anomaly'):
                            anomaly_item = {
                                'symbol': data.get('symbol', 'Unknown'),
                                'anomaly_score': data.get('anomaly_score', 0),
                                'timestamp': data.get('timestamp', ''),
                                'priority': 'HIGH' if data.get('anomaly_score', 0) > 4 else 'MEDIUM' if data.get('anomaly_score', 0) > 2 else 'LOW',
                                'message': data.get('ai_analysis', 'Alert generated'),
                                'explanation': data.get('explanation', ''),
                                'price': data.get('price', 0),
                                'risk_assessment': data.get('risk_assessment', 'LOW')
                            }
                            anomalies_data.append(anomaly_item)
                    except:
                        continue
        except Exception as e:
            st.error(f"Error loading alerts data: {str(e)}")
    
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
    
    # Group by symbol and sort by timestamp
    symbols = {}
    for item in market_data:
        symbol = item.get('symbol', 'Unknown')
        if symbol not in symbols:
            symbols[symbol] = {'timestamps': [], 'prices': [], 'anomaly_scores': []}
        
        # Parse timestamp
        timestamp_str = item.get('timestamp', '')
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            timestamp = datetime.now()
        
        symbols[symbol]['timestamps'].append(timestamp)
        symbols[symbol]['prices'].append(float(item.get('price', 0)))
        symbols[symbol]['anomaly_scores'].append(float(item.get('anomaly_score', 0)))
    
    # Sort data by timestamp for each symbol
    for symbol in symbols:
        data = symbols[symbol]
        sorted_indices = sorted(range(len(data['timestamps'])), key=lambda i: data['timestamps'][i])
        data['timestamps'] = [data['timestamps'][i] for i in sorted_indices]
        data['prices'] = [data['prices'][i] for i in sorted_indices]
        data['anomaly_scores'] = [data['anomaly_scores'][i] for i in sorted_indices]
    
    fig = go.Figure()
    
    # Add price traces with color coding based on anomaly score
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, (symbol, data) in enumerate(symbols.items()):
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=data['prices'],
                mode='lines+markers',
                name=symbol,
                line=dict(width=2, color=colors[i % len(colors)]),
                marker=dict(size=4),
                hovertemplate=f"<b>{symbol}</b><br>Price: $%{{y:.2f}}<br>Time: %{{x}}<br>Anomaly Score: %{{customdata:.2f}}<extra></extra>",
                customdata=data['anomaly_scores']
            )
        )
    
    fig.update_layout(
        title="Real-Time Market Prices",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=400,
        showlegend=True,
        xaxis=dict(type='date'),
        hovermode='x unified'
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
    messages = []
    
    for item in anomalies_data:
        symbols.append(item.get('symbol', 'Unknown'))
        score = float(item.get('anomaly_score', 0))
        scores.append(score)
        
        # Parse timestamp
        timestamp_str = item.get('timestamp', '')
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            timestamp = datetime.now()
        timestamps.append(timestamp)
        
        messages.append(item.get('message', item.get('explanation', 'Anomaly detected')))
        
        # Color based on severity
        if score > 4:
            colors.append('red')
        elif score > 2:
            colors.append('orange')
        else:
            colors.append('green')
    
    fig = go.Figure(data=go.Scatter(
        x=timestamps,
        y=scores,
        mode='markers',
        marker=dict(
            size=12,
            color=colors,
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        text=symbols,
        customdata=messages,
        hovertemplate="<b>%{text}</b><br>Anomaly Score: %{y:.2f}<br>Time: %{x}<br>Details: %{customdata}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Anomaly Detection Timeline",
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        height=300,
        xaxis=dict(type='date'),
        yaxis=dict(range=[0, max(scores) + 1 if scores else 5])
    )
    
    return fig
    
    return fig


def display_alerts(alerts_data: List[Dict]):
    """Display alert cards."""
    if not alerts_data:
        st.info("No active alerts")
        return
    
    # Sort alerts by timestamp (most recent first)
    sorted_alerts = sorted(alerts_data, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    for alert in sorted_alerts[:10]:  # Show last 10 alerts
        priority = alert.get('priority', 'LOW')
        symbol = alert.get('symbol', 'Unknown')
        message = alert.get('message', 'No message')
        timestamp = alert.get('timestamp', '')
        explanation = alert.get('explanation', '')
        anomaly_score = alert.get('anomaly_score', 0)
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%H:%M:%S")
        except:
            time_str = timestamp
        
        # Create alert container
        with st.container():
            # Color based on priority
            if priority == 'HIGH':
                st.error(f"🚨 **{priority}** - {symbol}: {message}")
            elif priority == 'MEDIUM':
                st.warning(f"⚠️ **{priority}** - {symbol}: {message}")
            else:
                st.info(f"ℹ️ **{priority}** - {symbol}: {message}")
            
            # Additional details
            col1, col2 = st.columns(2)
            with col1:
                if time_str:
                    st.caption(f"⏰ Time: {time_str}")
            with col2:
                st.caption(f"📊 Score: {anomaly_score:.2f}")
            
            if explanation:
                st.caption(f"💡 {explanation}")
            
            st.divider()


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header"><h1>🚀 Real-Time Market Anomaly Detector</h1><p>AI-Powered Financial Market Monitoring</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    # Auto-refresh toggle with live update capability
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_rate = st.sidebar.selectbox("Refresh Rate", [5, 10, 15, 30], index=0, format_func=lambda x: f"{x} seconds")
    
    # Update refresh interval based on selection
    REFRESH_INTERVAL = refresh_rate
    
    # Force refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()  # Clear any cached data
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
    
    # Load data with better error handling and cache busting
    try:
        # Load fresh data using cached function
        market_data, anomalies_data = get_fresh_data()
        
        # Store in session state for reference
        st.session_state.market_data = market_data
        st.session_state.anomalies_data = anomalies_data
        
        # Debug info
        if st.sidebar.checkbox("Show Debug Info", value=False):
            st.sidebar.text(f"Market data: {len(market_data)} points")
            st.sidebar.text(f"Anomalies: {len(anomalies_data)} points")
            if market_data:
                latest_time = market_data[-1].get('timestamp', 'Unknown')
                st.sidebar.text(f"Latest: {latest_time[-8:] if len(latest_time) > 8 else latest_time}")
        
        # Also try API for additional data if available
        try:
            market_response = fetch_api_data("/market-data")
            alerts_response = fetch_api_data("/alerts")
            
            if "error" not in market_response and market_response.get('data'):
                api_market_data = market_response.get('data', [])
                # Merge with file data (prefer more recent data)
                all_symbols = set(item.get('symbol') for item in market_data + api_market_data)
                market_data = api_market_data if api_market_data else market_data
            
            if "error" not in alerts_response and alerts_response.get('alerts'):
                alerts_data = alerts_response.get('alerts', [])
                anomalies_data = alerts_data if alerts_data else anomalies_data
        except:
            # Use file data as fallback
            pass
        
        alerts_data = anomalies_data
        
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
        
        # Charts with real-time data
        chart_container = st.container()
        with chart_container:
            # Add timestamp to show when charts were last updated
            col_chart1, col_chart2 = st.columns([3, 1])
            with col_chart1:
                st.plotly_chart(create_price_chart(market_data), use_container_width=True, key=f"price_chart_{datetime.now().timestamp()}")
            with col_chart2:
                st.metric("📊 Data Points", len(market_data))
                st.metric("🔄 Last Update", datetime.now().strftime("%H:%M:%S"))
        
        st.plotly_chart(create_anomaly_chart(anomalies_data), use_container_width=True, key=f"anomaly_chart_{datetime.now().timestamp()}")
        
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
        
        # Simple and reliable auto-refresh with JavaScript timer
        if auto_refresh:
            # Get current time and last refresh time
            current_time = datetime.now()
            time_since_refresh = (current_time - st.session_state.last_refresh).total_seconds()
            
            # Show countdown in sidebar
            remaining = max(0, int(REFRESH_INTERVAL - time_since_refresh))
            
            if remaining > 0:
                st.sidebar.info(f"🔄 Auto-refresh in {remaining}s")
                
                # Add a simple JavaScript timer that refreshes the page
                st.markdown(
                    f"""
                    <script>
                    setTimeout(function(){{
                        window.location.reload(1);
                    }}, {remaining * 1000});
                    </script>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.sidebar.success("🔄 Refreshing...")
                st.session_state.last_refresh = current_time
                st.rerun()
        
    except Exception as e:
        st.error(f"❌ Dashboard error: {str(e)}")
        st.info("💡 Make sure the system is running with `./start_system.sh`")


# Cache function for real-time data loading
@st.cache_data(ttl=1)  # Cache for only 1 second to ensure fresh data
def get_fresh_data():
    return load_data_from_files()


if __name__ == "__main__":
    main()
