import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Market Anomaly Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 60  # seconds

def load_data_from_files():
    market_data, anomalies_data = [], []
    # Use absolute paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    market_file = project_root / "data" / "output" / "market_data.jsonl"
    alerts_file = project_root / "data" / "output" / "alerts.jsonl"
    
    if market_file.exists():
        with open(market_file, 'r') as f:
            for line in f.readlines()[-100:]:  # Last 100 entries
                try: 
                    market_data.append(json.loads(line.strip()))
                except: continue
    
    if alerts_file.exists():
        with open(alerts_file, 'r') as f:
            for line in f.readlines()[-50:]:  # Last 50 entries
                try: 
                    data = json.loads(line.strip())
                    # Transform the alert data to match expected format
                    if data.get('alert_sent') and data.get('is_anomaly'):
                        alert_item = {
                            'symbol': data.get('symbol', 'Unknown'),
                            'anomaly_score': data.get('anomaly_score', 0),
                            'timestamp': data.get('timestamp', ''),
                            'priority': 'HIGH' if data.get('anomaly_score', 0) > 4 else 'MEDIUM' if data.get('anomaly_score', 0) > 2 else 'LOW',
                            'message': data.get('ai_analysis', 'Alert generated'),
                            'explanation': data.get('explanation', ''),
                            'price': data.get('price', 0),
                            'risk_assessment': data.get('risk_assessment', 'LOW')
                        }
                        anomalies_data.append(alert_item)
                except: continue
    return market_data, anomalies_data

def fetch_api_data(endpoint: str) -> Dict:
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return {"error": "API not available", "data": []}

def query_knowledge_base(query: str, symbol: str = None) -> Dict:
    try:
        payload = {"query": query}
        if symbol: payload["symbol"] = symbol
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"Unable to process query: {str(e)}", "context": []}

def create_price_chart(market_data: List[Dict]) -> go.Figure:
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
    
    # Add price traces with color coding
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
    if not anomalies_data:
        fig = go.Figure()
        fig.add_annotation(text="No anomalies detected", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    symbols, scores, timestamps, colors, messages = [], [], [], [], []
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
        
        messages.append(item.get('explanation', item.get('message', 'Anomaly detected')))
        
        # Color based on severity (matching priority levels)
        if score > 4:
            colors.append('red')    # HIGH priority
        elif score > 2:
            colors.append('orange') # MEDIUM priority  
        else:
            colors.append('green')  # LOW priority
    
    fig = go.Figure(data=go.Scatter(
        x=timestamps,
        y=scores,
        mode='markers',
        marker=dict(size=12, color=colors, opacity=0.8, line=dict(width=1, color='white')),
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

def display_alerts(alerts_data: List[Dict]):
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
        price = alert.get('price', 0)
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%H:%M:%S")
        except:
            time_str = timestamp[-8:] if len(timestamp) > 8 else timestamp
        
        # Create alert container
        with st.container():
            if priority == 'HIGH':
                st.error(f"🚨 **{priority}** - {symbol}: Anomaly Score {anomaly_score:.2f}")
            elif priority == 'MEDIUM':
                st.warning(f"⚠️ **{priority}** - {symbol}: Anomaly Score {anomaly_score:.2f}")
            else:
                st.info(f"ℹ️ **{priority}** - {symbol}: Anomaly Score {anomaly_score:.2f}")
            
            # Additional details
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"⏰ Time: {time_str}")
                if price > 0:
                    st.caption(f"💰 Price: ${price:.2f}")
            with col2:
                st.caption(f"📊 Risk: {alert.get('risk_assessment', 'MEDIUM')}")
            
            if explanation and explanation != message:
                st.caption(f"💡 {explanation}")
            
            st.divider()

def main():
    st.markdown('<div class="main-header"><h1>🚀 Real-Time Market Anomaly Detector</h1><p>AI-Powered Financial Market Monitoring</p></div>', unsafe_allow_html=True)
    st.sidebar.title("🔧 Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
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
        except: st.sidebar.error("API not available")
    col1, col2, col3, col4 = st.columns(4)
    try:
        market_response = fetch_api_data("/market-data")
        alerts_response = fetch_api_data("/alerts")
        if "error" in market_response:
            market_data, anomalies_data = load_data_from_files()
            alerts_data = anomalies_data
        else:
            market_data = market_response.get('data', [])
            alerts_data = alerts_response.get('alerts', [])
            anomalies_data = alerts_data
        with col1:
            st.markdown('<div class="metric-card"><span style="font-size:2em;">📊</span><br><b>Symbols Monitored</b><br>' + str(len(set(item.get('symbol', '') for item in market_data))) + '</div>', unsafe_allow_html=True)
        with col2:
            active_alerts = len([a for a in alerts_data if a.get('priority') in ['HIGH', 'MEDIUM']])
            st.markdown('<div class="metric-card"><span style="font-size:2em;">🚨</span><br><b>Active Alerts</b><br>' + str(active_alerts) + '</div>', unsafe_allow_html=True)
        with col3:
            total_anomalies = len(anomalies_data)
            st.markdown('<div class="metric-card"><span style="font-size:2em;">🔍</span><br><b>Anomalies Detected</b><br>' + str(total_anomalies) + '</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><span style="font-size:2em;">⏰</span><br><b>Last Update</b><br>' + datetime.now().strftime("%H:%M:%S") + '</div>', unsafe_allow_html=True)
        st.plotly_chart(create_price_chart(market_data), use_container_width=True)
        st.plotly_chart(create_anomaly_chart(anomalies_data), use_container_width=True)
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("🚨 Recent Alerts")
            display_alerts(alerts_data)
        with col_right:
            st.subheader("🤖 AI Assistant")
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
                            if result.get('context'):
                                with st.expander("📄 Sources Used"):
                                    for ctx in result['context'][:3]:
                                        st.write(f"• {ctx.get('content', '')[:200]}...")
                        else:
                            st.error(f"❌ {result['error']}")
        if st.checkbox("Show Raw Data"):
            st.subheader("📋 Market Data")
            if market_data:
                df = pd.DataFrame(market_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No market data available")
        if auto_refresh:
            time.sleep(REFRESH_INTERVAL)
            st.rerun()
    except Exception as e:
        st.error(f"❌ Dashboard error: {str(e)}")
        st.info("💡 Make sure the system is running with `./start_system.sh`")

if __name__ == "__main__":
    main()

st.markdown("""
---
<div style='text-align:center;'>
    <small>Made for Pathway LiveAI Hackathon | <a href='https://github.com/your-repo' target='_blank'>GitHub</a> | <a href='https://pathway.com' target='_blank'>Pathway</a></small>
</div>
""", unsafe_allow_html=True)
