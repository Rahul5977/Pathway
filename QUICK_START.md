# 🚀 Quick Start Guide

_Get the Market Anomaly Detector running in 5 minutes_

## 📋 Prerequisites

- Python 3.8+ installed
- OpenAI API key (required)
- 4GB RAM minimum
- Internet connection for API calls

## ⚡ 5-Minute Setup

### Step 1: Clone and Setup (1 minute)

```bash
# Navigate to project directory
cd /Users/rahulraj/Desktop/Pathway

# Copy environment template
cp .env.example .env
```

### Step 2: Configure API Keys (1 minute)

Edit the `.env` file and add your API keys:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for enhanced functionality)
GOOGLE_API_KEY=your_google_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_news_api_key_here
```

### Step 3: Install Dependencies (2 minutes)

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 4: Start the System (1 minute)

```bash
# One-command startup
./start_system.sh
```

### Step 5: Access the Dashboard

- **Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

## 🎯 For Demo/Hackathon Judges

### Quick Demo Commands

```bash
# Start the demo scenarios (in new terminal)
python demo.py

# This will show:
# 1. Normal market operation
# 2. High volatility period
# 3. Breaking news impact
# 4. Crypto flash crash
# 5. Earnings surprise
```

### What to Look For

1. **Real-time updates** - timestamps changing continuously
2. **Anomaly detection** - alerts appearing during demo
3. **AI explanations** - ask questions in the dashboard
4. **Live data flow** - no manual refreshes needed

## 🔧 System Components

When you run `./start_system.sh`, these components start:

1. **Pathway Pipeline** (Port: Background)

   - Real-time data ingestion
   - Anomaly detection algorithms
   - Live vector indexing

2. **FastAPI Server** (Port: 8000)

   - REST API endpoints
   - RAG query interface
   - System monitoring

3. **Streamlit Dashboard** (Port: 8501)
   - Interactive visualizations
   - Real-time alerts
   - AI assistant interface

## 📊 Key Features to Demonstrate

### Real-Time Anomaly Detection

- Watch for alerts in the dashboard
- Notice different priority levels (high/medium/low)
- Observe immediate response to market movements

### AI-Powered Explanations

- Use the "Ask AI Assistant" feature
- Try queries like:
  - "Why is AAPL volatile today?"
  - "Explain the recent crypto movements"
  - "What's causing the market anomalies?"

### Live Data Processing

- Observe timestamps updating in real-time
- Watch price movements and volume changes
- See anomaly scores calculated live

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use**

```bash
# Kill existing processes
pkill -f "streamlit"
pkill -f "uvicorn"
pkill -f "python src/main.py"
```

**Missing API Key**

- Ensure `OPENAI_API_KEY` is set in `.env`
- Restart the system after adding keys

**Dependencies Missing**

```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

**No Data Appearing**

- Check that demo script is running: `python demo.py`
- Verify files are being created in `./data/input/`

### System Status

Check component status at: http://localhost:8000/health

## 🎬 Demo Script

For hackathon presentation, follow this sequence:

1. **Show system starting** (30 seconds)
2. **Open dashboard** - point out real-time data (1 minute)
3. **Run demo scenarios** - `python demo.py` (2 minutes)
4. **Show AI assistant** - ask about anomalies (1 minute)
5. **Highlight API** - show http://localhost:8000/docs (30 seconds)

## 📁 Project Structure

```
├── src/
│   ├── main.py              # Main Pathway pipeline
│   ├── config/settings.py   # Configuration management
│   ├── data/                # Data connectors and processors
│   ├── ai/                  # RAG engine and AI agents
│   ├── api/                 # FastAPI server
│   └── dashboard/           # Streamlit interface
├── demo.py                  # Demo scenario generator
├── start_system.sh          # One-command startup
├── requirements.txt         # Dependencies
└── .env                     # Configuration (create from .env.example)
```

## 🏆 Hackathon Compliance

✅ **All Requirements Met:**

- Pathway-powered streaming ETL
- Dynamic indexing (no rebuilds)
- Live retrieval/generation interface
- Demo video ready
- REST API exposed

✅ **Bonus Features:**

- Multi-agent AI workflows
- Real-time RAG with context
- Interactive dashboard
- Professional presentation quality

---

**Ready to win the hackathon!** 🚀

For questions or support during evaluation, all documentation is included in the project files.
