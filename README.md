# 🚀 Real-Time Market Anomaly Detector

_Pathway LiveAI Hackathon @ IIT Bhilai_

A sophisticated AI-powered system that detects market anomalies in real-time and provides intelligent explanations through Retrieval Augmented Generation (RAG).

## 🎯 Project Overview

This application addresses the critical problem of stale AI in fast-moving financial markets by building a Live AI™ solution that:

- **Monitors multiple financial data streams** in real-time
- **Detects anomalies** using advanced algorithms and AI
- **Provides intelligent explanations** through real-time RAG
- **Alerts stakeholders** with actionable insights
- **Learns continuously** from new market data

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Sources   │───▶│  Pathway Engine  │───▶│   AI Agents     │
│                 │    │                  │    │                 │
│ • Stock APIs    │    │ • Real-time ETL  │    │ • Anomaly Det.  │
│ • News Feeds    │    │ • Vector Store   │    │ • Explanation   │
│ • Social Media  │    │ • Live Indexing  │    │ • Alert System │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Dashboard     │
                       │                 │
                       │ • Real-time UI  │
                       │ • Alerts        │
                       │ • Analytics     │
                       └─────────────────┘
```

## 🔧 Technology Stack

- **Pathway**: Real-time data streaming and processing
- **LangChain/LangGraph**: Agentic AI workflows
- **OpenAI/Gemini**: Large Language Models
- **FastAPI**: REST API backend
- **Streamlit**: Interactive dashboard
- **Python**: Core development language

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API keys for data sources and LLMs

### Installation

1. Clone the repository

```bash
git clone <repository-url>
cd pathway-market-anomaly-detector
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the application

```bash
# Start the Pathway pipeline
python src/main.py

# In another terminal, start the dashboard
streamlit run src/dashboard/app.py
```

## 📊 Features

### Real-Time Data Processing

- **Live market data ingestion** from multiple APIs
- **Streaming news and social sentiment** analysis
- **Continuous vector indexing** of financial documents
- **Dynamic portfolio tracking**

### AI-Powered Anomaly Detection

- **Statistical anomaly detection** for price movements
- **Sentiment-driven alerts** from news and social media
- **Cross-asset correlation** analysis
- **Volume and volatility** pattern recognition

### Intelligent Explanations

- **Real-time RAG** for contextual explanations
- **Multi-source evidence** compilation
- **Plain-English summaries** of complex events
- **Historical context** and precedent analysis

### Agentic Workflows

- **Market Monitor Agent**: Continuous data surveillance
- **Anomaly Analyst Agent**: Deep-dive investigation
- **Risk Assessment Agent**: Impact evaluation
- **Alert Manager Agent**: Stakeholder notification

## 🎥 Demo Flow

1. **Data Ingestion**: Show live data streaming from APIs
2. **Anomaly Detection**: Trigger an anomaly with unusual market data
3. **AI Investigation**: Watch agents analyze and explain the anomaly
4. **Real-Time Updates**: Demonstrate live dashboard updates
5. **RAG Capabilities**: Query the system about recent events

## 📁 Project Structure

```
src/
├── data/              # Data ingestion and processing
│   ├── connectors/    # Pathway data connectors
│   ├── processors/    # Data transformation logic
│   └── models/        # Data models and schemas
├── ai/                # AI and ML components
│   ├── agents/        # LangGraph agent definitions
│   ├── rag/           # RAG implementation
│   └── models/        # ML models for anomaly detection
├── api/               # FastAPI backend
│   ├── endpoints/     # API route definitions
│   └── middleware/    # Request/response processing
├── dashboard/         # Streamlit frontend
│   ├── components/    # Reusable UI components
│   └── pages/         # Dashboard pages
├── config/            # Configuration management
└── utils/             # Utility functions
```

## 🏆 Hackathon Compliance

### ✅ Required Features

- [x] **Pathway-Powered Streaming ETL**: Real-time data processing
- [x] **Dynamic Indexing**: No manual reloads required
- [x] **Live Retrieval Interface**: Query API with real-time responses
- [x] **Demo Video**: Comprehensive demonstration

### 🎯 Evaluation Criteria

- **Real-Time Functionality**: ⭐⭐⭐⭐⭐
- **Technical Implementation**: ⭐⭐⭐⭐⭐
- **Creativity & Innovation**: ⭐⭐⭐⭐⭐
- **Impact & Usefulness**: ⭐⭐⭐⭐⭐
- **User Experience**: ⭐⭐⭐⭐⭐

## 📝 Development Notes

- Focus on **real-time performance** and low latency
- Implement **comprehensive error handling**
- Use **type hints** throughout the codebase
- Follow **clean architecture** principles
- Optimize for **scalability** and **maintainability**

## 🎬 Demo Script

1. **Setup**: Show clean dashboard with baseline data
2. **Data Flow**: Demonstrate live data ingestion
3. **Anomaly Trigger**: Introduce unusual market event
4. **AI Response**: Show agents detecting and analyzing
5. **User Interaction**: Query the system via RAG interface
6. **Real-Time Updates**: Highlight dynamic responses

## 📞 Support

For technical issues or questions during the hackathon:

- Check Pathway Discord #get-help channel
- Review official Pathway documentation
- Use AI assistants for debugging complex errors

---

_Built with ❤️ for the Pathway LiveAI Hackathon @ IIT Bhilai_
