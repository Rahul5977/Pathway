#!/bin/bash

# Start the Market Anomaly Detector System
# This script starts all components of the real-time market anomaly detection system

echo "🚀 Starting Market Anomaly Detector System"
echo "=========================================="

# Check if Python environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected. Consider using a virtual environment."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/input
mkdir -p data/input/news
mkdir -p data/input/sentiment
mkdir -p data/output
mkdir -p data/output/alerts
mkdir -p data/persistence
mkdir -p logs

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your API keys before continuing."
    echo "   Required: OPENAI_API_KEY"
    echo "   Optional: GOOGLE_API_KEY, ALPHA_VANTAGE_API_KEY, NEWS_API_KEY"
    read -p "Press Enter to continue after editing .env file..."
fi

# Install dependencies if not already installed
echo "📦 Checking dependencies..."
/Users/rahulraj/Desktop/Pathway/.venv/bin/python -c "import pathway" 2>/dev/null || {
    echo "Installing dependencies..."
    /Users/rahulraj/Desktop/Pathway/.venv/bin/pip install -r requirements.txt
}

# Start the main Pathway pipeline in background
echo "🔄 Starting Pathway pipeline..."
/Users/rahulraj/Desktop/Pathway/.venv/bin/python src/main.py &
PATHWAY_PID=$!
echo "   Pathway PID: $PATHWAY_PID"

# Wait for the pipeline to initialize
sleep 5

# Start the FastAPI server in background
echo "🌐 Starting API server..."
cd src && /Users/rahulraj/Desktop/Pathway/.venv/bin/python -m api.server &
API_PID=$!
cd ..
echo "   API PID: $API_PID"

# Wait for API to start
sleep 3

# Start the Streamlit dashboard
echo "📊 Starting dashboard..."
echo "   Dashboard will be available at: http://localhost:8501"
echo "   API will be available at: http://localhost:8000"
echo ""
echo "🎯 System Components:"
echo "   • Real-time data ingestion and processing"
echo "   • AI-powered anomaly detection"
echo "   • Live RAG for intelligent explanations"
echo "   • Interactive dashboard with real-time updates"
echo ""
echo "📱 Demo Instructions:"
echo "   1. Open http://localhost:8501 in your browser"
echo "   2. Watch the real-time data streams"
echo "   3. Observe anomaly detection in action"
echo "   4. Try the AI assistant for explanations"
echo "   5. Monitor alerts and notifications"
echo ""

# Store PIDs for cleanup
echo "$PATHWAY_PID" > .pathway.pid
echo "$API_PID" > .api.pid

# Start Streamlit dashboard (this will block)
/Users/rahulraj/Desktop/Pathway/.venv/bin/streamlit run src/dashboard/app.py --server.port 8501

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down system..."
    
    if [ -f .pathway.pid ]; then
        PATHWAY_PID=$(cat .pathway.pid)
        kill $PATHWAY_PID 2>/dev/null
        rm .pathway.pid
        echo "   Stopped Pathway pipeline"
    fi
    
    if [ -f .api.pid ]; then
        API_PID=$(cat .api.pid)
        kill $API_PID 2>/dev/null
        rm .api.pid
        echo "   Stopped API server"
    fi
    
    echo "✅ System shutdown complete"
}

# Set up cleanup on script exit
trap cleanup EXIT INT TERM
