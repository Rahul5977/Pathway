# 🎬 Demo Video Script

_For Pathway LiveAI Hackathon Submission_

## 📝 Video Structure (Target: 3-5 minutes)

### **Opening (30 seconds)**

**[Screen: Show title slide with project name]**

**Narrator:** "Financial markets move in milliseconds, but most AI tools are hours behind. I'm excited to show you our Real-Time Market Anomaly Detector - a system that detects market anomalies and explains them using AI, all in real-time."

**[Screen: Show architecture diagram]**

**Narrator:** "Built with Pathway for streaming data processing, this system combines real-time RAG, AI agents, and live anomaly detection to solve the $500 billion problem of stale AI in finance."

### **System Overview (45 seconds)**

**[Screen: Show code in VS Code - main.py]**

**Narrator:** "Let me show you how we've implemented this using Pathway. Here's our main pipeline - we're ingesting live market data, news feeds, and social sentiment in real-time."

**[Screen: Point to Pathway streaming code]**

**Narrator:** "Notice how Pathway handles multiple data streams simultaneously. As new data arrives, it's immediately processed for anomalies and indexed for our RAG system - no manual rebuilds required."

**[Screen: Show terminal starting the system]**

**Narrator:** "Let's start the system with our one-command deployment."

```bash
./start_system.sh
```

### **Live Demo - Normal Operation (30 seconds)**

**[Screen: Open dashboard at localhost:8501]**

**Narrator:** "Here's our real-time dashboard. You can see live market data flowing in - notice the timestamps updating continuously. We're monitoring multiple assets including stocks and cryptocurrencies."

**[Screen: Point to real-time data streams]**

**Narrator:** "Each data point is processed immediately by our anomaly detection algorithms - we use statistical analysis, machine learning, and rule-based detection in combination."

### **Anomaly Detection in Action (60 seconds)**

**[Screen: Run demo script]**

**Narrator:** "Now let me trigger our demo scenarios to show anomaly detection in action."

```bash
python demo.py
```

**[Screen: Show volatility scenario starting]**

**Narrator:** "I'm simulating a high volatility period - watch as our system immediately detects these anomalies. See how the anomaly scores spike and alerts are generated automatically."

**[Screen: Point to alerts appearing in real-time]**

**Narrator:** "The alerts show up instantly in our dashboard. Notice the different priority levels - high priority for severe anomalies, medium for significant movements."

**[Screen: Show breaking news scenario]**

**Narrator:** "Now I'm simulating breaking news. Watch how our AI agents correlate the news sentiment with market movements and provide intelligent explanations."

### **AI Intelligence Demo (45 seconds)**

**[Screen: Open RAG interface in dashboard]**

**Narrator:** "Here's where our real-time RAG shines. Let me ask the AI assistant about recent market activity."

**[Screen: Type query: "Why is AAPL showing high volatility?"]**

**Narrator:** "I'm asking about Apple's volatility. The system is using our live vector database to find relevant context from recent news, market data, and previous analyses."

**[Screen: Show AI response with sources]**

**Narrator:** "Look at this response - it's not just generic AI output. It's referencing specific market data, news sentiment, and providing contextual explanations based on real-time information. And see these sources - it shows exactly what data informed this analysis."

### **Technical Excellence (30 seconds)**

**[Screen: Show API documentation at localhost:8000/docs]**

**Narrator:** "Our system also exposes a complete REST API for enterprise integration. This isn't just a demo - it's production-ready infrastructure."

**[Screen: Show code architecture]**

**Narrator:** "The architecture is modular and scalable. We have separate components for data ingestion, AI processing, and user interfaces, all orchestrated through Pathway's streaming engine."

### **Flash Crash Demo (30 seconds)**

**[Screen: Show crypto flash crash scenario]**

**Narrator:** "Let me show you one more scenario - a cryptocurrency flash crash. Watch how our system responds to extreme market movements."

**[Screen: Show rapid price declines and immediate alerts]**

**Narrator:** "You can see the system immediately detects the anomaly, generates high-priority alerts, and provides AI explanations for what's happening. This is exactly the kind of rapid response that traders and risk managers need."

### **Closing (30 seconds)**

**[Screen: Show multiple components running]**

**Narrator:** "In summary, we've built a complete real-time AI system that demonstrates all the hackathon requirements: Pathway-powered streaming, dynamic indexing, live RAG responses, and intelligent anomaly detection."

**[Screen: Show results dashboard]**

**Narrator:** "This system could be deployed in financial institutions tomorrow to help traders, analysts, and risk managers make better decisions with always-current AI insights. The future of financial AI is real-time, intelligent, and explainable - and we've built it today."

**[Screen: End with project logo and GitHub link]**

---

## 🎥 Recording Instructions

### **Pre-Recording Setup**

1. **Clean desktop** - close unnecessary applications
2. **Prepare browser tabs**:
   - Dashboard (localhost:8501)
   - API docs (localhost:8000/docs)
   - VS Code with project open
3. **Start system** and verify everything is running
4. **Prepare demo data** by running a quick test

### **Recording Settings**

- **Resolution**: 1920x1080 (Full HD)
- **Frame rate**: 30 fps
- **Audio**: Clear microphone, minimal background noise
- **Screen recording**: Use OBS or similar professional tool

### **During Recording**

- **Speak clearly** and at moderate pace
- **Use cursor highlighting** to point out important elements
- **Show timestamps** to prove real-time functionality
- **Demonstrate live updates** by waiting for data refresh
- **Highlight Pathway code** to show technical implementation

### **Key Visual Elements to Capture**

1. **Real-time timestamps** changing in dashboard
2. **Live data flowing** through the system
3. **Anomaly alerts** appearing immediately
4. **AI responses** with source attribution
5. **Code structure** showing Pathway usage
6. **Multiple scenarios** demonstrating versatility

### **Post-Recording**

- **Edit for clarity** - remove any delays or errors
- **Add captions** for key technical terms
- **Highlight important UI elements** with annotations
- **Keep final video under 5 minutes**
- **Export in high quality** for submission

---

## 🎯 Demo Success Checklist

### **Technical Demonstration**

- [ ] System starts successfully with one command
- [ ] Real-time data updates are visible
- [ ] Anomaly detection triggers correctly
- [ ] AI explanations are contextual and accurate
- [ ] Dashboard updates without manual refresh
- [ ] API endpoints respond correctly

### **Pathway Integration**

- [ ] Streaming data pipeline is shown
- [ ] Live indexing demonstrates no rebuilds
- [ ] Multiple data sources are integrated
- [ ] Real-time processing is evident

### **AI Capabilities**

- [ ] RAG provides relevant, contextual answers
- [ ] AI agents analyze anomalies intelligently
- [ ] Explanations reference real data sources
- [ ] Multi-agent workflows are demonstrated

### **User Experience**

- [ ] Dashboard is visually appealing
- [ ] Real-time updates are smooth
- [ ] Alerts are clear and actionable
- [ ] Interface is intuitive and responsive

This demo script ensures we showcase all the winning elements of our project while staying within the time limit and maintaining judge engagement throughout.
