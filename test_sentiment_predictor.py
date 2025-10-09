"""
Test script for the Predictive Sentiment AI Agent
"""

import asyncio
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai.sentiment_predictor import sentiment_agent

async def test_sentiment_prediction():
    """Test the sentiment prediction functionality"""
    print("🧠 Testing Predictive Sentiment AI Agent")
    print("=" * 50)
    
    test_symbols = ['AAPL', 'TSLA']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        
        try:
            # Update sentiment history
            print("  📈 Updating sentiment data...")
            await sentiment_agent.update_sentiment_history(symbol)
            
            # Get sentiment summary
            print("  📋 Getting sentiment summary...")
            summary = await sentiment_agent.get_sentiment_summary(symbol)
            
            if 'error' not in summary:
                print(f"  ✅ Current sentiment: {summary['current_sentiment']['overall']:.2f}")
                print(f"  📈 Sentiment trend: {summary['sentiment_trend']:.2f}")
                print(f"  ⚡ Momentum: {summary['sentiment_momentum']:.2f}")
            
            # Generate prediction
            print("  🔮 Generating prediction...")
            prediction = await sentiment_agent.predict_sentiment_trajectory(symbol, 15)
            
            if prediction:
                print(f"  🎯 Predicted sentiment: {prediction.predicted_sentiment:.2f}")
                print(f"  💰 Price impact: {prediction.price_impact_prediction:.2%}")
                print(f"  🎪 Confidence: {prediction.confidence_score:.1%}")
                print(f"  📝 Factors: {', '.join(prediction.contributing_factors[:2])}")
            
            # Check for alerts
            print("  🚨 Checking alerts...")
            alerts = await sentiment_agent.generate_sentiment_alerts(symbol)
            
            if alerts:
                for alert in alerts:
                    print(f"  🔔 {alert['priority']}: {alert['message']}")
            else:
                print("  ✅ No sentiment alerts")
        
        except Exception as e:
            print(f"  ❌ Error testing {symbol}: {str(e)}")
    
    print("\n🎉 Sentiment prediction test completed!")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the test
    asyncio.run(test_sentiment_prediction())
