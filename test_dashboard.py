#!/usr/bin/env python3
"""Quick test script to verify dashboard data is updating."""

import sys
import os
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dashboard.app import load_data_from_files

def test_data_updates():
    """Test that data is updating in real-time."""
    print("🧪 Testing Dashboard Data Updates")
    print("=" * 50)
    
    # Load initial data
    print("📊 Loading initial data...")
    market_data, anomalies_data = load_data_from_files()
    
    if not market_data:
        print("❌ No market data found!")
        return False
    
    initial_count = len(market_data)
    initial_timestamp = market_data[-1]['timestamp'] if market_data else None
    
    print(f"✅ Found {initial_count} market data points")
    print(f"✅ Found {len(anomalies_data)} anomalies")
    print(f"📅 Latest data timestamp: {initial_timestamp}")
    
    print("\n⏳ Waiting 10 seconds for new data...")
    time.sleep(10)
    
    # Load data again
    print("📊 Loading updated data...")
    new_market_data, new_anomalies_data = load_data_from_files()
    
    new_count = len(new_market_data)
    new_timestamp = new_market_data[-1]['timestamp'] if new_market_data else None
    
    print(f"📈 Market data points: {initial_count} → {new_count}")
    print(f"🚨 Anomalies: {len(anomalies_data)} → {len(new_anomalies_data)}")
    print(f"📅 Latest timestamp: {new_timestamp}")
    
    # Check if data updated
    if new_count > initial_count or new_timestamp != initial_timestamp:
        print("\n✅ SUCCESS: Data is updating in real-time!")
        
        # Show some recent data
        print("\n📊 Recent Market Data:")
        for item in new_market_data[-3:]:
            print(f"  {item['symbol']}: ${item['price']:.2f} (Score: {item['anomaly_score']:.2f}) at {item['timestamp'][-8:]}")
        
        if new_anomalies_data:
            print("\n🚨 Recent Anomalies:")
            for item in new_anomalies_data[-3:]:
                print(f"  {item['symbol']}: Score {item['anomaly_score']:.2f} - {item['priority']} priority")
        
        return True
    else:
        print("\n⚠️  Data doesn't seem to be updating. Check if the Pathway pipeline is running.")
        return False

if __name__ == "__main__":
    success = test_data_updates()
    print(f"\n🎯 Test {'PASSED' if success else 'FAILED'}")
