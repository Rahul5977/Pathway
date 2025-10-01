#!/usr/bin/env python3
"""Test script to verify real-time chart updates are working."""

import sys
import os
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_realtime_updates():
    """Test that the dashboard is getting real-time data."""
    print("🧪 Testing Real-Time Chart Updates")
    print("=" * 50)
    
    from dashboard.app import load_data_from_files
    
    # Test 1: Initial data load
    print("📊 Test 1: Loading initial data...")
    market_data, anomalies_data = load_data_from_files()
    
    if not market_data:
        print("❌ No market data found!")
        return False
    
    initial_timestamp = market_data[-1]['timestamp']
    initial_count = len(market_data)
    
    print(f"✅ Loaded {initial_count} market data points")
    print(f"✅ Latest timestamp: {initial_timestamp}")
    
    # Test 2: Wait and check for updates
    print("\n⏳ Test 2: Waiting 8 seconds for new data...")
    time.sleep(8)
    
    market_data2, anomalies_data2 = load_data_from_files()
    new_timestamp = market_data2[-1]['timestamp']
    new_count = len(market_data2)
    
    print(f"📈 Data points: {initial_count} → {new_count}")
    print(f"🕒 Timestamp: {initial_timestamp[-8:]} → {new_timestamp[-8:]}")
    
    # Test 3: Verify charts would update
    if new_timestamp != initial_timestamp:
        print("\n✅ SUCCESS: Real-time data is flowing!")
        print("🎯 Charts should be updating with new data")
        
        # Show recent data sample
        recent_symbols = set(item['symbol'] for item in market_data2[-5:])
        print(f"📊 Recent symbols: {', '.join(recent_symbols)}")
        
        return True
    else:
        print("\n⚠️  Data timestamp hasn't changed")
        print("💡 Check if the Pathway pipeline is running")
        return False

if __name__ == "__main__":
    print("🚀 Real-Time Chart Update Test")
    print(f"🕒 Test started at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    success = test_realtime_updates()
    
    print(f"\n🎯 Test Result: {'PASSED ✅' if success else 'FAILED ❌'}")
    
    if success:
        print("\n📱 Dashboard Status:")
        print("• Charts should update automatically every 5 seconds")
        print("• New data points appear in real-time")
        print("• Timestamps advance continuously")
        print("• Auto-refresh countdown shows in sidebar")
    else:
        print("\n🔧 Troubleshooting:")
        print("• Check if Pathway pipeline is running")
        print("• Verify data files are being updated")
        print("• Restart system with ./start_system.sh")
