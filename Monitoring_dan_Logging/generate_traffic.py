import requests
import time
import random

url = "http://localhost:5001/predict"

print(f"Starting traffic generator to {url}...")
print("Press Ctrl+C to stop.")

try:
    while True:
        # Dummy features sesuai 7.inference.py (5 fitur)
        features = [random.random() for _ in range(5)]
        payload = {"features": features}
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print(f"Request sent: {response.json()}")
            else:
                print(f"Failed: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
            
        time.sleep(random.uniform(0.5, 2.0))
except KeyboardInterrupt:
    print("\nStopped.")
