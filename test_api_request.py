import requests
import json

url = "http://localhost:5001/predict"
data = {"features": [3, 0, 22.0, 1, 0, 7.25, 1, 0]}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
