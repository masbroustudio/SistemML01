import requests
import json

url = "http://localhost:5001/predict"
data = {"wrong_key": [1, 2, 3]}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
