
import numpy as np
import requests

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "data/processed/test_cf_9.npz"

# Load a real FI-2010 test sequence
data = np.load(DATA_PATH)
X = data["X"]

sequence = X[:100].astype(np.float32)

print("Input sequence shape:", sequence.shape)

# Send the sequence to the API
response = requests.post(
    API_URL,
    json={"sequence": sequence.tolist()},
    timeout=30,
)

print("\nHTTP status:", response.status_code)
print("API response:")

print(response.text)

response.raise_for_status()

result = response.json()

print("\nPredicted direction:", result["prediction"])
print("Class ID:", result["class_id"])
print("Probabilities:")

for direction, probability in result["probabilities"].items():
    print(f"{direction}: {probability:.4f}")
