import json

import requests

r = requests.get("http://127.0.0.1:8000")

print("GET request status code:", r.status_code)

print("Welcome message:", r.json())

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

r = requests.post("http://127.0.0.1:8000/data/", json=data)

print("POST request status code:", r.status_code)

print("Prediction result:", r.json())