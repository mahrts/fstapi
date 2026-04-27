"""This script fetch an example of get response from deployed app."""
import requests

url = "https://salaryrangecensusprediction.onrender.com/predict"

data = {"age": [20, 34],
        "workclass": ["Private", "Private"], 
        "fnlwgt": [162282, 195860], 
        "education": ["Some-college", "HS-grad"], 
        "education-num": [10, 9],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Machine-op-inspct", "Craft-repair"],
        "relationship": ["Own-child", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [60, 40],
        "native-country": ["United-States", "United-States"]}

if __name__ == "__main__":
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
