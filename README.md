This repo contains code that predict salary range for census income data, which is
publicly available at https://archive.ics.uci.edu/dataset/20/census+income.

The code trains sklearn randomforestclassifier, and deploys with FastAPI.

## Usage:
The app is deployed on render, and can be accessed at: [Live App](https://salaryrangecensusprediction.onrender.com/docs)
The live app can be used to dispay  testing score, to compute testing score on data slice, to do prediction on new data.

For the prediction part, one can for example paste the following input and get prediction back for 
two new observed data:
```bash
{"age": [20, 34],
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
        "native-country": ["United-States", "United-States"]
}
```

This project can also be runed locally as follow.

1- Clone the repo.

```bash
git clone <the repo>
```

2- Move to fstapi folder
```bash
cd fstapi
```

3- Install dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e .
```

4- Tranin the model

```bash
python src/train_model.py
```
This will create a folder 'deployment', where necessary file deployment and testing are saved.

5- Finally, run the following and interact with full web interface 
(usually at http://127.0.0.1:8000). All usefull info to interact with the deployed model
can be found

```bash
uvicorn src.API.endpoints:app --reload
```
